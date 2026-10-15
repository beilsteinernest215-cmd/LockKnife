from __future__ import annotations

import contextlib
import os
import pathlib
import stat
import sys
import time
import uuid
from collections.abc import Iterator
from typing import Any

_POSIX = os.name == "posix"


def _windows_api() -> tuple[Any, Any, Any]:
    if sys.platform != "win32":
        raise OSError("Windows handle access requires a Windows host")
    import ctypes
    from ctypes import wintypes

    class FileInfo(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("created", wintypes.FILETIME),
            ("accessed", wintypes.FILETIME),
            ("written", wintypes.FILETIME),
            ("volume", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD),
            ("index_low", wintypes.DWORD),
        ]

    class FileIdInfo(ctypes.Structure):
        _fields_ = [("volume", ctypes.c_ulonglong), ("identifier", ctypes.c_ubyte * 16)]

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    api.CreateFileW.restype = wintypes.HANDLE
    api.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(FileInfo)]
    api.GetFileInformationByHandle.restype = wintypes.BOOL
    api.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    api.GetFileInformationByHandleEx.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    return api, FileInfo, FileIdInfo


def _windows_details(handle: int) -> tuple[int, int, int, int]:
    if sys.platform != "win32":
        raise OSError("Windows handle access requires a Windows host")
    import ctypes

    api, info_type, id_type = _windows_api()
    info, identity = info_type(), id_type()
    if not api.GetFileInformationByHandle(handle, ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not api.GetFileInformationByHandleEx(
        handle, 18, ctypes.byref(identity), ctypes.sizeof(identity)
    ):
        raise OSError(
            "Secure workspace requires Windows file identity support; use a local NTFS workspace"
        )
    return (
        info.attributes,
        info.links,
        identity.volume,
        int.from_bytes(bytes(identity.identifier), "little"),
    )


def _windows_fd_details(fd: int) -> tuple[int, int, int, int]:
    if sys.platform != "win32":
        raise OSError("Windows handle access requires a Windows host")
    import msvcrt

    return _windows_details(msvcrt.get_osfhandle(fd))


def _windows_fd(path: pathlib.Path, *, directory: bool = False, create: bool = False) -> int:
    """Open without following reparse points; deny directory rename while held."""
    if sys.platform != "win32":
        raise OSError("Windows handle access requires a Windows host")
    import ctypes
    import msvcrt

    api, _, _ = _windows_api()
    access = 0x80 if directory else 0x80000000 | (0x40000000 if create else 0)
    handle = api.CreateFileW(
        str(path),
        access,
        3 if directory or create else 1,
        None,
        4 if create else 3,
        0x00200000 | (0x02000000 if directory else 0),
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        attributes, _, _, _ = _windows_details(handle)
        if attributes & 0x400 or bool(attributes & 0x10) != directory:
            raise PermissionError("Access denied: reparse point or unexpected file type")
        fd = msvcrt.open_osfhandle(handle, os.O_BINARY | (os.O_RDWR if create else os.O_RDONLY))
    except BaseException:
        api.CloseHandle(handle)
        raise
    return fd


class WorkspaceFiles:
    """Case-relative I/O with pinned directory identities and no link traversal."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = pathlib.Path(root).absolute()
        self._pins: dict[tuple[str, ...], tuple[int, int]] = {}
        with self._directory(()) as (fd, _):
            self._pins[()] = self._identity(fd)

    @staticmethod
    def _identity(fd: int) -> tuple[int, int]:
        if not _POSIX:
            _, _, volume, identifier = _windows_fd_details(fd)
            return volume, identifier
        info = os.fstat(fd)
        return info.st_dev, info.st_ino

    def _parts(self, path: str | pathlib.Path) -> tuple[str, ...]:
        value = pathlib.PurePosixPath(path) if _POSIX else pathlib.PureWindowsPath(path)
        if value.anchor and not value.is_absolute():
            raise PermissionError("Access denied: drive-relative or rooted Windows path")
        if value.is_absolute():
            try:
                value = value.relative_to(self.root)
            except ValueError as exc:
                raise PermissionError("Access denied: path escapes case workspace") from exc
        parts: list[str] = []
        for part in value.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                if not parts:
                    raise PermissionError("Access denied: path escapes case workspace")
                parts.pop()
            else:
                if not _POSIX and (":" in part or part.endswith((".", " "))):
                    raise PermissionError("Access denied: ambiguous Windows path")
                parts.append(part)
        return tuple(parts)

    @contextlib.contextmanager
    def _directory(
        self, parts: tuple[str, ...], *, create: bool = False
    ) -> Iterator[tuple[int, pathlib.Path]]:
        with contextlib.ExitStack() as stack:
            path = pathlib.Path(self.root.anchor)
            if _POSIX:
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                fd = os.open(path, flags)
            else:
                fd = _windows_fd(path, directory=True)
            stack.callback(os.close, fd)
            root_parts = self.root.parts[1:]
            for index, part in enumerate((*root_parts, *parts)):
                path = path / part
                is_relative = index >= len(root_parts)
                if _POSIX:
                    if create and is_relative:
                        try:
                            os.mkdir(part, mode=0o700, dir_fd=fd)
                        except FileExistsError:
                            pass
                    fd = os.open(part, flags, dir_fd=fd)
                else:
                    if create and is_relative:
                        try:
                            path.mkdir(mode=0o700)
                        except FileExistsError:
                            pass
                    fd = _windows_fd(path, directory=True)
                stack.callback(os.close, fd)
                if index >= len(root_parts) - 1:
                    key = tuple(parts[: index - len(root_parts) + 1])
                    expected = self._pins.get(key)
                    if expected is not None and expected != self._identity(fd):
                        raise PermissionError("Access denied: workspace directory was replaced")
            yield fd, path

    def ensure_directory(self, path: str, *, pin: bool = False) -> None:
        parts = self._parts(path)
        with self._directory(parts, create=True) as (fd, _):
            if pin:
                self._pins[parts] = self._identity(fd)

    @staticmethod
    def _regular(fd: int) -> None:
        if not _POSIX:
            attributes, links, _, _ = _windows_fd_details(fd)
            if attributes & (0x400 | 0x10) or links != 1:
                raise PermissionError("Access denied: expected a single-link regular file")
            return
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PermissionError("Access denied: expected a single-link regular file")

    @contextlib.contextmanager
    def _file(self, path: str | pathlib.Path, *, create: bool = False) -> Iterator[int]:
        parts = self._parts(path)
        if not parts:
            raise PermissionError("Access denied: a file path is required")
        with self._directory(parts[:-1]) as (parent, directory):
            if _POSIX:
                flags = (os.O_RDWR if create else os.O_RDONLY) | os.O_NOFOLLOW | os.O_NONBLOCK
                for attempt in range(8):
                    try:
                        fd = os.open(parts[-1], flags, dir_fd=parent)
                        break
                    except FileNotFoundError:
                        if not create:
                            raise
                    try:
                        # Separate creation from reopening: concurrent Darwin openat
                        # calls with O_CREAT alone can spuriously return ENOENT.
                        fd = os.open(
                            parts[-1], flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent
                        )
                        break
                    except (FileExistsError, FileNotFoundError):
                        if attempt == 7:
                            raise
                        time.sleep(0.001)
            else:
                fd = _windows_fd(directory / parts[-1], create=create)
            try:
                self._regular(fd)
                yield fd
            finally:
                os.close(fd)

    def read_text(self, path: str | pathlib.Path) -> str:
        with self._file(path) as fd:
            with os.fdopen(os.dup(fd), "r", encoding="utf-8", errors="replace") as stream:
                text = stream.read()
            self._regular(fd)
            return text

    def write_text(self, path: str | pathlib.Path, text: str, *, exclusive: bool = False) -> None:
        parts = self._parts(path)
        if not parts:
            raise PermissionError("Access denied: a file path is required")
        with self._directory(parts[:-1], create=True) as (parent, directory):
            name = parts[-1]
            temporary = f".agent-write-{uuid.uuid4().hex}"
            target = temporary if not exclusive else name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if _POSIX:
                fd = os.open(target, flags | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            else:
                fd = os.open(directory / target, flags | os.O_BINARY, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                if not exclusive:
                    if _POSIX:
                        os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
                    else:
                        os.replace(directory / temporary, directory / name)
                if _POSIX:
                    os.fsync(parent)
            finally:
                if not exclusive:
                    if _POSIX:
                        try:
                            os.unlink(temporary, dir_fd=parent)
                        except FileNotFoundError:
                            pass
                    else:
                        (directory / temporary).unlink(missing_ok=True)

    def names(self, path: str) -> list[str]:
        with self._directory(self._parts(path)) as (fd, directory):
            return os.listdir(fd if _POSIX else directory)

    @contextlib.contextmanager
    def lock_file(self, path: str) -> Iterator[None]:
        with self._file(path, create=True) as fd:
            if _POSIX:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            else:
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
