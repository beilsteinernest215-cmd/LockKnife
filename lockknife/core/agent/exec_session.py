from __future__ import annotations

import codecs
import collections
import logging
import os
import select
import shlex
import signal
import subprocess  # nosec B404
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger("lockknife.agent.exec_session")
_POSIX = os.name == "posix"


class ExecSession:
    """A single stateful interactive process session with non-blocking I/O buffers."""

    def __init__(self, session_id: str, command: str, proc: subprocess.Popen) -> None:
        self.session_id = session_id
        self.command = command
        self.proc = proc
        self.created_at = time.time()
        self.last_active_at = time.time()
        self._output_buffer: collections.deque[str] = collections.deque(maxlen=2000)
        self._lock = threading.Lock()
        self._input_lock = threading.Lock()
        if _POSIX and self.proc.stdin is not None:
            os.set_blocking(self.proc.stdin.fileno(), False)
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

    def _read_stdout(self) -> None:
        try:
            stdout = self.proc.stdout
            if stdout is not None:
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                for chunk in iter(lambda: os.read(stdout.fileno(), 1024), b""):
                    line = decoder.decode(chunk)
                    with self._lock:
                        self._output_buffer.append(line)
                        self.last_active_at = time.time()
        except Exception as exc:
            logger.debug("ExecSession %s stdout closed: %s", self.session_id, exc)

    def write_input(self, text: str) -> bool:
        """Write stdin text to the running interactive process."""
        if self.proc.poll() is not None or not self.proc.stdin:
            return False
        try:
            if not text.endswith("\n"):
                text += "\n"
            data = text.encode("utf-8")
            if len(data) > 65536:
                return False
            if not _POSIX:
                return self._write_blocking_pipe(text)
            with self._input_lock:
                deadline = time.monotonic() + 1.0
                fd = self.proc.stdin.fileno()
                while data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                        return False
                    try:
                        data = data[os.write(fd, data) :]
                    except BlockingIOError:
                        continue
            self.last_active_at = time.time()
            return True
        except Exception as exc:
            logger.warning("Failed to write to ExecSession %s stdin: %s", self.session_id, exc)
            return False

    def _write_blocking_pipe(self, text: str) -> bool:
        """Windows pipes cannot be passed to select; bound the caller's wait instead."""
        stream = self.proc.stdin
        if stream is None or not self._input_lock.acquire(blocking=False):
            return False
        completed = threading.Event()
        success = []

        def write() -> None:
            try:
                stream.write(text)
                stream.flush()
                success.append(True)
            except (OSError, ValueError):
                pass
            finally:
                self._input_lock.release()
                completed.set()

        threading.Thread(target=write, daemon=True).start()
        return completed.wait(timeout=1.0) and bool(success)

    def _terminate(self, *, force: bool = False) -> None:
        if _POSIX:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif self.proc.poll() is None:
            if force:
                self.proc.kill()
            else:
                self.proc.terminate()

    def poll_output(self, wait_s: float = 0.5) -> dict[str, Any]:
        """Poll and drain buffered output lines since last read."""
        if wait_s > 0:
            time.sleep(min(wait_s, 5.0))

        lines: list[str] = []
        with self._lock:
            while self._output_buffer:
                lines.append(self._output_buffer.popleft())

        returncode = self.proc.poll()
        is_running = returncode is None

        return {
            "session_id": self.session_id,
            "output": "".join(lines),
            "is_running": is_running,
            "returncode": returncode,
            "elapsed_s": time.time() - self.created_at,
        }

    def close(self) -> None:
        """Terminate the running process cleanly."""
        self._terminate()
        if self.proc.poll() is None:
            try:
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._terminate(force=True)
                    self.proc.wait(timeout=1.0)
                except Exception:
                    pass
        # The group may outlive its original parent, including shell-launched children.
        self._terminate(force=True)
        for stream in (self.proc.stdin, self.proc.stdout):
            if stream is not None:
                stream.close()
        self._reader_thread.join(timeout=1.0)


class ExecSessionManager:
    """Manages active interactive device/shell sessions across agent turns."""

    def __init__(self) -> None:
        self._sessions: dict[str, ExecSession] = {}
        self._lock = threading.Lock()

    def start_session(
        self,
        command: str,
        session_id: str | None = None,
        use_shell: bool = False,
    ) -> dict[str, Any]:
        sid = session_id or f"sess_{str(uuid.uuid4())[:6]}"
        with self._lock:
            if sid in self._sessions:
                return {"ok": False, "error": f"Session {sid} already exists", "session_id": sid}
            if len(self._sessions) >= 32:
                return {"ok": False, "error": "Session limit reached", "session_id": sid}

            try:
                # Use shlex or list if not shell
                args = command if use_shell or not _POSIX else shlex.split(command)
                proc = subprocess.Popen(  # nosec B603 B602
                    args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    shell=use_shell,
                    start_new_session=_POSIX,
                )
                session = ExecSession(sid, command, proc)
                self._sessions[sid] = session
                logger.info("Started ExecSession %s", sid)
                return {"ok": True, "session_id": sid, "command": command}
            except Exception as exc:
                logger.error("Failed to start ExecSession %s (%s)", sid, type(exc).__name__)
                return {"ok": False, "error": str(exc), "session_id": sid}

    def write_session(self, session_id: str, input_text: str) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": f"Session {session_id} not found"}
        ok = sess.write_input(input_text)
        return {
            "ok": ok,
            "session_id": session_id,
            "error": None if ok else "Session closed, input exceeded 64 KiB, or write timed out",
        }

    def poll_session(self, session_id: str, wait_s: float = 0.5) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": f"Session {session_id} not found", "output": ""}
        result = sess.poll_output(wait_s=wait_s)
        result["ok"] = True
        return result

    def close_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.pop(session_id, None)
        if not sess:
            return {"ok": False, "error": f"Session {session_id} not found"}
        sess.close()
        logger.info("Closed ExecSession %s", session_id)
        return {"ok": True, "session_id": session_id}

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "session_id": sid,
                    "command": s.command,
                    "is_running": s.proc.poll() is None,
                    "elapsed_s": round(time.time() - s.created_at, 2),
                }
                for sid, s in self._sessions.items()
            ]

    def close_all(self) -> None:
        """Cleanup all running sessions."""
        with self._lock:
            for s in list(self._sessions.values()):
                s.close()
            self._sessions.clear()
