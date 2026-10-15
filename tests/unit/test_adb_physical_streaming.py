"""Unit tests for ADB TCP Physical Forensics Stream Extraction."""

from __future__ import annotations

import pathlib
import subprocess
from unittest.mock import patch

import pytest

from lockknife.modules.exploitation.adb_tcp.extractor import (
    DataExtractor,
    ExtractionStatus,
    ExtractionType,
)


def test_physical_extraction_streaming(tmp_path: pathlib.Path) -> None:
    """Test physical extraction streaming directly without staging."""
    extractor = DataExtractor(output_dir=tmp_path)
    payload = b"RAW_PARTITION_DATA" * 64

    def fake_stream_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if "id" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="uid=0(root) gid=0(root)")
        if cmd[-1].startswith("blockdev --getsize64 "):
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{len(payload)}\n")
        if "exec-out" in cmd:
            # Simulate streaming dd raw bytes to stdout file
            f_out = kwargs.get("stdout")
            if f_out and hasattr(f_out, "write"):
                f_out.write(payload)
            return subprocess.CompletedProcess(cmd, 0)
        pytest.fail(f"Unexpected command during direct streaming: {cmd}")

    with patch("subprocess.run", side_effect=fake_stream_run) as mock_run:
        extracted = extractor.extract("192.0.2.100:5555", ExtractionType.PHYSICAL)

        assert extracted.status == ExtractionStatus.COMPLETED
        assert extracted.total_files == 4  # system, data, cache, boot
        assert extracted.total_bytes == 4 * len(payload)
        assert all(path.read_bytes() == payload for path in tmp_path.rglob("*.img"))
        # Verify exec-out was called
        calls = [c[0][0] for c in mock_run.call_args_list]
        assert sum("exec-out" in cmd for cmd in calls) == 4
        assert sum(cmd[-1].startswith("blockdev --getsize64 ") for cmd in calls) == 4
        assert not any("rm -f" in " ".join(cmd) for cmd in calls)
        assert not list(tmp_path.rglob(".acquiring-*"))


def test_physical_extraction_fallback_cleanup(tmp_path: pathlib.Path) -> None:
    """Test physical extraction fallback staging cleans up remote files on device."""
    extractor = DataExtractor(output_dir=tmp_path)
    payload = b"STAGED_DATA"
    owned: set[str] = set()
    copied: set[str] = set()
    cleaned: set[str] = set()

    def fake_fallback_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if "id" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="uid=0(root) gid=0(root)")
        if cmd[-1].startswith("blockdev --getsize64 "):
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{len(payload)}\n")
        if "exec-out" in cmd:
            return subprocess.CompletedProcess(cmd, 1)  # exec-out unsupported
        action = cmd[-1]
        if action.startswith("mkdir -m 700 "):
            directory = action.split()[-1]
            assert directory.startswith("/data/local/tmp/lockknife-")
            assert directory not in owned
            owned.add(directory)
        elif action.startswith("dd if="):
            destination = next(part[3:] for part in action.split() if part.startswith("of="))
            assert str(pathlib.PurePosixPath(destination).parent) in owned
            copied.add(destination)
        elif action.startswith("rm -f "):
            _, _, remote, conjunction, rmdir, directory = action.split()
            assert remote in copied
            assert str(pathlib.PurePosixPath(remote).parent) == directory
            assert directory in owned
            assert conjunction == "&&" and rmdir == "rmdir"
            cleaned.add(directory)
        else:
            pytest.fail(f"Unexpected fallback command: {cmd}")
        return subprocess.CompletedProcess(cmd, 0)

    def fake_pull(target: str, remote: str, local: pathlib.Path) -> tuple[bool, pathlib.Path]:
        assert remote in copied
        assert str(pathlib.PurePosixPath(remote).parent) not in cleaned
        local.write_bytes(payload)
        return True, local

    with patch("subprocess.run", side_effect=fake_fallback_run) as mock_run:
        with patch.object(extractor, "pull_file", side_effect=fake_pull):
            extracted = extractor.extract("192.0.2.100:5555", ExtractionType.PHYSICAL)

            assert extracted.status == ExtractionStatus.COMPLETED
            assert extracted.total_files == 4
            assert extracted.total_bytes == 4 * len(payload)
            assert all(path.read_bytes() == payload for path in tmp_path.rglob("*.img"))
            assert len(owned) == 4 and cleaned == owned
            calls = [c[0][0] for c in mock_run.call_args_list]
            # Ensure cleanup rm -f was invoked for all partitions
            cleanup_calls = [cmd for cmd in calls if "rm -f" in " ".join(cmd)]
            assert len(cleanup_calls) == 4
            assert not any("/sdcard/" in " ".join(cmd) for cmd in calls)
            assert not list(tmp_path.rglob(".acquiring-*"))
