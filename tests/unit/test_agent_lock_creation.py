from __future__ import annotations

import os

import pytest

from lockknife.core.agent import _workspace_fs
from lockknife.core.agent.memory import MemoryStore


@pytest.mark.skipif(os.name != "posix", reason="Tests POSIX exclusive lock-file creation")
@pytest.mark.parametrize("race", ["another_creator", "transient_missing", "symlink"])
def test_lock_file_creation_retries_without_weakening_anchoring(monkeypatch, tmp_path, race):
    store = MemoryStore(tmp_path / "case")
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    real_open = os.open
    attempts = []
    injected = False

    def open_with_race(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal injected
        if path == "facts.lock":
            attempts.append((flags, dir_fd))
            assert flags & os.O_NOFOLLOW
            assert dir_fd is not None
            if flags & os.O_CREAT:
                assert flags & os.O_EXCL
                if not injected:
                    injected = True
                    if race == "another_creator":
                        fd = real_open(path, flags, mode, dir_fd=dir_fd)
                        os.close(fd)
                        raise FileExistsError("Concurrent creator won")
                    if race == "symlink":
                        os.symlink(outside, path, dir_fd=dir_fd)
                        raise FileExistsError("Untrusted link appeared")
                    raise FileNotFoundError("Transient concurrent creation miss")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(_workspace_fs.os, "open", open_with_race)
    if race == "symlink":
        with pytest.raises(OSError):
            store.set_fact("key", "value")
    else:
        store.set_fact("key", "value")
        assert store.get_facts() == {"key": "value"}
    assert injected
    assert len(attempts) >= 3
    assert len({fd for _, fd in attempts}) == 1
    assert outside.read_text() == "unchanged"
