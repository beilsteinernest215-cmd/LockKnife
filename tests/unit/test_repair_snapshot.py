from __future__ import annotations

import json
import pathlib
import shlex

import pytest

from lockknife.modules.forensics.snapshot import create_snapshot
from lockknife_headless_cli.tui_callback import build_tui_callback


class Device:
    def __init__(self, *, fail_pull=False):
        self.commands = []
        self.remotes = []
        self.fail_pull = fail_pull

    def has_root(self, serial):
        return True

    def shell(self, serial, command, **kwargs):
        self.commands.append(command)
        return ""

    def pull(self, serial, remote, local, **kwargs):
        self.remotes.append(remote)
        if self.fail_pull:
            raise RuntimeError("DUMMY transfer failure")
        local.write_bytes(b"DUMMY snapshot")


@pytest.mark.parametrize("encrypt", [False, True])
def test_snapshot_callback_returns_actual_path_and_passes_selected_paths(
    tmp_path, monkeypatch, encrypt
):
    import lockknife_headless_cli.tui_callback as cb

    device = Device()
    monkeypatch.setattr(cb, "_devices", lambda app: device)
    output = tmp_path / "snapshot.tar"
    result = build_tui_callback(None)(
        "forensics.snapshot",
        {
            "serial": "DUMMY",
            "output": str(output),
            "paths": "/data/local/tmp/EXAMPLE",
            "encrypt": encrypt,
        },
    )
    assert result["ok"], result
    data = json.loads(result["data_json"])
    assert pathlib.Path(data["output"]).is_file()
    assert data["encrypted"] == encrypt
    assert "/data/local/tmp/EXAMPLE" in shlex.split(device.commands[0])[-1]
    if encrypt:
        assert data["output"].endswith(".lkenc")


def test_snapshot_quotes_nested_shell_and_cleans_up_failed_transfer(tmp_path):
    device = Device(fail_pull=True)
    path = "/data/local/tmp/EXAMPLE$(unused)"
    with pytest.raises(RuntimeError, match="transfer failure"):
        create_snapshot(device, "DUMMY", output_path=tmp_path / "out.tar", paths=[path])
    outer = shlex.split(device.commands[0])
    assert outer[:2] == ["su", "-c"]
    inner = shlex.split(outer[2])
    assert inner[-1] == path
    assert device.remotes[0] in shlex.split(device.commands[-1])[-1]
    assert not (tmp_path / "out.tar").exists()
