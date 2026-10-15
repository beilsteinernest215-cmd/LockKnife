from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from lockknife.modules.exploitation.auto_exploit import AutoExploiter
from lockknife.modules.exploitation.bluetooth.gatt import GATTClient, GATTResult
from lockknife_headless_cli.tui_callback import build_tui_callback


def test_failed_wifi_operation_is_not_successful_dispatch(monkeypatch):
    from lockknife.modules.exploitation.wifi.wps import WpsAttacker, WpsAttackResult

    monkeypatch.setattr(
        WpsAttacker,
        "attack",
        lambda *a, **kw: WpsAttackResult(target="DUMMY", success=False, error="DUMMY failure"),
    )
    result = build_tui_callback(None)("exploit.run.wifi", {"target": "00:00:00:00:00:00"})
    assert result["ok"] is False
    assert "DUMMY failure" in result["error"]
    assert json.loads(result["data_json"])["success"] is False


def test_unknown_cve_never_falls_back_to_different_capability():
    result = build_tui_callback(None)(
        "exploit.run.zeroclick", {"target": "DUMMY", "cve": "CVE-DUMMY"}
    )
    assert not result["ok"]
    assert "Unsupported" in result["error"]


def test_bluetooth_invalid_method_does_not_silently_change_method():
    result = build_tui_callback(None)(
        "exploit.run.bluetooth", {"target": "DUMMY", "method": "invalid"}
    )
    assert not result["ok"]
    assert "pairing method" in result["error"]


def test_ble_discovers_and_disconnects_using_real_result_type(monkeypatch):
    connect = AsyncMock(return_value=GATTResult(success=True))
    discover = AsyncMock(return_value=[object(), object()])
    disconnect = AsyncMock(return_value=GATTResult(success=True))
    monkeypatch.setattr(GATTClient, "connect", connect)
    monkeypatch.setattr(GATTClient, "discover_services", discover)
    monkeypatch.setattr(GATTClient, "disconnect", disconnect)
    result = AutoExploiter()._try_ble("DUMMY", False)
    assert result.success
    assert result.metadata["services_count"] == 2
    disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_ble_sync_entrypoint_also_works_inside_event_loop(monkeypatch):
    monkeypatch.setattr(
        GATTClient,
        "connect",
        AsyncMock(return_value=GATTResult(success=False, error="DUMMY unavailable")),
    )
    result = AutoExploiter()._try_ble("DUMMY", False)
    assert not result.success
    assert result.error == "DUMMY unavailable"
