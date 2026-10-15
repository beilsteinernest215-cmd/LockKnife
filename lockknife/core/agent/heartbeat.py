from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from lockknife.core.adb import AdbClient, AdbDevice

logger = logging.getLogger("lockknife.agent.heartbeat")


class DeviceHeartbeatDaemon:
    """Proactive autonomy daemon watching for connected ADB devices and state changes."""

    def __init__(
        self,
        *,
        interval_s: float = 10.0,
        adb_client: AdbClient | None = None,
        on_device_attached: Callable[[AdbDevice], None] | None = None,
        on_device_detached: Callable[[str], None] | None = None,
    ) -> None:
        self.interval_s = max(1.0, interval_s)
        self.adb = adb_client or AdbClient()
        self.on_device_attached = on_device_attached
        self.on_device_detached = on_device_detached
        self._known_devices: dict[str, AdbDevice] = {}
        self._running = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_result: dict[str, Any] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()

    def status(self) -> dict[str, Any]:
        alive = self._thread is not None and self._thread.is_alive()
        return {
            "status": "stopping"
            if alive and self._stop_event.is_set()
            else "active"
            if alive
            else "stopped",
            "interval_s": self.interval_s,
            "last_result": self.last_result,
        }

    def tick(self) -> dict[str, Any]:
        """Execute one polling tick to check device state."""
        events: list[dict[str, Any]] = []
        try:
            current_devices = {d.serial: d for d in self.adb.list_devices()}
        except Exception as exc:
            logger.warning("Heartbeat ADB check failed: %s", exc)
            return {"ok": False, "error": str(exc), "events": []}

        with self._lock:
            # Check for new or updated devices
            for serial, device in current_devices.items():
                if serial not in self._known_devices:
                    logger.info(
                        "Proactive Heartbeat: Device attached: %s (%s)", serial, device.model
                    )
                    events.append({"event": "attached", "device": serial, "model": device.model})
                    if self.on_device_attached:
                        try:
                            self.on_device_attached(device)
                        except Exception as exc:
                            logger.error("Error in on_device_attached hook: %s", exc)

            # Check for disconnected devices
            for serial in list(self._known_devices.keys()):
                if serial not in current_devices:
                    logger.info("Proactive Heartbeat: Device detached: %s", serial)
                    events.append({"event": "detached", "device": serial})
                    if self.on_device_detached:
                        try:
                            self.on_device_detached(serial)
                        except Exception as exc:
                            logger.error("Error in on_device_detached hook: %s", exc)

            self._known_devices = current_devices

        return {"ok": True, "events": events, "active_count": len(current_devices)}

    def run_forever(self, max_ticks: int | None = None) -> None:
        """Run the polling loop until stopped or max_ticks reached."""
        self._running = True
        logger.info("Starting proactive heartbeat loop (interval: %ss)", self.interval_s)
        ticks = 0

        try:
            while not self._stop_event.is_set():
                self.last_result = self.tick()
                ticks += 1
                if max_ticks is not None and ticks >= max_ticks:
                    break
                self._stop_event.wait(self.interval_s)
        except KeyboardInterrupt:
            logger.info("Heartbeat loop stopped by operator.")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
