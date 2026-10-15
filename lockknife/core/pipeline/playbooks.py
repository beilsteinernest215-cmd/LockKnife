from __future__ import annotations

import json
import pathlib
from typing import Any

from lockknife.core.exceptions import LockKnifeError
from lockknife.core.pipeline.models import PlaybookDefinition, StepDefinition


def _step(
    action: str, label: str, *, deps: tuple[str, ...] = (), device: bool = False,
    params: dict[str, Any] | None = None, optional: bool = False,
) -> StepDefinition:
    return StepDefinition(
        step_id=action, label=label, action_id=action, category=action.split(".")[0],
        depends_on=deps, requires_device=device, params=params or {}, optional=optional,
    )


_DEVICE = _step("device.info", "Device Identification", device=True)
_POSTURE = tuple(
    _step(action, label, deps=("device.info",), device=True,
          params={"output": "${run_dir}/" + action + ".json"})
    for action, label in (
        ("security.audit", "Security Posture"),
        ("security.bootloader", "Verified Boot Audit"),
        ("security.hardware", "Hardware Security Assessment"),
        ("security.network_scan", "Network Exposure"),
    )
)
_EXTRACTIONS = tuple(
    _step(action, label, deps=("device.info",), device=True,
          params={"output": "${run_dir}/evidence/" + filename, "format": "json"})
    for action, label, filename in (
        ("extraction.sms", "SMS Extraction", "sms.json"),
        ("extraction.contacts", "Contacts Extraction", "contacts.json"),
        ("extraction.call_logs", "Call History Extraction", "call_logs.json"),
        ("extraction.browser", "Browser History Extraction", "browser.json"),
        ("extraction.location", "Location Extraction", "location.json"),
        ("extraction.messaging", "Messaging Extraction", "messaging.json"),
    )
)
_SNAPSHOT = _step(
    "forensics.snapshot", "Acquire Snapshot for File Carving", deps=("device.info",),
    device=True, params={"output": "${run_dir}/evidence/snapshot.tar", "full": True},
)
_CARVE = _step(
    "forensics.carve", "Signature-Based Snapshot File Carving",
    deps=("forensics.snapshot",), params={
        "path": {"$result": "forensics.snapshot", "path": "output"},
        "output_dir": "${run_dir}/carved", "source": "auto",
    },
)
_PARSE = _step(
    "forensics.parse", "Normalize Extracted Artifacts",
    deps=tuple(s.step_id for s in _EXTRACTIONS),
    params={"path": "${run_dir}/evidence", "output": "${run_dir}/parsed.json"},
)
_TIMELINE = _step(
    "forensics.timeline", "Correlate Extracted Timeline",
    deps=("forensics.parse",), params={
        **{key: {"$result": action, "path": "output"} for key, action in (
            ("sms", "extraction.sms"), ("calls", "extraction.call_logs"),
            ("browser", "extraction.browser"), ("messaging", "extraction.messaging"),
            ("location", "extraction.location"), ("parsed_artifacts", "forensics.parse"),
        )},
        "output": "${run_dir}/timeline.json",
    },
)
_KEYSTORE = _step("credentials.keystore", "Keystore Inventory", deps=("device.info",), device=True)
_PASSKEYS = _step("credentials.passkeys", "Passkey Inventory", deps=("device.info",), device=True)
_CRYPTO = _step(
    "crypto.scan_device", "Device Wallet Discovery", deps=("device.info",), device=True,
    params={"output": "${run_dir}/wallets.json"},
)
_ENRICH = _step(
    "case.enrich", "Enrich Case Evidence", deps=("forensics.parse",),
    params={"output": "${run_dir}/enrichment.json"},
)
_IOC = _step(
    "intelligence.ioc", "Detect Network Indicators", deps=("security.network_scan",),
    params={"input": {"$result": "security.network_scan", "path": "output"},
            "output": "${run_dir}/indicators.json"},
)


def _playbook(name: str, title: str, description: str, steps: tuple[StepDefinition, ...]) -> PlaybookDefinition:
    integrity = _step("report.integrity", "Verify Evidence Integrity", deps=tuple(s.step_id for s in steps))
    report = _step("report.generate", "Generate Investigation Report", deps=(integrity.step_id,),
                   params={"template": "technical", "format": "json", "output": "${run_dir}/report.json"})
    return PlaybookDefinition(name, title, description, "investigation", (*steps, integrity, report))


BUILTIN_PLAYBOOKS = {pb.name: pb for pb in (
    _playbook("triage", "Rapid Security & Device Triage", "Device posture, verified boot, hardware and network exposure.",
              (_DEVICE, *_POSTURE)),
    _playbook("deep-forensics", "Deep Forensic Extraction & Carving",
              "Extract records, carve snapshot file signatures, normalize and correlate a timeline.",
              (_DEVICE, *_EXTRACTIONS, _SNAPSHOT, _CARVE, _PARSE, _TIMELINE, _PASSKEYS, _ENRICH)),
    _playbook("incident-response", "Incident Response & Threat Hunt",
              "Device posture and attack-surface assessment; custom recipes can add supplied malware rules.",
              (_DEVICE, *_POSTURE, _IOC, _step("security.attack_surface", "Device Attack Surface",
                                        deps=("device.info",), device=True))),
    _playbook("crypto-audit", "Cryptocurrency Vault Forensics",
              "Discover device wallets and inventory keystore and passkey evidence.",
              (_DEVICE, _CRYPTO, _KEYSTORE, _PASSKEYS)),
    _playbook("full-spectrum", "Full-Spectrum Autonomous Investigation",
              "Posture, extraction, snapshot carving, wallet discovery and timeline correlation.",
              (_DEVICE, *_POSTURE, *_EXTRACTIONS, _SNAPSHOT, _CARVE, _PARSE, _TIMELINE,
               _CRYPTO, _KEYSTORE, _PASSKEYS, _ENRICH, _IOC)),
)}


def get_playbook(name: str) -> PlaybookDefinition:
    if name not in BUILTIN_PLAYBOOKS:
        raise LockKnifeError(f"Unknown playbook {name!r}. Available playbooks: {list(BUILTIN_PLAYBOOKS)}")
    return BUILTIN_PLAYBOOKS[name]


def list_playbooks() -> list[PlaybookDefinition]:
    return list(BUILTIN_PLAYBOOKS.values())


def load_custom_playbook(path: pathlib.Path) -> PlaybookDefinition:
    if not path.exists():
        raise LockKnifeError(f"Playbook file not found: {path}")
    try:
        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Playbook must contain a mapping/object")
        return PlaybookDefinition.from_dict(data)
    except Exception as exc:
        raise LockKnifeError(f"Invalid playbook definition in {path}: {exc}") from exc
