from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from lockknife.core.exceptions import LockKnifeError
from lockknife.core.pipeline.models import PlaybookDefinition

# The legacy catalog omits several requirements enforced by the handlers.
_REQUIRED = {
    "device.info": ("serial",),
    "crypto.scan_device": ("serial",),
    "forensics.carve": ("path", "output_dir"),
    "forensics.sqlite": ("path",),
    "forensics.parse": ("path",),
    "forensics.import_aleapp": ("input_dir",),
    "forensics.decode_protobuf": ("path",),
    "forensics.recover": ("path",),
    "forensics.correlate": ("inputs",),
    "security.malware": ("rule", "target"),
    "security.owasp": ("artifacts",),
    "crypto.wallets": ("path",),
    "intelligence.ioc": ("input",),
}


@lru_cache(maxsize=1)
def _action_registry() -> Any:
    from lockknife_headless_cli.tui_callback import build_action_registry

    return build_action_registry()


def validate_action(action_id: str, params: dict[str, Any]) -> None:
    definition = _action_registry().get(action_id)
    if definition is None or definition.handler is None:
        raise LockKnifeError(f"Unregistered pipeline action: {action_id}")
    required = set(_REQUIRED.get(action_id, ()))
    required.update(field.key for field in definition.fields if field.required)
    if definition.requires_device:
        required.add("serial")
    if action_id == "forensics.parse" and (params.get("input_dir") or params.get("aleapp")):
        required.discard("path")
    if action_id == "forensics.import_aleapp" and params.get("path"):
        required.discard("input_dir")
    if action_id == "crypto.wallets" and params.get("serial"):
        required.discard("path")
    for key in required:
        if params.get(key) is None or params.get(key) == "":
            raise LockKnifeError(f"Action {action_id} requires parameter {key}")
    for field in definition.fields:
        value = params.get(field.key)
        if (
            value is not None
            and not isinstance(value, dict)
            and field.choices
            and value not in field.choices
        ):
            raise LockKnifeError(f"Invalid {field.key} for {action_id}: {value}")


def validate_recipe(playbook: PlaybookDefinition) -> None:
    steps = {step.step_id: step for step in playbook.steps}

    def ancestors(step_id: str) -> set[str]:
        deps = set(steps[step_id].depends_on)
        return deps.union(*(ancestors(dep) for dep in deps))

    def check(value: Any, allowed: set[str]) -> None:
        if isinstance(value, dict):
            if "$result" in value:
                if value["$result"] not in allowed or not isinstance(value.get("path"), str):
                    raise LockKnifeError(f"Invalid result reference: {value}")
            else:
                for item in value.values():
                    check(item, allowed)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item, allowed)

    for step in playbook.steps:
        if not math.isfinite(step.timeout_s) or step.timeout_s <= 0:
            raise LockKnifeError(f"Invalid timeout for {step.step_id}")
        if (
            not isinstance(step.retries, int)
            or step.retries < 0
            or not math.isfinite(step.retry_backoff_s)
            or step.retry_backoff_s < 0
        ):
            raise LockKnifeError(f"Invalid retry settings for {step.step_id}")
        check(step.params, ancestors(step.step_id))
