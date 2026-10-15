from __future__ import annotations

import contextlib
import contextvars
import json
import re
from collections.abc import Iterator, Mapping
from typing import Any

_secrets: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "log_secrets", default=()
)
_sensitive = re.compile(
    r"(?:^|_)(?:password|passwd|passphrase|secret|token|credential|authorization|api_key|private_key|signing_key|access_key)(?:$|_)",
    re.IGNORECASE,
)


def sensitive_key(key: str) -> bool:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).replace("-", "_")
    return bool(_sensitive.search(normalized))


def redact(value: Any) -> Any:
    """Remove credential fields and active command secrets from diagnostic data."""
    if isinstance(value, Mapping):
        return {
            key: "[REDACTED]" if sensitive_key(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        variants: set[str] = set()
        for secret in _secrets.get():
            # Exceptions may render arguments using repr(), and JSON clients can
            # embed a serialized credential in their diagnostic message.
            variants.update((secret, repr(secret)[1:-1], ascii(secret)[1:-1]))
            variants.update(
                json.dumps(secret, ensure_ascii=ascii_only)[1:-1] for ascii_only in (True, False)
            )
        for variant in sorted(variants, key=len, reverse=True):
            if variant:
                value = value.replace(variant, "[REDACTED]")
    return value


@contextlib.contextmanager
def credential_context(params: Mapping[str, Any]) -> Iterator[None]:
    values = tuple(
        value
        for key, value in params.items()
        if sensitive_key(key) and isinstance(value, str) and value
    )
    token = _secrets.set((*_secrets.get(), *values))
    try:
        yield
    finally:
        _secrets.reset(token)


def redact_log_event(_logger: Any, _method: str, event: dict[str, Any]) -> dict[str, Any]:
    return redact(event)
