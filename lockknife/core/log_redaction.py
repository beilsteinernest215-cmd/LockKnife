from __future__ import annotations

import contextlib
import contextvars
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
        for secret in sorted(_secrets.get(), key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
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
