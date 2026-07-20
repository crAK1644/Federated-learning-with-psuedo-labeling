"""Structured JSON-lines logging.

Every log record can carry run/client/algorithm/scenario/round/phase/message identity via
:func:`bind`. Records are written as one JSON object per line so ``events.jsonl`` in a run
directory is directly machine-readable (per the M10 reporting/reproducibility-bundle contract).

Forbidden-field guard: :func:`log_event` rejects payloads containing any key that looks like it
carries private data, raw model tensors, or secrets (see ``_FORBIDDEN_KEY_SUBSTRINGS``). This is a
defense-in-depth check, not a substitute for callers simply not passing that data — see the
protocol/security tests in M10 for the authoritative guarantee.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_FORBIDDEN_KEY_SUBSTRINGS = (
    "private_feature",
    "private_label",
    "private_x",
    "private_y",
    "raw_features",
    "model_state",
    "state_dict",
    "weights",
    "gradient",
    "password",
    "secret",
    "api_key",
    "token",
)


class ForbiddenLogFieldError(ValueError):
    pass


def _check_forbidden(payload: dict[str, Any]) -> None:
    for key in payload:
        lowered = key.lower()
        for bad in _FORBIDDEN_KEY_SUBSTRINGS:
            if bad in lowered:
                raise ForbiddenLogFieldError(
                    f"refusing to log field {key!r}: matches forbidden pattern {bad!r} "
                    "(private data, model tensors, and secrets must never be logged)"
                )


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.Formatter) -> str:  # type: ignore[override]
        payload = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(level: int = logging.INFO, stream: Any = None) -> logging.Logger:
    root = logging.getLogger("ssfl")
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonLinesFormatter())
    root.addHandler(handler)
    root.propagate = False
    return root


def bind(logger: logging.Logger, **fields: Any) -> logging.LoggerAdapter:
    """Return a LoggerAdapter that merges ``fields`` into every record's ``fields`` extra.

    Typical fields: run_id, client_id, algorithm, scenario, round, phase, message_id,
    dataset_hash, config_hash.
    """
    _check_forbidden(fields)

    class _Adapter(logging.LoggerAdapter):
        def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            merged = dict(self.extra)
            merged.update(kwargs.pop("fields", {}) or {})
            _check_forbidden(merged)
            kwargs["extra"] = {"fields": merged}
            return msg, kwargs

    return _Adapter(logger, fields)


def log_event(logger: logging.LoggerAdapter | logging.Logger, message: str, **fields: Any) -> None:
    _check_forbidden(fields)
    logger.info(message, extra={"fields": fields} if not isinstance(logger, logging.LoggerAdapter) else {"fields": fields})
