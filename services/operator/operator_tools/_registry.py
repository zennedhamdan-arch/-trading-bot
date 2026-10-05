"""
services/operator/operator_tools/_registry.py

The controlled internal tool layer of the Trading Partner.

EVERY tool is:
  * registered explicitly (no dynamic dispatch, no eval, no shell, no
    filesystem access, no arbitrary HTTP — only the fixed functions below);
  * read-only with respect to the trading system (no order submission, no
    configuration mutation — those functions simply do not exist here);
  * validated (unknown tools / unknown or malformed arguments are rejected);
  * error-contained (a tool failure returns ToolResult(ok=False) and never
    breaks the conversation);
  * sanitized (results pass schemas.sanitize() before reaching the model).
"""

import logging
import time

from services.operator.schemas import ToolSpec, ToolResult, validate_tool_args

logger = logging.getLogger("operator")

_REGISTRY = {}          # name -> (ToolSpec, callable)


def tool(name: str, category: str, description: str, args: dict = None,
         label: str = ""):
    """Decorator registering a read-only operator tool."""
    def deco(fn):
        register(ToolSpec(name=name, description=description, category=category,
                          args=args or {}, label=label or name), fn)
        return fn
    return deco


def register(spec: ToolSpec, fn) -> None:
    if spec.name in _REGISTRY:
        raise RuntimeError(f"duplicate operator tool '{spec.name}'")
    _REGISTRY[spec.name] = (spec, fn)


def list_tools() -> list:
    """The public tool catalog (for the model and /api/operator/tools)."""
    return [spec.public() for spec, _ in
            sorted(_REGISTRY.values(), key=lambda pair: pair[0].name)]


def get_spec(name: str):
    entry = _REGISTRY.get(name)
    return entry[0] if entry else None


def run_tool(name: str, args: dict = None) -> ToolResult:
    """Dispatches one tool call safely. Returns ToolResult; never raises."""
    spec_fn = _REGISTRY.get(name)
    if spec_fn is None:
        return ToolResult(ok=False, data={}, error=f"unknown tool '{name}'")
    spec, fn = spec_fn
    started = time.monotonic()

    def _ms() -> int:
        return int((time.monotonic() - started) * 1000)

    try:
        arg_error = validate_tool_args(spec, args)
        if arg_error:
            logger.warning("operator tool %s rejected args: %s", name, arg_error)
            return ToolResult(ok=False, data={"tool": name}, error=arg_error,
                              ms=_ms())
        data = fn(args or {})
        if not isinstance(data, dict):
            return ToolResult(ok=False, data={},
                              error=f"tool '{name}' returned a non-dict result",
                              ms=_ms())
        logger.info("operator tool ok: %s (%.0fms)", name, _ms())
        return ToolResult(ok=True, data=data, ms=_ms())
    except Exception as exc:  # noqa: BLE001 — tools never break the loop
        logger.warning("operator tool %s failed: %s", name, exc)
        return ToolResult(ok=False, data={"tool": name}, error=str(exc)[:500],
                          ms=_ms())
