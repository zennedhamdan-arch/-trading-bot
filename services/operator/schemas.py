"""
services/operator/schemas.py

Trading Partner schemas + the security boundary for every tool result.

Everything a tool returns (and everything that reaches the operator model's
context) passes through `sanitize()`:

  * secret-looking keys are REDACTED anywhere in the structure
    (API keys, tokens, passwords, credentials) — value replaced with
    "***configured***" / "***not set***";
  * long strings and large lists are capped (the model context is finite;
    a multi-megabyte log line must never reach it);
  * control characters are stripped (log/view injection hardening);
  * non-JSON-serializable values are converted defensively.

The operator has NO shell, filesystem or HTTP tools — dispatch happens only
through the explicit registry in operator_tools/__init__.py.
"""

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|credential|private[_-]?key|"
    r"session[_-]?key|auth[_-]?header|authorization|bearer)", re.IGNORECASE)

# Secret-looking VALUES: an API key echoed inside an error string, a
# "Bearer <token>" header, common vendor key prefixes. Matched substrings
# are masked (the surrounding error text stays useful for diagnosis).
_SECRET_VALUE_RE = re.compile(
    r"(bearer\s+\S{4,}|\b(?:sk|nvai|xai|rsk)-[A-Za-z0-9_-]{8,}"
    r"|\bgsk_[A-Za-z0-9]{8,}|\bAIza[A-Za-z0-9_-]{10,})", re.IGNORECASE)

REDACTED_SET = "***not set***"
REDACTED_CONFIGURED = "***configured***"

MAX_STRING = 4000          # per value
MAX_LIST = 60              # per list
MAX_DEPTH = 8


def redact_value(value):
    """Redacts a scalar that IS a secret (a key found inside a value, e.g.
    an API key echoed inside an error message)."""
    return REDACTED_CONFIGURED


def sanitize(obj, depth: int = 0):
    """Recursively redacts + sizes any tool result. Returns a structure that
    is safe to embed in the operator's context and to return via the API."""
    if depth > MAX_DEPTH:
        return "[truncated: depth]"
    if obj is None or isinstance(obj, (bool, int)):
        return obj
    if isinstance(obj, float):
        return round(obj, 6) if obj == obj else None   # NaN -> None
    if isinstance(obj, str):
        text = "".join(ch for ch in obj if ch >= " " or ch in "\n\t")
        if _SECRET_VALUE_RE.search(text):
            text = _SECRET_VALUE_RE.sub("***", text)
        if len(text) > MAX_STRING:
            text = text[:MAX_STRING] + f"…[truncated {len(obj) - MAX_STRING} chars]"
        return text
    if isinstance(obj, dict):
        out = {}
        for key, value in list(obj.items())[:MAX_LIST * 2]:
            key_s = str(key)
            if _SECRET_KEY_RE.search(key_s):
                out[key_s] = REDACTED_CONFIGURED if value else REDACTED_SET
            else:
                out[key_s] = sanitize(value, depth + 1)
        return out
    if isinstance(obj, (list, tuple, set)):
        items = list(obj)[:MAX_LIST]
        if len(obj) > MAX_LIST:
            return [sanitize(i, depth + 1) for i in items] + [
                f"…[{len(obj) - MAX_LIST} more]"]
        return [sanitize(i, depth + 1) for i in items]
    # datetime / enums / SDK objects — best-effort string, never a crash
    return sanitize(str(obj), depth + 1)


# ---------------------------------------------------------------------------
# Tool protocol types
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    description: str
    category: str                 # system | agents | llm | market | portfolio | configuration | history | diagnostics
    args: dict = field(default_factory=dict)   # arg name -> {"type", "description", "required"}
    label: str = ""               # human-readable activity label for the UI

    def public(self) -> dict:
        return {"tool": self.name, "description": self.description,
                "category": self.category, "args": self.args,
                "label": self.label or self.name}


@dataclass
class ToolResult:
    ok: bool
    data: dict
    error: str = None
    ms: int = 0              # execution time, set by run_tool

    def public(self) -> dict:
        out = {"ok": self.ok, "data": sanitize(self.data)}
        if self.error:
            out["error"] = sanitize(self.error)
        return out


def validate_tool_args(spec: ToolSpec, args: dict) -> str:
    """Returns an error string when args are invalid, else None. Unknown
    argument names are rejected (typo/injection hardening)."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return "tool arguments must be a JSON object"
    allowed = set(spec.args or {})
    unknown = [k for k in args if k not in allowed]
    if unknown:
        return f"unknown argument(s) {unknown}; allowed: {sorted(allowed) or 'none'}"
    for name, meta in (spec.args or {}).items():
        if meta.get("required") and (args.get(name) in (None, "", [])):
            return f"missing required argument '{name}'"
        if name in args and args[name] is not None:
            kind = meta.get("type", "string")
            value = args[name]
            if kind == "string" and not isinstance(value, str):
                return f"argument '{name}' must be a string"
            if kind == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
                return f"argument '{name}' must be an integer"
            if kind == "number" and not isinstance(value, (int, float)):
                return f"argument '{name}' must be a number"
            if isinstance(value, str) and len(value) > 200:
                return f"argument '{name}' too long (max 200 chars)"
    return None
