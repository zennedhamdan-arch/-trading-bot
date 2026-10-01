"""
operator_tools — the read-only diagnostic tool registry.

Importing this package registers all tool modules. No tool here can trade,
write configuration, or reach the network directly — everything goes through
existing service abstractions, error-contained by _registry.run_tool.
"""

from services.operator.operator_tools import (  # noqa: F401 — imports register tools
    _registry,
    agents,
    configuration,  # includes history tools
    llm,
    market,
    portfolio,
    system,
)
from services.operator.operator_tools._registry import get_spec, list_tools, run_tool

__all__ = ["get_spec", "list_tools", "run_tool"]
