"""Permission policy checked by the harness before tool execution."""

from dataclasses import dataclass
from typing import Literal

from app.harness.config import AgentConfig


@dataclass(frozen=True)
class PermissionDecision:
    """A policy decision, not an approval supplied by the model."""

    effect: Literal["ALLOW", "DENY", "ASK"]
    reason: str


def check_tool_permission(
    config: AgentConfig,
    tool_name: str,
) -> PermissionDecision:
    """Determine whether a tool may execute."""
    if tool_name not in config.tools:
        return PermissionDecision(
            effect="DENY",
            reason="Tool is not enabled for this agent.",
        )

    if tool_name in {"pdf_info", "view_pdf_page"}:
        return PermissionDecision(
            effect="ALLOW",
            reason="Read-only PDF tool is enabled.",
        )

    if tool_name == "save_report":
        if config.permission.report_export == "deny":
            return PermissionDecision(
                effect="DENY",
                reason="Report export is disabled.",
            )

        return PermissionDecision(
            effect="ASK",
            reason="Report export requires user approval.",
        )

    return PermissionDecision(
        effect="DENY",
        reason="Tool has no registered permission policy.",
    )