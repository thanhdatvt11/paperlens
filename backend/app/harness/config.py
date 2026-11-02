"""Validated settings for the document review harness."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PermissionConfig(BaseModel):
    """Policy for exporting a report."""

    model_config = ConfigDict(extra="forbid")

    report_export: Literal["deny", "ask"] = "ask"


class ContextConfig(BaseModel):
    """Limits applied when preparing model input."""

    model_config = ConfigDict(extra="forbid")

    num_ctx: int = Field(
        default=8192,
        ge=4096,
        le=8192,
        strict=True,
    )
    num_predict: int = Field(
        default=3072,
        ge=256,
        le=3072,
        strict=True,
    )
    recent_messages: int = Field(
        default=6,
        ge=0,
        le=12,
        strict=True,
    )


class AgentConfig(BaseModel):
    """Configuration shared by the API and harness runtime."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    name: str = Field(
        default="PDF Reviewer",
        min_length=1,
        max_length=80,
    )

    model: Literal["qwen3-vl:2b-instruct"] = (
        "qwen3-vl:2b-instruct"
    )

    voice: Literal["concise", "teacher", "formal"] = "concise"

    max_steps: int = Field(
        default=6,
        ge=1,
        le=12,
        strict=True,
    )

    memory_enabled: bool = Field(
        default=True,
        strict=True,
    )

    tools: list[
        Literal["pdf_info", "view_pdf_page", "save_report"]
    ] = Field(
        default_factory=lambda: [
            "pdf_info",
            "view_pdf_page",
            "save_report",
        ],
        min_length=1,
        max_length=3,
    )

    permission: PermissionConfig = Field(
        default_factory=PermissionConfig,
    )

    context: ContextConfig = Field(
        default_factory=ContextConfig,
    )

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: list[str]) -> list[str]:
        """Reject duplicate tools and missing PDF viewing capability."""
        if len(value) != len(set(value)):
            raise ValueError("tools must not contain duplicates")

        if "view_pdf_page" not in value:
            raise ValueError(
                "PDF reviewer requires view_pdf_page"
            )

        return value