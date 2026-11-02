"""Validated actions and reports returned by the model."""

from typing import Literal, Self

from pydantic import(
    BaseModel,
    ConfigDict,
    Field,
    model_validator
)


class ReviewFinding(BaseModel):
    """One issue observed on a PDF page."""
    
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    
    page_number: int = Field(ge=1, strict=True)
    severity: Literal["minor", "major"]

    problem: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)


class ReviewReport(BaseModel):
    """Structured report produced by the agent."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    summary: str = Field(min_length=1)

    findings: list[ReviewFinding] = Field(
        default_factory=list,
        max_length=2,
    )


class HarnessAction(BaseModel):
    """One model decision: request a tool or return a report."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["tool", "final"]

    tool: Literal[
        "pdf_info",
        "view_pdf_page",
    ] | None

    arguments: dict[str, object]

    report: ReviewReport | None

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        """Check fields according to the selected action."""
        if self.action == "final":
            if self.tool is not None:
                raise ValueError(
                    "Final action must have tool=null."
                )

            if self.arguments:
                raise ValueError(
                    "Final action must have arguments={}."
                )

            if self.report is None:
                raise ValueError(
                    "Final action requires a report."
                )

            return self

        if self.report is not None:
            raise ValueError(
                "Tool action must have report=null."
            )

        if self.tool == "pdf_info":
            if self.arguments:
                raise ValueError(
                    "pdf_info requires arguments={}."
                )

        elif self.tool == "view_pdf_page":
            if set(self.arguments) != {"page"}:
                raise ValueError(
                    "view_pdf_page requires only page."
                )

            page = self.arguments["page"]

            if type(page) is not int or page < 1:
                raise ValueError(
                    "page must be a positive integer."
                )

        else:
            raise ValueError(
                "Tool action requires a registered tool."
            )

        return self