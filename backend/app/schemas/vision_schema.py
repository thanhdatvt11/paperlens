"""API input for page-image reviews; all page numbers are PDF positions."""

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class VisionReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1, max_length=500)
    pages: list[StrictInt] | None = Field(default=None, min_length=1, max_length=80)
    timeout_seconds: int = Field(default=14400, ge=600, le=86400)

    @field_validator("instruction")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction must not be blank")
        return value.strip()

    @field_validator("pages")
    @classmethod
    def page_positions(cls, value: list[int] | None) -> list[int] | None:
        if value is not None:
            if any(page < 1 for page in value):
                raise ValueError("pages must contain positive PDF positions")
            return sorted(set(value))
        return value

