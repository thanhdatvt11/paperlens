"""Schemas for Ollama-backed endpoints."""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AIModelsResponse(BaseModel):
    models: list[str]


class AITestRequest(BaseModel):
    message: str

    @field_validator('message', mode='before')
    @classmethod
    def validate_message(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        trimmed = value.strip()
        if not 1 <= len(trimmed) <= 2000:
            raise ValueError('message must contain 1 to 2000 characters')
        return trimmed


class AITestResponse(BaseModel):
    model: str
    content: str


class ChunkReviewPreviewRequest(BaseModel):
    instruction: str

    @field_validator('instruction', mode='before')
    @classmethod
    def validate_instruction(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        trimmed = value.strip()
        if not 1 <= len(trimmed) <= 500:
            raise ValueError('instruction must contain 1 to 500 characters')
        return trimmed


class DocumentReviewRequest(BaseModel):
    instruction: str

    @field_validator('instruction', mode='before')
    @classmethod
    def validate_instruction(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        trimmed = value.strip()
        if not 1 <= len(trimmed) <= 500:
            raise ValueError('instruction must contain 1 to 500 characters')
        return trimmed


class ReviewFinding(BaseModel):
    model_config = ConfigDict(extra='forbid')
    severity: Literal['minor', 'major']
    comment: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)

    @field_validator('comment', 'quote', 'suggestion')
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('review fields must not be blank')
        return value


class ChunkReviewContent(BaseModel):
    model_config = ConfigDict(extra='forbid')
    summary: str = Field(min_length=1)
    findings: list[ReviewFinding] = Field(max_length=3)

    @field_validator('summary')
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('summary must not be blank')
        return value


class ReviewSource(BaseModel):
    model_config = ConfigDict(extra='forbid')
    chunk_id: int = Field(ge=1)
    page_number: int = Field(ge=1)


class ReviewEvidence(ReviewSource):
    quote: str = Field(min_length=1)

    @field_validator('quote')
    @classmethod
    def validate_quote_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('evidence quote must not be blank')
        return value


class DocumentKeyPoint(BaseModel):
    model_config = ConfigDict(extra='forbid')
    point: str = Field(min_length=1)
    sources: list[ReviewSource] = Field(min_length=1)

    @field_validator('point')
    @classmethod
    def validate_point_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('key point must not be blank')
        return value


class DocumentFinding(BaseModel):
    model_config = ConfigDict(extra='forbid')
    severity: Literal['minor', 'major']
    comment: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)
    evidence: list[ReviewEvidence] = Field(min_length=1)

    @field_validator('comment', 'suggestion')
    @classmethod
    def validate_finding_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('finding fields must not be blank')
        return value


class DocumentReviewContent(BaseModel):
    model_config = ConfigDict(extra='forbid')
    summary: str = Field(min_length=1)
    key_points: list[DocumentKeyPoint]
    findings: list[DocumentFinding] = Field(max_length=10)

    @field_validator('summary')
    @classmethod
    def validate_document_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('document summary must not be blank')
        return value


class DocumentReviewResult(DocumentReviewContent):
    document_id: int
    version_id: int
    model: str


class ChunkReviewPreviewResponse(ChunkReviewContent):
    document_id: int
    version_id: int
    chunk_id: int
    page_number: int
    model: str


class ReviewRunAcceptedResponse(BaseModel):
    run_id: int
    status: Literal['QUEUED']


class ReviewRunStatusResponse(BaseModel):
    run_id: int
    job_type: Literal['CHUNK_PREVIEW', 'DOCUMENT_REVIEW']
    document_id: int
    version_id: int
    status: Literal['QUEUED', 'RUNNING', 'COMPLETED', 'FAILED']
    stage: Literal['QUEUED', 'READING_CHUNKS', 'SYNTHESIZING', 'FINISHED']
    total_chunks: int
    processed_chunks: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: dict[str, object] | None
    error: str | None
