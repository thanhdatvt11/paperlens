"""Local Ollama settings rebuilt for the existing adapter/worker contract.

Only OLLAMA_BASE_URL and OLLAMA_MODEL are read. Process environment takes
precedence over backend/.env, then documented local defaults are used.
No new dependencies beyond the Pydantic already used by this application.
"""
from functools import lru_cache
import os
from pathlib import Path
import shlex
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    ollama_base_url: str = 'http://localhost:11434'
    ollama_model: str = Field(default='qwen3:1.7b', min_length=1)

    @field_validator('ollama_base_url')
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip('/')
        parsed = urlsplit(value)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc:
            raise ValueError('OLLAMA_BASE_URL must be an http(s) URL')
        return value

    @field_validator('ollama_model')
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('OLLAMA_MODEL must not be blank')
        return value


def _dotenv_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, separator, raw = line.partition('=')
        key = key.strip()
        if not separator or key not in ('OLLAMA_BASE_URL', 'OLLAMA_MODEL'):
            continue
        tokens = shlex.split(raw, comments=True, posix=True)
        if len(tokens) != 1:
            raise ValueError(f'Invalid or empty {key} in backend/.env')
        values[key] = tokens[0]
    return values


@lru_cache
def get_settings() -> Settings:
    values = _dotenv_values(BACKEND_DIR / '.env')
    return Settings(
        ollama_base_url=os.environ.get(
            'OLLAMA_BASE_URL', values.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        ),
        ollama_model=os.environ.get(
            'OLLAMA_MODEL', values.get('OLLAMA_MODEL', 'qwen3:1.7b')
        ),
    )
