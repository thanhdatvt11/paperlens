"""HTTP adapter for the local Ollama API."""

import base64
from typing import Any

import httpx

from app.core.config import Settings


class OllamaAdapterError(RuntimeError):
    """Base class for errors raised while calling Ollama."""


class OllamaConnectionError(OllamaAdapterError):
    """Ollama could not be reached."""


class OllamaTimeoutError(OllamaAdapterError):
    """Ollama did not respond within the configured timeout."""


class OllamaProviderError(OllamaAdapterError):
    """Ollama returned an unsuccessful HTTP response."""


class OllamaModelUnavailableError(OllamaAdapterError):
    """The configured model is not available in Ollama."""


class OllamaResponseError(OllamaAdapterError):
    """Ollama returned JSON with an unexpected shape."""


class OllamaAdapter:
    """Call Ollama with a short-lived, correctly closed HTTP client."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
        *,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model
        self._timeout = httpx.Timeout(timeout_seconds, connect=5.0)
        self._transport = transport

    def list_models(self) -> list[str]:
        """Return model names reported by Ollama."""
        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError from exc
        except httpx.HTTPError as exc:
            raise OllamaProviderError from exc

        try:
            payload = response.json()
            models = payload["models"]
            names = [model["name"] for model in models]
            if not isinstance(models, list) or not all(
                isinstance(name, str) for name in names
            ):
                raise TypeError
            return names
        except (KeyError, TypeError, ValueError) as exc:
            raise OllamaResponseError from exc

    def chat(
        self,
        message: str,
        *,
        num_predict: int = 256,
        response_format: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        images: list[bytes] | None = None,
        num_ctx: int = 4096,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send one user message and return Ollama's chat result."""
        messages = [{"role": "user", "content": message}]
        if images is not None:
            if not images or not all(isinstance(value, bytes) and value for value in images):
                raise ValueError("images must contain nonempty image bytes")
            messages[0]["images"] = [base64.b64encode(value).decode("ascii") for value in images]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        request_body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"num_ctx": num_ctx, "num_predict": num_predict},
        }
        if temperature is not None:
            request_body["options"]["temperature"] = temperature
        if response_format is not None:
            request_body["format"] = response_format
        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = client.post(
                    "/api/chat", json=request_body,
                    timeout=httpx.Timeout(request_timeout, connect=min(5.0, request_timeout))
                    if request_timeout is not None else self._timeout,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = exc.response.text[:1000]

            if status == 404:
                raise OllamaModelUnavailableError(
                    f"Ollama HTTP {status}: {detail}"
                ) from exc

            raise OllamaProviderError(
                f"Ollama HTTP {status}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaProviderError from exc

        try:
            payload = response.json()
            model = payload["model"]
            content = payload["message"]["content"]
            if not isinstance(model, str) or not isinstance(content, str):
                raise TypeError
            return {
                "model": model,
                "content": content,
                "done": payload.get("done"),
                "done_reason": payload.get("done_reason"),
                "eval_count": payload.get("eval_count"),
                "prompt_eval_count": payload.get("prompt_eval_count"),
                "thinking_characters": len(payload["message"].get("thinking") or ""),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise OllamaResponseError from exc
