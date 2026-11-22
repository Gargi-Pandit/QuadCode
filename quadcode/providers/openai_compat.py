"""Client for OpenAI-compatible chat completion APIs (Groq, NVIDIA NIM, OpenRouter, OpenAI, ...).

The endpoint comes from ``base_url`` in config.yaml; nothing here names a vendor.
"""

from __future__ import annotations

from typing import Any

import httpx

from quadcode.config import ProviderConfig
from quadcode.providers.base import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    PermanentError,
    empty_output_message,
    estimate_tokens,
)
from quadcode.providers.http import make_http_client, request_json, snippet, token_count


def _content_text(content: Any) -> str | None:
    """``message.content`` is a string, or a list of typed parts in some implementations."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text") for p in content if isinstance(p, dict) and p.get("type", "text") == "text"]
        joined = "".join(p for p in parts if isinstance(p, str))
        return joined or None
    return None


class OpenAICompatClient(LLMClient):
    def __init__(self, provider_name: str, cfg: ProviderConfig, api_key: str,
                 transport: httpx.BaseTransport | None = None) -> None:
        super().__init__(provider_name)
        if not cfg.base_url:
            raise ValueError("openai_compat providers need base_url")
        self._max_retry_after_s = cfg.max_retry_after_s
        self._http = make_http_client(cfg.base_url, {"Authorization": f"Bearer {api_key}"}, cfg.timeout_s,
                                      transport)

    def complete(self, request: LLMRequest) -> LLMResponse:
        system, user = request.messages()
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        body: dict[str, Any] = {**request.options, "model": request.model, "messages": messages,
                                "max_tokens": request.max_tokens, "temperature": request.temperature}
        payload = request_json(self._http, "POST", "chat/completions", self._max_retry_after_s, body=body)

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise PermanentError(f"response has no choices: {snippet(str(payload))}")
        choice = choices[0]
        finish = choice.get("finish_reason")
        finish_reason = finish if isinstance(finish, str) else None
        message = choice.get("message")
        text = _content_text(message.get("content")) if isinstance(message, dict) else None
        if not text:
            raise PermanentError(empty_output_message("finish_reason", finish_reason))

        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        tin, tout = token_count(usage.get("prompt_tokens")), token_count(usage.get("completion_tokens"))
        estimated = tin is None or tout is None
        model = payload.get("model")
        return LLMResponse(
            text=text,
            tokens_in=tin if tin is not None else estimate_tokens(request.system + request.prompt),
            tokens_out=tout if tout is not None else estimate_tokens(text),
            tokens_estimated=estimated,
            model=model if isinstance(model, str) and model else request.model,
            finish_reason=finish_reason,
        )

    def list_models(self) -> list[str]:
        payload = request_json(self._http, "GET", "models", self._max_retry_after_s)
        data = payload.get("data")
        if not isinstance(data, list):
            raise PermanentError(f"model list has no 'data' array: {snippet(str(payload))}")
        return sorted({m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)})

    def close(self) -> None:
        self._http.close()
