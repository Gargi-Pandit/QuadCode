"""Client for the Anthropic Messages API."""

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

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"


class AnthropicClient(LLMClient):
    def __init__(self, provider_name: str, cfg: ProviderConfig, api_key: str,
                 transport: httpx.BaseTransport | None = None) -> None:
        super().__init__(provider_name)
        self._max_retry_after_s = cfg.max_retry_after_s
        self._http = make_http_client(cfg.base_url or DEFAULT_BASE_URL,
                                      {"x-api-key": api_key, "anthropic-version": API_VERSION},
                                      cfg.timeout_s, transport)

    def complete(self, request: LLMRequest) -> LLMResponse:
        system, user = request.messages()
        body: dict[str, Any] = {**request.options, "model": request.model, "max_tokens": request.max_tokens,
                                "temperature": request.temperature,
                                "messages": [{"role": "user", "content": user}]}
        if system is not None:
            body["system"] = system
        payload = request_json(self._http, "POST", "messages", self._max_retry_after_s, body=body)

        stop = payload.get("stop_reason")
        finish_reason = stop if isinstance(stop, str) else None
        content = payload.get("content")
        texts = [b["text"] for b in content or [] if isinstance(b, dict) and b.get("type") == "text"
                 and isinstance(b.get("text"), str)] if isinstance(content, list) else []
        text = "".join(texts)
        if not text:
            raise PermanentError(empty_output_message("stop_reason", finish_reason))

        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        tin, tout = token_count(usage.get("input_tokens")), token_count(usage.get("output_tokens"))
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
        payload = request_json(self._http, "GET", "models", self._max_retry_after_s, params={"limit": 1000})
        data = payload.get("data")
        if not isinstance(data, list):
            raise PermanentError(f"model list has no 'data' array: {snippet(str(payload))}")
        return sorted({m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)})

    def close(self) -> None:
        self._http.close()
