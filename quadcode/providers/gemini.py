"""Client for the Gemini API (``generateContent``), which also serves Gemma models.

The key goes in the ``x-goog-api-key`` header, never in the URL.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _model_path(model: str) -> str:
    name = model.removeprefix("models/")
    return f"models/{quote(name, safe='-._')}"


class GeminiClient(LLMClient):
    def __init__(self, provider_name: str, cfg: ProviderConfig, api_key: str,
                 transport: httpx.BaseTransport | None = None) -> None:
        super().__init__(provider_name)
        self._max_retry_after_s = cfg.max_retry_after_s
        self._http = make_http_client(cfg.base_url or DEFAULT_BASE_URL, {"x-goog-api-key": api_key},
                                      cfg.timeout_s, transport)

    def complete(self, request: LLMRequest) -> LLMResponse:
        system, user = request.messages()
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {**request.options, "maxOutputTokens": request.max_tokens,
                                 "temperature": request.temperature},
        }
        if system is not None:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        payload = request_json(self._http, "POST", f"{_model_path(request.model)}:generateContent",
                               self._max_retry_after_s, body=body)

        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
            feedback = payload.get("promptFeedback")
            reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
            raise PermanentError(f"response has no candidates (blockReason={reason})")
        cand = candidates[0]
        finish = cand.get("finishReason")
        finish_reason = finish if isinstance(finish, str) else None
        content = cand.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        texts = [p["text"] for p in parts or [] if isinstance(p, dict) and isinstance(p.get("text"), str)
                 and not p.get("thought")]
        text = "".join(texts)
        if not text:
            raise PermanentError(empty_output_message("finishReason", finish_reason))

        usage = payload.get("usageMetadata") if isinstance(payload.get("usageMetadata"), dict) else {}
        tin, tout = token_count(usage.get("promptTokenCount")), token_count(usage.get("candidatesTokenCount"))
        thoughts = token_count(usage.get("thoughtsTokenCount")) or 0  # thinking tokens are output tokens
        estimated = tin is None or tout is None
        version = payload.get("modelVersion")
        return LLMResponse(
            text=text,
            tokens_in=tin if tin is not None else estimate_tokens(request.system + request.prompt),
            tokens_out=(tout + thoughts) if tout is not None else estimate_tokens(text),
            tokens_estimated=estimated,
            model=version if isinstance(version, str) and version else request.model,
            finish_reason=finish_reason,
        )

    def list_models(self) -> list[str]:
        names: set[str] = set()
        page_token: str | None = None
        for _ in range(20):  # the catalog fits in a page or two; the cap guards against a looping token
            params: dict[str, Any] = {"pageSize": 1000}
            if page_token:
                params["pageToken"] = page_token
            payload = request_json(self._http, "GET", "models", self._max_retry_after_s, params=params)
            models = payload.get("models")
            if not isinstance(models, list):
                raise PermanentError(f"model list has no 'models' array: {snippet(str(payload))}")
            for m in models:
                if not isinstance(m, dict) or not isinstance(m.get("name"), str):
                    continue
                methods = m.get("supportedGenerationMethods")
                if isinstance(methods, list) and "generateContent" not in methods:
                    continue
                names.add(m["name"].removeprefix("models/"))
            token = payload.get("nextPageToken")
            if not isinstance(token, str) or not token:
                break
            page_token = token
        return sorted(names)

    def close(self) -> None:
        self._http.close()
