"""OpenAIProvider -- OpenAI & Azure OpenAI cloud connector with full v2.1 telemetry."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from aurarouter_openai.models import (
    OPENAI_MODELS,
    get_default_model_id,
    get_model_info,
    is_reasoning_model,
)

try:
    import openai as _openai_sdk  # type: ignore[import-not-found]

    _REAL_OPENAI_CLS: tuple[type, ...] = tuple(
        cls
        for cls in (
            getattr(_openai_sdk, "OpenAI", None),
            getattr(_openai_sdk, "AzureOpenAI", None),
        )
        if isinstance(cls, type)
    )
except ImportError:
    _openai_sdk = None
    _REAL_OPENAI_CLS = ()

_OPENAI_CTX_KEYWORDS = (
    "context_length_exceeded",
    "maximum context length",
    "too many tokens",
    "prompt is too long",
    "context window",
    "reduce the length of the messages",
)

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_AZURE_API_VERSION = "2025-01-01-preview"


class OpenAIRateLimitError(Exception):
    """Raised when OpenAI or Azure OpenAI returns HTTP 429 rate limit exceeded."""

    def __init__(
        self,
        message: str,
        headers: dict[str, str] | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.headers: dict[str, str] = headers or {}
        self.retry_after_s = retry_after_s


class OpenAIContextLengthError(Exception):
    """Raised when prompt + completion exceeds the target OpenAI model context window."""

    def __init__(self, message: str, max_tokens: int | None = None) -> None:
        super().__init__(message)
        self.max_tokens = max_tokens


class OpenAITransientError(Exception):
    """Raised on upstream 5xx, timeout, or connection failures."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _extract_ratelimit_headers(raw_headers: Any) -> dict[str, str]:
    """Filter or normalize response headers relevant to rate-limit tracking (QA 3.3)."""
    if raw_headers is None or not hasattr(raw_headers, "items"):
        return {}
    kept: dict[str, str] = {}
    for k, v in raw_headers.items():
        kl = str(k).lower()
        if "ratelimit" in kl or kl in ("retry-after", "retry-after-ms"):
            kept[str(k)] = str(v)
    return kept


def _contains_json_keyword(texts: list[str]) -> bool:
    return any("json" in (t or "").lower() for t in texts)


class OpenAIProvider:
    """Feature-complete wrapper around OpenAI Platform & Azure OpenAI APIs.

    Uses the official ``openai`` Python SDK when installed (or when *client*
    is explicitly injected) and falls back cleanly to ``httpx`` against the
    documented REST endpoints so zero-SDK environments remain operational.
    """

    _extract_ratelimit_headers = staticmethod(_extract_ratelimit_headers)

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "",
        *,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        azure_ad_token: str | None = None,
        api_mode: str = "chat_completions",
        timeout: float = 120.0,
        client: Any = None,
    ) -> None:
        self._azure_endpoint = (
            azure_endpoint
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
            or os.environ.get("OPENAI_AZURE_ENDPOINT")
            or ""
        ).rstrip("/")
        self._azure_ad_token = azure_ad_token or os.environ.get("AZURE_OPENAI_AD_TOKEN") or ""
        self._api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("AZURE_OPENAI_API_KEY")
            or ""
        )

        if client is None and not self._api_key and not self._azure_ad_token:
            raise ValueError(
                "No OpenAI credentials provided. Set OPENAI_API_KEY (or "
                "AZURE_OPENAI_API_KEY / AZURE_OPENAI_AD_TOKEN) or pass api_key=."
            )

        self._base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or _DEFAULT_OPENAI_BASE_URL
        ).rstrip("/")
        self._organization = organization or os.environ.get("OPENAI_ORG_ID") or ""
        self._project = project or os.environ.get("OPENAI_PROJECT_ID") or ""
        self._api_version = (
            api_version
            or os.environ.get("OPENAI_API_VERSION")
            or _DEFAULT_AZURE_API_VERSION
        )
        if api_mode not in ("chat_completions", "responses"):
            raise ValueError(
                f"Invalid api_mode '{api_mode}'. Must be 'chat_completions' or 'responses'."
            )
        self._api_mode = api_mode
        self._timeout = float(timeout)
        self._default_model = default_model or get_default_model_id()

        if client is not None:
            self._client = client
        elif _openai_sdk is not None:
            if self._azure_endpoint:
                az_kwargs: dict[str, Any] = {
                    "azure_endpoint": self._azure_endpoint,
                    "api_version": self._api_version,
                    "timeout": self._timeout,
                    "max_retries": 0,  # AuraRouter ComputeFabric owns failover/retry policy
                }
                if self._azure_ad_token:
                    az_kwargs["azure_ad_token"] = self._azure_ad_token
                else:
                    az_kwargs["api_key"] = self._api_key
                self._client = _openai_sdk.AzureOpenAI(**az_kwargs)
            else:
                oa_kwargs: dict[str, Any] = {
                    "api_key": self._api_key,
                    "base_url": self._base_url,
                    "timeout": self._timeout,
                    "max_retries": 0,
                }
                if self._organization:
                    oa_kwargs["organization"] = self._organization
                if self._project:
                    oa_kwargs["project"] = self._project
                self._client = _openai_sdk.OpenAI(**oa_kwargs)
        else:
            self._client = None

    @property
    def is_azure(self) -> bool:
        return bool(self._azure_endpoint)

    @property
    def api_mode(self) -> str:
        return self._api_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: str = "",
        json_mode: bool = False,
        *,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        reasoning_effort: str = "",
        prompt_cache_key: str = "",
        api_mode: str = "",
    ) -> dict[str, Any]:
        """Single-shot text generation."""
        return self.generate_with_history(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="",
            model=model,
            json_mode=json_mode,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            prompt_cache_key=prompt_cache_key,
            api_mode=api_mode,
        )

    def generate_with_history(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str = "",
        model: str = "",
        json_mode: bool = False,
        *,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        reasoning_effort: str = "",
        prompt_cache_key: str = "",
        api_mode: str = "",
    ) -> dict[str, Any]:
        """Multi-turn generation adapting automatically to reasoning vs chat models
        and Chat Completions vs Responses APIs.
        """
        model_id = model or self._default_model
        effective_mode = api_mode or self._api_mode
        reasoning = is_reasoning_model(model_id)
        model_meta = get_model_info(model_id)
        ctx_limit = int(model_meta["context_window"]) if model_meta else 128_000

        if effective_mode == "responses":
            payload = self._build_responses_payload(
                model_id=model_id,
                messages=messages,
                system_prompt=system_prompt,
                reasoning=reasoning,
                json_mode=json_mode,
                response_schema=response_schema,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                prompt_cache_key=prompt_cache_key,
            )
            data, rl_headers, elapsed = self._invoke_responses(model_id, payload)
            parsed = self._parse_responses_output(data, model_id, ctx_limit, elapsed, rl_headers)
            return parsed

        payload = self._build_chat_payload(
            model_id=model_id,
            messages=messages,
            system_prompt=system_prompt,
            reasoning=reasoning,
            json_mode=json_mode,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            prompt_cache_key=prompt_cache_key,
        )
        data, rl_headers, elapsed = self._invoke_chat_completions(model_id, payload)
        return self._parse_chat_output(data, model_id, ctx_limit, elapsed, rl_headers)

    def list_models(self, *, live: bool = False) -> list[dict[str, Any]]:
        """Return available OpenAI models with metadata (optionally merging live ``/models``)."""
        catalog = [dict(m) for m in OPENAI_MODELS]
        if not live:
            return catalog

        known_ids = {m["model_id"] for m in catalog}
        live_ids = self._fetch_live_model_ids()
        for mid in live_ids:
            if mid not in known_ids:
                catalog.append({
                    "id": mid,
                    "model_id": mid,
                    "display_name": mid,
                    "context_window": 128_000,
                    "capabilities": ["chat"],
                    "reasoning": is_reasoning_model(mid),
                    "hosting_tier": "cloud",
                })
        return catalog

    def health_check(self, *, probe_upstream: bool = False) -> dict[str, Any]:
        """Verify credential configuration and optionally ping ``GET /models``."""
        has_creds = bool(self._api_key or self._azure_ad_token or self._client is not None)
        if not has_creds:
            return {"healthy": False, "provider": "openai", "message": "Missing credentials"}
        if not probe_upstream:
            return {
                "healthy": True,
                "provider": "azure_openai" if self.is_azure else "openai",
                "default_model": self._default_model,
                "api_mode": self._api_mode,
                "message": "ready",
            }
        try:
            models = self._fetch_live_model_ids()
            return {
                "healthy": True,
                "provider": "azure_openai" if self.is_azure else "openai",
                "models_discovered": len(models),
                "message": "upstream reachable",
            }
        except Exception as exc:
            return {
                "healthy": False,
                "provider": "azure_openai" if self.is_azure else "openai",
                "message": str(exc),
            }

    # ------------------------------------------------------------------
    # Payload Builders (Dialect Adaptation)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chat_payload(
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        system_prompt: str,
        reasoning: bool,
        json_mode: bool,
        response_schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float | None,
        reasoning_effort: str,
        prompt_cache_key: str,
    ) -> dict[str, Any]:
        instruction_role = "developer" if reasoning else "system"
        formatted_msgs: list[dict[str, Any]] = []
        all_texts: list[str] = []

        sys_parts: list[str] = [system_prompt] if system_prompt else []

        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            all_texts.append(content)
            if role in ("system", "developer"):
                if content:
                    sys_parts.append(content)
                continue
            formatted_msgs.append({"role": role, "content": content})

        if json_mode and response_schema is None:
            combined_so_far = sys_parts + all_texts
            if not _contains_json_keyword(combined_so_far):
                sys_parts.append("Respond in valid JSON.")

        if sys_parts:
            formatted_msgs.insert(
                0, {"role": instruction_role, "content": "\n".join(sys_parts)}
            )

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": formatted_msgs,
        }

        if reasoning:
            payload["max_completion_tokens"] = max_tokens
            if reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
            # Reasoning models reject arbitrary temperatures (only 1.0 is accepted on some)
            if temperature == 1.0:
                payload["temperature"] = 1.0
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = 0.7 if temperature is None else float(temperature)

        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "aurarouter_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}

        if prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key

        return payload

    @staticmethod
    def _build_responses_payload(
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        system_prompt: str,
        reasoning: bool,
        json_mode: bool,
        response_schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float | None,
        reasoning_effort: str,
        prompt_cache_key: str,
    ) -> dict[str, Any]:
        instructions_parts: list[str] = [system_prompt] if system_prompt else []
        input_items: list[dict[str, Any]] = []
        all_texts: list[str] = []

        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            all_texts.append(content)
            if role in ("system", "developer"):
                if content:
                    instructions_parts.append(content)
                continue
            input_items.append({"role": role, "content": content})

        if json_mode and response_schema is None:
            if not _contains_json_keyword(instructions_parts + all_texts):
                instructions_parts.append("Respond in valid JSON.")

        payload: dict[str, Any] = {
            "model": model_id,
            "input": input_items,
            "max_output_tokens": max_tokens,
        }
        if instructions_parts:
            payload["instructions"] = "\n".join(instructions_parts)

        if reasoning:
            if reasoning_effort:
                payload["reasoning"] = {"effort": reasoning_effort}
        elif temperature is not None:
            payload["temperature"] = float(temperature)

        if response_schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "aurarouter_response",
                    "strict": True,
                    "schema": response_schema,
                }
            }
        elif json_mode:
            payload["text"] = {"format": {"type": "json_object"}}

        if prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key

        return payload

    # ------------------------------------------------------------------
    # Transport Invokers (SDK + httpx fallback)
    # ------------------------------------------------------------------

    def _build_http_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.is_azure:
            if self._azure_ad_token:
                headers["Authorization"] = f"Bearer {self._azure_ad_token}"
            elif self._api_key:
                headers["api-key"] = self._api_key
        else:
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            if self._organization:
                headers["OpenAI-Organization"] = self._organization
            if self._project:
                headers["OpenAI-Project"] = self._project
        return headers

    def _build_http_url(self, model_id: str, path: str) -> tuple[str, dict[str, str]]:
        if self.is_azure:
            deployment = model_id.split("/")[-1]
            if path == "/models":
                url = f"{self._azure_endpoint}/openai/models"
            else:
                url = f"{self._azure_endpoint}/openai/deployments/{deployment}{path}"
            return url, {"api-version": self._api_version}
        return f"{self._base_url}{path}", {}

    def _invoke_chat_completions(
        self, model_id: str, payload: dict[str, Any]
    ) -> tuple[Any, dict[str, str], float]:
        if self._client is not None:
            return self._invoke_sdk_endpoint(
                resource=self._client.chat.completions, payload=payload
            )
        return self._invoke_httpx(model_id, "/chat/completions", payload)

    def _invoke_responses(
        self, model_id: str, payload: dict[str, Any]
    ) -> tuple[Any, dict[str, str], float]:
        if self._client is not None and hasattr(self._client, "responses"):
            return self._invoke_sdk_endpoint(
                resource=self._client.responses, payload=payload
            )
        return self._invoke_httpx(model_id, "/responses", payload)

    def _invoke_sdk_endpoint(
        self, resource: Any, payload: dict[str, Any]
    ) -> tuple[Any, dict[str, str], float]:
        t0 = time.monotonic()
        try:
            if _REAL_OPENAI_CLS and isinstance(self._client, _REAL_OPENAI_CLS):
                raw = resource.with_raw_response.create(**payload)
                elapsed = time.monotonic() - t0
                parsed = raw.parse()
                headers = _extract_ratelimit_headers(getattr(raw, "headers", None))
                return parsed, headers, elapsed

            response = resource.create(**payload)
            elapsed = time.monotonic() - t0
            mock_hdrs = getattr(response, "_raw_headers", None)
            headers = _extract_ratelimit_headers(mock_hdrs) if isinstance(mock_hdrs, dict) else {}
            return response, headers, elapsed
        except Exception as exc:
            self._translate_sdk_exception(exc)
            raise

    def _invoke_httpx(
        self, model_id: str, path: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, str], float]:
        url, query_params = self._build_http_url(model_id, path)
        headers = self._build_http_headers()
        req_body = dict(payload)
        if self.is_azure and path != "/responses":
            # Azure deployment URL already encodes the deployment; keep model optional
            req_body.pop("model", None)

        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=self._timeout) as http_client:
                resp = http_client.post(
                    url, params=query_params or None, json=req_body, headers=headers
                )
                elapsed = time.monotonic() - t0
                rl_headers = _extract_ratelimit_headers(resp.headers)
                if resp.status_code >= 400:
                    self._raise_from_http_error(resp, rl_headers)
                return resp.json(), rl_headers, elapsed
        except (OpenAIRateLimitError, OpenAIContextLengthError, OpenAITransientError):
            raise
        except httpx.TimeoutException as exc:
            raise OpenAITransientError(f"OpenAI request timed out: {exc}", status_code=504) from exc
        except httpx.RequestError as exc:
            raise OpenAITransientError(f"OpenAI connection error: {exc}", status_code=503) from exc

    def _fetch_live_model_ids(self) -> list[str]:
        if self._client is not None and hasattr(self._client, "models"):
            page = self._client.models.list()
            items = getattr(page, "data", page) or []
            return [str(getattr(m, "id", "")) for m in items if getattr(m, "id", "")]

        url, query_params = self._build_http_url(self._default_model, "/models")
        headers = self._build_http_headers()
        headers.pop("Content-Type", None)
        with httpx.Client(timeout=min(10.0, self._timeout)) as http_client:
            resp = http_client.get(url, params=query_params or None, headers=headers)
            resp.raise_for_status()
            body = resp.json()
        return [
            str(item["id"])
            for item in (body.get("data") or [])
            if isinstance(item, dict) and item.get("id")
        ]

    # ------------------------------------------------------------------
    # Response Parsers & Error Mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_chat_output(
        data: Any,
        model_id: str,
        ctx_limit: int,
        elapsed: float,
        rl_headers: dict[str, str],
    ) -> dict[str, Any]:
        if isinstance(data, dict):
            choices = data.get("choices") or []
            msg = (choices[0].get("message") or {}) if choices else {}
            text = msg.get("content") or ""
            usage = data.get("usage") or {}
            in_tok = int(usage.get("prompt_tokens", 0) or 0)
            out_tok = int(usage.get("completion_tokens", 0) or 0)
            prompt_details = usage.get("prompt_tokens_details") or {}
            comp_details = usage.get("completion_tokens_details") or {}
            cached_tok = int(prompt_details.get("cached_tokens", 0) or 0)
            reasoning_tok = int(comp_details.get("reasoning_tokens", 0) or 0)
            resolved_model = str(data.get("model") or model_id)
        else:
            choices = getattr(data, "choices", None) or []
            first_msg = getattr(choices[0], "message", None) if choices else None
            text = getattr(first_msg, "content", "") or ""
            usage = getattr(data, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            p_det = getattr(usage, "prompt_tokens_details", None)
            c_det = getattr(usage, "completion_tokens_details", None)
            cached_tok = int(getattr(p_det, "cached_tokens", 0) or 0)
            reasoning_tok = int(getattr(c_det, "reasoning_tokens", 0) or 0)
            resolved_model = str(getattr(data, "model", None) or model_id)

        result: dict[str, Any] = {
            "text": text,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cached_input_tokens": cached_tok,
            "reasoning_tokens": reasoning_tok,
            "model_id": resolved_model,
            "context_limit": ctx_limit,
            "ttft_s": elapsed,
        }
        if rl_headers:
            result["rate_limit"] = rl_headers
        return result

    @staticmethod
    def _parse_responses_output(
        data: Any,
        model_id: str,
        ctx_limit: int,
        elapsed: float,
        rl_headers: dict[str, str],
    ) -> dict[str, Any]:
        if isinstance(data, dict):
            text = data.get("output_text") or ""
            if not text:
                parts: list[str] = []
                for item in data.get("output") or []:
                    if isinstance(item, dict):
                        for blk in item.get("content") or []:
                            if isinstance(blk, dict) and blk.get("text"):
                                parts.append(str(blk["text"]))
                text = "".join(parts)
            usage = data.get("usage") or {}
            in_tok = int(usage.get("input_tokens", 0) or 0)
            out_tok = int(usage.get("output_tokens", 0) or 0)
            in_det = usage.get("input_tokens_details") or {}
            out_det = usage.get("output_tokens_details") or {}
            cached_tok = int(in_det.get("cached_tokens", 0) or 0)
            reasoning_tok = int(out_det.get("reasoning_tokens", 0) or 0)
            resolved_model = str(data.get("model") or model_id)
        else:
            text = getattr(data, "output_text", "") or ""
            usage = getattr(data, "usage", None)
            in_tok = int(getattr(usage, "input_tokens", 0) or 0)
            out_tok = int(getattr(usage, "output_tokens", 0) or 0)
            in_det = getattr(usage, "input_tokens_details", None)
            out_det = getattr(usage, "output_tokens_details", None)
            cached_tok = int(getattr(in_det, "cached_tokens", 0) or 0)
            reasoning_tok = int(getattr(out_det, "reasoning_tokens", 0) or 0)
            resolved_model = str(getattr(data, "model", None) or model_id)

        result: dict[str, Any] = {
            "text": text,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cached_input_tokens": cached_tok,
            "reasoning_tokens": reasoning_tok,
            "model_id": resolved_model,
            "context_limit": ctx_limit,
            "ttft_s": elapsed,
        }
        if rl_headers:
            result["rate_limit"] = rl_headers
        return result

    @staticmethod
    def _raise_from_http_error(
        resp: httpx.Response, rl_headers: dict[str, str]
    ) -> None:
        status = resp.status_code
        body_text = ""
        err_code = ""
        try:
            body_text = resp.text
            parsed = json.loads(body_text)
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
                err_code = str(parsed["error"].get("code") or "")
        except Exception:
            pass

        msg = f"HTTP {status}: {body_text[:500]}" if body_text else f"HTTP {status}"
        err_code_l = err_code.lower()
        if status in (429, 529) or err_code_l in ("rate_limit_exceeded", "insufficient_quota"):
            hdrs_out = dict(rl_headers)
            retry_s: float | None = None
            if err_code_l == "insufficient_quota":
                hdrs_out["is_daily_quota"] = "true"  # type: ignore[assignment]
                retry_s = 3600.0
            elif status == 529 and not any(k.lower().startswith("retry-after") for k in hdrs_out):
                hdrs_out["retry-after"] = "5"
                retry_s = 5.0
            raise OpenAIRateLimitError(msg, headers=hdrs_out, retry_after_s=retry_s)
        if status in (400, 413, 422):
            combined = f"{err_code} {body_text}".lower()
            if any(kw in combined for kw in _OPENAI_CTX_KEYWORDS):
                raise OpenAIContextLengthError(msg)
        if status >= 500:
            raise OpenAITransientError(msg, status_code=status)
        resp.raise_for_status()

    @staticmethod
    def _translate_sdk_exception(exc: Exception) -> None:
        cls_name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        resp = getattr(exc, "response", None)
        hdrs = _extract_ratelimit_headers(getattr(resp, "headers", None))
        err_code = str(getattr(exc, "code", "") or "")
        err_code_l = err_code.lower()

        if (
            cls_name == "RateLimitError"
            or status in (429, 529)
            or err_code_l in ("rate_limit_exceeded", "insufficient_quota")
        ):
            retry_s: float | None = None
            if err_code_l == "insufficient_quota":
                retry_s = 3600.0
            elif status == 529 and not any(k.lower().startswith("retry-after") for k in hdrs):
                hdrs["retry-after"] = "5"
                retry_s = 5.0
            raise OpenAIRateLimitError(str(exc), headers=hdrs, retry_after_s=retry_s) from exc
        lower = f"{err_code} {exc}".lower()
        if cls_name == "BadRequestError" or status in (400, 413, 422):
            if any(kw in lower for kw in _OPENAI_CTX_KEYWORDS):
                raise OpenAIContextLengthError(str(exc)) from exc

        if cls_name in ("InternalServerError", "APIConnectionError", "APITimeoutError") or (
            isinstance(status, int) and status >= 500
        ):
            raise OpenAITransientError(
                str(exc), status_code=status if isinstance(status, int) else 503
            ) from exc

