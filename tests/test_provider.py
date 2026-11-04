"""Unit tests for OpenAIProvider (chat, reasoning o-series/GPT-5, Responses API, Azure, httpx)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from aurarouter_openai.models import get_default_model_id, get_model_info, is_reasoning_model
from aurarouter_openai.provider import (
    OpenAIContextLengthError,
    OpenAIProvider,
    OpenAIRateLimitError,
    OpenAITransientError,
)
try:
    from conftest import make_mock_chat_completion
except ImportError:
    from tests.conftest import make_mock_chat_completion


class TestModelCatalog:
    def test_reasoning_detection(self):
        assert is_reasoning_model("o1") is True
        assert is_reasoning_model("o3") is True
        assert is_reasoning_model("o3-2025-04-16") is True
        assert is_reasoning_model("o4-mini") is True
        assert is_reasoning_model("gpt-5") is True
        assert is_reasoning_model("gpt-5-mini") is True
        assert is_reasoning_model("openai/o3") is True

        assert is_reasoning_model("gpt-4o") is False
        assert is_reasoning_model("gpt-4o-mini") is False
        assert is_reasoning_model("gpt-4.1") is False

    def test_model_info_lookup_and_dated_snapshots(self):
        info = get_model_info("gpt-4o-2024-11-20")
        assert info is not None
        assert info["model_id"] == "gpt-4o"
        assert info["hosting_tier"] == "cloud"
        assert get_default_model_id() == "gpt-4o"


class TestOpenAIProviderSDKAndDialect:
    def test_missing_credentials_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
        with pytest.raises(ValueError, match="No OpenAI credentials"):
            OpenAIProvider()

    def test_standard_chat_model_payload(self, sdk_mock_provider):
        prov, mock_client = sdk_mock_provider
        mock_client.chat.completions.create.return_value = make_mock_chat_completion(
            text='{"answer": 42}',
            prompt_tokens=200,
            completion_tokens=15,
            cached_tokens=128,
        )

        out = prov.generate_with_history(
            messages=[{"role": "user", "content": "Compute 6*7"}],
            system_prompt="You are a calculator.",
            model="gpt-4o",
            json_mode=True,
            prompt_cache_key="sess-abc",
        )

        assert out["text"] == '{"answer": 42}'
        assert out["input_tokens"] == 200
        assert out["output_tokens"] == 15
        assert out["cached_input_tokens"] == 128
        assert out["context_limit"] == 128_000

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["max_tokens"] == 4096
        assert "max_completion_tokens" not in call_kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["prompt_cache_key"] == "sess-abc"
        assert call_kwargs["response_format"] == {"type": "json_object"}
        # System role used for non-reasoning model + auto-injected 'JSON' keyword
        assert call_kwargs["messages"][0]["role"] == "system"
        assert "json" in call_kwargs["messages"][0]["content"].lower()

    def test_reasoning_model_dialect_adaptation_o3_and_gpt5(self, sdk_mock_provider):
        prov, mock_client = sdk_mock_provider
        mock_client.chat.completions.create.return_value = make_mock_chat_completion(
            text="Proof complete.",
            prompt_tokens=100,
            completion_tokens=300,
            reasoning_tokens=220,
            model="o3",
        )

        out = prov.generate_with_history(
            messages=[
                {"role": "system", "content": "Be rigorous."},
                {"role": "user", "content": "Prove sqrt(2) is irrational."},
            ],
            system_prompt="Top-level instruction.",
            model="o3",
            temperature=0.4,  # Must be stripped for reasoning models!
            reasoning_effort="high",
            max_tokens=8192,
        )

        assert out["text"] == "Proof complete."
        assert out["reasoning_tokens"] == 220

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "o3"
        assert call_kwargs["max_completion_tokens"] == 8192
        assert "max_tokens" not in call_kwargs
        assert "temperature" not in call_kwargs  # 0.4 stripped to prevent OpenAI 400!
        assert call_kwargs["reasoning_effort"] == "high"
        # System instructions folded into 'developer' role!
        assert call_kwargs["messages"][0]["role"] == "developer"
        assert "Top-level instruction." in call_kwargs["messages"][0]["content"]
        assert "Be rigorous." in call_kwargs["messages"][0]["content"]

    def test_structured_outputs_response_schema(self, sdk_mock_provider):
        prov, mock_client = sdk_mock_provider
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        prov.generate("Check status", model="gpt-4.1-mini", response_schema=schema)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True
        assert call_kwargs["response_format"]["json_schema"]["schema"] == schema


class TestHttpxNativeResponsesAndAzure:
    @respx.mock
    def test_httpx_responses_api_mode(self):
        route = respx.post("https://api.openai.com/v1/responses").mock(
            return_value=httpx.Response(
                200,
                headers={
                    "x-ratelimit-limit-requests": "500",
                    "x-ratelimit-remaining-requests": "498",
                },
                json={
                    "id": "resp_1",
                    "model": "gpt-5",
                    "output_text": "Responses API output",
                    "usage": {
                        "input_tokens": 80,
                        "output_tokens": 30,
                        "input_tokens_details": {"cached_tokens": 64},
                        "output_tokens_details": {"reasoning_tokens": 12},
                    },
                },
            )
        )

        prov = OpenAIProvider(api_key="sk-test", api_mode="responses")
        prov._client = None  # Force native httpx path
        res = prov.generate(
            "Hello responses API",
            model="gpt-5",
            reasoning_effort="medium",
            prompt_cache_key="cache-1",
        )

        assert route.called
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["model"] == "gpt-5"
        assert sent_body["max_output_tokens"] == 4096
        assert sent_body["reasoning"] == {"effort": "medium"}
        assert sent_body["prompt_cache_key"] == "cache-1"

        assert res["text"] == "Responses API output"
        assert res["input_tokens"] == 80
        assert res["output_tokens"] == 30
        assert res["cached_input_tokens"] == 64
        assert res["reasoning_tokens"] == 12
        assert res["rate_limit"]["x-ratelimit-remaining-requests"] == "498"

    @respx.mock
    def test_azure_openai_deployment_url_and_api_key_header(self):
        route = respx.post(
            "https://contoso.openai.azure.com/openai/deployments/my-gpt4o/chat/completions"
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "gpt-4o",
                    "choices": [{"message": {"content": "Azure OK"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                },
            )
        )

        prov = OpenAIProvider(
            api_key="az-secret-key",
            azure_endpoint="https://contoso.openai.azure.com",
            api_version="2025-01-01-preview",
        )
        prov._client = None
        out = prov.generate("Ping Azure", model="my-gpt4o")

        assert out["text"] == "Azure OK"
        req = route.calls[0].request
        assert req.headers["api-key"] == "az-secret-key"
        assert "Authorization" not in req.headers
        assert req.url.params["api-version"] == "2025-01-01-preview"

    @respx.mock
    def test_httpx_live_model_discovery_and_health_check(self):
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": "gpt-4o"}, {"id": "ft:gpt-4o-mini:custom-org"}]},
            )
        )
        prov = OpenAIProvider(api_key="sk-test")
        prov._client = None

        models = prov.list_models(live=True)
        ids = {m["model_id"] for m in models}
        assert "ft:gpt-4o-mini:custom-org" in ids

        hc = prov.health_check(probe_upstream=True)
        assert hc["healthy"] is True
        assert hc["models_discovered"] == 2

    @respx.mock
    def test_http_error_classification_429_400_503(self):
        prov = OpenAIProvider(api_key="sk-test")
        prov._client = None

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                429,
                headers={"retry-after": "12", "x-ratelimit-remaining-tokens": "0"},
                json={"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}},
            )
        )
        with pytest.raises(OpenAIRateLimitError) as rl_exc:
            prov.generate("burst")
        assert rl_exc.value.headers["x-ratelimit-remaining-tokens"] == "0"

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                400,
                json={
                    "error": {
                        "message": "This model's maximum context length is 128000 tokens.",
                        "code": "context_length_exceeded",
                    }
                },
            )
        )
        with pytest.raises(OpenAIContextLengthError):
            prov.generate("huge")

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(503, text="Upstream overloaded")
        )
        with pytest.raises(OpenAITransientError) as tr_exc:
            prov.generate("oops")
        assert tr_exc.value.status_code == 503

