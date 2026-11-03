"""Cross-MCP v2.1 telemetry & error-envelope integration tests between
aurarouter-openai and aurarouter's McpProvider + RateLimitTracker.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
import respx

from aurarouter.errors import (
    ContextLengthExceededError,
    ProviderTransientError,
    RateLimitExceededError,
)
from aurarouter.providers.mcp_provider import McpProvider
from aurarouter.ratelimit.models import RateState
from aurarouter.ratelimit.tracker import RateLimitTracker
from aurarouter_openai.provider import OpenAIProvider
from aurarouter_openai.server import create_server


def _wire_inprocess_mcp_provider(openai_prov: OpenAIProvider, model_name: str = "gpt-4o") -> McpProvider:
    """Wire aurarouter's McpProvider directly to aurarouter-openai's FastMCP tools."""
    mcp_srv = create_server(provider=openai_prov)
    tool_objs = mcp_srv._tool_manager.list_tools()
    tool_fns = {t.name: t.fn for t in tool_objs}

    mcp_prov = McpProvider({"mcp_endpoint": "http://openai-mcp.local", "model_name": model_name})
    fake_client = MagicMock()
    fake_client.connected = True
    fake_client.connect.return_value = True
    fake_client.get_tools.return_value = [{"name": t.name} for t in tool_objs]
    fake_client.get_capabilities.return_value = set(tool_fns.keys())
    fake_client.last_response_headers = {}
    fake_client.call_tool.side_effect = lambda name, **kwargs: tool_fns[name](**kwargs)
    mcp_prov._client = fake_client
    return mcp_prov


class TestCrossMcpOpenAITelemetry:
    @respx.mock
    def test_200_go_duration_headers_flow_into_generate_result_and_tracker(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                headers={
                    "x-ratelimit-limit-requests": "500",
                    "x-ratelimit-remaining-requests": "50",
                    "x-ratelimit-reset-requests": "6s",
                    "x-ratelimit-limit-tokens": "200000",
                    "x-ratelimit-remaining-tokens": "20000",
                    "x-ratelimit-reset-tokens": "1m30s",
                },
                json={
                    "model": "gpt-4o",
                    "choices": [{"message": {"content": "Telemetry OK"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 100},
                },
            )
        )

        oa_prov = OpenAIProvider(api_key="sk-test")
        oa_prov._client = None
        mcp_prov = _wire_inprocess_mcp_provider(oa_prov, model_name="gpt-4o")

        res = mcp_prov.generate_with_usage("Hello")
        assert res.text == "Telemetry OK"
        assert res.rate_limit is not None
        assert res.rate_limit.rpm_limit == 500
        assert res.rate_limit.rpm_remaining == 50
        assert res.rate_limit.rpm_reset_s == pytest.approx(6.0)
        assert res.rate_limit.itpm_limit == 200000
        assert res.rate_limit.itpm_remaining == 20000
        assert res.rate_limit.itpm_reset_s == pytest.approx(90.0)

        tracker = RateLimitTracker()
        lease = tracker.reserve("openai/gpt-4o", "openai", est_input_tokens=500, est_output_tokens=100)
        tracker.settle(
            lease,
            actual_input_tokens=res.input_tokens,
            actual_output_tokens=res.output_tokens,
            snapshot=res.rate_limit,
            ttft_s=res.ttft_s,
        )
        util = tracker.get_utilization("openai/gpt-4o", "openai")
        # 450/500 = 0.90 utilization -> NEARING_SATURATION
        assert util.state == RateState.NEARING_SATURATION

    @respx.mock
    def test_429_400_503_envelopes_raise_typed_aurarouter_exceptions(self):
        oa_prov = OpenAIProvider(api_key="sk-test")
        oa_prov._client = None
        mcp_prov = _wire_inprocess_mcp_provider(oa_prov, model_name="gpt-4o")

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                429,
                headers={"retry-after": "25", "x-ratelimit-remaining-requests": "0"},
                json={"error": {"message": "429 quota", "code": "rate_limit_exceeded"}},
            )
        )
        with pytest.raises(RateLimitExceededError) as rl_err:
            mcp_prov.generate_with_usage("test 429")
        assert rl_err.value.snapshot is not None
        assert rl_err.value.snapshot.retry_after_s == pytest.approx(25.0)
        assert rl_err.value.snapshot.rpm_remaining == 0

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                400,
                json={"error": {"message": "maximum context length is 128000", "code": "context_length_exceeded"}},
            )
        )
        with pytest.raises(ContextLengthExceededError):
            mcp_prov.generate_with_history([{"role": "user", "content": "huge"}])

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(502, text="Bad Gateway")
        )
        with pytest.raises(ProviderTransientError) as tr_err:
            mcp_prov.generate_with_usage("test 502")
        assert tr_err.value.status_code == 502
