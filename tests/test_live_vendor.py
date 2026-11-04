"""Live vendor canary test for aurarouter-openai.

Requires OPENAI_API_KEY environment variable.
Exercises live OpenAI Platform API auth, response generation, and Go-duration ratelimit headers.
Hard spend guard: max_tokens=16, lowest cost mini tier (< $0.0001 per test).
"""

from __future__ import annotations

import os
import pytest
from aurarouter_openai.provider import OpenAIProvider


@pytest.mark.live_vendor
class TestLiveOpenAIVendor:
    """Canary tests against live OpenAI Platform API."""

    @pytest.fixture(autouse=True)
    def skip_if_no_key(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")

    def test_live_openai_generation_and_ratelimit_headers(self):
        prov = OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            default_model="gpt-4o-mini",
        )
        result = prov.generate_with_usage(
            "Respond with the single word: OK",
            max_tokens=16,
        )
        assert result.text is not None
        assert len(result.text.strip()) > 0
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        # If headers were returned by upstream, ensure they were captured
        if result.rate_limit is not None:
            assert result.rate_limit.rpm_remaining is not None or result.rate_limit.itpm_remaining is not None

