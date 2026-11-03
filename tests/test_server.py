"""Tests for aurarouter-openai FastMCP server, protocol compliance, and entry point."""

from __future__ import annotations

import json
import subprocess
import sys

from aurarouter.providers.protocol import ALL_TOOLS, validate_provider_tools

from aurarouter_openai import __version__, get_provider_metadata
from aurarouter_openai.server import create_server


class TestProtocolComplianceAndServer:
    def test_entry_point_metadata(self):
        meta = get_provider_metadata()
        assert meta.name == "openai"
        assert meta.provider_type == "mcp"
        assert meta.version == __version__
        assert meta.command == ["python", "-m", "aurarouter_openai"]
        assert "api_key" in meta.requires_config

    def test_lazy_server_creation_and_protocol_validation(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mcp = create_server()
        tools = [{"name": t.name} for t in mcp._tool_manager.list_tools()]
        valid, errors = validate_provider_tools(tools)
        assert valid is True, errors
        assert {t["name"] for t in tools} == ALL_TOOLS

    def test_mcp_tool_invocations_and_capabilities(self, sdk_mock_provider):
        prov, _mock_client = sdk_mock_provider
        mcp = create_server(provider=prov)
        tool_map = {t.name: t.fn for t in mcp._tool_manager.list_tools()}

        gen_out = json.loads(tool_map["provider.generate"](prompt="Hi"))
        assert gen_out["text"] == "Hello from GPT!"
        assert gen_out["model_id"] == "gpt-4o"

        # Accepts both list[dict] and JSON-encoded string for messages
        hist_list = json.loads(
            tool_map["provider.generate_with_history"](
                messages=[{"role": "user", "content": "Turn 1"}]
            )
        )
        assert hist_list["text"] == "Hello from GPT!"

        hist_str = json.loads(
            tool_map["provider.generate_with_history"](
                messages=json.dumps([{"role": "user", "content": "Turn 1"}])
            )
        )
        assert hist_str["text"] == "Hello from GPT!"

        models = json.loads(tool_map["provider.list_models"]())
        assert any(m["model_id"] == "gpt-5" for m in models)
        assert any(m["model_id"] == "o3" for m in models)

        hc = json.loads(tool_map["provider.health_check"]())
        assert hc["healthy"] is True

        caps = json.loads(tool_map["provider.capabilities"]())
        assert caps["provider"] == "openai"
        assert caps["features"]["reasoning_models"] is True
        assert caps["features"]["responses_api"] is True

    def test_cli_help_flag(self):
        import os

        env = dict(os.environ)
        env["PYTHONPATH"] = r"c:\projects\auracore\aurarouter-openai\src"
        res = subprocess.run(
            [sys.executable, "-m", "aurarouter_openai", "--help"],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        assert res.returncode == 0
        assert "aurarouter-openai" in res.stdout
        assert "OPENAI_API_KEY" in res.stdout

