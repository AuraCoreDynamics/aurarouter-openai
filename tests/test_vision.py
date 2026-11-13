"""Unit tests for aurarouter-openai multimodal vision support."""

from __future__ import annotations

import base64
import json

from aurarouter_openai.models import OPENAI_MODELS
from aurarouter_openai.provider import OpenAIProvider
from aurarouter_openai.server import create_server


class TestOpenAIVision:
    def test_catalog_vision_capabilities(self):
        by_id = {m["model_id"]: m for m in OPENAI_MODELS}
        assert "vision" in by_id["gpt-4o"]["capabilities"]
        assert "vision" in by_id["gpt-4.1"]["capabilities"]
        assert "vision" not in by_id["o3-mini"].get("capabilities", [])

    def test_content_formatting_helpers(self):
        b64 = base64.b64encode(b"img").decode("ascii")
        imgs = [{"data": b64, "mime_type": "image/jpeg"}]

        chat_blocks, _ = OpenAIProvider._format_chat_content("Describe", imgs)
        assert isinstance(chat_blocks, list)
        assert chat_blocks[0] == {"type": "text", "text": "Describe"}
        assert chat_blocks[1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }

        resp_blocks, _ = OpenAIProvider._format_responses_content("Describe", imgs)
        assert isinstance(resp_blocks, list)
        assert resp_blocks[0] == {"type": "input_text", "text": "Describe"}
        assert resp_blocks[1] == {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}",
        }

    def test_provider_generate_and_history_with_images(self, sdk_mock_provider):
        prov, mock_client = sdk_mock_provider
        b64 = base64.b64encode(b"img").decode("ascii")
        res = prov.generate("Extract objects", images=[{"data": b64, "mime_type": "image/png"}])
        assert res["text"] == "Hello from GPT!"
        sent_msgs = mock_client.chat.completions.create.call_args[1]["messages"]
        assert isinstance(sent_msgs[0]["content"], list)
        assert sent_msgs[0]["content"][1]["type"] == "image_url"

        prov.generate_with_history([
            {"role": "user", "content": "Analyze", "images": [{"data": b64, "mime_type": "image/png"}]}
        ])
        hist_msgs = mock_client.chat.completions.create.call_args[1]["messages"]
        assert isinstance(hist_msgs[0]["content"], list)
        assert hist_msgs[0]["content"][1]["type"] == "image_url"

    def test_mcp_generate_tool_forwards_images(self, sdk_mock_provider):
        prov, mock_client = sdk_mock_provider
        mcp = create_server(provider=prov)
        tool_map = {t.name: t.fn for t in mcp._tool_manager.list_tools()}
        b64 = base64.b64encode(b"img").decode("ascii")
        out = json.loads(
            tool_map["provider.generate"](
                prompt="OCR",
                images=[{"data": b64, "mime_type": "image/png"}],
            )
        )
        assert out["text"] == "Hello from GPT!"
        sent_msgs = mock_client.chat.completions.create.call_args[1]["messages"]
        assert sent_msgs[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
