"""Shared fixtures for aurarouter-openai tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest


@dataclass
class MockPromptDetails:
    cached_tokens: int = 0


@dataclass
class MockCompletionDetails:
    reasoning_tokens: int = 0


@dataclass
class MockChatUsage:
    prompt_tokens: int = 120
    completion_tokens: int = 40
    prompt_tokens_details: MockPromptDetails = field(default_factory=MockPromptDetails)
    completion_tokens_details: MockCompletionDetails = field(default_factory=MockCompletionDetails)


@dataclass
class MockChatMessage:
    content: str = "Hello from GPT!"
    role: str = "assistant"


@dataclass
class MockChatChoice:
    message: MockChatMessage = field(default_factory=MockChatMessage)
    finish_reason: str = "stop"


@dataclass
class MockChatCompletion:
    id: str = "chatcmpl-test123"
    model: str = "gpt-4o"
    choices: list[MockChatChoice] = field(default_factory=lambda: [MockChatChoice()])
    usage: MockChatUsage = field(default_factory=MockChatUsage)


def make_mock_chat_completion(
    text: str = "Hello from GPT!",
    prompt_tokens: int = 120,
    completion_tokens: int = 40,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    model: str = "gpt-4o",
) -> MockChatCompletion:
    return MockChatCompletion(
        model=model,
        choices=[MockChatChoice(message=MockChatMessage(content=text))],
        usage=MockChatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=MockPromptDetails(cached_tokens=cached_tokens),
            completion_tokens_details=MockCompletionDetails(reasoning_tokens=reasoning_tokens),
        ),
    )


@pytest.fixture
def sdk_mock_provider():
    """OpenAIProvider wired to an injected mock SDK client."""
    from aurarouter_openai.provider import OpenAIProvider

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_mock_chat_completion()
    prov = OpenAIProvider(api_key="sk-test-openai-key", client=mock_client)
    return prov, mock_client

