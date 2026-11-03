"""OpenAI model catalog -- IDs, context windows, reasoning flags, and pricing."""

from __future__ import annotations

import re

OPENAI_MODELS: list[dict] = [
    {
        "id": "gpt-5",
        "model_id": "gpt-5",
        "display_name": "GPT-5",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "cost_per_1m_input": 1.25,
        "cost_per_1m_output": 10.00,
        "capabilities": ["code", "reasoning", "chat", "structured_outputs"],
        "reasoning": True,
        "hosting_tier": "cloud",
    },
    {
        "id": "gpt-5-mini",
        "model_id": "gpt-5-mini",
        "display_name": "GPT-5 Mini",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "cost_per_1m_input": 0.25,
        "cost_per_1m_output": 2.00,
        "capabilities": ["code", "reasoning", "chat", "structured_outputs"],
        "reasoning": True,
        "hosting_tier": "cloud",
    },
    {
        "id": "gpt-4.1",
        "model_id": "gpt-4.1",
        "display_name": "GPT-4.1",
        "context_window": 1_047_576,
        "max_output_tokens": 32_768,
        "cost_per_1m_input": 2.00,
        "cost_per_1m_output": 8.00,
        "capabilities": ["code", "reasoning", "chat", "structured_outputs"],
        "reasoning": False,
        "hosting_tier": "cloud",
    },
    {
        "id": "gpt-4.1-mini",
        "model_id": "gpt-4.1-mini",
        "display_name": "GPT-4.1 Mini",
        "context_window": 1_047_576,
        "max_output_tokens": 32_768,
        "cost_per_1m_input": 0.40,
        "cost_per_1m_output": 1.60,
        "capabilities": ["code", "chat", "structured_outputs"],
        "reasoning": False,
        "hosting_tier": "cloud",
    },
    {
        "id": "gpt-4.1-nano",
        "model_id": "gpt-4.1-nano",
        "display_name": "GPT-4.1 Nano",
        "context_window": 1_047_576,
        "max_output_tokens": 32_768,
        "cost_per_1m_input": 0.10,
        "cost_per_1m_output": 0.40,
        "capabilities": ["chat", "structured_outputs"],
        "reasoning": False,
        "hosting_tier": "cloud",
    },
    {
        "id": "gpt-4o",
        "model_id": "gpt-4o",
        "display_name": "GPT-4o",
        "context_window": 128_000,
        "max_output_tokens": 16_384,
        "cost_per_1m_input": 2.50,
        "cost_per_1m_output": 10.00,
        "capabilities": ["code", "reasoning", "chat", "structured_outputs"],
        "reasoning": False,
        "hosting_tier": "cloud",
    },
    {
        "id": "gpt-4o-mini",
        "model_id": "gpt-4o-mini",
        "display_name": "GPT-4o Mini",
        "context_window": 128_000,
        "max_output_tokens": 16_384,
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
        "capabilities": ["code", "chat", "structured_outputs"],
        "reasoning": False,
        "hosting_tier": "cloud",
    },
    {
        "id": "o3",
        "model_id": "o3",
        "display_name": "OpenAI o3",
        "context_window": 200_000,
        "max_output_tokens": 100_000,
        "cost_per_1m_input": 2.00,
        "cost_per_1m_output": 8.00,
        "capabilities": ["code", "reasoning", "structured_outputs"],
        "reasoning": True,
        "hosting_tier": "cloud",
    },
    {
        "id": "o4-mini",
        "model_id": "o4-mini",
        "display_name": "OpenAI o4-mini",
        "context_window": 200_000,
        "max_output_tokens": 100_000,
        "cost_per_1m_input": 1.10,
        "cost_per_1m_output": 4.40,
        "capabilities": ["code", "reasoning", "structured_outputs"],
        "reasoning": True,
        "hosting_tier": "cloud",
    },
    {
        "id": "o3-mini",
        "model_id": "o3-mini",
        "display_name": "OpenAI o3-mini",
        "context_window": 200_000,
        "max_output_tokens": 100_000,
        "cost_per_1m_input": 1.10,
        "cost_per_1m_output": 4.40,
        "capabilities": ["code", "reasoning", "structured_outputs"],
        "reasoning": True,
        "hosting_tier": "cloud",
    },
    {
        "id": "o1",
        "model_id": "o1",
        "display_name": "OpenAI o1",
        "context_window": 200_000,
        "max_output_tokens": 100_000,
        "cost_per_1m_input": 15.00,
        "cost_per_1m_output": 60.00,
        "capabilities": ["code", "reasoning", "structured_outputs"],
        "reasoning": True,
        "hosting_tier": "cloud",
    },
]

_REASONING_PREFIX_RE = re.compile(r"^(?:o1|o3|o4|gpt-5)(?:[-_]|$)", re.IGNORECASE)


def is_reasoning_model(model_id: str) -> bool:
    """Return True if *model_id* uses OpenAI's reasoning parameter dialect.

    Reasoning models (``o1*``, ``o3*``, ``o4*``, ``gpt-5*``):
    - require ``max_completion_tokens`` instead of ``max_tokens`` in Chat Completions
    - map ``system`` messages to ``developer`` instructions
    - support ``reasoning_effort`` and reject arbitrary ``temperature`` values
    """
    if not model_id:
        return False
    # Strip optional provider prefix like 'openai/o3'
    leaf = model_id.split("/")[-1].strip()
    info = get_model_info(leaf)
    if info is not None:
        return bool(info.get("reasoning", False))
    return bool(_REASONING_PREFIX_RE.match(leaf))


def get_model_info(model_id: str) -> dict | None:
    """Look up a model by its ID (or dated snapshot prefix). Returns None if not found."""
    leaf = (model_id or "").split("/")[-1].strip()
    for model in OPENAI_MODELS:
        if model["model_id"] == leaf:
            return dict(model)
    # Match dated snapshots like 'gpt-4o-2024-11-20' or 'o3-2025-04-16'
    for model in sorted(OPENAI_MODELS, key=lambda m: len(m["model_id"]), reverse=True):
        if leaf.startswith(model["model_id"] + "-"):
            return dict(model)
    return None


def get_default_model_id() -> str:
    """Return the default OpenAI model ID."""
    return "gpt-4o"

