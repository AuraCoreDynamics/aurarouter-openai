"""FastMCP server factory for aurarouter-openai.

Exposes all five AuraRouter Provider Protocol MCP tools:
- ``provider.generate``
- ``provider.generate_with_history``
- ``provider.list_models``
- ``provider.health_check``
- ``provider.capabilities``
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from aurarouter_openai import __version__
from aurarouter_openai.provider import (
    OpenAIContextLengthError,
    OpenAIProvider,
    OpenAIRateLimitError,
    OpenAITransientError,
)


def _serialize_error(exc: Exception) -> str:
    if isinstance(exc, OpenAIRateLimitError):
        payload: dict[str, Any] = {
            "error_type": "rate_limit_exceeded",
            "error": str(exc),
            "rate_limit": exc.headers,
        }
        if exc.retry_after_s is not None:
            payload["retry_after_s"] = exc.retry_after_s
        return json.dumps(payload)
    if isinstance(exc, OpenAIContextLengthError):
        payload = {
            "error_type": "context_length_exceeded",
            "error": str(exc),
        }
        if exc.max_tokens is not None:
            payload["max_tokens"] = exc.max_tokens
        return json.dumps(payload)
    if isinstance(exc, OpenAITransientError):
        return json.dumps({
            "error_type": "provider_transient_error",
            "error": str(exc),
            "status_code": exc.status_code,
        })
    raise exc


def create_server(
    api_key: str | None = None,
    *,
    provider: OpenAIProvider | None = None,
    **provider_kwargs: Any,
) -> FastMCP:
    """Create and return a configured FastMCP server instance for OpenAI / Azure OpenAI.

    Lazily instantiates :class:`OpenAIProvider` on first tool invocation when
    neither *provider* nor *api_key* is eagerly supplied, allowing tool discovery
    and ``--help`` inspection without requiring ``OPENAI_API_KEY`` upfront.
    """
    mcp = FastMCP("aurarouter-openai")
    holder: dict[str, OpenAIProvider | None] = {
        "instance": provider
        if provider is not None
        else (OpenAIProvider(api_key=api_key, **provider_kwargs) if api_key else None)
    }

    def _get_provider() -> OpenAIProvider:
        inst = holder["instance"]
        if inst is None:
            inst = OpenAIProvider(api_key=api_key, **provider_kwargs)
            holder["instance"] = inst
        return inst

    @mcp.tool(name="provider.generate")
    def generate(
        prompt: str,
        model: str = "",
        json_mode: bool = False,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        reasoning_effort: str = "",
        prompt_cache_key: str = "",
        api_mode: str = "",
    ) -> str:
        """Single-shot text generation via OpenAI or Azure OpenAI."""
        try:
            prov = _get_provider()
            result = prov.generate(
                prompt,
                model=model,
                json_mode=json_mode,
                response_schema=response_schema,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                prompt_cache_key=prompt_cache_key,
                api_mode=api_mode,
            )
            return json.dumps(result)
        except (OpenAIRateLimitError, OpenAIContextLengthError, OpenAITransientError) as exc:
            return _serialize_error(exc)

    @mcp.tool(name="provider.generate_with_history")
    def generate_with_history(
        messages: list[dict[str, Any]] | str,
        system_prompt: str = "",
        model: str = "",
        json_mode: bool = False,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        reasoning_effort: str = "",
        prompt_cache_key: str = "",
        api_mode: str = "",
    ) -> str:
        """Multi-turn generation with conversation history (accepts list[dict] or JSON str)."""
        parsed_messages: list[dict[str, Any]] = (
            json.loads(messages) if isinstance(messages, str) else list(messages)
        )
        try:
            prov = _get_provider()
            result = prov.generate_with_history(
                parsed_messages,
                system_prompt=system_prompt,
                model=model,
                json_mode=json_mode,
                response_schema=response_schema,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                prompt_cache_key=prompt_cache_key,
                api_mode=api_mode,
            )
            return json.dumps(result)
        except (OpenAIRateLimitError, OpenAIContextLengthError, OpenAITransientError) as exc:
            return _serialize_error(exc)

    @mcp.tool(name="provider.list_models")
    def list_models(live: bool = False) -> str:
        """List available OpenAI models with pricing, context window, and reasoning metadata."""
        prov = _get_provider()
        return json.dumps(prov.list_models(live=live))

    @mcp.tool(name="provider.health_check")
    def health_check(probe_upstream: bool = False) -> str:
        """Liveness and readiness probe for the OpenAI MCP provider."""
        try:
            prov = _get_provider()
            return json.dumps(prov.health_check(probe_upstream=probe_upstream))
        except Exception as exc:
            return json.dumps({
                "healthy": False,
                "provider": "openai",
                "message": str(exc),
            })

    @mcp.tool(name="provider.capabilities")
    def capabilities() -> str:
        """Advertise supported AuraRouter Provider Protocol tools and OpenAI features."""
        return json.dumps({
            "provider": "openai",
            "version": __version__,
            "tools": [
                "provider.generate",
                "provider.generate_with_history",
                "provider.list_models",
                "provider.health_check",
                "provider.capabilities",
            ],
            "features": {
                "json_mode": True,
                "structured_outputs": True,
                "multi_turn": True,
                "reasoning_models": True,
                "responses_api": True,
                "azure_openai": True,
                "prompt_caching": True,
                "rate_limit_telemetry": True,
            },
        })

    return mcp

