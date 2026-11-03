"""aurarouter-openai -- OpenAI & Azure OpenAI MCP provider for AuraRouter."""

from __future__ import annotations

__version__ = "0.5.1"
__package_name__ = "aurarouter-openai"


def get_provider_metadata():
    """Entry point callable for ``aurarouter.providers`` discovery.

    Returns a ``ProviderMetadata``-compatible object that AuraRouter's
    ``ProviderCatalog`` uses to register and launch this provider.
    """
    try:
        from aurarouter.providers.protocol import ProviderMetadata

        return ProviderMetadata(
            name="openai",
            provider_type="mcp",
            version=__version__,
            description="OpenAI & Azure OpenAI models (GPT-5, GPT-4.1, GPT-4o, o3, o4-mini)",
            command=["python", "-m", "aurarouter_openai"],
            requires_config=["api_key"],
            homepage="https://github.com/auracore-dynamics/aurarouter-openai",
        )
    except ImportError:
        return {
            "name": "openai",
            "provider_type": "mcp",
            "version": __version__,
            "description": "OpenAI & Azure OpenAI models (GPT-5, GPT-4.1, GPT-4o, o3, o4-mini)",
            "command": ["python", "-m", "aurarouter_openai"],
            "requires_config": ["api_key"],
            "homepage": "https://github.com/auracore-dynamics/aurarouter-openai",
        }

