"""Entry point for ``python -m aurarouter_openai``."""

from __future__ import annotations

import sys


def main() -> None:
    """Create and run the OpenAI MCP provider server."""
    from aurarouter_openai.server import create_server

    server = create_server()
    server.run()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("aurarouter-openai: OpenAI & Azure OpenAI MCP provider server")
        print()
        print("Usage:")
        print("  python -m aurarouter_openai          Start the MCP server")
        print("  python -m aurarouter_openai --help   Show this help message")
        print()
        print("Environment variables:")
        print("  OPENAI_API_KEY          OpenAI (or Azure OpenAI) API key")
        print("  OPENAI_BASE_URL         Optional custom OpenAI API base URL")
        print("  OPENAI_ORG_ID           Optional OpenAI organization ID")
        print("  OPENAI_PROJECT_ID       Optional OpenAI project ID")
        print("  AZURE_OPENAI_ENDPOINT   Optional Azure OpenAI resource endpoint")
        print("  AZURE_OPENAI_AD_TOKEN   Optional Microsoft Entra ID token for Azure")
        print("  OPENAI_API_VERSION      Azure API version (default: 2025-01-01-preview)")
        print()
        print("MCP tools exposed:")
        print("  provider.generate              Single-shot text generation")
        print("  provider.generate_with_history Multi-turn conversation")
        print("  provider.list_models           List available OpenAI models")
        print("  provider.health_check          Provider readiness probe")
        print("  provider.capabilities          Advertise provider features")
        sys.exit(0)

    main()

