# aurarouter-openai

OpenAI Platform and Azure OpenAI MCP provider server for [AuraRouter](https://github.com/AuraCoreDynamics/aurarouter).

`aurarouter-openai` exposes OpenAI and Azure OpenAI models as an MCP (Model Context Protocol) provider server compatible with the AuraRouter Provider Protocol. It separates cloud vendor specifics (such as reasoning model dialects, `/v1/responses`, Structured Outputs, and prompt caching) from AuraRouter core's generic local `OpenAPIProvider`.

---

## Features

- **Standard MCP Provider Protocol**:
  - `provider.generate`: Single-shot text generation
  - `provider.generate_with_history`: Multi-turn conversational generation
  - `provider.list_models`: Catalog discovery with context limits, pricing, and capabilities
  - `provider.health_check`: Liveness and credential readiness probe
  - `provider.capabilities`: Tool and feature advertisement
- **Reasoning Model Dialect Adaptation**:
  - Automatically switches `o1`, `o3`, `o4-mini`, and `gpt-5*` models to `max_completion_tokens`.
  - Maps system instructions to the `developer` role.
  - Supports `reasoning_effort` (`low`, `medium`, `high`) and suppresses unsupported `temperature` parameters.
- **Dual API Modes**:
  - Chat Completions: `/v1/chat/completions` (default)
  - Responses API: `/v1/responses` (`api_mode="responses"`)
- **Azure OpenAI Support**:
  - Configurable `azure_endpoint`, `api_version`, and deployment routing.
  - Supports both `api-key` header authentication and Microsoft Entra ID (`AZURE_OPENAI_AD_TOKEN`).
- **Prompt Cache Affinity**:
  - Passes `prompt_cache_key` to upstream endpoints.
  - Returns `cached_input_tokens` and `reasoning_tokens` telemetry.
- **AuraRouter v2.1 Telemetry Envelopes**:
  - Normalizes Go-duration rate limit headers (`x-ratelimit-*`) and `Retry-After`.
  - Propagates typed error envelopes: `rate_limit_exceeded`, `context_length_exceeded`, and `provider_transient_error`.

---

## Installation

```bash
pip install aurarouter-openai
```

Or install from source:

```bash
cd aurarouter-openai
pip install -e .
```

---

## Configuration

### Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (or Azure OpenAI API key) |
| `OPENAI_BASE_URL` | Optional custom base URL (default: `https://api.openai.com/v1`) |
| `OPENAI_ORG_ID` | Optional OpenAI organization ID |
| `OPENAI_PROJECT_ID` | Optional OpenAI project ID |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint (e.g. `https://my-resource.openai.azure.com`) |
| `AZURE_OPENAI_AD_TOKEN` | Optional Microsoft Entra ID bearer token for Azure OpenAI |
| `OPENAI_API_VERSION` | Azure OpenAI API version (default: `2025-01-01-preview`) |

---

## Running the Server

Start the standalone MCP server:

```bash
python -m aurarouter_openai
```

Display command line options and help:

```bash
python -m aurarouter_openai --help
```

---

## AuraRouter Integration

In your AuraRouter configuration file (`auraconfig.yaml`):

```yaml
models:
  gpt-4o:
    provider: mcp
    mcp_endpoint: http://localhost:8000
    model_name: gpt-4o
    hosting_tier: cloud
    cost_per_1m_input: 2.50
    cost_per_1m_output: 10.00
    context_limit: 128000

  o3:
    provider: mcp
    mcp_endpoint: http://localhost:8000
    model_name: o3
    hosting_tier: cloud
    cost_per_1m_input: 2.00
    cost_per_1m_output: 8.00
    context_limit: 200000
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
