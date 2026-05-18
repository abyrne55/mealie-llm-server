# mealie-llm-server

OpenAI-compatible API gateway for [Mealie](https://mealie.io/) that intercepts ingredient parsing requests and processes them locally using a small language model ([NuExtract-2.0-2B](https://huggingface.co/numind/NuExtract-2.0-2B)) with GBNF grammar-constrained decoding. All other requests are proxied to an upstream cloud API.

## Quick Start

```bash
docker run -d \
  -p 8000:8000 \
  -v ./models:/models \
  -e MEALIE_URL=http://mealie:9000 \
  -e MEALIE_API_KEY=your-mealie-api-key \
  -e UPSTREAM_URL=https://api.openai.com/v1 \
  -e UPSTREAM_API_KEY=sk-... \
  ghcr.io/abyrne/mealie-llm-server:latest
```

Then point Mealie at this server:

```
OPENAI_BASE_URL=http://mealie-llm-server:8000/v1
```

## How It Works

```
Mealie ──POST /v1/chat/completions──► mealie-llm-server
                                          │
                                    ┌─────┴─────┐
                                    │   Router   │  (jaccard similarity
                                    │            │   on system prompt)
                                    └─────┬─────┘
                                   match? │ no match?
                                    ┌─────┴─────┐
                              ┌─────┤           ├─────┐
                              ▼                       ▼
                        Local LLM              Upstream API
                     (NuExtract-2B)          (OpenAI, etc.)
                     GBNF grammar
                     constrained
```

Ingredient parsing requests are identified by matching the system prompt against known Mealie prompts. Matched requests are processed locally with enum constraints from your Mealie database (foods, units) enforced via GBNF grammar. Everything else passes through to the upstream API.

## Configuration

All environment variables support `_FILE` variants for container secrets (e.g. `MEALIE_API_KEY_FILE=/run/secrets/key`).

| Variable | Required | Default | Description |
|---|---|---|---|
| `MEALIE_URL` | Yes | — | Mealie instance URL |
| `MEALIE_API_KEY` | Yes | — | Mealie API key |
| `MEALIE_CACHE_TTL` | No | `300` | Foods/units cache TTL (seconds) |
| `UPSTREAM_URL` | No | — | Upstream OpenAI-compatible API base URL |
| `UPSTREAM_API_KEY` | No | — | Upstream API key |
| `UPSTREAM_TIMEOUT` | No | `300` | Upstream request timeout (seconds) |
| `MODEL_INGREDIENT_PARSING` | No | `numind/NuExtract-2.0-2B-GGUF:Q6_K` | HuggingFace model ID for ingredient parsing |
| `MODEL_LOADING_STRATEGY` | No | `all` | `all` (preload at startup) or `swap` (load on demand) |
| `MODEL_CONTEXT_SIZE` | No | `4096` | LLM context window size |
| `MODEL_THREADS` | No | auto | Number of CPU threads for inference |
| `MODEL_CACHE_DIR` | No | `/models` | Directory for downloaded models |
| `ROUTER_THRESHOLD` | No | `0.6` | Jaccard similarity threshold for prompt matching |
| `LOG_LEVEL` | No | `info` | Logging level |

## Development

```bash
uv sync
uv run pytest -v
uv run uvicorn mealie_llm_server.app:app --reload
```

## License

MIT
