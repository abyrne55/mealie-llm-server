# mealie-llm-server

OpenAI-compatible API gateway for Mealie that routes ingredient parsing to a local LLM with GBNF grammar-enforced JSON output and embedding-based food matching, and proxies all other requests to an upstream cloud API.

## Development

```bash
uv sync                                           # install deps
uv run uvicorn mealie_llm_server.app:app --reload  # run locally
uv run pytest -v                                   # run tests
uv run ruff check .                                # lint
```

## Testing with a Live Mealie Instance

```bash
./scripts/start-mealie.sh   # start ephemeral Mealie container, seed data, print env
source <(grep '^export' <(./scripts/start-mealie.sh))  # or source the exports directly
```

Launches a pre-seeded Mealie container (`mealie-test`) on port 9797 with foods, units, labels, and a long-lived API token. Writes `.env.test` with `MEALIE_URL` and `MEALIE_API_KEY`. Idempotent — re-runs exit early if credentials are still valid.

Cleanup: `podman stop mealie-test && podman rm mealie-test`

## Container Build

```bash
podman build -t mealie-llm-server -f Containerfile .
```

## Architecture

Request flow: `POST /v1/chat/completions` → Router (jaccard similarity on system prompt) → Handler or Proxy.

- **Router** (`router.py`): Matches incoming system prompts against reference Mealie prompts using jaccard similarity.
- **Handlers** (`handlers/`): Process matched requests locally using llama-cpp-python. `IngredientParsingHandler` extracts ingredients via LLM with GBNF JSON schema enforcement, then resolves foods against the Mealie database using embedding similarity.
- **Food Matcher** (`food_matcher.py`): Matches extracted food strings to Mealie database entries using model2vec static embeddings (exact match first, then cosine similarity).
- **Proxy** (`proxy.py`): Forwards unmatched/multimodal requests to an upstream OpenAI-compatible API.
- **Model Manager** (`model_manager.py`): Loads/unloads GGUF models with `all` (preload) or `swap` (on-demand) strategies.
- **Mealie Client** (`mealie_client.py`): Fetches foods/units from Mealie with TTL cache for enum constraints.

## Key Files

| File | Responsibility |
|---|---|
| `mealie_llm_server/app.py` | FastAPI app, lifespan, endpoints |
| `mealie_llm_server/config.py` | Pydantic Settings with `_FILE` support |
| `mealie_llm_server/models.py` | OpenAI request/response Pydantic models |
| `mealie_llm_server/router.py` | Jaccard similarity prompt router |
| `mealie_llm_server/handlers/ingredient_parsing.py` | LLM extraction + post-processing pipeline |
| `mealie_llm_server/food_matcher.py` | Embedding-based food entity resolution |
| `mealie_llm_server/mealie_client.py` | Mealie API client with TTL cache |
| `mealie_llm_server/model_manager.py` | LLM loading/unloading |
| `mealie_llm_server/proxy.py` | Upstream reverse proxy |

## Environment Variables

All env vars support `_FILE` variants for container secrets (e.g. `MEALIE_API_KEY_FILE`).

| Variable | Required | Default | Description |
|---|---|---|---|
| `MEALIE_URL` | Yes | — | Mealie instance URL |
| `MEALIE_API_KEY` | Yes | — | Mealie API key |
| `MEALIE_CACHE_TTL` | No | `300` | Cache TTL in seconds |
| `UPSTREAM_URL` | No | — | Upstream OpenAI-compatible API URL |
| `UPSTREAM_API_KEY` | No | — | Upstream API key |
| `UPSTREAM_TIMEOUT` | No | `300` | Upstream request timeout (seconds) |
| `MODEL_INGREDIENT_PARSING` | No | `openbmb/MiniCPM-V-4.6-gguf:Q6_K` | HF model ID (see below) |
| `MODEL_LOADING_STRATEGY` | No | `all` | `all` or `swap` |
| `MODEL_CONTEXT_SIZE` | No | `4096` | Context window size |
| `MODEL_THREADS` | No | auto | CPU threads for inference |
| `MODEL_CACHE_DIR` | No | `/models` | Model download directory |
| `ROUTER_THRESHOLD` | No | `0.6` | Jaccard similarity threshold |
| `LOG_LEVEL` | No | `info` | Logging level |

## Supported Extraction Models

Set `MODEL_INGREDIENT_PARSING` to switch. The handler auto-detects the prompt format.

| Model | ID | Speed | Strengths | Weaknesses |
|---|---|---|---|---|
| **MiniCPM-V-4.6** (default) | `openbmb/MiniCPM-V-4.6-gguf:Q6_K` | ~5.5s/ingredient | Better unit extraction (bunch, pinch), better notes, smarter food names | Occasional typos, needs few-shot prompt for fractions |
| NuExtract-2.0-2B | `numind/NuExtract-2.0-2B-GGUF:Q6_K` | ~3.4s/ingredient | Faster, reliable fraction handling, no hallucinations | Misses some units (bunch, pinch), weak notes extraction |
