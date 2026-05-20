# mealie-llm-server

OpenAI-compatible API gateway for [Mealie](https://mealie.io/) that intercepts ingredient parsing requests and processes them locally using a fine-tuned [NuExtract-tiny-v1.5](https://huggingface.co/numind/NuExtract-tiny-v1.5) model with GBNF grammar-enforced JSON output and embedding-based food matching. All other requests are proxied to an upstream cloud API.

## Quick Start

```bash
docker run -d \
  -p 8000:8000 \
  -v ./models:/models \
  -e MEALIE_URL=http://mealie:9000 \
  -e MEALIE_API_KEY=your-mealie-api-key \
  -e UPSTREAM_URL=https://api.openai.com/v1 \
  -e UPSTREAM_API_KEY=sk-... \
  ghcr.io/abyrne/mealie-llm-server:main
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
                   (NuExtract-tiny)          (OpenAI, etc.)
                        │
                  ┌─────┴─────┐
                  │  Extract   │  GBNF grammar → JSON
                  └─────┬─────┘
                  ┌─────┴─────┐
                  │  Resolve   │  model2vec embeddings
                  └─────┬─────┘     → Mealie foods DB
                        ▼
                   Structured
                    response
```

Ingredient parsing requests are identified by matching the system prompt against known Mealie prompts. Matched requests go through a two-step pipeline:

1. **Extract** — A fine-tuned NuExtract-tiny-v1.5 model extracts quantity, unit, food, and note from the ingredient text, with GBNF grammar enforcing valid JSON output.
2. **Resolve** — Extracted food strings are matched to your Mealie database using [model2vec](https://github.com/MinishLab/model2vec) static embeddings (exact match first, then cosine similarity).

Everything else passes through to the upstream API.

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
| `MODEL_INGREDIENT_EXTRACTOR` | No | `abyrne55/nuextract-1.5-tiny-mealie-ingredient-parser:q8_0` | HuggingFace GGUF model for ingredient extraction (see [Supported Models](#supported-extraction-models)) |
| `MODEL_INGREDIENT_RESOLVER` | No | `minishlab/potion-retrieval-32M` | model2vec embedding model for food resolution |
| `MODEL_LOADING_STRATEGY` | No | `all` | `all` (preload at startup) or `swap` (load on demand) |
| `MODEL_CONTEXT_SIZE` | No | `4096` | LLM context window size |
| `MODEL_THREADS` | No | auto | Number of CPU threads for inference |
| `MODEL_CACHE_DIR` | No | `/models` | Directory for downloaded models |
| `ROUTER_THRESHOLD` | No | `0.6` | Jaccard similarity threshold for prompt matching |
| `LOG_LEVEL` | No | `info` | Logging level |

## Supported Extraction Models

Set `MODEL_INGREDIENT_EXTRACTOR` to switch.

| Model | ID | Speed | Notes |
|---|---|---|---|
| **NuExtract-tiny-v1.5 fine-tuned** (default) | `abyrne55/nuextract-1.5-tiny-mealie-ingredient-parser:q8_0` | ~1.8s/ingredient | ~0.5GB, LoRA fine-tuned on ingredient dataset, 93% test pass rate |
| NuExtract-tiny-v1.5 (base) | `DevQuasar-3/numind.NuExtract-tiny-v1.5-GGUF:Q6_K` | ~1.8s/ingredient | ~0.5GB, base model without fine-tuning, 49% test pass rate |
| NuExtract-v1.5 | `DevQuasar-3/numind.NuExtract-v1.5-GGUF:Q6_K` | ~3s/ingredient | ~1.5GB, larger base model |

## Development

```bash
uv sync
uv run pytest -v
uv run uvicorn mealie_llm_server.app:app --reload
```

### Testing with a Live Mealie Instance

To spin up an ephemeral, pre-seeded Mealie container for local testing:

```bash
./scripts/start-mealie.sh
```

This starts a Mealie container on port 9797, seeds it with foods/units/labels, creates an API token, and writes `.env.test`. Re-runs are idempotent.

Clean up when done:

```bash
podman stop mealie-test && podman rm mealie-test
```

### Fine-Tuning

The extraction model can be fine-tuned on the ingredient dataset in `tests/integration/ingredients.jsonl` using LoRA:

```bash
uv sync --group train
uv run python scripts/finetune.py        # CPU
```

Or use `notebooks/finetune.ipynb` for GPU training on Google Colab (~2 min on T4).

## License

MIT
