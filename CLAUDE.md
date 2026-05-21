# mealie-local-ai

OpenAI-compatible API gateway for Mealie that routes ingredient parsing to a local LLM with GBNF grammar-enforced JSON output and embedding-based food matching, and proxies all other requests to an upstream cloud API.

## Development

```bash
uv sync                                           # install deps
uv run uvicorn mealie_local_ai.app:app --reload  # run locally
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
podman build -t mealie-local-ai -f Containerfile .
```

## Architecture

Request flow: `POST /v1/chat/completions` → Router (jaccard similarity on system prompt) → Handler or Proxy.

- **Router** (`router.py`): Matches incoming system prompts against reference Mealie prompts using jaccard similarity.
- **Handlers** (`handlers/`): Process matched requests locally using llama-cpp-python. `IngredientParsingHandler` extracts ingredients via LLM with GBNF JSON schema enforcement, then resolves foods against the Mealie database using embedding similarity.
- **Food Resolver** (`food_resolver.py`): Resolves extracted food strings to Mealie database entries using model2vec static embeddings (exact match first, then cosine similarity).
- **Proxy** (`proxy.py`): Forwards unmatched/multimodal requests to an upstream OpenAI-compatible API.
- **Model Manager** (`model_manager.py`): Loads/unloads GGUF models with `all` (preload) or `swap` (on-demand) strategies.
- **Mealie Client** (`mealie_client.py`): Fetches foods/units from Mealie with TTL cache for enum constraints.

## Key Files

| File | Responsibility |
|---|---|
| `mealie_local_ai/app.py` | FastAPI app, lifespan, endpoints |
| `mealie_local_ai/config.py` | Pydantic Settings with `_FILE` support |
| `mealie_local_ai/models.py` | OpenAI request/response Pydantic models |
| `mealie_local_ai/router.py` | Jaccard similarity prompt router |
| `mealie_local_ai/handlers/ingredient_parsing.py` | NuExtract extraction + post-processing pipeline |
| `mealie_local_ai/food_resolver.py` | Embedding-based food entity resolution |
| `mealie_local_ai/mealie_client.py` | Mealie API client with TTL cache |
| `mealie_local_ai/model_manager.py` | LLM loading/unloading |
| `mealie_local_ai/proxy.py` | Upstream reverse proxy |
| `scripts/extract_training_data.py` | Extract JSONL training data from Mealie API + recipe pages |
| `scripts/validate_jsonl.py` | Validate JSONL dataset format and constraints |

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
| `MODEL_INGREDIENT_EXTRACTOR` | No | `abyrne55/nuextract-1.5-tiny-mealie-ingredient-parser:q8_0` | HF GGUF model for extraction (see below). Also accepts local `.gguf` file paths. |
| `MODEL_INGREDIENT_RESOLVER` | No | `minishlab/potion-retrieval-32M` | model2vec embedding model for food resolution |
| `MODEL_LOADING_STRATEGY` | No | `all` | `all` or `swap` |
| `MODEL_CONTEXT_SIZE` | No | `4096` | Context window size |
| `MODEL_THREADS` | No | auto | CPU threads for inference |
| `MODEL_CACHE_DIR` | No | `/models` | Model download directory |
| `ROUTER_THRESHOLD` | No | `0.6` | Jaccard similarity threshold |
| `LOG_LEVEL` | No | `info` | Logging level |

## Fine-Tuning Dataset

`tests/integration/ingredients.jsonl` is the training dataset for fine-tuning the extraction model (HuggingFace `trl.SFTTrainer` + `peft` LoRA). Each line is a JSON object with an OpenAI `messages` array: a `user` message containing the full NuExtract prompt and an `assistant` message containing the expected JSON output. Fine-tune with `uv sync --group train && uv run python scripts/finetune.py` (CPU) or use `notebooks/finetune.ipynb` (Colab GPU, ~5 min on T4). The dataset is shuffled (seed=42) during training.

### Expanding the Dataset

To add new training examples from hand-corrected Mealie recipes:

1. Import recipes into a running Mealie instance and correct the parsed ingredients by hand
2. Run the extraction script against the Mealie API:

```bash
uv run python scripts/extract_training_data.py              # uses .env.test
uv run python scripts/extract_training_data.py --dry-run     # preview without writing
uv run python scripts/validate_jsonl.py tests/integration/ingredients.jsonl  # validate
```

The script fetches all recipes from Mealie, scrapes the original recipe pages for raw ingredient text, filters to foods in the default en-US seed database (from `../mealie`), and appends new deduplicated entries to the JSONL. Requires `MEALIE_URL` and `MEALIE_API_KEY` (from env, `.env.test`, or `--mealie-url`/`--mealie-api-key` flags).

**When adding or editing entries manually, all of the following must hold:**

1. **Schema format**: The assistant content must be single-line JSON with keys in exactly this order: `quantity`, `unit`, `food`, `note`. Use `": "` after colons and `", "` between fields (Python `json.dumps` defaults). No trailing whitespace.
2. **Key casing**: All keys lowercase — `quantity`, `unit`, `food`, `note`.
3. **Empty values**: Use `""` (empty string) for absent unit/note, not `null`.
4. **Prompt generation**: Always use `build_messages()` from `handlers/ingredient_parsing.py` to generate the user message. Do not hand-write the NuExtract template.
5. **Food values must be exact Mealie DB entries**: Every `food` value must exist verbatim (case-insensitive) in the Mealie test instance seeded by `scripts/start-mealie.sh`. Query the API (`/api/foods?search=<term>`) to verify. If the ingredient text uses a synonym or variant not in the DB, map `food` to the correct DB entry and capture the original qualifier in `note` (e.g., "arborio rice" → `food: "risotto rice"`, `note: "arborio"`).
6. **Unit values must be known aliases**: Every non-empty `unit` value must appear in the Mealie unit alias map (canonical name, plural, or abbreviation). Check via `/api/units?per_page=-1`.
7. **No unresolvable foods**: If an ingredient's food cannot be matched to any DB entry, do not include it in the dataset.

## Supported Extraction Models

Set `MODEL_INGREDIENT_EXTRACTOR` to switch.

| Model | ID | Speed | Notes |
|---|---|---|---|
| **NuExtract-tiny-v1.5 fine-tuned** (default) | `abyrne55/nuextract-1.5-tiny-mealie-ingredient-parser:q8_0` | ~1.8s/ingredient | ~0.5GB, LoRA fine-tuned on 162-example ingredient dataset (shuffled), 93% training / 97% novel ingredient pass rate |
| NuExtract-tiny-v1.5 (base) | `DevQuasar-3/numind.NuExtract-tiny-v1.5-GGUF:Q6_K` | ~1.8s/ingredient | ~0.5GB, base model without fine-tuning, 49% test pass rate |
| NuExtract-v1.5 | `DevQuasar-3/numind.NuExtract-v1.5-GGUF:Q6_K` | ~3s/ingredient | ~1.5GB, larger base model |
