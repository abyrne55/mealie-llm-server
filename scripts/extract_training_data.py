#!/usr/bin/env python3
"""Extract training data from a live Mealie instance.

Fetches hand-corrected recipes from the Mealie API, scrapes the original
recipe pages for raw ingredient text, filters to foods present in the
default en-US seed database, and appends new entries to the training JSONL.

Usage:
    uv run python scripts/extract_training_data.py

Requires MEALIE_URL and MEALIE_API_KEY environment variables (or .env.test).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mealie_llm_server.handlers.ingredient_parsing import build_messages  # noqa: E402

SEED_FOODS_PATH = (
    PROJECT_ROOT / ".." / "mealie" / "mealie" / "repos" / "seed" / "resources" / "foods" / "locales" / "en-US.json"
)
JSONL_PATH = PROJECT_ROOT / "tests" / "integration" / "ingredients.jsonl"
ENV_FILE = PROJECT_ROOT / ".env.test"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def load_seed_foods(path: Path) -> set[str]:
    with open(path) as f:
        data = json.load(f)
    foods: set[str] = set()
    for category in data.values():
        if isinstance(category, dict) and "foods" in category:
            for food_data in category["foods"].values():
                foods.add(food_data["name"].lower())
    return foods


def mealie_get(base_url: str, api_key: str, path: str) -> dict | list:
    url = f"{base_url}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def fetch_recipes(base_url: str, api_key: str) -> list[dict]:
    data = mealie_get(base_url, api_key, "/api/recipes?per_page=-1")
    slugs = [r["slug"] for r in data.get("items", [])]
    recipes = []
    for slug in slugs:
        recipe = mealie_get(base_url, api_key, f"/api/recipes/{slug}")
        recipes.append(recipe)
    return recipes


def fetch_page_ingredients(url: str) -> list[str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; training-data-extractor/1.0)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  Warning: failed to fetch {url}: {e}", file=sys.stderr)
        return []

    for m in re.finditer(r'"recipeIngredient"\s*:\s*(\[.*?\])', html, re.DOTALL):
        try:
            ingredients = json.loads(m.group(1))
            return [s.strip() for s in ingredients if isinstance(s, str)]
        except json.JSONDecodeError:
            pass
    return []


def match_fetched_to_db(db_ingredients: list[dict], fetched: list[str]) -> dict[int, str]:
    matches: dict[int, str] = {}
    used: set[int] = set()

    for i, ing in enumerate(db_ingredients):
        original = (ing.get("originalText") or "").strip()
        if not original:
            continue
        best_score = 0.0
        best_idx = -1
        for j, fetched_str in enumerate(fetched):
            if j in used:
                continue
            score = SequenceMatcher(None, original.lower(), fetched_str.lower()).ratio()
            if score > best_score:
                best_score = score
                best_idx = j
        if best_score > 0.5 and best_idx >= 0:
            used.add(best_idx)
            matches[i] = fetched[best_idx]

    return matches


def find_unit_form(text: str, unit: dict) -> str:
    aliases = [unit["name"]]
    if unit.get("pluralName"):
        aliases.append(unit["pluralName"])
    if unit.get("abbreviation"):
        aliases.append(unit["abbreviation"])

    text_lower = text.lower()
    text_words = text_lower.split()
    for alias in sorted(aliases, key=len, reverse=True):
        if alias.lower() in text_words:
            return alias
    for alias in sorted(aliases, key=len, reverse=True):
        if alias.lower() in text_lower:
            return alias
    return aliases[0]


def clean_quantity(q: float | None) -> int | float | None:
    if q is None:
        return None
    if isinstance(q, float) and q == int(q):
        return int(q)
    return q


def load_existing_ingredients(path: Path) -> set[str]:
    texts: set[str] = set()
    if not path.exists():
        return texts
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            content = entry["messages"][0]["content"]
            m = re.search(r"### Text:\n(.+?)\n\n<\|output\|>", content, re.DOTALL)
            if m:
                texts.add(m.group(1).strip().lower())
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract training data from Mealie API")
    parser.add_argument("--mealie-url", default=None, help="Mealie URL (default: MEALIE_URL env var or .env.test)")
    parser.add_argument(
        "--mealie-api-key", default=None, help="Mealie API key (default: MEALIE_API_KEY env var or .env.test)"
    )
    parser.add_argument("--seed-foods", default=str(SEED_FOODS_PATH), help="Path to en-US.json seed foods")
    parser.add_argument("--output", default=str(JSONL_PATH), help="Path to output JSONL")
    parser.add_argument("--dry-run", action="store_true", help="Print entries without writing")
    args = parser.parse_args()

    load_env_file(ENV_FILE)

    base_url = args.mealie_url or os.environ.get("MEALIE_URL")
    api_key = args.mealie_api_key or os.environ.get("MEALIE_API_KEY")
    if not base_url or not api_key:
        print("Error: MEALIE_URL and MEALIE_API_KEY required (env, .env.test, or --flags)", file=sys.stderr)
        sys.exit(1)

    seed_foods = load_seed_foods(Path(args.seed_foods))
    print(f"Loaded {len(seed_foods)} seed food names")

    print("Fetching recipes from Mealie...")
    recipes = fetch_recipes(base_url, api_key)
    total_ingredients = sum(len(r.get("recipeIngredient", [])) for r in recipes)
    print(f"Loaded {len(recipes)} recipes, {total_ingredients} ingredients")

    output_path = Path(args.output)
    existing = load_existing_ingredients(output_path)
    print(f"Found {len(existing)} existing entries in {output_path.name}")

    new_entries: list[str] = []
    stats = {
        "added": 0,
        "skipped_no_food": 0,
        "skipped_not_seeded": 0,
        "skipped_no_text": 0,
        "skipped_duplicate": 0,
    }

    for recipe in recipes:
        url = recipe.get("orgURL")
        if not url:
            continue

        ingredients = recipe.get("recipeIngredient", [])
        if not ingredients:
            continue

        print(f"\n{recipe['name']}")
        print(f"  {url}")
        fetched = fetch_page_ingredients(url)
        print(f"  {len(fetched)} from page, {len(ingredients)} in DB")

        matches = match_fetched_to_db(ingredients, fetched)

        for i, ing in enumerate(ingredients):
            food = ing.get("food")
            if not food or not food.get("name"):
                stats["skipped_no_food"] += 1
                continue

            food_name = food["name"]
            if food_name.lower() not in seed_foods:
                print(f"  skip (not seeded): {food_name}")
                stats["skipped_not_seeded"] += 1
                continue

            ingredient_text = matches.get(i) or (ing.get("originalText") or "").strip()
            if not ingredient_text:
                stats["skipped_no_text"] += 1
                continue

            if ingredient_text.strip().lower() in existing:
                stats["skipped_duplicate"] += 1
                continue

            unit = ing.get("unit")
            if unit and unit.get("name"):
                unit_str = find_unit_form(ingredient_text, unit)
            else:
                unit_str = ""

            messages = build_messages(ingredient_text)
            output = {
                "quantity": clean_quantity(ing.get("quantity")),
                "unit": unit_str,
                "food": food_name,
                "note": ing.get("note") or "",
            }
            messages.append({"role": "assistant", "content": json.dumps(output)})
            entry = json.dumps({"messages": messages})

            new_entries.append(entry)
            existing.add(ingredient_text.strip().lower())
            stats["added"] += 1
            print(f"  + {ingredient_text[:70]}")

        time.sleep(0.5)

    if new_entries and not args.dry_run:
        with open(output_path, "a") as f:
            for entry in new_entries:
                f.write(entry + "\n")

    print("\n--- Summary ---")
    print(f"Added:                {stats['added']}")
    print(f"Skipped (no food):    {stats['skipped_no_food']}")
    print(f"Skipped (not seeded): {stats['skipped_not_seeded']}")
    print(f"Skipped (no text):    {stats['skipped_no_text']}")
    print(f"Skipped (duplicate):  {stats['skipped_duplicate']}")
    if args.dry_run:
        print("\n(Dry run — no files modified)")


if __name__ == "__main__":
    main()
