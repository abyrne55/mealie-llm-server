import pytest
from importlib.resources import files
from mealie_llm_server.router import Router, tokenize, jaccard_similarity


def _load_prompt(name: str) -> str:
    return files("mealie_llm_server.prompts").joinpath(name).read_text()


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello world") == {"hello", "world"}

    def test_strips_punctuation(self):
        tokens = tokenize("Parse ingredient strings into components.")
        assert "components" in tokens
        assert "components." not in tokens

    def test_lowercases(self):
        assert tokenize("Parse Ingredient") == {"parse", "ingredient"}


class TestJaccardSimilarity:
    def test_identical(self):
        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial(self):
        result = jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert result == pytest.approx(2 / 4)

    def test_empty(self):
        assert jaccard_similarity(set(), set()) == 0.0


class TestRouter:
    def test_match_ingredient_parsing_prompt(self):
        router = Router(threshold=0.6)
        prompt = _load_prompt("parse-recipe-ingredients.txt")
        handler_name = router.match(prompt)
        assert handler_name == "parse-recipe-ingredients"

    def test_no_match_returns_none(self):
        router = Router(threshold=0.6)
        handler_name = router.match("You are a helpful assistant.")
        assert handler_name is None

    def test_strips_mealie_data_injection(self):
        router = Router(threshold=0.6)
        prompt = _load_prompt("parse-recipe-ingredients.txt")
        prompt += (
            "\n###\n"
            "Below is a list of units found in the units database.\n"
            "---\n\n"
            "['tablespoon', 'teaspoon', 'cup', 'gram', 'ounce']"
        )
        handler_name = router.match(prompt)
        assert handler_name == "parse-recipe-ingredients"

    def test_match_scrape_recipe(self):
        router = Router(threshold=0.6)
        prompt = _load_prompt("scrape-recipe.txt")
        handler_name = router.match(prompt)
        assert handler_name == "scrape-recipe"

    def test_threshold_respected(self):
        router = Router(threshold=0.99)
        prompt = _load_prompt("parse-recipe-ingredients.txt")
        handler_name = router.match(prompt)
        assert handler_name == "parse-recipe-ingredients"

    def test_short_prompt_below_threshold(self):
        router = Router(threshold=0.6)
        handler_name = router.match("Parse ingredient strings into components.")
        assert handler_name is None
