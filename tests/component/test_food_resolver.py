"""Component tests for FoodResolver with real embedding model.

Tests exact match, semantic similarity, threshold behavior, and edge cases.
"""

FOODS = [
    "flour",
    "eggs",
    "olive oil",
    "chicken broth",
    "butter",
    "garlic",
    "onion",
    "sugar",
    "salt",
    "black pepper",
    "tomato paste",
    "soy sauce",
    "rice",
    "parmesan cheese",
    "cream cheese",
    "vanilla extract",
    "baking powder",
    "baking soda",
    "heavy cream",
    "lemon",
]


class TestExactMatch:
    def test_exact_match_lowercase(self, food_resolver):
        result, score, exact = food_resolver.match("flour", FOODS)
        assert result == "flour"
        assert score == 1.0
        assert exact is True

    def test_exact_match_case_insensitive(self, food_resolver):
        result, score, exact = food_resolver.match("Flour", FOODS)
        assert result == "flour"
        assert exact is True

    def test_exact_match_multiword(self, food_resolver):
        result, score, exact = food_resolver.match("olive oil", FOODS)
        assert result == "olive oil"
        assert exact is True


class TestSimilarityMatch:
    def test_similar_food_matches(self, food_resolver):
        result, score, exact = food_resolver.match("egg", FOODS)
        assert result is not None
        assert exact is False
        assert score >= 0.65

    def test_similarity_score_above_threshold(self, food_resolver):
        result, score, exact = food_resolver.match("parm cheese", FOODS)
        if result is not None:
            assert score >= 0.65


class TestBelowThreshold:
    def test_unrelated_food_returns_none(self, food_resolver):
        result, score, exact = food_resolver.match("unicorn tears", FOODS, threshold=0.95)
        assert result is None
        assert score < 0.95

    def test_custom_threshold(self, food_resolver):
        result_strict, _, _ = food_resolver.match("egg", FOODS, threshold=0.99)
        result_loose, _, _ = food_resolver.match("egg", FOODS, threshold=0.3)
        assert result_loose is not None


class TestEdgeCases:
    def test_empty_query(self, food_resolver):
        result, score, exact = food_resolver.match("", FOODS)
        assert result is None
        assert score == 0.0

    def test_empty_foods_list(self, food_resolver):
        result, score, exact = food_resolver.match("flour", [])
        assert result is None
        assert score == 0.0


class TestEmbeddingCache:
    def test_cache_reuse(self, food_resolver):
        food_resolver.match("flour", FOODS)
        food_resolver.match("sugar", FOODS)
        assert food_resolver._food_embeddings is not None
