import pytest

from mealie_local_ai.regex_parser import decimal_to_fraction, normalize_numeric_text


class TestNormalizeNumericText:
    def test_standalone_unicode_half(self):
        assert normalize_numeric_text("½ cup flour") == "0.5 cup flour"

    def test_standalone_unicode_third(self):
        assert normalize_numeric_text("⅓ cup wine") == "0.333 cup wine"

    def test_standalone_unicode_quarter(self):
        assert normalize_numeric_text("¼ teaspoon salt") == "0.25 teaspoon salt"

    def test_standalone_unicode_five_eighths(self):
        assert normalize_numeric_text("⅝ cup sugar") == "0.625 cup sugar"

    def test_standalone_unicode_seven_eighths(self):
        assert normalize_numeric_text("⅞ cup cream") == "0.875 cup cream"

    def test_standalone_unicode_two_thirds(self):
        assert normalize_numeric_text("⅔ cup yeast") == "0.667 cup yeast"

    def test_mixed_unicode_no_space(self):
        assert normalize_numeric_text("2½ cups broth") == "2.5 cups broth"

    def test_mixed_unicode_with_space(self):
        assert normalize_numeric_text("1 ½ cups rice") == "1.5 cups rice"

    def test_mixed_slash(self):
        assert normalize_numeric_text("1 1/2 cups broth") == "1.5 cups broth"

    def test_slash_fraction(self):
        assert normalize_numeric_text("1/4 teaspoon salt") == "0.25 teaspoon salt"

    def test_mid_text_fraction(self):
        assert normalize_numeric_text("sliced into ⅛-inch pieces") == "sliced into 0.125-inch pieces"

    def test_multiple_fractions(self):
        result = normalize_numeric_text("½ cup plus ¼ teaspoon")
        assert result == "0.5 cup plus 0.25 teaspoon"

    def test_plain_integer_unchanged(self):
        assert normalize_numeric_text("2 cups flour") == "2 cups flour"

    def test_plain_decimal_unchanged(self):
        assert normalize_numeric_text("1.5 cups flour") == "1.5 cups flour"

    def test_no_numbers(self):
        assert normalize_numeric_text("a pinch of salt") == "a pinch of salt"

    def test_slash_false_positive_dual_measure(self):
        assert normalize_numeric_text("1 cup/1.5 oz cheese") == "1 cup/1.5 oz cheese"

    def test_slash_false_positive_large_denominator(self):
        assert normalize_numeric_text("1 cup/120 grams flour") == "1 cup/120 grams flour"

    def test_rare_unicode_one_seventh(self):
        assert normalize_numeric_text("⅐ cup water") == "0.143 cup water"

    def test_rare_unicode_one_ninth(self):
        assert normalize_numeric_text("⅑ cup milk") == "0.111 cup milk"

    def test_rare_unicode_one_tenth(self):
        assert normalize_numeric_text("⅒ cup oil") == "0.1 cup oil"

    def test_three_eighths(self):
        assert normalize_numeric_text("⅜ cup butter") == "0.375 cup butter"

    def test_mixed_three_quarters(self):
        assert normalize_numeric_text("2¾ cups stock") == "2.75 cups stock"


class TestDecimalToFraction:
    def test_half(self):
        assert decimal_to_fraction(0.5) == "½"

    def test_third(self):
        assert decimal_to_fraction(0.333) == "⅓"

    def test_two_thirds(self):
        assert decimal_to_fraction(0.667) == "⅔"

    def test_quarter(self):
        assert decimal_to_fraction(0.25) == "¼"

    def test_three_quarters(self):
        assert decimal_to_fraction(0.75) == "¾"

    def test_five_eighths(self):
        assert decimal_to_fraction(0.625) == "⅝"

    def test_seven_eighths(self):
        assert decimal_to_fraction(0.875) == "⅞"

    def test_one_eighth(self):
        assert decimal_to_fraction(0.125) == "⅛"

    def test_three_eighths(self):
        assert decimal_to_fraction(0.375) == "⅜"

    def test_mixed_number(self):
        assert decimal_to_fraction(1.5) == "1½"

    def test_mixed_third(self):
        assert decimal_to_fraction(1.333) == "1⅓"

    def test_integer(self):
        assert decimal_to_fraction(2.0) == "2"

    def test_integer_from_int(self):
        assert decimal_to_fraction(3.0) == "3"

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.2, "⅕"),
            (0.4, "⅖"),
            (0.6, "⅗"),
            (0.8, "⅘"),
            (0.167, "⅙"),
            (0.833, "⅚"),
        ],
    )
    def test_other_vulgar_fractions(self, value, expected):
        assert decimal_to_fraction(value) == expected

    def test_no_vulgar_char_fallback(self):
        result = decimal_to_fraction(0.0625)
        assert result == "1/16"
