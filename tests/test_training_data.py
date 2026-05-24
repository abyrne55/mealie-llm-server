from __future__ import annotations

import textwrap
from pathlib import Path

from scripts.training_data import load_training_data


def test_load_training_data(tmp_path: Path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        textwrap.dedent("""\
        ingredient_text,quantity,unit,food,note
        1 cup flour,1,cup,flour,
        1.5 cups broth,1.5,cups,chicken broth,
        salt to taste,,,salt,to taste
        "2 cloves garlic, minced",2,cloves,garlic,minced
    """)
    )
    rows = load_training_data(csv_file)

    assert len(rows) == 4
    assert rows[0] == ("1 cup flour", 1, "cup", "flour", "")
    assert rows[1] == ("1.5 cups broth", 1.5, "cups", "chicken broth", "")
    assert rows[2] == ("salt to taste", None, "", "salt", "to taste")
    assert rows[3] == ("2 cloves garlic, minced", 2, "cloves", "garlic", "minced")


def test_load_training_data_quantity_types(tmp_path: Path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        textwrap.dedent("""\
        ingredient_text,quantity,unit,food,note
        a,1,,,
        b,1.5,,,
        c,,,,
    """)
    )
    rows = load_training_data(csv_file)

    assert isinstance(rows[0][1], int)
    assert isinstance(rows[1][1], float)
    assert rows[2][1] is None
