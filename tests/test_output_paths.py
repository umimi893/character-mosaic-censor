from pathlib import Path

from character_mosaic.output_paths import default_output_for_input


def test_default_output_follows_input_folder():
    input_path = str(Path("images") / "batch_a")
    assert default_output_for_input(input_path) == str(Path(input_path) / "_censored")


def test_default_output_ignores_surrounding_whitespace():
    input_path = str(Path("images") / "batch_b")
    assert default_output_for_input(f"  {input_path}  ") == str(Path(input_path) / "_censored")


def test_default_output_is_empty_without_input():
    assert default_output_for_input("") == ""
    assert default_output_for_input("   ") == ""
