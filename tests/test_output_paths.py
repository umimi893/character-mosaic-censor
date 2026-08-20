from pathlib import Path

from character_mosaic.output_paths import default_output_for_input, output_differs_from_default


def test_default_output_follows_input_folder():
    input_path = str(Path("images") / "batch_a")
    assert default_output_for_input(input_path) == str(Path(input_path) / "_censored")


def test_default_output_ignores_surrounding_whitespace():
    input_path = str(Path("images") / "batch_b")
    assert default_output_for_input(f"  {input_path}  ") == str(Path(input_path) / "_censored")


def test_default_output_is_empty_without_input():
    assert default_output_for_input("") == ""
    assert default_output_for_input("   ") == ""


def test_matching_saved_output_is_not_treated_as_custom():
    input_path = str(Path("images") / "batch_c")
    output_path = str(Path(input_path) / "_censored")
    assert output_differs_from_default(input_path, output_path) is False


def test_different_saved_output_is_treated_as_custom():
    input_path = str(Path("images") / "batch_d")
    output_path = str(Path("exports") / "finished")
    assert output_differs_from_default(input_path, output_path) is True
