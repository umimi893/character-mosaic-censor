import pytest

from character_mosaic import __version__
from character_mosaic.cli import build_parser


def test_cli_version(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_default_paths_and_flags(tmp_path):
    parser = build_parser()
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    args = parser.parse_args([str(input_dir), str(output_dir)])
    assert args.input == input_dir
    assert args.output == output_dir
    assert args.detect_threshold == 0.12
    assert args.auto_threshold == 0.30
    assert args.no_tiles is False
    assert args.no_flip_tta is False
