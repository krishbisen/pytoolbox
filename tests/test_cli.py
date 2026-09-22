from pytoolbox.cli import build_parser


def test_cli_parser_initialization():
    parser = build_parser()

    assert parser is not None
    assert parser.prog == "pytoolbox"


def test_cli_commands():
    parser = build_parser()

    commands = parser._subparsers._group_actions[0].choices

    assert "clamp" in commands
    assert "palindrome" in commands
    assert "chunk" in commands