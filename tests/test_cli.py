import pytest
from pytoolbox.cli import CLIApplication

def test_cli_parser_initialization():
    """Test that CLI argument parser initializes successfully."""
    app = CLIApplication()
    assert app.parser is not None