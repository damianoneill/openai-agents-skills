"""Version tests."""

import re

from openai_agents_skills import __version__


def test_version() -> None:
    """Test that version follows semver format."""
    assert re.match(r"^\d+\.\d+\.\d+$", __version__)
