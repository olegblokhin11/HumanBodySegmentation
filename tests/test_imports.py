"""Tests that the entry points import cleanly.

No unit test imports them, so without this a broken import in train.py or app.py would only
show up when somebody runs the script. All three keep their work behind main(), so importing
them has no side effects.
"""

import importlib

import pytest

ENTRY_POINTS = ["app", "evaluate", "train"]


@pytest.mark.parametrize("module_name", ENTRY_POINTS)
def test_entry_point_imports(module_name):
    """Importing an entry point succeeds and yields a module."""
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name
