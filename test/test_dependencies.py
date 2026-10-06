import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    ["bloqade.analog", "bloqade.squin", "bloqade.lanes", "bloqade.tsim"],
)
def test_component_packages_import(module):
    importlib.import_module(module)
