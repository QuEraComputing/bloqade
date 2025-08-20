"""Generate the code reference pages and navigation."""

import os
from pathlib import Path

import bloqade.analog
import mkdocs_gen_files

import bloqade

# NOTE: get module paths from actual imported packages
bloqade_circuit_path = bloqade.__file__
if bloqade_circuit_path is not None:
    BLOQADE_CIRCUIT_SRC_PATH = os.path.dirname(bloqade_circuit_path)
else:
    bloqade_paths = bloqade.__path__
    BLOQADE_CIRCUIT_SRC_PATH = next(
        path for path in bloqade_paths if "bloqade-circuit/src/bloqade" in path
    )

bloqade_analog_path = bloqade.analog.__file__
if bloqade_analog_path is not None:
    BLOQADE_ANALOG_SRC_PATH = os.path.dirname(bloqade_analog_path)
else:
    BLOQADE_ANALOG_SRC_PATH = bloqade.analog.__path__[0]


skip_keywords = [
    "julia",  ## [KHW] skip for now since we didn't have julia codegen rdy
    "builder/base",  ## hiding from user
    "builder/terminate",  ## hiding from user
    "ir/tree_print",  ## hiding from user
    "ir/visitor",  ## hiding from user
    "codegen/",  ## hiding from user
    "builder/factory",  ## hiding from user
    "builder_old",  ## deprecated from user
    "task_old",  ## deprecated from user
    "visualization",  ## hiding from user
    "submission/capabilities",  ## hiding from user
    "submission/quera_api_client",
    "test/",
    "tests/",
    "test_utils",
    "bloqade-analog/docs",
    "squin/cirq/emit/",  # NOTE: this fails when included because there is an __init__.py missing, but the files have no docs anyway and it will be moved so safe to ignore
]


def make_nav(
    bloqade_package_name: str, BLOQADE_PACKAGE_PATH: str, prefix="src/bloqade"
):
    """
    build the mkdocstrings nav object for the given package

    Arguments:
        bloqade_package_name (str): name of the bloqade package. This must match with the mkdocs path as the generated pages are put under reference/<bloqade_package_name>/<prefix>/
        BLOQADE_PACKAGE_PATH (str): the path to the module.
        prefix (str): the prefix at which the source files are located in the root directory of the sub-package. Usually, that's src/bloqade for bloqade-* packages, but in the case of e.g. analog it's bloqade-analog/src/bloqade/analog.
    """
    nav = mkdocs_gen_files.Nav()
    for path in sorted(Path(BLOQADE_PACKAGE_PATH).rglob("*.py")):
        module_path = Path(
            prefix, path.relative_to(BLOQADE_PACKAGE_PATH).with_suffix("")
        )
        doc_path = Path(
            bloqade_package_name, module_path.relative_to(".").with_suffix(".md")
        )
        full_doc_path = Path("reference/", doc_path)

        iskip = False

        for kwrd in skip_keywords:
            if kwrd in str(doc_path):
                iskip = True
                break
        if iskip:
            print("[Ignore]", str(doc_path))
            continue

        print("[>]", str(doc_path))

        parts = tuple(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1].startswith("_"):
            continue

        if len(parts) == 0:
            continue

        nav[parts] = doc_path.as_posix()
        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts[1:])
            fd.write(f"::: {ident}")

        mkdocs_gen_files.set_edit_path(full_doc_path, ".." / path)

    return nav


bloqade_circuit_nav = make_nav("bloqade-circuit", BLOQADE_CIRCUIT_SRC_PATH)
with mkdocs_gen_files.open("reference/SUMMARY_BLOQADE_CIRCUIT.md", "w") as nav_file:
    nav_file.writelines(bloqade_circuit_nav.build_literate_nav())

bloqade_analog_nav = make_nav(
    "bloqade-analog", BLOQADE_ANALOG_SRC_PATH, prefix="src/bloqade/analog"
)
with mkdocs_gen_files.open("reference/SUMMARY_BLOQADE_ANALOG.md", "w") as nav_file:
    nav_file.writelines(bloqade_analog_nav.build_literate_nav())
