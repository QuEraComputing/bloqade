"""Generate the code reference pages and navigation."""

import os
from pathlib import Path

import mkdocs_gen_files

if os.getenv("GITHUB_ACTIONS") == "true":
    BLOQADE_CIRCUIT_SRC_PATH = "submodules/bloqade-circuit/"
    BLOQADE_ANALOG_SRC_PATH = "submodules/bloqade-analog/"
else:
    """
    NOTE: we assume the following project structure when building locally:

    ../
    ├── bloqade
    ├── bloqade-analog
    └── bloqade-circuit
    """
    BLOQADE_CIRCUIT_SRC_PATH = "../bloqade-circuit/"
    BLOQADE_ANALOG_SRC_PATH = "../bloqade-analog/"


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
    "docs/",
    "squin/cirq/emit/",  # NOTE: this fails when included because there is an __init__.py missing, but the files have no docs anyway and it will be moved so safe to ignore
]


def make_nav(bloqade_package_name: str, BLOQADE_PACKAGE_PATH: str):
    """
    build the mkdocstrings nav object for the given package

    Arguments:
        bloqade_package_name (str): name of the bloqade package. This must match with the mkdocs path as the generated pages are put under reference/<bloqade_package_name>
        BLOQADE_PACKAGE_PATH (str): the path to the module.
    """
    nav = mkdocs_gen_files.Nav()
    for path in sorted(Path(BLOQADE_PACKAGE_PATH).rglob("*.py")):
        module_path = Path(path.relative_to(BLOQADE_PACKAGE_PATH).with_suffix(""))
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

bloqade_analog_nav = make_nav("bloqade-analog", BLOQADE_ANALOG_SRC_PATH)
with mkdocs_gen_files.open("reference/SUMMARY_BLOQADE_ANALOG.md", "w") as nav_file:
    nav_file.writelines(bloqade_analog_nav.build_literate_nav())
