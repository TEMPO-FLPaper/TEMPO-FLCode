"""Dependency-free syntax audit against the original notebook definitions."""

import ast
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Reuse only the normalization definitions, not the test module's scientific imports.
test_tree = ast.parse(
    (ROOT / "tests/test_structure_and_checkpoints.py").read_text(encoding="utf-8")
)
ns = {"ast": ast, "deepcopy": deepcopy}
exec(
    compile(
        ast.Module(
            body=[
                n
                for n in test_tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in {"UndoInstance", "normalize"}
            ],
            type_ignores=[],
        ),
        "<audit normalization>",
        "exec",
    ),
    ns,
)
index = json.loads((ROOT / "tests/reference_index.json").read_text())
references = json.loads((ROOT / "implementation/config/reference_sources.json").read_text())
originals = {}
source_trees = {}
for method, reference in references.items():
    nb = json.loads((ROOT / "references/Codes" / reference["notebook"]).read_text(encoding="utf-8"))
    functions = {}
    cells = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            tree = ast.parse(
                "\n".join(
                    line
                    for line in "".join(cell["source"]).splitlines()
                    if not line.lstrip().startswith(("!", "%"))
                )
            )
            cells[i] = tree
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    functions[node.name] = node
    originals[method] = functions
    source_trees[method] = cells


def converted(entry):
    tree = ast.parse((ROOT / "implementation" / (entry["group"] + ".py")).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == entry["export_name"]
    )


count = 0
failures = []
for entry in index:
    if entry["name"] not in next(iter(originals.values())) and not any(
        entry["name"] in x for x in originals.values()
    ):
        continue
    if entry["name"] in {"build_shift_profiles"}:
        continue
    for method in entry["origins"]:
        if entry["name"] not in originals[method]:
            continue
        if ns["normalize"](converted(entry), True) != ns["normalize"](
            originals[method][entry["name"]]
        ):
            failures.append(method + ":" + entry["name"])
        count += 1
print("Original function body comparisons:", count)
print("Divergences:", failures)
if failures:
    raise SystemExit(1)
