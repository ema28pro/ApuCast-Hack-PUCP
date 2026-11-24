"""Load the official notebook definitions without running its training cells."""

import json
from pathlib import Path


def load_starter(repo, data_dir=None):
    root = Path(repo)
    ns = {"data_dir": Path(data_dir).resolve() if data_dir else root / "data"}
    path = root / "train/starter_notebook.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "fig, ax = " in source:
            continue
        if "tr = np.load" in source:
            source = source.replace("tr = np.load", "def find(name): return data_dir / name\n\ntr = np.load")
        if "\nmodel = Net()" in source:
            exec(compile(source.split("\nmodel = Net()", 1)[0], str(path), "exec"), ns)
            return ns
        exec(compile(source, str(path), "exec"), ns)
    raise RuntimeError("No se encontró la definición de la CNN del starter")
