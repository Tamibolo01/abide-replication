"""Execute a notebook headlessly with the venv kernel.

    venv/bin/python notebooks/run_notebook.py notebooks/NAME.ipynb [OUT.ipynb]

Writes the executed notebook (outputs included) back to the same file unless OUT is given.

Runs with the notebook's folder as working directory (as Jupyter would), no per-cell
timeout, errors kept in the output notebook instead of aborting, then prints a summary
of every error output so the caller can see failures without opening the file.
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

source = Path(sys.argv[1])
target = Path(sys.argv[2]) if len(sys.argv) > 2 else source
nb = nbformat.read(source, as_version=4)
client = NotebookClient(
    nb,
    timeout=None,
    kernel_name="python3",
    allow_errors=True,
    resources={"metadata": {"path": str(source.parent)}},
)
start = time.time()
client.execute()
nbformat.write(nb, target)

errors = []
for i, cell in enumerate(nb.cells):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            errors.append((i, output.get("ename"), output.get("evalue"), "\n".join(output.get("traceback", [])[-3:])))
print(f"executed {len(nb.cells)} cells in {time.time() - start:.0f} s -> {target}")
if errors:
    print(f"{len(errors)} ERROR(S):")
    for i, name, value, tail in errors:
        print(f"--- cell {i}: {name}: {value}\n{tail}")
    sys.exit(1)
print("no errors")
