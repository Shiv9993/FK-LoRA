"""Turn kaggle_cells.py (cells marked `# %%`) into latent_lora_kaggle.ipynb, with the code files embedded in the notebook.
Usage: python3 make_notebook.py      (run it in this folder; it reads ../certified_switch/certified_switch.py and rewordings.json)"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EMBED = {"certified_switch.py": os.path.join(HERE, "..", "certified_switch", "certified_switch.py"),
         "latent_lora.py": os.path.join(HERE, "latent_lora.py"),
         "run_five.py": os.path.join(HERE, "run_five.py"),
         "rewordings.json": os.path.join(HERE, "rewordings.json")}

src = open(os.path.join(HERE, "kaggle_cells.py")).read()
for name, path in EMBED.items():
    text = open(path).read()
    assert "'''" not in text, f"{name} contains ''' and cannot be embedded in a raw string"
    assert not text.endswith("\\"), name
    old = f'FILES["{name}"] = None  # filled in by make_notebook.py'
    assert old in src, name
    src = src.replace(old, f'FILES["{name}"] = r\'\'\'' + text + "'''", 1)

cells, kind, buf = [], None, []


def flush():
    if kind is None:
        return
    text = "\n".join(buf).strip("\n")
    if kind == "markdown":
        text = "\n".join(re.sub(r"^# ?", "", l) for l in text.split("\n"))
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
    elif text:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})


for line in src.split("\n"):
    if line.startswith("# %%"):
        flush()
        kind, buf = ("markdown" if "[markdown]" in line else "code"), []
    else:
        buf.append(line)
flush()
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}}
json.dump(nb, open(os.path.join(HERE, "latent_lora_kaggle.ipynb"), "w"), indent=1)
print(len(cells), "cells written to latent_lora_kaggle.ipynb")
