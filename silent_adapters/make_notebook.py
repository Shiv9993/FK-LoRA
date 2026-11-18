"""Turn a script with cells marked `# %%` into a Colab notebook.
Usage: python3 make_notebook.py [name]   (default: stage0)"""
import json, re, sys

VARIANTS = {"five_tasks_seeds": ("quick_test", [            # the same code as quick_test, set to three seeds
    ("seeds=[42],                            # one full run per seed; for more seeds write for example [43, 44]",
     "seeds=[42, 43, 44],                    # one full run per seed"),
    ("# # Silent adapters — short tests on five tasks\n#\n# Five short SuperNI tasks learned one after another (t5-base, 10 epochs, one seed). Exploratory: it\n# shows whether an idea does anything on a real model before many GPU-hours are spent.",
     "# # Five tasks, three seeds: certified switch and the two published methods\n#\n# **This notebook repeats the five-task comparison with seeds 42, 43 and 44** (t5-base, 10 epochs): `certified2`,\n# InfLoRA and GainLoRA, nine runs of 36 minutes each. With two GPUs (Kaggle T4 x2) about 3 hours, with one GPU\n# about 5.5 hours. At the end it prints one block of tables per seed and then all seeds together. Nothing has\n# to be edited. The text below describes the shared code."),
]), "certified15": ("quick_test", [                 # the same code as quick_test, set to the 15-task sequence
    ('PRESET = "five"', 'PRESET = "fifteen"'),
    ("# # Silent adapters — short tests on five tasks\n#\n# Five short SuperNI tasks learned one after another (t5-base, 10 epochs, one seed). Exploratory: it\n# shows whether an idea does anything on a real model before many GPU-hours are spent.",
     "# # The 15-task sequence (SuperNI order 1): certified switch and the two published methods\n#\n# **This notebook runs `certified2`, InfLoRA and GainLoRA on the 15 tasks of SuperNI order 1** (t5-base, 10 epochs,\n# one seed), all through the same code and with the same re-test. With two GPUs (Kaggle T4 x2) about 7 hours in\n# one go; with one GPU about 10 hours (on Colab it continues after a disconnect). To run only our method, set\n# `methods=[\"certified2\"]` in the `fifteen` line of the first code cell (about 3 to 4 hours). The text below\n# describes the shared code. The re-test of `certified2` also gives plain adapters (`all`) and the exact switch\n# (`key`) from the same weights. Reworded definitions exist for five of the 15 tasks only."),
])}

for NAME in (sys.argv[1:] or ["stage0", "quick_test", "check_progress", "certified15", "five_tasks_seeds"]):
  if True:

    cells, kind, buf = [], None, []
    def flush():
        if kind is None: return
        text = "\n".join(buf).strip("\n")
        if kind == "markdown":
            text = "\n".join(re.sub(r"^# ?", "", l) for l in text.split("\n"))
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
        elif text:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})

    src = open(VARIANTS.get(NAME, (NAME,))[0] + ".py").read()
    for old, new in VARIANTS.get(NAME, (None, []))[1]:
        assert old in src, old[:60]
        src = src.replace(old, new, 1)
    if "SILENT_RUN_SOURCE = None  # filled in by make_notebook.py" in src:      # put our method's code inside the notebook
        code = open("silent_run.py").read()
        assert "\'\'\'" not in code
        src = src.replace("SILENT_RUN_SOURCE = None  # filled in by make_notebook.py", "SILENT_RUN_SOURCE = r\'\'\'" + code + "\'\'\'")
    for line in src.split("\n"):
        if line.startswith("# %%"):
            flush(); kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}}}
    if NAME == "check_progress":                    # read-only: must not ask Colab for a GPU
        nb["metadata"].pop("accelerator", None); nb["metadata"]["colab"].pop("gpuType", None)
    json.dump(nb, open(NAME + "_colab.ipynb", "w"), indent=1)
    print(len(cells), "cells written to", NAME + "_colab.ipynb")
