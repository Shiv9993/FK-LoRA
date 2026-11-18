"""Tables, figures and key numbers of the paper, made from the saved result files.

    python3 make_results.py

Reads   ../silent_adapters/results/*/small_files.zip   (GPU runs: every score and every answer)
        routing/*.json                                 (CPU checks of the routes, from routing_check.py)
        ../certified_switch/demo_results/*.json        (stand-alone code on small models)
        ../certified_switch/rewordings.json
Writes  tables/*.tex, figures/*.pdf (and .png), facts.txt (the numbers quoted in the text).
Nothing is typed in by hand: when a new result file arrives, run this again.
"""
import glob
import json
import os
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIVE, FIFTEEN = "superni_short5_t5base_e10", "superni_o1_t5base_e10_ours"
LEVELS = ["original", "light", "medium", "heavy"]
SHOW = [("certified2", "certified", "\\cert{} (ours)", "ClaimLoRA (ours)"), ("certified2", "nearest", "\\nearest{}", "Nearest zone"),
        ("certified2", "centre", "\\centre{}", "Nearest centre"), ("certified2", "key", "\\keyswitch{}", "Exact key"),
        ("certified2", "all", "\\plain{}", "Plain adapters"), ("inflora", "official", "InfLoRA", "InfLoRA"),
        ("gainlora_inflora", "official", "GainLoRA", "GainLoRA")]
MAIN = [0, 3, 4, 5, 6]                                # the rows shown in the main figures
SHORT = {"task1572": "samsum", "task363": "sst2", "task1290": "xsum", "task181": "outcome extr.", "task002": "quoref",
         "task1510": "relation extr.", "task639": "multi_woz", "task1729": "personachat", "task073": "commonsenseqa",
         "task1590": "diplomacy", "task748": "glucose", "task511": "reddit_tifu", "task591": "sciq",
         "task1687": "sentiment140", "task875": "emotion"}
import figstyle as fs

fs.apply()
INK, AXIS, GRID = fs.INK, fs.AXIS, fs.GRID


class _Look(dict):
    """thin wrapper: COLOR[name] / MARK[name] come from figstyle, so a method looks the same in every figure"""
    def __init__(self, get):
        super().__init__()
        self.get_ = get

    def __missing__(self, name):
        return self.get_(name)


COLOR, MARK = _Look(fs.color), _Look(fs.marker)
# the old look, kept only for the unused figure "concept" (so that its files do not change)
OLD_RC = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.titlesize": 9, "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.labelsize": 8.5,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "axes.labelcolor": "#1a1a1a", "text.color": "#1a1a1a",
    "xtick.color": "#444444", "ytick.color": "#444444", "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#d9d9d9", "grid.linewidth": 0.5, "grid.linestyle": (0, (1, 2)),
    "legend.fontsize": 8, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}
FACTS = []


def fact(text):
    FACTS.append(text)
    print(text)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


NUMBERS = {}
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def num(name, value, digits=None):
    """A number of the text: becomes the LaTeX command \\name, so that the prose cannot disagree with the tables."""
    if digits is not None:
        value = f"{value:.{digits}f}"
        if float(value) == 0:
            value = value.lstrip("-")
        if value.startswith("-"):
            value = "$-$" + value[1:]
    elif isinstance(value, int):
        value = f"{value:,}".replace(",", "{,}")
    assert name.isalpha(), name
    assert NUMBERS.get(name, value) == value, f"two values for {name}"
    NUMBERS[name] = value


def short(task):
    return SHORT.get(task.split("_")[0], task)


def short_tex(task):
    return short(task).replace("_", "\\_")


def save(fig, name):
    fs.save(fig, name)


def write(name, text):
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write(text)


# ----------------------------------------------------------------------------------------------
# reading the GPU runs
# ----------------------------------------------------------------------------------------------
def read_runs():
    """{(tag, method, seed): run}. A later archive replaces an earlier one; repeats are kept for a comparison."""
    runs, repeats = {}, []
    for path in sorted(glob.glob(os.path.join(ROOT, "silent_adapters", "results", "*", "small_files.zip"))):
        z, found, progress = zipfile.ZipFile(path), {}, {}
        for name in z.namelist():
            m = re.match(r"silent_adapters/(?P<tag>[^/]+)/(?P=tag)_(?P<method>.+)_seed(?P<seed>\d+)/outputs/(?P<i>\d+)-(?P<task>[^/]+)/"
                         r"(?P<file>all_results|extra_eval)\.json$", name)
            if m:
                key = (m["tag"], m["method"], int(m["seed"]))
                found.setdefault(key, {}).setdefault(int(m["i"]), {"task": m["task"]})[m["file"]] = json.loads(z.read(name))
            m = re.match(r"silent_adapters/(?P<tag>[^/]+)/(?P=tag)_(?P<method>.+)_seed(?P<seed>\d+)/progress\.json$", name)
            if m:
                progress[m["tag"], m["method"], int(m["seed"])] = json.loads(z.read(name)).get("tasks", {})
        for key, steps in found.items():
            order = [i for i in sorted(steps) if "all_results" in steps[i] and "extra_eval" in steps[i]]
            if order != list(range(1, len(order) + 1)) or not order:
                continue
            tasks = [steps[i]["task"] for i in order]
            run = {"tasks": tasks, "steps": [steps[i]["extra_eval"] for i in order], "source": os.path.basename(os.path.dirname(path)),
                   "progress": [progress.get(key, {}).get(f"{i}-{steps[i]['task']}", {}) for i in order],
                   "official": [[steps[i]["all_results"][f"predict_eval_rougeL_for_{tasks[j]}"] for j in range(i)] for i in order]}
            if key in runs:
                repeats.append((key, runs[key], run))
            runs[key] = run
    return runs, repeats


def metrics(run, level, rule):
    """Learned, AP, FT, answers changed, routes: for one run, one wording and one rule, from the saved answers."""
    steps, tasks = run["steps"], run["tasks"]
    T, last = len(steps), run["steps"][-1]["levels"].get(level)
    if not last or rule not in last[0]:
        return None
    have = [j for j in range(T) if any(r["dataset"] == tasks[j] for r in last)]
    score, by_id = {}, []
    for i in range(T):
        data = steps[i]["levels"].get(level, [])
        by_id.append({(r["dataset"], r["id"]): r for r in data})
        for j in have:
            if j <= i:
                score[i, j] = mean(r[rule]["score"] for r in data if r["dataset"] == tasks[j])
    earlier = [j for j in have if j < T - 1]
    changed = total = 0
    for j in earlier:
        for key, r in by_id[j].items():
            if key[0] == tasks[j]:
                total += 1
                changed += r[rule]["pred"] != by_id[-1][key][rule]["pred"]
    routed = [r for r in last if r[rule]["route"] is not None]
    return {"have": have, "T": T, "score": score, "learned": mean(score[j, j] for j in have), "AP": mean(score[T - 1, j] for j in have),
            "FT": mean(max(score[i, j] for i in range(j, T - 1)) - score[T - 1, j] for j in earlier) if earlier else 0.0,
            "changed": changed, "compared": total, "inputs": len(last), "zone": sum(r["in_zone"] for r in last),
            "right": mean(r[rule]["route"] == tasks.index(r["dataset"]) + 1 for r in routed) if routed else None,
            "per_task": {tasks[j]: (score[j, j], score[T - 1, j]) for j in have}}


def official(run):
    rows, T = run["official"], len(run["official"])
    return {"learned": mean(rows[i][i] for i in range(T)), "AP": mean(rows[-1]),
            "FT": mean(max(r[j] for r in rows[j:-1]) - rows[-1][j] for j in range(T - 1)) if T > 1 else 0.0}


RUNS, REPEATS = read_runs()


def seeds_of(tag, method, full):
    return sorted(s for (t, m, s), run in RUNS.items() if t == tag and m == method and len(run["steps"]) == full)


def collect(tag, full, method, rule, level):
    """The metrics of every finished seed."""
    out = [metrics(RUNS[tag, method, s], level, rule) for s in seeds_of(tag, method, full)]
    return [m for m in out if m]


def signed(value, digits):
    text = f"{value:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return "$-$" + text[1:] if text.startswith("-") else text


def cell(values, digits=2, ranges=True):
    """Mean, with the lowest and highest seed if there is more than one and they differ."""
    if not values:
        return "--"
    if len(values) == 1 or not ranges or signed(min(values), 1) == signed(max(values), 1):
        return signed(mean(values), digits)
    sep = " to " if min(values) < 0 else "--"
    return f"{signed(mean(values), digits)} {{\\scriptsize({signed(min(values), 1)}{sep}{signed(max(values), 1)})}}"


def count(n):
    return f"{n:,}".replace(",", "{,}")


# ----------------------------------------------------------------------------------------------
# tables for one sequence
# ----------------------------------------------------------------------------------------------
def sequence_tables(tag, full, label):
    n_seeds = {m: len(seeds_of(tag, m, full)) for m in ("certified2", "inflora", "gainlora_inflora")}
    if not n_seeds["certified2"]:
        print(f"[{label}] no finished run of certified2 yet")
        return None
    fact(f"[{label}] finished seeds: {n_seeds}; seeds of certified2: {seeds_of(tag, 'certified2', full)}")
    # numbers for the prose
    P = label.capitalize()
    names = {("certified2", "certified"): "Cert", ("certified2", "nearest"): "Near", ("certified2", "centre"): "Centre", ("certified2", "key"): "Key",
             ("certified2", "all"): "Plain", ("inflora", "official"): "Inf", ("gainlora_inflora", "official"): "Gain"}
    for (method, rule), M in names.items():
        ns = len(seeds_of(tag, method, full))
        num(f"{P}{M}Seeds", WORDS[ns] if ns < len(WORDS) else str(ns))
        for level in LEVELS:
            ms = collect(tag, full, method, rule, level)
            if not ms:
                continue
            W = "" if level == "original" else level.capitalize()
            for key, short_key in (("learned", "Learned"), ("AP", "AP"), ("FT", "FT")):
                num(f"{P}{M}{short_key}{W}", mean(m[key] for m in ms), 1)
                if len(ms) > 1:
                    num(f"{P}{M}{short_key}{W}Low", min(m[key] for m in ms), 1)
                    num(f"{P}{M}{short_key}{W}High", max(m[key] for m in ms), 1)
            num(f"{P}{M}Changed{W}", sum(m["changed"] for m in ms))
            num(f"{P}{M}Compared{W}", sum(m["compared"] for m in ms))
            if ms[0]["right"] is not None:
                num(f"{P}{M}Right{W}", 100 * mean(m["right"] for m in ms), 1)
                num(f"{P}{M}Wrong{W}", sum(round((1 - m["right"]) * m["inputs"]) for m in ms) // len(ms))
            if level == "original" and M in ("Cert", "Plain", "Inf", "Gain"):
                for task in ms[0]["per_task"]:
                    key = re.sub("[^a-z]", "", short(task).split()[0].lower()).capitalize()
                    num(f"{P}{M}{key}Learned", mean(m["per_task"][task][0] for m in ms), 1)
                    num(f"{P}{M}{key}End", mean(m["per_task"][task][1] for m in ms), 1)
            if M == "Cert":
                num(f"{P}CertZone{W}", sum(m["zone"] for m in ms))
                num(f"{P}CertInputs{W}", sum(m["inputs"] for m in ms))
                num(f"{P}CertZoneOne{W}", ms[0]["zone"])
                num(f"{P}CertInputsOne{W}", ms[0]["inputs"])
        offs = [official(RUNS[tag, method, s]) for s in seeds_of(tag, method, full)] if rule in ("certified", "official") else []
        for key, short_key in (("learned", "OffLearned"), ("AP", "OffAP"), ("FT", "OffFT")):
            if offs:
                num(f"{P}{M}{short_key}", mean(o[key] for o in offs), 1)
        same = [st.get("same_as_official") for s in seeds_of(tag, method, full) for st in RUNS[tag, method, s]["steps"]] if rule in ("certified", "official") else []
        same = [x for x in same if x]
        if same:
            num(f"{P}{M}Same", sum(a for a, _ in same))
            num(f"{P}{M}SameOf", sum(b for _, b in same))
        if rule in ("certified", "official"):
            runs = [RUNS[tag, method, s] for s in seeds_of(tag, method, full)]
            minutes = [sum(pr.get("minutes") or 0 for pr in r["progress"]) for r in runs]
            if runs and all(minutes):
                num(f"{P}{M}Minutes", round(mean(minutes)))
                fact(f"[{label}] minutes per run | {method}: " + ", ".join(f"{x:.0f}" for x in minutes))
            blocked = [re.search(r"older zone: (\d+) of (\d+)", pr.get("zone") or "") for r in runs for pr in r["progress"]]
            if M == "Cert" and blocked and all(blocked):
                num(f"{P}CertBlocked", sum(int(b[1]) for b in blocked))
                num(f"{P}CertTrainInputs", sum(int(b[2]) for b in blocked))
                radii = [float(re.search(r"zone radius ([0-9.]+)", pr["zone"])[1]) for pr in runs[0]["progress"]]
                fact(f"[{label}] certified2: training inputs inside an older zone {sum(int(b[1]) for b in blocked)} of {sum(int(b[2]) for b in blocked)}; "
                     f"radii {min(radii):.1f} to {max(radii):.1f}")
    # main table: original wording (the column with the number of seeds is left out if it is the same for every method)
    uniform = len(set(n_seeds.values())) == 1
    lines = []
    for method, rule, tex, plain_name in SHOW:
        ms = collect(tag, full, method, rule, "original")
        if not ms:
            continue
        zone = f"{count(sum(m['zone'] for m in ms))} of {count(sum(m['inputs'] for m in ms))}" if rule == "certified" else ""
        lines.append(f"    {tex} & " + ("" if uniform else f"{len(ms)} & ") + f"{cell([m['learned'] for m in ms])} & {cell([m['AP'] for m in ms])} & {cell([m['FT'] for m in ms])} & "
                     f"{count(sum(m['changed'] for m in ms))} of {count(sum(m['compared'] for m in ms))} & {zone} \\\\")
        fact(f"[{label}] original | {plain_name}: seeds {len(ms)} learned {mean(m['learned'] for m in ms):.2f} AP {mean(m['AP'] for m in ms):.2f} "
             f"FT {mean(m['FT'] for m in ms):.2f} changed {sum(m['changed'] for m in ms)}/{sum(m['compared'] for m in ms)}"
             + (f" right {mean(m['right'] for m in ms):.3f}" if ms[0]["right"] is not None else ""))
        if method == "certified2" and rule == "all":
            lines.append("    \\midrule")
    write(f"{label}_main", "\\begin{tabular}{l" + ("" if uniform else "c") + "ccccc}\n    \\toprule\n    Method & " + ("" if uniform else "Seeds & ")
          + "Learned & AP $\\uparrow$ & FT $\\downarrow$ & Changed & "
          "Certified \\\\\n    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    # by wording: AP and answers changed
    lines = []
    for method, rule, tex, plain_name in SHOW:
        aps, chg = [], []
        for level in LEVELS:
            ms = collect(tag, full, method, rule, level)
            aps.append(cell([m["AP"] for m in ms], 1, ranges=False) if ms else "--")
            chg.append(count(sum(m['changed'] for m in ms)) if ms else "--")
            if ms:
                fact(f"[{label}] {level} | {plain_name}: tasks {len(ms[0]['have'])} learned {mean(m['learned'] for m in ms):.2f} AP {mean(m['AP'] for m in ms):.2f} "
                     f"FT {mean(m['FT'] for m in ms):.2f} changed {sum(m['changed'] for m in ms)}/{sum(m['compared'] for m in ms)}"
                     + (f" right {mean(m['right'] for m in ms):.3f}" if ms[0]["right"] is not None else "")
                     + (f" in zone {sum(m['zone'] for m in ms)}/{sum(m['inputs'] for m in ms)}" if rule == "certified" else ""))
        if any(a != "--" for a in aps):
            lines.append(f"    {tex} & " + " & ".join(aps) + " & " + " & ".join(chg) + " \\\\")
            if method == "certified2" and rule == "all":
                lines.append("    \\midrule")
    ms = collect(tag, full, "certified2", "certified", "light")
    per = count(sum(m['compared'] for m in ms)) if ms else "?"
    ms0 = collect(tag, full, "certified2", "certified", "original")
    per0 = count(sum(m['compared'] for m in ms0)) if ms0 else "?"
    # the count goes into the header only when it is the same for every wording (it is not for the 15-task sequence)
    head = "Answers changed (of " + per + ")" if per0 == per else "Answers changed"
    write(f"{label}_wordings", "\\begin{tabular}{lcccccccc}\n    \\toprule\n    & \\multicolumn{4}{c}{AP $\\uparrow$} & \\multicolumn{4}{c}{"
          + head + "} \\\\\n    \\cmidrule(lr){2-5} \\cmidrule(lr){6-9}\n    Method & original & light & medium & heavy & original & light & medium & heavy \\\\\n"
          "    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    # learned and FT by wording (appendix)
    lines = []
    for method, rule, tex, _ in SHOW:
        cells = []
        for level in LEVELS:
            ms = collect(tag, full, method, rule, level)
            cells += [signed(mean(m['learned'] for m in ms), 1) if ms else "--", signed(mean(m['FT'] for m in ms), 1) if ms else "--"]
        if any(c != "--" for c in cells):
            lines.append(f"    {tex} & " + " & ".join(cells) + " \\\\")
    write(f"{label}_wordings_more", "\\begin{tabular}{lcccccccc}\n    \\toprule\n    & \\multicolumn{2}{c}{original} & \\multicolumn{2}{c}{light} & "
          "\\multicolumn{2}{c}{medium} & \\multicolumn{2}{c}{heavy} \\\\\n    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}\n"
          "    Method & Learned & FT & Learned & FT & Learned & FT & Learned & FT \\\\\n    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    # routes of the certified switch by wording
    lines = []
    for level in LEVELS:
        ms = collect(tag, full, "certified2", "certified", level)
        mc = collect(tag, full, "certified2", "centre", level)
        if ms:
            lines.append(f"    {level} & {len(ms[0]['have'])} & {count(sum(m['zone'] for m in ms))} of {count(sum(m['inputs'] for m in ms))} & {100 * mean(m['right'] for m in ms):.1f}\\% & "
                         f"{count(sum(m['changed'] for m in ms))} of {count(sum(m['compared'] for m in ms))} & {100 * mean(m['right'] for m in mc):.1f}\\% & "
                         f"{count(sum(m['changed'] for m in mc))} of {count(sum(m['compared'] for m in mc))} \\\\")
    write(f"{label}_routes", "\\begin{tabular}{lcccccc}\n    \\toprule\n    & & \\multicolumn{3}{c}{\\cert{}} & \\multicolumn{2}{c}{\\centre{}} \\\\\n"
          "    \\cmidrule(lr){3-5} \\cmidrule(lr){6-7}\n    Wording & Tasks & Certified inputs & Right task & Answers changed & Right task & Answers changed \\\\\n"
          "    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    # per task, original wording
    tasks = RUNS[tag, "certified2", seeds_of(tag, "certified2", full)[0]]["tasks"]
    cols = [(m, r, t) for m, r, t, _ in SHOW if (m, r) in (("certified2", "certified"), ("certified2", "all"), ("inflora", "official"), ("gainlora_inflora", "official"))
            and collect(tag, full, m, r, "original")]
    lines = []
    for task in tasks:
        cells = []
        for m, r, _ in cols:
            ms = collect(tag, full, m, r, "original")
            cells.append(f"{mean(x['per_task'][task][0] for x in ms):.1f} $\\to$ {mean(x['per_task'][task][1] for x in ms):.1f}")
        lines.append(f"    {short_tex(task)} & " + " & ".join(cells) + " \\\\")
    write(f"{label}_pertask", "\\begin{tabular}{l" + "c" * len(cols) + "}\n    \\toprule\n    Task (in order) & " + " & ".join(t for _, _, t in cols)
          + " \\\\\n    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    # the official scores (one randomly picked reference answer)
    lines = []
    for method, tex in (("certified2", "\\cert{} (ours)"), ("inflora", "InfLoRA"), ("gainlora_inflora", "GainLoRA")):
        offs = [official(RUNS[tag, method, s]) for s in seeds_of(tag, method, full)]
        if offs:
            lines.append(f"    {tex} & " + ("" if uniform else f"{len(offs)} & ") + " & ".join(cell([o[k] for o in offs]) for k in ("learned", "AP", "FT")) + " \\\\")
            fact(f"[{label}] official script | {method}: " + " ".join(f"{k} {mean(o[k] for o in offs):.2f}" for k in ("learned", "AP", "FT")))
    write(f"{label}_official", "\\begin{tabular}{l" + ("" if uniform else "c") + "ccc}\n    \\toprule\n    Method & " + ("" if uniform else "Seeds & ")
          + "Learned & AP $\\uparrow$ & FT $\\downarrow$ \\\\\n    \\midrule\n"
          + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    # does the re-test reproduce the official test?
    for method in ("certified2", "inflora", "gainlora_inflora"):
        same = [st.get("same_as_official") for s in seeds_of(tag, method, full) for st in RUNS[tag, method, s]["steps"]]
        same = [x for x in same if x]
        if same:
            fact(f"[{label}] re-test equals official test | {method}: {sum(a for a, _ in same)} of {sum(b for _, b in same)} answers")
    return tasks


# ----------------------------------------------------------------------------------------------
# figures for one sequence
# ----------------------------------------------------------------------------------------------
def figure_wordings(tag, full, label):
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    shown = [SHOW[i] for i in MAIN if collect(tag, full, SHOW[i][0], SHOW[i][1], "original")]
    width = 0.8 / len(shown)
    for k, (method, rule, _, name) in enumerate(shown):
        for x, level in enumerate(LEVELS):
            ms = collect(tag, full, method, rule, level)
            if not ms:
                continue
            aps = [m["AP"] for m in ms]
            pos = x - 0.4 + (k + 0.5) * width
            ax.bar(pos, mean(aps), width * 0.88, color=fs.color(name), linewidth=0, zorder=3)
            if len(aps) > 1:
                ax.plot([pos, pos], [min(aps), max(aps)], color=fs.INK, lw=0.8, zorder=4, solid_capstyle="butt")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["original\ninstruction", "light\nrewording", "medium\nrewording", "heavy\nrewording"])
    ax.set_ylabel("average score at the end (AP)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, 3.5)
    ax.tick_params(axis="x", length=0)
    fig.subplots_adjust(top=0.9)
    fs.legend_above(fig, [Patch(facecolor=fs.color(n)) for *_, n in shown], [n for *_, n in shown], y=0.93)
    save(fig, f"{label}_wordings")


def figure_learn_forget(tags):
    tags = [(tag, full, title) for tag, full, title in tags if seeds_of(tag, "certified2", full)]
    fig, axes = plt.subplots(1, len(tags), figsize=(5.5, 2.5), sharey=True, squeeze=False)
    for ax, (tag, full, title), letter in zip(axes[0], tags, "abcd"):
        shown = [SHOW[i] for i in MAIN if collect(tag, full, SHOW[i][0], SHOW[i][1], "original")]
        for y, (method, rule, _, name) in enumerate(shown):
            ms = collect(tag, full, method, rule, "original")
            learned, ap = mean(m["learned"] for m in ms), mean(m["AP"] for m in ms)
            col = fs.color(name)
            fs.dumbbell(ax, y, learned, ap, col, lw=3.0 if y == 0 else 2.2, ms=6 if y == 0 else 5.5)
            lo, hi = sorted((learned, ap))
            if f"{learned:.0f}" == f"{ap:.0f}":
                ax.text(hi + 3.5, y, f"{ap:.0f}", ha="left", va="center", fontsize=7.5, color=fs.INK)
            else:
                ax.text(lo - 3.5, y, f"{lo:.0f}", ha="right", va="center", fontsize=7.5, color=fs.INK)
                ax.text(hi + 3.5, y, f"{hi:.0f}", ha="left", va="center", fontsize=7.5, color=fs.INK)
            if name == "Plain adapters":
                plain = (y, (learned + ap) / 2, ap)
        fs.panel_title(ax, letter, title.split(") ", 1)[1])
        ax.set_xlim(0, 100)
        ax.set_ylim(len(shown) - 0.5, -0.6)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Rouge-L")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2.5)
        ax.set_yticks(range(len(shown)))
        ax.spines["left"].set_visible(False)
        if ax is axes[0][0]:
            ax.set_yticklabels([n for *_, n in shown], color=fs.INK)
    y, mid, low = plain
    fs.note(axes[0][-1], "the gap is what\nwas forgotten", xy=(mid, y + 0.12), xytext=(mid + 22, y + 1.15), ha="left")
    fs.legend_above(fig, fs.open_filled_handles(), ["open: right after learning", "filled: after the last task"], y=0.93)
    fig.subplots_adjust(wspace=0.08, top=0.84)
    save(fig, "learn_forget")


def figure_tracks(tag, full, label):
    from matplotlib.lines import Line2D
    seed = seeds_of(tag, "certified2", full)[0]
    shown = [SHOW[i] for i in (4, 0, 5, 6) if (tag, SHOW[i][0], seed) in RUNS and len(RUNS[tag, SHOW[i][0], seed]["steps"]) == full]
    fig, axes = plt.subplots(1, len(shown), figsize=(5.5, 2.1), sharey=True, squeeze=False)
    tasks = RUNS[tag, "certified2", seed]["tasks"]
    for ax, (method, rule, _, name), tagletter in zip(axes[0], shown, "abcd"):
        m = metrics(RUNS[tag, method, seed], "original", rule)
        for j, task in enumerate(tasks):
            vals = [m["score"][i, j] for i in range(j, full)]
            ax.plot(range(j + 1, full + 1), vals, color=fs.TASKS[j % 5], lw=1.5, marker=fs.TASK_MARKS[j % 5], ms=3.8, clip_on=False, zorder=3)
        fs.panel_title(ax, tagletter, name.replace(" (ours)", ""))
        ax.set_xticks(range(1, full + 1))
        ax.set_xlim(0.8, full + 0.2)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25))
        ax.set_xlabel("tasks learned")
        ax.tick_params(axis="x", length=2.5)
    axes[0][0].set_ylabel("Rouge-L")
    # the note sits in the empty strip between the lines at 63 and 99, where no grid line runs
    t = axes[0][1].text(3.0, 37.5, "flat: nothing changes", ha="center", va="center", fontsize=7.5, color=fs.INK2,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), zorder=5)
    handles = [Line2D([], [], color=fs.TASKS[j % 5], lw=1.5, marker=fs.TASK_MARKS[j % 5], ms=3.8) for j in range(len(tasks))]
    fig.legend(handles, [short(t_) for t_ in tasks], loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=len(tasks), handlelength=2.0,
               columnspacing=1.4, handletextpad=0.5, borderaxespad=0.0)
    fig.subplots_adjust(wspace=0.14)
    save(fig, f"{label}_tracks")


def figure_dumbbell(tag, full, label):
    seed = seeds_of(tag, "certified2", full)[0]
    shown = [SHOW[i] for i in (4, 0, 5, 6) if (tag, SHOW[i][0], seed) in RUNS and len(RUNS[tag, SHOW[i][0], seed]["steps"]) == full]
    tasks = RUNS[tag, "certified2", seed]["tasks"]
    fig, axes = plt.subplots(1, len(shown), figsize=(5.5, 3.3), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.5] + [1] * (len(shown) - 2)})
    for ax, (method, rule, _, name), tagletter in zip(axes[0], shown, "abcd"):
        m = metrics(RUNS[tag, method, seed], "original", rule)
        ours = "ours" in name
        for y, task in enumerate(tasks):
            a, b = m["per_task"][task]
            fs.dumbbell(ax, y, a, b, fs.color(name), lw=2.6 if ours else 2.0, ms=4.8 if ours else 4.2)
        fs.panel_title(ax, tagletter, name)
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("Rouge-L")
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0][0].set_yticks(range(len(tasks)))
    axes[0][0].set_yticklabels([short(t) for t in tasks], color=fs.INK)
    axes[0][0].invert_yaxis()
    ax = axes[0][0]
    a, b = metrics(RUNS[tag, shown[0][0], seed], "original", shown[0][1])["per_task"][tasks[5]]
    fs.note(ax, "the gap is what\nwas forgotten", xy=(80, 5.15), xytext=(103, 7.7), ha="right")
    fs.legend_above(fig, fs.open_filled_handles(), ["open: right after the task was learned", "filled: after the last task"], y=0.94)
    fig.subplots_adjust(wspace=0.1, top=0.86)
    save(fig, f"{label}_tasks")


def figure_concept():
    with plt.rc_context(OLD_RC):
        _figure_concept()


def _figure_concept():
    INK, AXIS = "#1a1a1a", "#444444"
    fig, ax = plt.subplots(figsize=(5.5, 2.35))
    ax.set_aspect("equal")
    ax.axis("off")
    zones = [((2.0, 2.2), 1.25, "#0072B2", "task 1"), ((5.2, 2.9), 1.0, "#009E73", "task 2"), ((3.9, 0.9), 0.85, "#CC79A7", "task 3")]
    for (cx, cy), r, c, name in zones:
        ax.add_patch(Circle((cx, cy), r, facecolor=c, alpha=0.13, edgecolor=c, lw=1.1))
        ax.plot(cx, cy, marker="*", ms=8, color=c)
        ax.text(cx, cy + r + 0.08, name, ha="center", va="bottom", fontsize=8, color=c)
    ax.add_patch(Circle((2.95, 1.55), 0.8, facecolor="none", edgecolor="#D55E00", lw=1.1, ls=(0, (4, 2))))
    ax.plot(2.95, 1.55, marker="*", ms=8, color="#D55E00")
    ax.text(2.42, 0.62, "task 4 (later)", ha="center", va="top", fontsize=8, color="#D55E00")
    for x, y in ((0.4, 0.5), (6.9, 1.0), (7.4, 3.3), (0.6, 3.7), (6.3, 0.2)):
        ax.plot(x, y, marker="x", ms=5, color="#7f7f7f")
    ax.text(7.4, 3.5, "unrelated tasks (x)", ha="center", va="bottom", fontsize=7.5, color="#7f7f7f")
    pts = [((1.35, 2.85), "A", "certified: task 1, for ever"), ((2.75, 1.95), "B", "in zones 1 and 4: the older claim wins"),
           ((5.9, 1.55), "C", "outside every zone: nearest zone (task 2), no certificate")]
    for (x, y), tag, _ in pts:
        ax.plot(x, y, marker="o", ms=5, color=INK)
        ax.text(x + 0.12, y + 0.08, tag, fontsize=8.5, fontweight="bold")
    ax.plot([5.9, 5.52], [1.55, 2.02], color=INK, lw=0.7, ls=(0, (2, 2)))
    for k, (_, tag, text) in enumerate(pts):
        ax.text(8.25, 3.0 - 0.62 * k, f"{tag}: {text}", fontsize=8, va="center")
    ax.text(8.25, 1.0, "star: signature of a task\ncircle: its protected zone (never changed)", fontsize=7.5, va="center", color=AXIS)
    ax.set_xlim(-0.2, 13.4)
    ax.set_ylim(-0.3, 4.4)
    with plt.rc_context(OLD_RC):
        save(fig, "concept")


# ----------------------------------------------------------------------------------------------
# CPU checks: routes for several models, the stress test, the radius factor
# ----------------------------------------------------------------------------------------------
def routing_tables():
    path = os.path.join(HERE, "routing", "models.json")
    if os.path.exists(path):
        data = json.load(open(path))
        lines = []
        for r in data["runs"]:
            f5, f15 = r["five"], r["fifteen"]
            pct = lambda d, k: "/".join(f"{100 * d[l][k]:.0f}" for l in LEVELS)
            lines.append(f"    \\texttt{{{r['model'].split('/')[-1]}}} & {r['dimensions']} & {pct(f5, 'zone')} & {pct(f5, 'certified')} & "
                         f"{pct(f15, 'certified')} & {sum(v['wrong'] for v in list(f5.values()) + list(f15.values()))} & "
                         f"{sum(v['certified_changed'] for v in list(f5.values()) + list(f15.values()))} \\\\")
            fact(f"[models] {r['model']}: five zone {pct(f5, 'zone')} right {pct(f5, 'certified')} | fifteen zone(orig) {100 * f15['original']['zone']:.0f} right {pct(f15, 'certified')} "
                 f"| wrong zone {sum(v['wrong'] for v in list(f5.values()) + list(f15.values()))} blocked {f5['original']['blocked'] + f15['original']['blocked']} "
                 f"changed {sum(v['certified_changed'] for v in list(f5.values()) + list(f15.values()))}")
        write("models", "\\begin{tabular}{lcccccc}\n    \\toprule\n    & & \\multicolumn{2}{c}{Five tasks} & Fifteen tasks & & \\\\\n"
              "    \\cmidrule(lr){3-4} \\cmidrule(lr){5-5}\n    Embedding table of & $d$ & certified, \\% & right task, \\% & right task, \\% & "
              "Wrong zone & Changed \\\\\n    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    path = os.path.join(HERE, "routing", "t5base.json")
    if os.path.exists(path):
        data = json.load(open(path))
        lines = []
        for r in data["runs"]:
            f5, f15 = r["five"], r["fifteen"]
            pct = lambda d, k: " / ".join(f"{100 * d[l][k]:.1f}" for l in LEVELS)
            lines.append(f"    {r['kappa']} & {pct(f5, 'zone')} & {pct(f5, 'certified')} & {pct(f15, 'certified')} & "
                         f"{sum(v['wrong'] for v in list(f5.values()) + list(f15.values()))} \\\\")
        write("kappa", "\\begin{tabular}{lcccc}\n    \\toprule\n    $\\kappa$ & Five: certified, \\% & Five: right task, \\% & Fifteen: right task, \\% & Wrong zone \\\\\n"
              "    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
        st = data["stress"].get("t5-base|0.5")
        if st:
            lines = []
            for look in ("light", "medium", "heavy"):
                s = st[look]
                row = [f"{s['blocked']} of {s['train']}"]
                for rule in ("certified", "nearest"):
                    row += [str(s[f"{rule}|{level}"]["changed"]) for level in LEVELS]
                lines.append(f"    {look} & " + " & ".join(row) + " \\\\")
                fact(f"[stress] look-alike with {look} rewording: blocked {s['blocked']}/{s['train']}; certified changed "
                     + "/".join(str(s[f'certified|{l}']['changed']) for l in LEVELS) + "; nearest changed " + "/".join(str(s[f'nearest|{l}']['changed']) for l in LEVELS)
                     + "; certified before (certified rule) " + "/".join(str(s[f'certified|{l}']['certified_before']) for l in LEVELS))
            write("stress", "\\begin{tabular}{lccccccccc}\n    \\toprule\n    & & \\multicolumn{4}{c}{Routes changed, \\cert{}} & \\multicolumn{4}{c}{Routes changed, \\nearest{}} \\\\\n"
                  "    \\cmidrule(lr){3-6} \\cmidrule(lr){7-10}\n    New task's definition & Blocked & orig. & light & med. & heavy & orig. & light & med. & heavy \\\\\n"
                  "    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")


def geometry_outputs():
    """How far the test inputs are from their own zone and from the nearest other zone (from extra_analysis.py)."""
    path = os.path.join(HERE, "routing", "geometry.json")
    if not os.path.exists(path):
        return
    g = json.load(open(path))
    lines = []
    for row in g["five"]:
        w = row["wordings"]
        lines.append(f"    {short_tex(row['task'])} & {row['opening']} & {row['radius']:.1f} & "
                     + " & ".join(f"{w[l]['own_median']:.2f} ({w[l]['own_max']:.2f})" for l in LEVELS[1:])
                     + f" & {min(w[l]['other_min'] for l in LEVELS):.2f} \\\\")
        fact(f"[geometry] {short(row['task'])}: opening {row['opening']} tokens, radius {row['radius']:.1f}; own zone median (max): "
             + "; ".join(f"{l} {w[l]['own_median']:.2f} ({w[l]['own_max']:.2f})" for l in LEVELS) + "; nearest other zone, smallest: "
             + "/".join(f"{w[l]['other_min']:.2f}" for l in LEVELS))
    write("geometry", "\\begin{tabular}{lcccccc}\n    \\toprule\n    & & & \\multicolumn{3}{c}{Distance from own signature, in radii, median (largest)} & Nearest other \\\\\n"
          "    \\cmidrule(lr){4-6}\n    Task & $n_k$ & $R_k$ & light & medium & heavy & signature, in its radii \\\\\n    \\midrule\n" + "\n".join(lines)
          + "\n    \\bottomrule\n\\end{tabular}\n")
    o15 = [r["wordings"]["original"]["other_min"] for r in g["fifteen"]]
    fact(f"[geometry] fifteen tasks, original wording: nearest other zone over all test inputs, smallest {min(o15):.2f}, largest {max(o15):.2f}; "
         f"radii {min(r['radius'] for r in g['fifteen']):.1f} to {max(r['radius'] for r in g['fifteen']):.1f}; "
         f"openings {min(r['opening'] for r in g['fifteen'])} to {max(r['opening'] for r in g['fifteen'])} tokens")
    c = g["cost"]
    fact(f"[cost] routing one input, {c['tasks']} tasks, CPU, one thread: {1000 * c['seconds_per_input_cpu']:.1f} ms; stored per task: {c['numbers_per_task']} numbers; "
         f"stored once: {c['numbers_once']} numbers (d = {c['dimensions']})")
    # figure: relative distances, five tasks
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ramp = [fs.SEQ(v) for v in (0.30, 0.55, 0.78, 1.0)]
    shades = dict(zip(LEVELS, ramp))
    marks = {"original": "o", "light": "s", "medium": "^", "heavy": "D"}
    ax.axhspan(0, 1.0, color=fs.KEPT, alpha=0.4, lw=0, zorder=0)
    for x, row in enumerate(g["five"]):
        for k, level in enumerate(LEVELS):
            w = row["wordings"][level]
            pos = x - 0.27 + 0.18 * k
            ax.plot([pos, pos], [w["own_median"], w["own_max"]], color=shades[level], lw=1.3, zorder=3, solid_capstyle="butt")
            ax.plot(pos, w["own_median"], marker=marks[level], ms=5, color=shades[level], mec=fs.SURFACE, ls="none", zorder=4)
            ax.plot([pos - 0.07, pos + 0.07], [w["other_min"], w["other_min"]], color=fs.INK2, lw=1.6, solid_capstyle="butt", zorder=3)
    ax.axhline(1.0, color=fs.MUTED, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.text(4.46, 1.04, "border of the zone", ha="right", va="bottom", fontsize=7.5, color=fs.INK2)
    ax.text(4.46, 0.94, "inside: certified", ha="right", va="top", fontsize=7.5, color=fs.INK2)
    ax.set_xticks(range(5))
    ax.set_xticklabels([short(r["task"]) for r in g["five"]])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("distance from a zone's signature,\nin units of the zone's radius")
    ax.set_ylim(-0.08, 3.0)
    ax.set_xlim(-0.5, 4.5)
    fs.note(ax, "the nearest other task stays far outside", xy=(3.09, 2.27), xytext=(2.15, 2.88), ha="left")
    handles = [Line2D([], [], ls="none", marker=marks[l], ms=5, color=shades[l], mec=fs.SURFACE) for l in LEVELS]
    handles.append(Line2D([], [], color=fs.INK2, lw=1.6))
    fs.legend_above(fig, handles, [f"{l} wording" for l in LEVELS] + ["nearest other task (closest test input)"], y=0.95)
    save(fig, "margins")


def orders_outputs():
    """Does the order of the tasks matter for the routes? (from order_analysis.py)"""
    path = os.path.join(HERE, "routing", "orders.json")
    if not os.path.exists(path):
        return
    o = json.load(open(path))

    def spread(d, scale=1.0, digits=0):
        lo, hi = scale * d["min"], scale * d["max"]
        short_number = lambda v: f"{v:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{v:.0f}"
        return short_number(lo) if abs(hi - lo) < 1e-9 else f"{short_number(lo)}--{short_number(hi)}"

    def row(name, n, r, many):
        get = (lambda level, key: r[f"{level}|{key}"]) if many else (lambda level, key: {"min": r[level][key], "max": r[level][key]})
        right = "/".join(spread(get(l, "right"), 100, 1) for l in LEVELS)
        own = "/".join(spread(get(l, "own_zone"), 100, 1) for l in LEVELS)
        wrong = max(get(l, "wrong_zone")["max"] for l in LEVELS)
        blocked = r["blocked_max"] if many else r["blocked"]
        lo, hi = sum(get(l, "changed")["min"] for l in LEVELS), sum(get(l, "changed")["max"] for l in LEVELS)
        changed = f"{lo:.0f}" if lo == hi else f"{lo:.0f}--{hi:.0f}"
        fact(f"[orders] {name}: orders {n}; right o/l/m/h {right}; own zone {own}; wrong zone (most) {wrong:.0f}; blocked (most) {blocked}; routes changed later {changed}")
        return f"    {name} & {n} & {right} & {own} & {wrong:.0f} & {blocked} & {changed} \\\\"

    lines = [row("5 tasks, all orders", "120", o["five_all_120"], True), "    \\midrule",
             row("15 tasks, official order 1", "1", o["fifteen_order1"], False),
             row("15 tasks, official order 2", "1", o["fifteen_order2"], False),
             row("15 tasks, random orders", "1,000", o["fifteen_random_1000"], True)]
    write("orders", "\\begin{tabular}{lcccccc}\n    \\toprule\n    Sequence & Orders & Right task, \\% & Certified, \\% & Wrong zone & Blocked & Changed \\\\\n"
          "    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    lines = []
    for r in o["fifteen_distances"]:
        nearer = r["nearest_task"] < r["nearest_unrelated"]
        lines.append(f"    {short_tex(r['task'])} & {r['nearest_unrelated']:.1f} & {r['nearest_task']:.1f} ({short_tex(r['which'])}) & "
                     + ("an earlier task, in some orders" if nearer else "an unrelated task, in every order") + " \\\\")
    fact("[orders] tasks whose nearest signature is another task (radius depends on the order): "
         + ", ".join(f"{short(r['task'])} ({r['nearest_task']:.1f} vs unrelated {r['nearest_unrelated']:.1f}, {short(r['which'])})"
                     for r in o["fifteen_distances"] if r["nearest_task"] < r["nearest_unrelated"]))
    write("distances", "\\begin{tabular}{lccl}\n    \\toprule\n    Task & Nearest unrelated & Nearest other task & The radius is set by \\\\\n    \\midrule\n"
          + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")


def edits_outputs():
    """The mechanical edit test of the routes (from edit_analysis.py)."""
    path = os.path.join(HERE, "routing", "edits.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    rows = {e["edit"]: e for e in d["edits"]}
    edited = [e for e in d["edits"] if e["edit"] != "none"]
    fact(f"[edits] edited inputs in total: {sum(e['inputs'] for e in edited)}; inside the zone of another task: {sum(e['foreign_zone'] for e in d['edits'])}; "
         f"test inputs per condition: {rows['none']['inputs']}")
    names = {"typo": "typos", "drop": "dropped words", "replace": "replaced words", "lower": "lower case", "polite": "sentence in front",
             "no-first": "first sentence removed", "no-last": "last sentence removed", "shuffle": "sentences shuffled"}
    ks = (1, 2, 4, 8, 16)
    for kind in ("typo", "drop", "replace"):
        fact(f"[edits] {names[kind]}, k = 1/2/4/8/16: certified " + "/".join(f"{100 * rows[f'{kind}-{k}']['own_zone']:.1f}" for k in ks)
             + "; right task " + "/".join(f"{100 * rows[f'{kind}-{k}']['right']:.1f}" for k in ks)
             + "; worst task right " + "/".join(f"{100 * rows[f'{kind}-{k}']['worst_right']:.0f} ({short(rows[f'{kind}-{k}']['worst_task'])})" for k in ks))
    lines = []
    for kind in ("lower", "polite", "no-first", "no-last", "shuffle"):
        e = rows[kind]
        lines.append(f"    {names[kind].capitalize()} & {e['tasks']} & {e['inputs']} & {100 * e['own_zone']:.1f} & {100 * e['right']:.1f} & {e['foreign_zone']} \\\\")
        fact(f"[edits] {names[kind]}: tasks {e['tasks']} inputs {e['inputs']} certified {100 * e['own_zone']:.1f} right {100 * e['right']:.1f} "
             f"worst task right {100 * e['worst_right']:.0f} ({short(e['worst_task'])}) foreign {e['foreign_zone']}")
    write("edits", "\\begin{tabular}{lccccc}\n    \\toprule\n    Edit of the instruction & Tasks & Inputs & Certified, \\% & Right task, \\% & Wrong zone \\\\\n    \\midrule\n"
          + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    lines = []
    for kind in ("typo", "drop", "replace"):
        for k in ks:
            e = rows[f"{kind}-{k}"]
            lines.append(f"    {names[kind].capitalize() if k == 1 else ''} & {k} & {100 * e['own_zone']:.1f} & {100 * e['worst_own']:.0f} & {100 * e['right']:.1f} & "
                         f"{100 * e['worst_right']:.0f} ({short_tex(e['worst_task'])}) & {e['foreign_zone']} \\\\")
        lines.append("    \\midrule")
    for kind in ("lower", "polite", "no-first", "no-last", "shuffle"):
        e = rows[kind]
        lines.append(f"    {names[kind].capitalize()} & -- & {100 * e['own_zone']:.1f} & {100 * e['worst_own']:.0f} & {100 * e['right']:.1f} & "
                     f"{100 * e['worst_right']:.0f} ({short_tex(e['worst_task'])}) & {e['foreign_zone']} \\\\")
    write("edits_full", "\\begin{tabular}{lcccccc}\n    \\toprule\n    & & \\multicolumn{2}{c}{Certified, \\%} & \\multicolumn{2}{c}{Right task, \\%} & \\\\\n"
          "    \\cmidrule(lr){3-4} \\cmidrule(lr){5-6}\n    Edit & $k$ & all inputs & worst task & all inputs & worst task & Wrong zone \\\\\n    \\midrule\n"
          + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    # per task: opening length and right-task share for the strongest edits
    tasks = [t["task"] for t in d["bound"]["tasks"]]
    info = {t["task"]: t for t in d["bound"]["tasks"]}
    per = lambda name, key: {t["task"]: 100 * t[key] / t["inputs"] for t in rows[name]["per_task"]}
    cols = [("typo-1", "own"), ("typo-4", "own"), ("typo-4", "right"), ("typo-8", "right"), ("typo-16", "right"), ("drop-16", "right"), ("replace-16", "right")]
    vals = [per(n, k) for n, k in cols]
    lines = [f"    {short_tex(t)} & {info[t]['opening']} & {100 * info[t]['share']:.1f} & " + " & ".join(f"{v[t]:.0f}" for v in vals) + " \\\\"
             for t in sorted(tasks, key=lambda t: info[t]["opening"])]
    write("edits_pertask", "\\begin{tabular}{lccccccccc}\n    \\toprule\n    & & & \\multicolumn{2}{c}{Certified, \\%} & \\multicolumn{5}{c}{Right task, \\%} \\\\\n"
          "    \\cmidrule(lr){4-5} \\cmidrule(lr){6-10}\n    Task & $n_k$ & $100\\,R_k/D$ & 1 typo & 4 typos & 4 typos & 8 typos & 16 typos & 16 drop. & 16 repl. \\\\\n"
          "    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    b = d["bound"]
    tokens = [t["opening"] * t["share"] for t in b["tasks"]]             # tokens of average weight that may be replaced
    num("BoundTokensLow", WORDS[int(min(tokens))])
    num("BoundTokensHigh", WORDS[int(max(tokens))])
    fact(f"[edits] bound in tokens of average weight (opening length x R_k/D): {min(tokens):.2f} to {max(tokens):.2f}")
    fact(f"[edits] bound: D = {b['diameter']:.1f} (largest norm {b['largest_norm']:.1f}, median norm {b['median_norm']:.1f}); R_k/D from "
         f"{min(t['share'] for t in b['tasks']):.3f} to {max(t['share'] for t in b['tasks']):.3f}")
    path = os.path.join(HERE, "routing", "bound_check.json")
    if os.path.exists(path):
        c = json.load(open(path))
        assert c["held"]
        num("BoundInputs", int(c["inputs"]))
        num("BoundWorstPct", 100 * c["largest_share_of_the_bound"], 0)
        num("BoundMedianPct", 100 * c["median_share_of_the_bound"], 0)
        num("BoundInside", int(c["inside"]))
        num("BoundGuaranteed", int(c["guaranteed"]))
        fact(f"[edits] numerical check of the bound: {c['inputs']} edited inputs, held for all; moved at most {100 * c['largest_share_of_the_bound']:.0f}% of the bound "
             f"(median {100 * c['median_share_of_the_bound']:.0f}%); inside their zone {c['inside']}, guaranteed by the bound {c['guaranteed']}")
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    style = {"typo": ("typos", "o"), "drop": ("dropped words", "s"), "replace": ("replaced words", "^")}
    for ax, key, letter, title in ((axes[0], "own_zone", "a", "still certified"), (axes[1], "right", "b", "given to the right task")):
        ends = []
        for kind, (name, m) in style.items():
            ys = [100 * rows[f"{kind}-{k}"][key] for k in ks]
            ax.plot(range(5), ys, color=fs.EDITS[name], marker=m, ms=4.3, lw=2.0 if kind == "typo" else 1.5, clip_on=False, zorder=3)
            ends.append(ys[-1])
        if ax is axes[0]:
            for name, ty in zip([n for n, _ in style.values()], fs.spread(ends, 8.5, lo=3, hi=97)):
                ax.annotate(name, xy=(4, ends[[n for n, _ in style.values()].index(name)]), xytext=(4.55, ty), ha="left", va="center", fontsize=7.5,
                            color=fs.INK, annotation_clip=False, arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.5, shrinkA=1, shrinkB=3))
            ax.set_xlim(-0.15, 6.2)
            ax.spines["bottom"].set_bounds(-0.15, 4.15)
        else:
            ax.set_xlim(-0.15, 4.15)
        ax.set_xticks(range(5))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("edited words in the instruction")
        ax.tick_params(axis="x", length=2.5)
        fs.panel_title(ax, letter, title)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("share of test inputs, %")
    fs.note(axes[1], "still the right task,\neven when no longer certified", xy=(0, 0), xytext=(0.0, 30), ha="left", arrow=False)
    fig.subplots_adjust(wspace=0.1)
    save(fig, "edits")


def background_outputs():
    """How much the routes depend on the unrelated inputs (from background_analysis.py)."""
    path = os.path.join(HERE, "routing", "background.json")
    if not os.path.exists(path):
        return
    rows = json.load(open(path))
    label = {"as in the paper (seed 0, 4,000)": "As in the paper", "another draw (seed 1)": "Second random draw",
             "another draw (seed 2)": "Third random draw", "500 unrelated inputs": "500 inputs", "1,000 unrelated inputs": "1{,}000 inputs",
             "2,000 unrelated inputs": "2{,}000 inputs", "first five unrelated tasks": "First five tasks only",
             "last five unrelated tasks": "Last five tasks only"}
    lines, heavy = [], []
    for r in rows:
        f5, f15 = r["five"], r["fifteen"]
        wrong = sum(v["wrong"] for v in list(f5.values()) + list(f15.values()))
        blocked = f5["original"]["blocked"] + f15["original"]["blocked"]
        heavy += [100 * f5["heavy"]["certified"], 100 * f15["heavy"]["certified"]]
        lines.append(f"    {label[r['name']]} & " + "/".join(f"{100 * f5[l]['zone']:.0f}" for l in LEVELS) + " & "
                     + "/".join(("100" if f5[l]["certified"] == 1 else f"{100 * f5[l]['certified']:.1f}") for l in LEVELS) + " & "
                     + f"{100 * f15['heavy']['certified']:.1f} & {wrong} & {blocked} & {r['fifteen_radius'][0]:.1f}--{r['fifteen_radius'][1]:.1f} \\\\")
        assert f15["original"]["zone"] == 1 and f15["light"]["zone"] == 1
        fact(f"[background] {r['name']}: five zone " + "/".join(f"{100 * f5[l]['zone']:.0f}" for l in LEVELS) + " right "
             + "/".join(f"{100 * f5[l]['certified']:.1f}" for l in LEVELS) + f"; fifteen zone(orig) {100 * f15['original']['zone']:.0f} right "
             + "/".join(f"{100 * f15[l]['certified']:.1f}" for l in LEVELS) + f"; wrong zone {wrong}; blocked {blocked}; radii {r['fifteen_radius'][0]:.1f}-{r['fifteen_radius'][1]:.1f}")
        if r is rows[0]:
            lines.append("    \\midrule")
    fact(f"[background] right task under heavy rewording over all variants: {min(heavy):.1f} to {max(heavy):.1f}")
    num("BackgroundHeavyLow", min(heavy), 1)
    num("BackgroundHeavyHigh", max(heavy), 1)
    write("background", "\\begin{tabular}{lcccccc}\n    \\toprule\n    & \\multicolumn{2}{c}{Five tasks} & Fifteen tasks & & & \\\\\n"
          "    \\cmidrule(lr){2-3} \\cmidrule(lr){4-4}\n    Unrelated inputs & certified, \\% & right task, \\% & right task, heavy, \\% & Wrong & Blocked & Radii \\\\\n"
          "    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")


def many_outputs():
    """The zones on many real tasks (from many_tasks_analysis.py)."""
    path = os.path.join(HERE, "routing", "many_tasks.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    K = str(d["tasks"])
    main, numeric = d["random order, seed 0"], d["order of the task numbers"]
    seeds = [d[f"random order, seed {i}"] for i in range(5)]
    pct = lambda a, b: 100 * a / b
    xpath = os.path.join(HERE, "routing", "exact_first.json")
    X = json.load(open(xpath)) if os.path.exists(xpath) else None
    xmain = X["many"]["random order, seed 0"] if X else None
    lines = []
    for N in sorted(main, key=int):
        r = main[N]
        if X:
            assert xmain[N]["certified"]["right"] == r["certified"]["right"] and xmain[N]["nearest"]["right"] == r["nearest"]["right"], "the two analyses disagree"
        lines.append(f"    {int(N):,} & {r['tasks_clean']} & {r['tasks_some_blocked']} & {r['tasks_all_blocked']} & {pct(r['certified']['right'], r['inputs']):.1f} & "
                     + (f"{pct(xmain[N]['exact']['right'], r['inputs']):.1f} & " if X else "-- & ")
                     + f"{pct(r['nearest']['right'], r['inputs']):.1f} & {r['certified']['changed']} / {xmain[N]['exact']['changed'] if X else '--'} / {r['nearest']['changed']} \\\\".replace(",", "{,}"))
        fact(f"[many] first {N} tasks (main order): clean {r['tasks_clean']} some {r['tasks_some_blocked']} all {r['tasks_all_blocked']}; inputs {r['inputs']} blocked "
             f"{r['inputs_blocked']}; right certified {pct(r['certified']['right'], r['inputs']):.1f}% nearest {pct(r['nearest']['right'], r['inputs']):.1f}%; "
             f"changed certified {r['certified']['changed']} nearest {r['nearest']['changed']}; zero radius {r['radius_zero']}; median radius {r['radius_median']:.1f}")
    write("many", "\\begin{tabular}{rccccccc}\n    \\toprule\n    & \\multicolumn{3}{c}{Tasks with blocked inputs} & \\multicolumn{3}{c}{Inputs to their own task, \\%} & "
          "Routes \\\\\n    \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
          "    Tasks & none & some & all & \\cert{} & \\exactfirst{} & \\nearest{} & changed later \\\\\n    \\midrule\n"
          + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    r = main[K]
    c = r["collisions"]
    same = sum(x["same_definition"] for x in c)
    high = sum((not x["same_definition"]) and x["alike"] >= 0.8 for x in c)
    mid = sum(0.5 <= x["alike"] < 0.8 for x in c)
    low = sum(x["alike"] < 0.5 for x in c)
    fact(f"[many] colliding tasks {len(c)} of {K}: same text {same}, 80% or more common words {high}, 50% to 80% {mid}, under 50% {low}; lowest {100 * min(x['alike'] for x in c):.0f}%; "
         f"mmmlu {sum('mmmlu' in x['task'] for x in c)}, atomic {sum('atomic' in x['task'] for x in c)}; same-definition pairs in the collection {d['same_definition_pairs']}; "
         f"openings {d['opening_tokens']}")
    for name, value in (("Tasks", d["tasks"]), ("Inputs", d["inputs"]), ("Clean", r["tasks_clean"]), ("SomeHeld", r["tasks_some_blocked"]), ("AllHeld", r["tasks_all_blocked"]),
                        ("Colliding", len(c)), ("SameText", same), ("AlikeHigh", high), ("AlikeMid", mid), ("AlikeLow", low),
                        ("FamilyMmlu", sum("mmmlu" in x["task"] for x in c)), ("FamilyAtomic", sum("atomic" in x["task"] for x in c)),
                        ("InputsHeld", r["inputs_blocked"]), ("ZeroRadius", r["radius_zero"]), ("SamePairs", d["same_definition_pairs"]),
                        ("OpeningMin", d["opening_tokens"]["min"]), ("OpeningMedian", d["opening_tokens"]["median"]), ("OpeningMax", d["opening_tokens"]["max"]),
                        ("NumericClean", numeric[K]["tasks_clean"]), ("NumericAllHeld", numeric[K]["tasks_all_blocked"]),
                        ("CleanLow", min(x[K]["tasks_clean"] for x in seeds)), ("CleanHigh", max(x[K]["tasks_clean"] for x in seeds))):
        num("Many" + name, int(value))
    for name, value in (("CleanPct", pct(r["tasks_clean"], d["tasks"])), ("CollidingPct", pct(len(c), d["tasks"])), ("CertRightPct", pct(r["certified"]["right"], r["inputs"])),
                        ("NearRightPct", pct(r["nearest"]["right"], r["inputs"])), ("AlikeLowest", 100 * min(x["alike"] for x in c)),
                        ("NumericCleanPct", pct(numeric[K]["tasks_clean"], d["tasks"])), ("NumericCertRightPct", pct(numeric[K]["certified"]["right"], numeric[K]["inputs"])),
                        ("CertRightPctLow", min(pct(x[K]["certified"]["right"], x[K]["inputs"]) for x in seeds)),
                        ("CertRightPctHigh", max(pct(x[K]["certified"]["right"], x[K]["inputs"]) for x in seeds))):
        num("Many" + name, value, 1 if name != "AlikeLowest" else 0)
    others = [100 - pct(x[K]["tasks_clean"], d["tasks"]) for x in seeds[1:]]
    num("ManyCollidingPctLow", min(others), 1)
    num("ManyCollidingPctHigh", max(others), 1)
    num("ManyNumericCollidingPct", 100 - pct(numeric[K]["tasks_clean"], d["tasks"]), 1)
    if X:
        xr = xmain[K]
        num("ExactRightPct", pct(xr["exact"]["right"], xr["inputs"]), 1)
        num("ExactTasksOwn", int(xr["exact"]["tasks_own_all"]))
        num("ExactTasksNone", int(xr["exact"]["tasks_own_none"]))
        num("ExactChanged", int(xr["exact"]["changed"]))
        num("ExactSameFive", int(X["five_same_routes"]))
        num("ExactSameFifteen", int(X["fifteen_same_routes"]))
        num("ExactStressOwn", int(X["stress"]["light"]["exact"]["new_task_own"]))
        num("CertStressOwn", int(X["stress"]["light"]["certified"]["new_task_own"]))
        num("StressTrain", int(X["stress"]["light"]["exact"]["new_task_inputs"]))
        assert X["five_same_routes"] == X["five_inputs"] and X["fifteen_same_routes"] == X["fifteen_inputs"]
        assert len({json.dumps(X["many"][f"random order, seed {i}"][K]["exact"], sort_keys=True) for i in range(5)}) == 1, "the variant should not depend on the order"
        for look in ("light", "medium", "heavy"):
            fact(f"[exact-first] stress, {look} look-alike: routes changed exact " + "/".join(map(str, X["stress"][look]["exact"]["changed"])) + " certified "
                 + "/".join(map(str, X["stress"][look]["certified"]["changed"])) + " nearest " + "/".join(map(str, X["stress"][look]["nearest"]["changed"]))
                 + f"; new task's own inputs: exact {X['stress'][look]['exact']['new_task_own']} certified {X['stress'][look]['certified']['new_task_own']} of {X['stress'][look]['exact']['new_task_inputs']}")
        fact(f"[exact-first] 875 tasks: right {pct(xr['exact']['right'], xr['inputs']):.1f}% ({xr['exact']['right']} of {xr['inputs']}); tasks with all own inputs {xr['exact']['tasks_own_all']}, "
             f"none {xr['exact']['tasks_own_none']}; changed later {xr['exact']['changed']}; same routes as certified on five {X['five_same_routes']}/{X['five_inputs']}, fifteen "
             f"{X['fifteen_same_routes']}/{X['fifteen_inputs']}")
    # the other orders
    lines = []
    for name in [f"random order, seed {i}" for i in range(5)] + ["order of the task numbers"]:
        x = d[name][K]
        lines.append(f"    {name.replace('random order, seed 0', 'Random, seed 0 (main)').replace('random order, seed', 'Random, seed').replace('order of the task numbers', 'By task number')} & {x['tasks_clean']} & {x['tasks_some_blocked']} & {x['tasks_all_blocked']} & "
                     f"{pct(x['certified']['right'], x['inputs']):.1f} & {pct(x['nearest']['right'], x['inputs']):.1f} & {x['certified']['changed']} & {x['nearest']['changed']} \\\\")
        fact(f"[many] {name}: clean {x['tasks_clean']} some {x['tasks_some_blocked']} all {x['tasks_all_blocked']}; right certified {pct(x['certified']['right'], x['inputs']):.1f}% "
             f"nearest {pct(x['nearest']['right'], x['inputs']):.1f}%; changed {x['certified']['changed']}/{x['nearest']['changed']}")
    write("many_orders", "\\begin{tabular}{lccccccc}\n    \\toprule\n    & \\multicolumn{3}{c}{Tasks with blocked inputs} & \\multicolumn{2}{c}{To own task, \\%} & "
          "\\multicolumn{2}{c}{Changed later} \\\\\n    \\cmidrule(lr){2-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}\n"
          "    Order of arrival & none & some & all & \\cert{} & \\nearest{} & \\cert{} & \\nearest{} \\\\\n    \\midrule\n"
          + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    # other radius factors (added after the run)
    lines = []
    for kappa, x in sorted(d["other_radius_factors"].items(), key=lambda kv: float(kv[0])):
        lines.append(f"    {kappa} & {x['tasks_clean']} & {pct(x['tasks_clean'], d['tasks']):.1f} & {x['tasks_all_blocked']} & {x['inputs_blocked']} & "
                     f"{pct(x['certified']['right'], x['inputs']):.1f} \\\\")
        word = {"0.1": "One", "0.2": "Two", "0.3": "Three", "0.4": "Four", "0.5": "Five"}[kappa]
        num(f"ManyCleanKappa{word}", int(x["tasks_clean"]))
        num(f"ManyCertRightPctKappa{word}", pct(x["certified"]["right"], x["inputs"]), 1)
        fact(f"[many] radius factor {kappa}: clean {x['tasks_clean']} ({pct(x['tasks_clean'], d['tasks']):.1f}%) all held {x['tasks_all_blocked']} inputs held {x['inputs_blocked']} "
             f"right certified {pct(x['certified']['right'], x['inputs']):.1f}%")
    write("many_kappa", "\\begin{tabular}{cccccc}\n    \\toprule\n    $\\kappa$ & Tasks with no collision & in \\% & Tasks with all inputs blocked & Inputs blocked & "
          "To their own task, \\% \\\\\n    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    # examples of collisions
    tex_name = lambda n: "\\texttt{" + (n if len(n) <= 30 else n[:29] + "\\ldots").replace("_", "\\_") + "}"
    least = sorted(c, key=lambda x: x["alike"])[:6]
    family = [x for x in c if "mmmlu" in x["task"]][:2] + [x for x in c if x["same_definition"]][:2]
    lines = [f"    {tex_name(x['task'])} & {tex_name(x['older'])} & {100 * x['alike']:.0f} & {x['blocked']} of {x['inputs']} \\\\" for x in least + family]
    lines.insert(len(least), "    \\midrule")
    write("many_examples", "\\begin{tabular}{llcc}\n    \\toprule\n    Task that arrives & Older task it collides with & Common words, \\% & Blocked \\\\\n    \\midrule\n"
          + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw={"width_ratios": [1, 1]})
    Ns = sorted(main, key=int)
    xs = [int(n) for n in Ns]
    ax = axes[0]
    xseeds = [X["many"][f"random order, seed {i}"] for i in range(5)] if X else None
    ends = []
    for key, name in (("certified", "ClaimLoRA (ours)"), ("exact", "Exact claims first (variant)"), ("nearest", "Nearest zone")):
        if key == "exact" and not X:
            continue
        src = xseeds if key == "exact" else seeds
        vals = [[pct(x[n][key]["right"], x[n]["inputs"]) for x in src] for n in Ns]
        col = fs.color(name)
        ax.fill_between(xs, [min(v) for v in vals], [max(v) for v in vals], color=col, alpha=0.15, linewidth=0)
        ax.plot(xs, [mean(v) for v in vals], color=col, marker=fs.marker(name), ms=4.2, lw=2.0 if key == "certified" else 1.3, clip_on=False, zorder=3)
        ends.append((name, col, fs.marker(name), mean(v[-1] for v in vals) if False else mean(vals[-1]), key == "certified"))
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=c_, lw=2.0 if bg else 1.3, marker=mk, ms=4.2) for (n_, c_, mk, y0, bg) in ends],
              [n_.replace(" (ours)", " (ours)").replace(" (variant)", "") for (n_, *_r) in ends], loc="lower left", frameon=False,
              handlelength=1.6, fontsize=7.5, labelspacing=0.35)
    fs.panel_title(ax, "a", "inputs given to their own task")
    ax.set_ylabel("share, %")
    ax.set_ylim(60, 100.5)
    ax = axes[1]
    vals = [[pct(x[n]["tasks_clean"], int(n)) for x in seeds] for n in Ns]
    col = fs.color("ClaimLoRA (ours)")
    ax.fill_between(xs, [min(v) for v in vals], [max(v) for v in vals], color=col, alpha=0.15, linewidth=0)
    ax.plot(xs, [mean(v) for v in vals], color=col, marker="o", ms=4.2, lw=2.0, clip_on=False, zorder=3)
    fs.panel_title(ax, "b", "tasks arriving without a collision")
    ax.set_ylim(60, 100.5)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlabel("tasks that have arrived")
        ax.tick_params(axis="x", length=2.5)
    for ax in axes:
        ax.set_xlim(xs[0] / 1.15, xs[-1] * 1.15)
    fig.subplots_adjust(wspace=0.2)
    save(fig, "many")


def demo_table():
    files = sorted(f for f in glob.glob(os.path.join(ROOT, "certified_switch", "demo_results", "*.json")) if not os.path.basename(f).startswith(("sibling_", "no_instruction_")))   # the no-instruction demo files have another format and are read by audit_results.py
    lines = []
    for f in files:
        d = json.load(open(f))
        rows = {(r["wording"], r["rule"]): r for r in d["table"]}
        c, a = rows["original", "certified"], rows["original", "all"]
        h = rows["heavy", "certified"]
        kind = {"encoder-decoder": "enc.-dec.", "decoder-only": "dec.-only"}.get(d["kind"], d["kind"])
        lines.append(f"    \\texttt{{{d['model'].split('/')[-1]}}} & {kind} & {c['learned']:.1f} & {c['AP']:.1f} & {c['changed']} of {c['compared']} & "
                     f"{a['AP']:.1f} & {a['changed']} of {a['compared']} & {c['certified']} of {c['inputs']} & {h['AP']:.1f} & {h['changed']} of {h['compared']} \\\\")
        key = {"t5-small": "Tfive", "gpt2": "Gpt", "SmolLM2-135M": "Smol"}.get(d["model"].split("/")[-1])
        if key:
            num(f"Demo{key}Learned", c["learned"], 1)
            num(f"Demo{key}AP", c["AP"], 1)
            num(f"Demo{key}Changed", int(c["changed"]))
            num(f"Demo{key}Compared", int(c["compared"]))
            num(f"Demo{key}PlainAP", a["AP"], 1)
            num(f"Demo{key}PlainChanged", int(a["changed"]))
            num(f"Demo{key}HeavyAP", h["AP"], 1)
            num(f"Demo{key}HeavyChanged", int(h["changed"]))
            num(f"Demo{key}HeavyRight", 100 * h["right_task"], 1)
            num(f"Demo{key}Certified", int(c["certified"]))
            num(f"Demo{key}Inputs", int(c["inputs"]))
            num(f"Demo{key}Minutes", round(d["minutes"]))
        fact(f"[demo] {d['model']} ({d['kind']}): certified learned {c['learned']:.1f} AP {c['AP']:.1f} changed {c['changed']}/{c['compared']} | plain AP {a['AP']:.1f} "
             f"changed {a['changed']}/{a['compared']} | certified inputs {c['certified']}/{c['inputs']} | heavy: AP {h['AP']:.1f} changed {h['changed']}/{h['compared']} "
             f"right {h['right_task']:.3f} | {d['minutes']:.0f} min")
    if lines:
        write("demo", "\\begin{tabular}{llcccccccc}\n    \\toprule\n    & & \\multicolumn{3}{c}{\\cert{}, original} & \\multicolumn{2}{c}{\\plain{}, original} & & "
              "\\multicolumn{2}{c}{\\cert{}, heavy} \\\\\n    \\cmidrule(lr){3-5} \\cmidrule(lr){6-7} \\cmidrule(lr){9-10}\n"
              "    Model & Kind & Learned & AP & Changed & AP & Changed & Certified & AP & Changed \\\\\n    \\midrule\n" + "\n".join(lines) + "\n    \\bottomrule\n\\end{tabular}\n")


def sibling_outputs():
    """The sibling demo of the stand-alone code (certified_switch/demo_sibling.py)."""
    path = os.path.join(ROOT, "certified_switch", "demo_results", "sibling_t5-small_seed0.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    c, x = d["rules"]["certified"], d["rules"]["exact_first"]
    num("SiblingTrain", int(c["train"]))
    num("SiblingTest", int(c["sibling_inputs"]))
    num("SiblingCertBlocked", int(c["blocked"]))
    num("SiblingCertRight", int(c["sibling_right"]))
    num("SiblingCertOldWords", int(c["sibling_answered_with_old_words"]))
    num("SiblingExactRight", int(x["sibling_right"]))
    num("SiblingOldInputs", int(x["changed"]["original"]["inputs"]))
    num("SiblingExactChangedLight", int(x["changed"]["light"]["changed"]))
    num("SiblingExactChangedLightTaskTwo", int(x["changed"]["light"]["task2_changed"]))
    for rule, r in (("certified", c), ("exact_first", x)):
        fact(f"[sibling] {rule}: blocked {r['blocked']}/{r['train']}; sibling right {r['sibling_right']}/{r['sibling_inputs']} (old words {r['sibling_answered_with_old_words']}); "
             "old answers changed " + "; ".join(f"{lvl} {v['changed']}/{v['inputs']} (task 2: {v['task2_changed']}/{v['task2_inputs']})" for lvl, v in r["changed"].items()))
    assert all(v["changed"] == 0 for v in c["changed"].values()), "the certified rule changed an old answer"
    assert all(v["changed"] == 0 for lvl, v in x["changed"].items() if lvl != "light"), "the variant changed an answer it should keep"
    assert x["changed"]["light"]["changed"] == x["changed"]["light"]["task2_changed"]


def rewordings_table():
    d = json.load(open(os.path.join(ROOT, "certified_switch", "rewordings.json")))
    esc = lambda s: s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace(">", "\\textgreater{}").replace('"', "''")
    out = []
    for task, texts in d["tasks"].items():
        out.append(f"\\paragraph{{{SHORT[task]} (\\texttt{{{task}}}).}}")
        out.append("\\begin{itemize}\n" + "\n".join(f"  \\item \\emph{{{lvl}:}} {esc(t)}" for lvl, t in zip(d["levels"], texts)) + "\n\\end{itemize}")
    write("rewordings", "\n".join(out) + "\n")


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    figure_concept()
    rewordings_table()
    for tag, full, label in ((FIVE, 5, "five"), (FIFTEEN, 15, "fifteen")):
        if sequence_tables(tag, full, label):
            figure_wordings(tag, full, label)
            figure_tracks(tag, full, label) if full == 5 else figure_dumbbell(tag, full, label)
    figure_learn_forget([(FIVE, 5, "(a) five tasks"), (FIFTEEN, 15, "(b) fifteen tasks")])
    for key, old, new in REPEATS:
        same = old["official"] == new["official"]
        fact(f"[repeat] {key} was run twice ({old['source']} and {new['source']}): official scores identical: {same}")
        rule = "certified" if key[1] == "certified2" else "official"
        M = {"certified2": "Cert", "inflora": "Inf", "gainlora_inflora": "Gain"}[key[1]]
        a, b = metrics(old, "original", rule), metrics(new, "original", rule)
        pa = {(r["dataset"], r["id"]): r[rule]["pred"] for r in old["steps"][-1]["levels"]["original"]}
        pb = {(r["dataset"], r["id"]): r[rule]["pred"] for r in new["steps"][-1]["levels"]["original"]}
        equal = sum(pa[k] == pb[k] for k in pa)
        exact_tasks = 0
        for j, task in enumerate(old["tasks"]):                     # how many tasks, from the first on, were reproduced answer for answer
            x = {r["id"]: r[rule]["pred"] for r in old["steps"][j]["levels"]["original"] if r["dataset"] == task}
            y = {r["id"]: r[rule]["pred"] for r in new["steps"][j]["levels"]["original"] if r["dataset"] == task}
            if x != y:
                break
            exact_tasks += 1
        fact(f"[repeat] {key}: re-test learned {a['learned']:.2f} vs {b['learned']:.2f}, AP {a['AP']:.2f} vs {b['AP']:.2f}, changed {a['changed']} vs {b['changed']}; "
             f"answers at the end equal in both runs {equal} of {len(pa)}; first tasks reproduced answer for answer: {exact_tasks}")
        if key[0] == FIVE:
            num(f"Repeat{M}First", a["AP"], 2)
            num(f"Repeat{M}Second", b["AP"], 2)
            num(f"Repeat{M}Same", equal)
            num(f"Repeat{M}Total", len(pa))
            num(f"Repeat{M}ChangedFirst", a["changed"])
            num(f"Repeat{M}ChangedSecond", b["changed"])
            num(f"Repeat{M}ExactTasks", WORDS[exact_tasks])
    routing_tables()
    geometry_outputs()
    orders_outputs()
    edits_outputs()
    background_outputs()
    many_outputs()
    demo_table()
    sibling_outputs()
    open(os.path.join(HERE, "facts.txt"), "w").write("\n".join(FACTS) + "\n")
    write("numbers", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(NUMBERS.items())))
