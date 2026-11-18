"""Four designed figures, real data only (run from paper_certified:  python3 make_figures.py).

    answer_grid    every square is one test answer (five tasks, no definitions, 100 epochs)
    stage_heatmap  what each task scores after each later task (same runs)
    route_matrix   which adapter each test input is sent to (fifteen tasks, no definitions)
    learn_keep     learned against kept (five tasks, without and with definitions)

Sources are read as audit_results.py reads them (audit_results.load_sources) and, for the per-stage and
per-example records, straight from the zip archives (extra_eval.json of every stage). Every number that
is drawn and that also exists as a macro in tables/numbers_audit.tex is compared with it; a wrong
reading stops the script.
"""
import json
import os
import re
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import audit_results as ar
import figstyle as fs

fs.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
S = ar.load_sources()
P = ar.PATHS

# ------------------------------------------------------------------ audited macros
MAC = {}
for line in open(os.path.join(HERE, "tables", "numbers_audit.tex"), encoding="utf-8"):
    m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line)
    if m:
        MAC[m.group(1)] = m.group(2)


def macro(name):
    return float(MAC[name].replace("{,}", "").replace("$-$", "-"))


def check(name, value, nd=2):
    """value (a float or int) must print like the macro"""
    want = MAC[name].replace("{,}", "").replace("$-$", "-")
    got = ("%d" % value) if nd == 0 else ("%.*f" % (nd, value))
    assert got == want, "%s: macro %s, data %s" % (name, want, got)


METHODS = ["InfLoRA", "GainLoRA", "FinderLoRA", "KeeperLoRA"]
SHORT5 = ["relation extr.", "sst2", "glucose", "sentiment140", "emotion"]
PUB = {"InfLoRA": "inflora", "GainLoRA": "gainlora_inflora"}
MAC_PUB = {"InfLoRA": "Inflora", "GainLoRA": "Gainlora"}
RULE = {"FinderLoRA": "table:ridge", "KeeperLoRA": "table:handicap"}      # ridge = Finder; handicap (nu = 2) = Keeper
MAC_OURS = {"FinderLoRA": "Ridge", "KeeperLoRA": "Handicap"}


def stage_records(zpath, sub, T):
    """extra_eval.json of every stage (records of every test input seen so far), in file order"""
    out = []
    with zipfile.ZipFile(zpath) as z:
        for i in range(1, T + 1):
            n = [x for x in z.namelist() if sub in x and ("/outputs/%d-" % i) in x and x.endswith("extra_eval.json")]
            assert len(n) == 1, (sub, i, n)
            out.append(json.load(z.open(n[0]))["levels"]["original"])
    return out


# ------------------------------------------------------------------ five tasks, 100 epochs, no definitions
REC = {}                      # method -> per-stage records
KEY = {}                      # method -> rule key inside a record
for mth in METHODS:
    if mth in PUB:
        REC[mth] = stage_records(P["full_zip"], "_e100_%s_seed42" % PUB[mth], 5)
        KEY[mth] = "official"
    else:
        REC[mth] = stage_records(P["fullours_zip"], "certified2_seed42", 5)
        KEY[mth] = RULE[mth]

DATASETS = []
for r in REC["InfLoRA"][-1]:
    if r["dataset"] not in DATASETS:
        DATASETS.append(r["dataset"])
assert len(DATASETS) == 5
for a, b in zip(DATASETS, ["1510", "363", "748", "1687", "875"]):
    assert a.startswith("task" + b), (a, b)


def retest_row(mth):
    if mth in PUB:
        return S["full"]["by_seed"]["42"]["retest"][PUB[mth]]["table"]["original|official"]
    return S["fullours"]["by_seed"]["42"]["retest"]["certified2"]["table"]["original|" + RULE[mth]]


# matrix of scores: M[i][j] = Rouge-L of task j after stage i (NaN before the task exists)
MAT = {}
for mth in METHODS:
    M = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(i + 1):
            s = [r[KEY[mth]]["score"] for r in REC[mth][i] if r["dataset"] == DATASETS[j]]
            assert len(s) == 100
            M[i, j] = sum(s) / len(s)
    MAT[mth] = M
    row = retest_row(mth)
    assert abs(np.mean(np.diag(M)) - row["learned"]) < 1e-9 and abs(np.mean(M[4]) - row["AP"]) < 1e-9, mth
    for j, t in enumerate(DATASETS):                     # the stored per-task numbers of the audit
        assert abs(M[j, j] - row["learned_scores"][t]) < 1e-9 and abs(M[4, j] - row["final_scores"][t]) < 1e-9

# answers changed: answer of task j right after learning it against the answer at the end (quick_test.retest_table)
FLAGS, CHANGED_N = {}, {}
ORDER = {}                    # (dataset) -> ids in dataset order
for j in range(4):
    ORDER[j] = [r["id"] for r in REC["InfLoRA"][j] if r["dataset"] == DATASETS[j]]
    assert len(ORDER[j]) == 100
for mth in METHODS:
    fl = []
    for j in range(4):
        first = {r["id"]: r[KEY[mth]]["pred"] for r in REC[mth][j] if r["dataset"] == DATASETS[j]}
        last = {r["id"]: r[KEY[mth]]["pred"] for r in REC[mth][4] if r["dataset"] == DATASETS[j]}
        ids = [r["id"] for r in REC[mth][j] if r["dataset"] == DATASETS[j]]
        assert ids == ORDER[j] and [r["id"] for r in REC[mth][4] if r["dataset"] == DATASETS[j]] == ORDER[j]   # same position in every panel
        fl.append(np.array([first[i] != last[i] for i in ids]))
    FLAGS[mth] = fl
    CHANGED_N[mth] = int(sum(f.sum() for f in fl))
    assert CHANGED_N[mth] == retest_row(mth)["changed"] and retest_row(mth)["compared"] == 400, mth

# the audited headline values
assert [CHANGED_N[m] for m in METHODS] == [382, 178, 6, 1], CHANGED_N
for mth in PUB:
    check("AudFull%sChanged" % MAC_PUB[mth], CHANGED_N[mth], 0)
for mth in RULE:
    check("AudFullOurs%sChanged" % MAC_OURS[mth], CHANGED_N[mth], 0)
for mth, (lrn, fin) in {"InfLoRA": (78.37, 25.37), "GainLoRA": (76.29, 56.41), "FinderLoRA": (80.78, 80.18), "KeeperLoRA": (80.78, 80.98)}.items():
    row = retest_row(mth)
    assert "%.2f" % row["learned"] == "%.2f" % lrn and "%.2f" % row["AP"] == "%.2f" % fin, (mth, row["learned"], row["AP"])
    pre = "AudFull" + MAC_PUB[mth] if mth in PUB else "AudFullOurs" + MAC_OURS[mth]
    check(pre + "Learned", row["learned"])
    check(pre + "AP", row["AP"])

# the official-test per-task numbers of tables/noinst_e100_pertask.tex, for the comparison in the report
PT = {"relation extr.": "Relation", "sst2": "Sst", "glucose": "Glucose", "sentiment140": "Sentiment", "emotion": "Emotion"}
PT_DIFF = []
for mth in PUB:
    for j, nm in enumerate(SHORT5):
        for suffix, val in (("Learned", MAT[mth][j, j]), ("Final", MAT[mth][4, j])):
            tab = macro("AudFull%sPt%s%s" % (MAC_PUB[mth], suffix, PT[nm]))
            if abs(tab - val) > 0.005:
                PT_DIFF.append((mth, nm, suffix, tab, round(val, 2)))

# ------------------------------------------------------------------ FIGURE 1  answer_grid
GAP_B, GAP_Y, SQ = 1.6, 3.0, 0.86
fig = plt.figure(figsize=(5.5, 2.55))
gs = fig.add_gridspec(1, 4, left=0.005, right=0.995, top=0.80, bottom=0.10, wspace=0.07)
side = 2 * 10 + GAP_B
sidey = 2 * 10 + GAP_Y
for k, mth in enumerate(METHODS):
    ax = fig.add_subplot(gs[0, k])
    ax.set_xlim(0, side)
    ax.set_ylim(sidey, -1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    patches, colors = [], []
    for b in range(4):
        bx, by = (b % 2) * (10 + GAP_B), (b // 2) * (10 + GAP_Y)
        for n, flag in enumerate(FLAGS[mth][b]):
            r, c = divmod(n, 10)
            patches.append(Rectangle((bx + c + (1 - SQ) / 2, by + r + (1 - SQ) / 2), SQ, SQ))
            colors.append(fs.CHANGED if flag else fs.KEPT)
        ax.text(bx, by - 0.45, SHORT5[b], fontsize=6.3, color=fs.INK2, ha="left", va="bottom")
    ax.add_collection(PatchCollection(patches, facecolors=colors, edgecolors="none"))
    ax.text(0.0, 1.22, mth, transform=ax.transAxes, fontsize=9, fontweight="bold", color=fs.INK, ha="left", va="bottom")
    ax.text(0.0, 1.13, "%d of 400 changed" % CHANGED_N[mth], transform=ax.transAxes, fontsize=7.5, color=fs.INK2, ha="left", va="bottom")
fig.patches.extend([])
lx = 0.005
for lab, col in (("answer kept", fs.KEPT), ("answer changed", fs.CHANGED)):
    fig.patches.append(Rectangle((lx, 0.035), 0.016, 0.016 * 5.5 / 2.55, transform=fig.transFigure, facecolor=col, edgecolor="none"))
    fig.text(lx + 0.022, 0.043, lab, fontsize=7.5, color=fs.INK2, ha="left", va="center")
    lx += 0.16
fig.text(0.995, 0.043, "one square = one test answer of the four older tasks, same place in every panel", fontsize=7, color=fs.INK2, ha="right", va="center")
fs.save(fig, "answer_grid")

# ------------------------------------------------------------------ FIGURE 2  stage_heatmap
fig = plt.figure(figsize=(5.5, 1.95))
gs = fig.add_gridspec(1, 5, left=0.135, right=0.935, top=0.80, bottom=0.215, wspace=0.12, width_ratios=[1, 1, 1, 1, 0.06])
for k, mth in enumerate(METHODS):
    ax = fig.add_subplot(gs[0, k])
    M = np.ma.masked_invalid(MAT[mth].T)      # rows = task, columns = after which task
    mesh = ax.pcolormesh(np.arange(6), np.arange(6), M, cmap=fs.SEQ, vmin=0, vmax=100, edgecolors=fs.SURFACE, linewidth=0.8)
    ax.set_xlim(0, 5)
    ax.set_ylim(5, 0)
    ax.set_aspect("equal")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(5):
        for j in range(5):
            if not np.isnan(MAT[mth][j, i]):
                v = MAT[mth][j, i]
                ax.text(j + 0.5, i + 0.5, "%d" % round(v), ha="center", va="center", fontsize=6.6, color=fs.ink_on(v / 100))
    ax.set_xticks(np.arange(5) + 0.5)
    ax.set_xticklabels([str(i) for i in range(1, 6)], fontsize=7)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks(np.arange(5) + 0.5)
    ax.set_yticklabels(SHORT5 if k == 0 else [], fontsize=7)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("after learning task", fontsize=7.5, labelpad=2)
    fs.panel_title(ax, "abcd"[k], mth)
cax = fig.add_subplot(gs[0, 4])
cb = fig.colorbar(mesh, cax=cax)
cb.set_ticks([0, 50, 100])
cb.ax.tick_params(labelsize=7, length=2)
cb.outline.set_visible(False)
cb.set_label("Rouge-L", fontsize=7.5)
fs.save(fig, "stage_heatmap")

# ------------------------------------------------------------------ FIGURE 3  route_matrix (fifteen tasks)
zf = P["fifteen_zip"]
REC15 = stage_records(zf, "certified2_seed42", 15)[-1]
TASKS15 = []
for r in REC15:
    if r["dataset"] not in TASKS15:
        TASKS15.append(r["dataset"])
assert len(TASKS15) == 15
NAME15 = [ar.NAMES15[re.match(r"task(\d+)_", t).group(1)] for t in TASKS15]
RT = json.load(zipfile.ZipFile(zf).open([n for n in zipfile.ZipFile(zf).namelist() if n.endswith("route_table.json")][0]))["routes"]


def route_matrix(rule):
    M = np.zeros((15, 15))
    n = np.zeros(15)
    for r in REC15:
        t = TASKS15.index(r["dataset"])
        M[t, r[rule]["route"] - 1] += 1
        n[t] += 1
    own = float(np.trace(M))
    return M / n[:, None] * 100, own, int(n.sum())


MC, own_c, tot = route_matrix("centre")
MR, own_r, tot2 = route_matrix("table:ridge")
assert tot == tot2 == 1197
# the stored route table (rule ridge, stage 15) must agree with the per-example records
cnt = {t: 0 for t in TASKS15}
for r in REC15:
    pass
for t in TASKS15:
    stored = [d["15"] for d in RT["ridge"][t]]
    mine = [r["table:ridge"]["route"] for r in REC15 if r["dataset"] == t]
    assert stored == mine, t
SHARE_C, SHARE_R = 100 * own_c / tot, 100 * own_r / tot
check("AudFifteenNoDefCentreRight", SHARE_C, 1)
check("AudFifteenNoDefRidgeRight", SHARE_R, 1)
t15 = S["fifteen"]["by_seed"]["42"]["retest"]["certified2"]["table"]
assert abs(own_r / tot - t15["original|table:ridge"]["right_task"]) < 1e-12 and abs(own_c / tot - t15["original|centre"]["right_task"]) < 1e-12

# headline of the fifteen tasks
check("AudFifteenNoDefRidgeChanged", t15["original|table:ridge"]["changed"], 0)
check("AudFifteenNoDefRidgeAP", t15["original|table:ridge"]["AP"])
assert t15["original|table:ridge"]["changed"] == 8 and t15["original|table:ridge"]["compared"] == 1097
assert "%.2f" % t15["original|table:ridge"]["AP"] == "42.16"

fig = plt.figure(figsize=(5.5, 3.25))
gs = fig.add_gridspec(1, 3, left=0.145, right=0.955, top=0.885, bottom=0.145, wspace=0.07, width_ratios=[1, 1, 0.04])
for k, (M, rule, share) in enumerate(((MC, "Nearest centre", SHARE_C), (MR, "FinderLoRA", SHARE_R))):
    ax = fig.add_subplot(gs[0, k])
    mesh = ax.pcolormesh(np.arange(16), np.arange(16), np.where(M > 0, M, np.nan), cmap=fs.SEQ, vmin=0, vmax=100,
                         edgecolors=fs.SURFACE, linewidth=0.5)
    # empty cells (no input) are drawn in the empty colour so the grid stays visible
    ax.set_xlim(0, 15)
    ax.set_ylim(15, 0)
    ax.set_aspect("equal")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(15):
        for j in range(15):
            if M[i, j] >= 1:
                ax.text(j + 0.5, i + 0.5, "%d" % round(M[i, j]), ha="center", va="center", fontsize=5.0, color=fs.ink_on(M[i, j] / 100))
    ax.set_xticks(np.arange(15) + 0.5)
    ax.set_xticklabels([str(i) for i in range(1, 16)], fontsize=5.8)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_yticks(np.arange(15) + 0.5)
    ax.set_yticklabels(NAME15 if k == 0 else [], fontsize=6.6)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("route chosen (the model as it was after this task)", fontsize=7.5, labelpad=2)
    fs.panel_title(ax, "ab"[k], rule)
    ax.text(1.0, 1.03, "%.1f %% to own task" % share, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=fs.INK2)
    if k == 0:
        ax.set_ylabel("task the test input belongs to", fontsize=7.5, labelpad=2)
cax = fig.add_subplot(gs[0, 2])
cb = fig.colorbar(mesh, cax=cax)
cb.set_ticks([0, 50, 100])
cb.ax.tick_params(labelsize=6.5, length=2)
cb.outline.set_visible(False)
cb.set_label("share of the task's inputs, %", fontsize=7)
fs.save(fig, "route_matrix")
# the mistakes that remain for FinderLoRA, for the report
OFF = [(NAME15[i], int(j + 1), round(MR[i, j], 1)) for i in range(15) for j in range(15) if i != j and MR[i, j] > 0]
OFF_C = sum(1 for i in range(15) for j in range(15) if i != j and MC[i, j] >= 1)

# ------------------------------------------------------------------ FIGURE 4  learn_keep
def mean(xs):
    return sum(xs) / len(xs)


def pt(rows):
    return mean([r["learned"] for r in rows]), mean([r["AP"] for r in rows])


NOD = {}                      # (a) without definitions: method -> (10 epochs, 100 epochs) as (learned, final)
sds = ar.seeds_complete(S)
assert sds == ["42", "43", "44"], sds
for mth in METHODS:
    if mth in PUB:
        r10 = pt(ar.seed_rows(S, "pub", PUB[mth]))
        check("AudSeeds%sLearned" % MAC_PUB[mth], r10[0])
        check("AudSeeds%sAP" % MAC_PUB[mth], r10[1])
    else:
        r10 = pt(ar.seed_rows(S, "ours", RULE[mth]))
        check("AudSeeds%sLearned" % MAC_OURS[mth], r10[0])
        check("AudSeeds%sAP" % MAC_OURS[mth], r10[1])
    row = retest_row(mth)
    NOD[mth] = (r10, (row["learned"], row["AP"]))

# (b) with definitions
seeds_u = sorted(S["runu_seeds"])
assert seeds_u == ["42", "43", "44"] and all(sd in S["defs3"] for sd in seeds_u)
DEF = {}                      # method -> (10 epochs or None, 100 epochs or None)
for mth in ("InfLoRA", "GainLoRA"):
    m = PUB[mth]
    r10 = pt([S["defs3"][sd]["retest"][m]["table"]["original|official"] for sd in seeds_u])
    check("AudUThreeOriginal%sAP" % MAC_PUB[mth], r10[1])
    r100row = S["fulldef"]["by_seed"]["42"]["retest"][m]["table"]["original|official"]
    check("AudFullDefOriginal%sLearned" % MAC_PUB[mth], r100row["learned"])
    check("AudFullDefOriginal%sAP" % MAC_PUB[mth], r100row["AP"])
    DEF[mth] = (r10, (r100row["learned"], r100row["AP"]))
for mth, rule, mn in (("ClaimLoRA", "certified", "Cert"), ("FinderLoRA", "table:ridge", "Ridge")):
    rows = [S["runu_seeds"][sd]["retest"]["certified2"]["table"]["original|" + rule] for sd in seeds_u]
    r10 = pt(rows)
    check("AudUThreeOriginal%sAP" % mn, r10[1])
    DEF[mth] = (r10, None)
# learned values of the run with definitions are the re-test's own "learned" field (same table the audit reads AP from)

fig = plt.figure(figsize=(5.5, 2.95))
gs = fig.add_gridspec(1, 2, left=0.075, right=0.995, top=0.90, bottom=0.145, wspace=0.16)
LO, HI = 20, 90
LABEL_AT = {          # label position in data coordinates (text is ink, a thin leader joins it to its point)
    "a": {"InfLoRA": (89, 32, "right"), "GainLoRA": (72, 46, "right")},
    "b": {"InfLoRA": (89, 68, "right"), "GainLoRA": (86, 60, "right")},
}
CLUSTER = {"a": ("FinderLoRA and KeeperLoRA:\non the line", (80.8, 80.6), (70, 85.0)),
           "b": ("ClaimLoRA and FinderLoRA\n(10 epochs)", (79.6, 79.5), (70, 85.0))}
for k, (letter, ttl, data) in enumerate((("a", "without definitions", NOD), ("b", "with definitions", DEF))):
    ax = fig.add_subplot(gs[0, k])
    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_aspect("equal")
    ax.grid(True, axis="both")
    ax.plot([LO, HI], [LO, HI], color=fs.MUTED, lw=0.7, zorder=1)
    ax.text(LO + 2, LO + 4, "nothing lost", fontsize=7, color=fs.INK2, rotation=45, rotation_mode="anchor", ha="left", va="bottom")
    for mth, (a10, a100) in data.items():
        col, mk = fs.color(mth), fs.marker(mth)
        if a10 and a100:
            ax.annotate("", xy=a100, xytext=a10, arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, mutation_scale=6, shrinkA=3, shrinkB=3), zorder=2)
        if a10:
            ax.plot(*a10, marker=mk, ms=6, mfc=fs.SURFACE, mec=col, mew=1.1, ls="none", zorder=3)
        if a100:
            ax.plot(*a100, marker=mk, ms=5, mfc=col, mec=fs.SURFACE, mew=0.5, ls="none", zorder=4)
        anchor = a100 if a100 else a10
        if mth in LABEL_AT[letter]:
            x, y, ha = LABEL_AT[letter][mth]
            fs.note(ax, mth, xy=anchor, xytext=(x, y), ha=ha, color=fs.INK)
    txt, xy, xyt = CLUSTER[letter]
    fs.note(ax, txt, xy=xy, xytext=xyt, ha="right", color=fs.INK)
    ax.text(0.72, 0.02, "below the line = forgotten", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=fs.INK2)
    ax.set_xticks(range(20, 91, 10))
    ax.set_yticks(range(20, 91, 10))
    ax.tick_params(axis="y", length=2.5)
    ax.set_xlabel("score right after learning (Rouge-L)", fontsize=7.8, labelpad=2)
    if k == 0:
        ax.set_ylabel("score after the last task (Rouge-L)", fontsize=7.8, labelpad=2)
    fs.panel_title(ax, letter, ttl)
handles = [Line2D([], [], marker="o", ls="none", ms=5, mfc=fs.SURFACE, mec=fs.INK2, mew=1.1, label="10 epochs (mean of 3 seeds)"),
           Line2D([], [], marker="o", ls="none", ms=5, mfc=fs.INK2, mec=fs.SURFACE, label="100 epochs (seed 42)")]
fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.54, -0.045), fontsize=7.5, handletextpad=0.3, columnspacing=1.6)
fs.save(fig, "learn_keep")

# ------------------------------------------------------------------ summary for the report
print("answers changed of 400:", CHANGED_N)
print("final / learned:", {m: (round(NOD[m][1][0], 2), round(NOD[m][1][1], 2)) for m in METHODS})
print("route share to own task: centre %.1f, FinderLoRA %.1f; FinderLoRA off-diagonal cells:" % (SHARE_C, SHARE_R), OFF)
print("centre off-diagonal cells >= 1 %%: %d" % OFF_C)
print("heat map vs tables/noinst_e100_pertask.tex (official test, one reference) differs in:", PT_DIFF)
print("learn_keep (a):", {m: [tuple(round(x, 2) for x in p) for p in v] for m, v in NOD.items()})
print("learn_keep (b):", {m: [tuple(round(x, 2) for x in p) if p else None for p in v] for m, v in DEF.items()})
print("all assertions passed")
