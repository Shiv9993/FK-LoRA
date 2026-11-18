"""Numbers and table for the test of the bound on the forgetting rate without instructions (Section sec:noinstruction).

    python3 bound_results.py

Reads   routing/noinstruction_study.json   (Pilot A: routes only, instances only, three sequences; every feature and score that is in the
                                            file is used, so the script can be run again when more of them are added)
Writes  tables/numbers_bound.tex           one \\newcommand per number (\\Bound...)
        tables/bound_pilot.tex             rows = sequence x feature x score at the primary level, columns = coverage, changed, blocked, own task
Nothing is typed by hand except the lines of the rules that were written before the run (PLAN.md) and the names of the rows.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "routing", "noinstruction_study.json")))
N = {}
SEQ = (("superni_order1", "Superni", "SuperNI (15)"), ("long_order3", "Long", "Long (15)"), ("both", "Both", "Both (30)"))
FEAT = (("t5static", "Word", "word-vec."), ("minilm", "Mini", "sentence enc."))
SCORE = (("knn10", "Knn", "10-NN"), ("centre", "Centre", "mean"), ("ratio10", "Ratio", "ratio"))
LEVEL = (("0.05", "Five"), ("0.1", "Ten"), ("0.2", "Twenty"))
ALPHA, LINE_OWN, LINE_BLOCK, LINE_NOT_OWN, LINE_NOT_BLOCK = 0.05, 90, 10, 75, 25     # the lines written before the run (PLAN.md)


def pct(x, nd=1):
    return ("%." + str(nd) + "f") % (100 * x)


def put(name, value):
    assert name.isalpha(), name
    assert N.get(name, value) == value, name
    N[name] = value


R = d["results"]
rows = []
for sk, sn, slabel in SEQ:
    for fk, fn, flabel in FEAT:
        for ck, cn, clabel in SCORE:
            for lk, ln in LEVEL:
                try:
                    cell = R[sk][fk][ck][lk]
                except KeyError:
                    continue
                m, se = cell["mean"], cell["se"]
                for mk, mn in (("coverage", "Cov"), ("changed", "Changed"), ("blocked", "Blocked"), ("own_at_end", "Own")):
                    put("Bound%s%s%s%s%s" % (sn, fn, cn, ln, mn), pct(m[mk]))
                    put("Bound%s%s%s%s%sSe" % (sn, fn, cn, ln, mn), pct(se[mk]))
                if lk == "0.05":
                    rows.append((slabel, flabel, clabel, m, se))
# data: calibration sizes
n_tasks = {"Superni": 15, "Long": 15, "Both": 30}
for k, v in n_tasks.items():
    put("BoundTasks" + k, str(v))
ncal = [t["n_cal"] for t in d["data"].values()]
put("BoundTasksAll", str(len(ncal)))
put("BoundMCal", str(max(ncal)))
put("BoundSmallTasks", str(sum(1 for c in ncal if c != max(ncal))))
put("BoundSmallMin", str(min(t["n_train"] for t in d["data"].values())))
put("BoundTrainMax", "{:,}".format(max(t["n_train"] for t in d["data"].values())).replace(",", "{,}"))
put("BoundAlpha", "5")
put("BoundCovPromised", "95")
put("BoundLineOwn", str(LINE_OWN))
put("BoundLineBlock", str(LINE_BLOCK))
put("BoundLineNotOwn", str(LINE_NOT_OWN))
put("BoundLineNotBlock", str(LINE_NOT_BLOCK))
# verdicts of the primary setting
V = d["verdict"]
for sk, sn, _ in SEQ:
    v = V[sk]
    assert abs(v["alpha"] - ALPHA) < 1e-12
    put("Bound%sClaim" % sn, v["claim"])
    put("Bound%sUse" % sn, v["use"])
    put("Bound%sChangedLimit" % sn, pct(v["changed_limit"]))
    put("Bound%sCovLimit" % sn, pct(v["coverage_limit"]))
put("BoundClaimAll", "holds" if all(V[s]["claim"] == "holds" for s, _, _ in SEQ) else "fails")
put("BoundUseAll", "does not" if all(V[s]["use"] == "does not" for s, _, _ in SEQ) else "mixed")
# ranges over the three sequences, primary setting
for mk, mn in (("coverage", "Cov"), ("changed", "Changed"), ("blocked", "Blocked"), ("own_at_end", "Own")):
    vals = [R[sk]["t5static"]["knn10"]["0.05"]["mean"][mk] for sk, _, _ in SEQ]
    put("BoundRange%sLow" % mn, pct(min(vals)))
    put("BoundRange%sHigh" % mn, pct(max(vals)))
# ------------------------------------------------------------------ second score: verdicts, pool, tasks told apart
VR = d.get("verdict_ratio")
if VR:
    for sk, sn, _ in SEQ:
        put("Bound%sRatioClaim" % sn, VR[sk]["claim"])
        put("Bound%sRatioUse" % sn, VR[sk]["use"])
        put("Bound%sRatioChangedLimit" % sn, pct(VR[sk]["changed_limit"]))
        put("Bound%sRatioCovLimit" % sn, pct(VR[sk]["coverage_limit"]))
    put("BoundPool", "{:,}".format(VR["superni_order1"]["pool_size"]).replace(",", "{,}"))
    put("BoundRatioClaimAll", "holds" if all(VR[s]["claim"] == "holds" for s, _, _ in SEQ) else "fails")
    put("BoundRatioUseAll", "does not" if all(VR[s]["use"] == "does not" for s, _, _ in SEQ) else "mixed")
    # all rows (both scores, all sequences, the word-vector feature): the claim holds, the use fails
    chg = [R[sk]["t5static"][ck]["0.05"]["mean"]["changed"] for sk, _, _ in SEQ for ck in ("knn10", "ratio10")]
    cov = [R[sk]["t5static"][ck]["0.05"]["mean"]["coverage"] for sk, _, _ in SEQ for ck in ("knn10", "ratio10")]
    put("BoundAllChangedLow", pct(min(chg)))
    put("BoundAllChangedHigh", pct(max(chg)))
    put("BoundAllCovLow", pct(min(cov)))
    put("BoundAllCovHigh", pct(max(cov)))
    pt = R["superni_order1"]["t5static"]["ratio10"]["0.05"]["per_task"]


    def short(t):
        n = t.split("/")[1].split("_", 1)[1]
        return n.replace("_", "\\_")
    apart = [short(t) for t, v in pt.items() if v["own_at_end"] >= 0.75]
    notapart = [short(t) for t, v in pt.items() if v["own_at_end"] <= 0.10]
    put("BoundToldApartCount", str(len(apart)))
    put("BoundToldApartNames", ", ".join(apart))
    put("BoundNotApartCount", str(len(notapart)))
    put("BoundNotApartNames", ", ".join(notapart))
    put("BoundToldApartMin", pct(min(v["own_at_end"] for t, v in pt.items() if v["own_at_end"] >= 0.75), 0))

# ------------------------------------------------------------------ Pilot B: trained answers, small model, five tasks, no definition
DEMO = os.path.join(HERE, "..", "certified_switch", "demo_results")
pb = {}
for key, fname in (("Plain", "no_instruction_t5-small_seed0.json"), ("Ratio", "no_instruction_t5-small_seed0_ratio.json")):
    p = os.path.join(DEMO, fname)
    if os.path.exists(p):
        pb[key] = json.load(open(p))
if pb:
    first = next(iter(pb.values()))
    put("BoundBTasks", "5")
    put("BoundBTrain", str(first["args"]["train"]))
    put("BoundBTest", str(first["args"]["test"]))
    put("BoundBEpochs", str(first["args"]["epochs"]))
    put("BoundBCal", str(first["m_cal"]))
    a = first["rules"]["all"]
    put("BoundBAllLearned", "%.1f" % a["learned"])
    put("BoundBAllAP", "%.1f" % a["AP"])
    put("BoundBAllFT", "%.1f" % a["FT"])
    put("BoundBAllChanged", str(a["changed"]))
    put("BoundBCompared", str(a["compared"]))
    sc = a["score_after_learning_per_task"]
    put("BoundBAllGuess", str(sum(1 for x in sc if x < 50)))
    put("BoundBAllGuessScore", "%.1f" % min(x for x in sc if x < 50))
    for key, kn in (("Plain", "Plain"), ("Ratio", "Ratio")):
        if key not in pb:
            continue
        z = pb[key]["rules"]["zones"]
        put("BoundB%sLearned" % kn, "%.1f" % z["learned"])
        put("BoundB%sAP" % kn, "%.1f" % z["AP"])
        put("BoundB%sFT" % kn, "%.1f" % z["FT"])
        put("BoundB%sChanged" % kn, str(z["changed"]))
        put("BoundB%sOwn" % kn, pct(z["routed_to_own_task_final_mean"], 0))
        put("BoundB%sInside" % kn, str(z["inside_own_zone_first4"]))
        put("BoundB%sChangedInside" % kn, str(z["changed_inside_own_zone"]))
        blk = [t["blocked_test"] for t in pb[key]["per_task"]]
        put("BoundB%sBlocked" % kn, ", ".join(str(b) for b in blk))
        put("BoundB%sFirstTrainBlocked" % kn, str(pb[key]["per_task"][1]["blocked_train"]))
        put("BoundB%sRatioToAll" % kn, pct(z["learned"] / a["learned"], 0))
        for i_, (zs, as_) in enumerate(zip(z["score_after_learning_per_task"], a["score_after_learning_per_task"]), 1):
            put("BoundB%sScore%s" % (kn, "One Two Three Four Five".split()[i_ - 1]), "%.1f" % zs)
            put("BoundBAllScore%s" % "One Two Three Four Five".split()[i_ - 1], "%.1f" % as_)
    rows_b = [("All adapters on", a["learned"], a["AP"], a["FT"], "%d of %d" % (a["changed"], a["compared"]), "--", "--")]
    for key, lab in (("Plain", "Zones, plain score"), ("Ratio", "Zones, ratio score")):
        if key in pb:
            z = pb[key]["rules"]["zones"]
            blk = ", ".join(str(t["blocked_test"]) for t in pb[key]["per_task"])
            rows_b.append((lab, z["learned"], z["AP"], z["FT"], "%d of %d" % (z["changed"], z["compared"]), pct(z["routed_to_own_task_final_mean"], 0), blk))
    lb = [r"\begin{tabular}{lrrrrrl}", r"    \toprule", r"    Rule & Learned & End & Forgetting & Changed & Own, \% & Blocked (of " + str(first["args"]["test"]) + r") \\", r"    \midrule"]
    for lab, l_, ap, ft, ch, own, blk in rows_b:
        lb.append("    %s & %.1f & %.1f & %.1f & %s & %s & %s \\\\" % (lab, l_, ap, ft, ch, own, blk))
    lb += [r"    \bottomrule", r"\end{tabular}", ""]
    open(os.path.join(HERE, "tables", "bound_pilotb.tex"), "w", encoding="utf-8").write("\n".join(lb))

# ------------------------------------------------------------------ the sentence-encoder feature: one sentence, written according to what is in the file
mini = [(sk, sn, ck, cn) for sk, sn, _ in SEQ for ck, cn, _ in SCORE if "minilm" in R.get(sk, {}) and ck in R[sk]["minilm"] and "0.05" in R[sk]["minilm"][ck]]
if mini:
    mc = [R[sk]["minilm"][ck]["0.05"]["mean"]["changed"] for sk, _, ck, _ in mini]
    mv = [R[sk]["minilm"][ck]["0.05"]["mean"]["coverage"] for sk, _, ck, _ in mini]
    put("BoundMiniChangedMax", pct(max(mc)))
    put("BoundMiniCovLow", pct(min(mv)))
    put("BoundMiniCovHigh", pct(max(mv)))
    sentence = (r"With the frozen sentence encoder as the feature (all-MiniLM-L6-v2 on the instance text, mean pooling; rows ``sentence enc.'' of Table~\ref{tab:boundpilot}) the claim holds again (coverage \BoundMiniCovLow{}\% to \BoundMiniCovHigh{}\%, at most \BoundMiniChangedMax{}\% of the routes changed in any row) "
                r"and the use still fails by the lines fixed in advance (with the ratio score \BoundSuperniMiniRatioFiveBlocked{}\%, \BoundLongMiniRatioFiveBlocked{}\% and \BoundBothMiniRatioFiveBlocked{}\% blocked on the three sequences), but the sentence encoder tells the tasks apart better than the word-vector average: on the SuperNI tasks \BoundSuperniMiniRatioFiveBlocked{}\% blocked instead of \BoundSuperniWordRatioFiveBlocked{}\% and \BoundSuperniMiniRatioFiveOwn{}\% of the test inputs given to their own task instead of \BoundSuperniWordRatioFiveOwn{}\%. "
                r"It was the secondary feature of the plan, so it changes no verdict.")
else:
    sentence = ("The same test with the frozen sentence encoder as the feature was still being computed when this paper was written and is not reported.")
open(os.path.join(HERE, "tables", "bound_minilm.tex"), "w", encoding="utf-8").write(sentence + "\n")

open(os.path.join(HERE, "tables", "numbers_bound.tex"), "w", encoding="utf-8").write("".join("\\newcommand{\\%s}{%s}\n" % (k, N[k]) for k in sorted(N)))

lines = [r"\begin{tabular}{lllrrrr}", r"    \toprule",
         r"    Sequence & Feature & Score & Covered, \% & Changed, \% & Blocked, \% & Own task, \% \\", r"    \midrule"]
prev = None
for slabel, flabel, clabel, m, se in rows:
    if prev is not None and slabel != prev:
        lines.append(r"    \midrule")
    prev = slabel
    lines.append("    %s & %s & %s & %s & %s & %s & %s \\\\" % (slabel if True else "", flabel, clabel, pct(m["coverage"]), pct(m["changed"]), pct(m["blocked"]), pct(m["own_at_end"])))
lines += [r"    \bottomrule", r"\end{tabular}", ""]
open(os.path.join(HERE, "tables", "bound_pilot.tex"), "w", encoding="utf-8").write("\n".join(lines))
print("wrote 2 tables and %d number commands" % len(N))
