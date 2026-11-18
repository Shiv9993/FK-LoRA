"""Tables, figure and number commands for the tests of wider zones (Section sec:paraphrase, Appendix app:widen).

    python3 widen_results.py

Reads   routing/feature_study.json   idea 1: hidden states of a frozen causal model (SmolLM2-135M)
        routing/ruler_study.json     idea 2: paraphrase ruler (also holds the reference numbers of the paper's rule)
        routing/encoder_study.json   idea 3: frozen sentence encoder, split-conformal radius
        routing/judge_study.json     idea 4: zone decided by a frozen inference (NLI) model
        routing/paraphrases.json     statistics of the machine paraphrases
Writes  tables/numbers_widen.tex     one \\newcommand per number used in the text (\\Widen...)
        tables/widen_main.tex        main table: the paper's rule and the five attempts, primary settings
        tables/widen_encoder.tex     the six settings of the sentence encoder, French and German pivot
        tables/widen_judge.tex       threshold curve of the judge (French pivot)
        tables/widen_siblings.tex    examples of sibling tasks that collide
        figures/widen_tradeoff.pdf   correct against wrong certificates, one marker per family
Every number is read from the json files; nothing is typed by hand except the thresholds of the rules that were
fixed before the runs (see PLAN.md). Run it again if a json file changes.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "routing")
load = lambda name: json.load(open(os.path.join(R, name + ".json")))
feat, ruler, enc, judge, para, span, rr = (load(n) for n in ("feature_study", "ruler_study", "encoder_study", "judge_study", "paraphrases", "spanish_check", "reroute_study"))

# thresholds of the rules written before the runs (silent_adapters/PLAN.md)
BAR_P, BAR_M, BAR_C = 0.75, 0.05, 0.19

N = {}


def pct(x):
    return "%.1f" % (100 * x)


def put(name, value):
    assert name.isalpha(), name
    assert N.get(name, value) == value, "two values for " + name
    N[name] = value


def thousands(n):
    return "{:,}".format(int(n)).replace(",", "{,}")


def measures(prefix, r, keys="CUPMRZ"):
    for k in keys:
        if k in r:
            put(prefix + k, pct(r[k]))


# ------------------------------------------------------------------ the machine paraphrases
ev, cal = len(feat["split"]["evaluation"]), len(feat["split"]["calibration"])
put("TasksAll", thousands(ev + cal))
put("TasksCal", str(cal))
put("TasksEval", str(ev))
put("HalfA", str(enc["notes"]["split"]["half_A"]))
put("HalfB", str(enc["notes"]["split"]["half_B"]))
for p, name in (("de", "De"), ("fr", "Fr")):
    s = para["stats"][p]
    put("Kept" + name, "%.0f" % (100 * s["mean_share_original_words_kept"]))
    put("Same" + name, str(s["identical"]))
assert para["stats"]["de"]["of"] == para["stats"]["fr"]["of"] == ev + cal
ref = ruler["tables"]["t5-base"]["reference"]
put("ParaInputs", thousands(ref["paraphrased_inputs"]))
put("ParaInputsPivot", thousands(enc["minilm"]["table"]["plain"]["0.1"]["French"]["paraphrased_inputs"]))
put("UneditedInputs", thousands(enc["minilm"]["table"]["plain"]["0.1"]["French"]["unedited_inputs"]))

# ------------------------------------------------------------------ the paper's rule (T5-base word-vectors)
assert ref["ruler"] == "OLD" and ref["rule"] == "B" and ref["kappa"] == 0.5
measures("RefTfive", ref, "CUPMR")
put("RefTfivePM", pct(ref["P"] + ref["M"]))
put("RefTfiveRest", pct(1 - ref["P"] - ref["M"]))
tab = ruler["tables"]["t5-base"]["matched"]["OLD"]
for piv, name in (("fr", "Fr"), ("de", "De")):
    measures("RefTfive%s" % name, tab[piv]["rule_B"], "PMR")
put("RefTfiveZero", str(ref["radius_zero"]))

# ------------------------------------------------------------------ idea 1: hidden states
fe = feat["evaluation"]
assert feat["chosen"]["config"] == "L30-mean" and fe["chosen"]["config"] == "L30-mean"
measures("RefSmol", fe["reference"], "CUPMR")
measures("Feat", fe["chosen"], "CUPMR")
measures("FeatCal", feat["chosen"]["calibration"], "CUPMR")
put("FeatGain", "%.1f" % (100 * (fe["chosen"]["P"] - fe["reference"]["P"])))
put("FeatRadiusQuantile", feat["chosen"]["setting"].split("(")[1].rstrip(")"))
put("FeatCalRefC", pct(feat["C_ref_calibration"]))
put("FeatLayers", str(len(feat["R2_radii"]) // 2))
put("FeatSettings", str(len(feat["R2_radii"])))
assert fe["chosen"]["M"] > fe["reference"]["M"] and fe["chosen"]["C"] > fe["reference"]["C"]   # so: not a success

# ------------------------------------------------------------------ idea 2: paraphrase ruler
rt = ruler["tables"]
for t, name in (("t5-base", "Tfive"), ("smollm2", "Smol")):
    v = rt[t]["verdict"]
    assert v["outcome"] == "against"
    put("Ruler%sPOld" % name, pct(v["P_old"]))
    put("Ruler%sPNew" % name, pct(v["P_new"]))
    put("Ruler%sMOld" % name, pct(v["M_old"]))
    put("Ruler%sMNew" % name, pct(v["M_new"]))
    put("Ruler%sGain" % name, "%.1f" % (100 * v["P_gain"]))
    m = rt[t]["matched"]
    put("Ruler%sROld" % name, pct(m["OLD"]["fr"]["rule_A"]["R"]))
    put("Ruler%sRNew" % name, pct(m["NEW_beta0.1"]["fr"]["rule_A"]["R"]))
    put("Ruler%sC" % name, pct(rt[t]["C_ref"]))
    # with the paper's own radius rule (rule B), both pivots
    put("Ruler%sRuleBOld" % name, pct(m["OLD"]["fr"]["rule_B"]["P"]))
    put("Ruler%sRuleBNew" % name, pct(m["NEW_beta0.1"]["fr"]["rule_B"]["P"]))
sec = rt["t5-base"]["secondary"]
put("RulerHeavyOld", pct(sec["OLD"]["five"]["heavy"]["certified"]))
put("RulerHeavyNew", pct(sec["NEW_beta0.1"]["five"]["heavy"]["certified"]))
for t, name in (("t5-base", "Tfive"), ("smollm2", "Smol")):
    put("RulerTopTen%s" % name, "%.0f" % (100 * rt[t]["diagnostics_sigma"]["top10_cumulative"]))

# ------------------------------------------------------------------ idea 3: sentence encoder
mi = enc["minilm"]
assert mi["primary_variant"] == "ruler"
VAR = (("plain", "Plain"), ("ruler", "Ruler"))
ALPHA = (("0.05", "Five"), ("0.1", "Ten"), ("0.2", "Twenty"))
PIV = (("French", "Fr"), ("German", "De"))
for v, vn in VAR:
    for a, an in ALPHA:
        put("Enc%s%sRadius" % (vn, an), "%.3g" % mi["radius"][v][a]["r"])
        for pv, pn in PIV:
            r = mi["table"][v][a][pv]
            for k in "CUPMRZ":
                put("Enc%s%s%s%s" % (vn, an, pn, k), pct(r[k]))
hi = [mi["table"][v][a]["French"]["R"] for v, _ in VAR for a in ("0.1", "0.2")]
put("EncRHighLow", pct(min(hi)))
put("EncRHighHigh", pct(max(hi)))
primary = mi["table"]["ruler"]["0.1"]["French"]
vd = mi["verdict"]
assert abs(vd["P"] - primary["P"]) < 1e-12 and vd["outcome"] == "against"
measures("EncPrim", primary, "CUPMRZ")
put("EncPrimRadius", "%.1f" % mi["radius"]["ruler"]["0.1"]["r"])
put("EncRatioRuler", "%.2f" % mi["ratio"]["ruler"]["ratio_to_r_0.1"])
put("EncRatioPlain", "%.2f" % mi["ratio"]["plain"]["ratio_to_r_0.1"])
put("EncRulerEigLow", "%.0f" % mi["ruler"]["ruler_eigs_min_max"][0])
put("EncRulerEigHigh", "%.0f" % mi["ruler"]["ruler_eigs_min_max"][1])
put("EncDisplacements", str(mi["ruler"]["n_displacements"]))
put("EncUneditedZero", thousands(mi["unedited_own_distance_check"]["ruler"]["exactly_0"]))
put("EncParams", "22")
put("EncPrefixes", str(len(mi["opening_tokens"]["L_k_counts"])))
put("EncMs", "%.0f" % enc["view_f_cost"]["mean_ms"])
put("EncMsHigh", "%.0f" % enc["view_f_cost"]["p90_ms"])
# second encoder, instruction as a separate field
bg = enc["bge"]
assert bg["verdict"]["outcome"] == "against"
put("BgeP", pct(bg["verdict"]["P"]))
put("BgeM", pct(bg["verdict"]["M"]))
put("BgeC", pct(bg["verdict"]["C"]))
vb = [enc["view_b_instruction_field"][e]["table"]["0.1"]["French"] for e in ("minilm", "bge", "mpnet")]
put("FieldPLow", pct(min(r["P"] for r in vb)))
put("FieldPHigh", pct(max(r["P"] for r in vb)))
put("FieldMLow", pct(min(r["M"] for r in vb)))
put("FieldMHigh", pct(max(r["M"] for r in vb)))
# benchmark tasks with the machine-written rewordings
db = enc["view_d_benchmark"]
for lv, ln in (("original", "Original"), ("light", "Light"), ("medium", "Medium"), ("heavy", "Heavy")):
    put("EncBench%sZone" % ln, pct(db["five"][lv]["in_own_zone"]))
    put("EncBench%sRouted" % ln, pct(db["five"][lv]["routed_own"]))
    assert db["five"][lv]["certified_wrong"] == 0 and db["fifteen"][lv]["certified_wrong"] == 0
# collisions (ruler, alpha 0.1)
col = enc["view_e_collisions"]
put("EncColliding", str(col["tasks_colliding"]))
put("EncCollidingPct", pct(col["share"]))

# ------------------------------------------------------------------ idea 4: judge
jc = judge["calibration"]
jv = judge["step3_final"]["chosen"]["table"]
assert judge["step3_final"]["verdict_chosen_French"]["outcome"] == "against" and jc["chosen_judge"] == "A"
fr = jv["French"]
put("JudgeC", pct(jv["C_U"]["C"]))
put("JudgeU", pct(jv["C_U"]["U"]))
for k in "ZPMR":
    put("JudgeFr" + k, pct(fr[k]))
    put("JudgeDe" + k, pct(jv["German"][k]))
put("JudgeTau", "%.2f" % jc["tau"]["A"])
put("JudgeRw", "%.2f" % jc["r_w"])
put("JudgeAucNli", "%.3f" % jc["auc"]["A"])
put("JudgeAucFree", "%.3f" % jc["auc"]["B"])
put("JudgeNegSame", pct(jc["judge_A_share_negatives_ge_tau"]))
put("JudgePosSame", pct(jc["judge_A_share_positives_ge_tau"]))
put("JudgeLimitFr", str(judge["limit_reached"]["fr"]["texts_with_more_than_20_inside_r_w"]))
put("JudgeLimit", "20")
put("JudgePairs", "%.0f" % judge["cost_5g"]["mean_judged_pairs_per_text_all_non_identical"])
put("JudgeSeconds", "%.1f" % (judge["cost_5g"]["mean_ms_per_text_filter_plus_judge"] / 1000))
put("JudgeCollisionPairs", str(judge["collisions_5d"]["per_judge"]["A"]["n_pairs"]))
put("JudgeCollisionSame", str(judge["collisions_5d"]["per_judge"]["A"]["n_ge_tau"]))
put("JudgeFilterOnlyP", pct(judge["step3_final"]["filter_alone_5b"]["table"]["French"]["P"]))
put("JudgeFilterOnlyM", pct(judge["step3_final"]["filter_alone_5b"]["table"]["French"]["M"]))
sweep = judge["sweep_5c"]["curve_French"]
put("SweepSteps", str(len(sweep)))
put("SweepPMax", pct(max(r["P"] for r in sweep)))
last = sweep[-1]
put("SweepLastP", pct(last["P"]))
put("SweepLastTau", "%.2f" % last["tau"])
below = [r for r in sweep[:-1] if r["M"] < 0.09]
best_m = min(sweep[:-1], key=lambda r: r["M"])
put("SweepBestTau", "%.2f" % best_m["tau"])
for k in "CPMR":
    put("SweepBest" + k, pct(best_m[k]))
mid = min(sweep, key=lambda r: abs(r["C"] - BAR_C))      # the threshold where C comes closest to the 19% of the rule
put("SweepMidTau", "%.2f" % mid["tau"])
for k in "CPM":
    put("SweepMid" + k, pct(mid[k]))
jb = judge["benchmark_5f"]["summary"]["all"]
inside = sum(jb[l]["in_own_zone"] for l in ("light", "medium", "heavy"))
total = sum(jb[l]["n"] for l in ("light", "medium", "heavy"))
put("JudgeBenchInside", str(inside))
put("JudgeBenchTotal", str(total))
for l in jb:
    assert jb[l]["certified_wrong_five"] == 0 and jb[l]["certified_wrong_fifteen"] == 0 and jb[l]["routed_correct_five"] == jb[l]["n"]


# ------------------------------------------------------------------ the check on Spanish-pivot paraphrases
SP_RULE_P, SP_RULE_R = 0.15, 0.08          # the rule written before the run (PLAN.md)
pr = span["paper_rule"]
assert span["reproduced"]["ok"]
for piv, pn in (("French", "Fr"), ("German", "De"), ("Spanish", "Es")):
    for k in "PMR":
        put("RefTfive%s%s" % (pn, k), pct(pr[piv][k]))
assert abs(pr["French"]["P"] - tab["fr"]["rule_B"]["P"]) < 1e-9 and abs(pr["German"]["M"] - tab["de"]["rule_B"]["M"]) < 1e-9
it_ = rr["italian_paper_measures_certified"]
for k in "PMR":
    put("RefTfiveIt%s" % k, pct(it_[k]))
for k in "PMR":
    vals = [pr[p][k] for p in ("French", "German", "Spanish")] + [it_[k]]
    put("RefTfive%sLow" % k, pct(min(vals)))
    put("RefTfive%sHigh" % k, pct(max(vals)))
put("SpInputs", thousands(span["verdict"]["n_spanish_inputs"]))
put("SpKept", "%.0f" % (100 * span["paraphrase_stats_es"]["es"]["mean_share_original_words_kept"]))
put("SpSame", str(span["paraphrase_stats_es"]["es"]["identical"]))
er = span["encoder"]["ruler"]["0.2"]
sv = span["verdict"]
assert sv["outcome"] == "not confirmed" and sv["P_gain"] < SP_RULE_P and sv["R_gain"] >= SP_RULE_R and sv["M_b"] <= sv["M_a"]
put("SpC", pct(er["C"]))
put("SpCRule", pct(span["paper_rule"]["C"]))
for piv, pn in (("French", "Fr"), ("German", "De"), ("Spanish", "Es")):
    for k in "PMRZ":
        put("SpEnc%s%s" % (pn, k), pct(er[piv][k]))
put("SpGainP", "%.1f" % (100 * sv["P_gain"]))
put("SpGainR", "%.1f" % (100 * sv["R_gain"]))
put("SpAskP", "%d" % round(100 * SP_RULE_P))
put("SpAskR", "%d" % round(100 * SP_RULE_R))
put("SpPromise", "80")
for var, vn in (("plain", "Plain"), ("ruler", "Ruler")):
    for a, an in (("0.05", "Five"), ("0.1", "Ten"), ("0.2", "Twenty")):
        for k in "PMRZ":
            put("SpAll%s%s%s" % (vn, an, k), pct(span["encoder"][var][a]["Spanish"][k]))
sm = span["splits"]["summary"]
for key, kn in (("paper_rule", "Rule"), ("encoder_ruler_0.2", "Enc")):
    for k in "CPMRZ":
        if key == "paper_rule" and k == "Z":
            continue
        d = sm[key][k] if k == "C" else sm[key]["Spanish"][k]
        put("Split%s%sMean" % (kn, k), pct(d["mean"]))
        put("Split%s%sLow" % (kn, k), pct(d["min"]))
        put("Split%s%sHigh" % (kn, k), pct(d["max"]))
put("SplitPasses", str(sum(1 for v in span["splits"]["summary"]["per_seed_spanish_rule"].values() if v)))
put("SplitCount", str(len(span["splits"]["seeds"])))
sb = span["benchmark"]["five"]
for lv, ln in (("light", "Light"), ("medium", "Medium"), ("heavy", "Heavy")):
    put("SpBench%sZone" % ln, pct(sb[lv]["in_own_zone"]))
    put("SpBench%sRouted" % ln, pct(sb[lv]["routed_own"]))
    assert sb[lv]["certified_wrong"] == 0


# ------------------------------------------------------------------ additions asked by the second reader
ps = span["splits"]["per_seed"]
more_m = more_c = 0
for sd, nm in zip(span["splits"]["seeds"], ("One", "Two", "Three", "Four", "Five")):
    a, b_ = ps[str(sd)]["paper_rule"], ps[str(sd)]["encoder_ruler_0.2"]
    put("SplitSeed%sRuleM" % nm, pct(a["Spanish"]["M"]))
    put("SplitSeed%sEncM" % nm, pct(b_["Spanish"]["M"]))
    put("SplitSeed%sRuleC" % nm, pct(a["C"]))
    put("SplitSeed%sEncC" % nm, pct(b_["C"]))
    more_m += b_["Spanish"]["M"] > a["Spanish"]["M"]
    more_c += b_["C"] > a["C"]
put("SplitMoreM", str(more_m))
put("SplitMoreC", str(more_c))
for key, kn in (("paper_rule", "Rule"), ("encoder_ruler_0.2", "Enc")):
    for k in "PMR":
        d_ = sm[key]["French"][k]
        put("SplitFr%s%sMean" % (kn, k), pct(d_["mean"]))
        put("SplitFr%s%sLow" % (kn, k), pct(d_["min"]))
        put("SplitFr%s%sHigh" % (kn, k), pct(d_["max"]))
cv = enc["view_a_curve"]
put("EncCurveC", "%.1f" % (100 * 0.189))   # the C of SmolLM2's reference, as set in encoder_study.py (C_REF)
for k in "PMR":
    put("EncCurveRuler" + k, pct(cv["ruler"]["at_C_0.189"][k]))
    put("EncCurvePlain" + k, pct(cv["plain"]["at_C_0.189"][k]))
bh = rt["t5-base"]["matched"]["NEW_beta0.5"]["fr"]["rule_A"]
put("RulerHalfP", pct(bh["P"]))
put("RulerHalfM", pct(bh["M"]))
cp = judge["collisions_5d"]["per_judge"]["A"]
put("JudgeCollisionNotSame", str(cp["n_pairs_not_same_definition"]))
put("JudgeCollisionNotSameSame", str(cp["n_ge_tau_not_same_definition"]))
put("JudgeCollisionSameDef", str(cp["n_pairs"] - cp["n_pairs_not_same_definition"]))
put("EncGrid", ", ".join(str(g) for g in enc["notes"]["grid"]))
put("EncGridCount", str(len(enc["notes"]["grid"])))
pl = [(mi["table"]["plain"][a][pv]["Z"], 1 - float(a)) for a, _ in ALPHA for pv, _ in PIV]
below = [(1 - z / 1.0, p) for z, p in pl if z < p - 1e-9]
put("PlainZBelowCount", str(len(below)))
put("PlainZBelowMax", "%.1f" % (100 * max(p - z for z, p in pl)))
put("RulerDeTenZ", pct(mi["table"]["ruler"]["0.1"]["German"]["Z"]))

# ------------------------------------------------------------------ the re-routing study
RR = {"signature": "Sig", "nearest": "Near", "certified": "Cert", "hybrid": "Hyb", "normalised": "Norm", "certified3": "CertThree", "background": "Bg", "certified4": "CertFour"}
assert rr["reproduced"]["ok"]
for rule, rn in RR.items():
    fg = rr["rules"][rule]["French+German"]
    es = rr["rules"][rule]["Spanish"]
    un = rr["rules"][rule]["unedited"]
    put("Reroute%sOwnStart" % rn, pct(fg["own_at_arrival"]))
    put("Reroute%sOwnEnd" % rn, pct(fg["own_at_end"]))
    put("Reroute%sChanged" % rn, "%.2f" % (100 * fg["changed"]))
    put("Reroute%sNChanged" % rn, thousands(fg["n_changed"]))
    put("Reroute%sLost" % rn, "%.2f" % (100 * fg["lost"]))
    put("Reroute%sNLost" % rn, str(fg["n_lost"]))
    put("Reroute%sTasks" % rn, str(fg["tasks_with_changed"]))
    put("Reroute%sUnOwn" % rn, pct(un["own_at_end"]))
    put("Reroute%sUnChanged" % rn, str(un["n_changed"]))
    put("Reroute%sEsOwn" % rn, pct(es["own_at_end"]))
    put("Reroute%sEsChanged" % rn, "%.2f" % (100 * es["changed"]))
    put("Reroute%sEsNChanged" % rn, str(es["n_changed"]))
    itd = rr["rules"][rule]["Italian"]
    put("Reroute%sItOwn" % rn, pct(itd["own_at_end"]))
    put("Reroute%sItChanged" % rn, "%.2f" % (100 * itd["changed"]))
    put("Reroute%sItNChanged" % rn, str(itd["n_changed"]))
    if "certified_at_arrival" in fg:
        put("Reroute%sInZone" % rn, pct(fg["certified_at_arrival"]))
        assert fg["changed_among_certified"] == 0
    for wd, wn in (("medium", "Medium"), ("heavy", "Heavy")):
        for sq, sn in (("five", "Five"), ("fifteen", "Fifteen")):
            b_ = rr["benchmark"][sq][wd][rule]
            put("Reroute%sBench%s%sOwn" % (rn, sn, wn), pct(b_["own_at_end"]))
            put("Reroute%sBench%s%sChanged" % (rn, sn, wn), str(b_["changed"]))
for rule_ in RR:
    assert rr["rules"][rule_]["unedited"]["n_changed"] == 0, rule_
def texname(t):
    return t.split("_", 1)[1].replace("_", "\\_")
for nm, i in (("One", 0), ("Two", 2), ("Three", 3)):
    ex = rr["examples_signature"][i]
    put("RerouteEx%sTask" % nm, texname(ex["task"]))
    put("RerouteEx%sEnd" % nm, texname(ex["task_at_end"]))
    put("RerouteEx%sBetween" % nm, str(ex["tasks_arrived_in_between"]))
    put("RerouteEx%sShare" % nm, "%.0f" % (100 * ex["word_share_of_definitions"]))
    put("RerouteEx%sPivot" % nm, ex["pivot"])
put("RerouteHybGain", "%.1f" % (100 * (rr["rules"]["hybrid"]["French+German"]["own_at_end"] - rr["rules"]["certified"]["French+German"]["own_at_end"])))
put("RerouteHybLoss", "%.1f" % (100 * (rr["benchmark"]["five"]["heavy"]["certified"]["own_at_end"] - rr["benchmark"]["five"]["heavy"]["hybrid"]["own_at_end"])))
put("RerouteExactEsOwn", pct(rr["rules"]["exactfirst"]["Spanish"]["own_at_end"]))
put("RerouteInputs", thousands(rr["rules"]["certified"]["French+German"]["n"]))
put("RerouteInputsPivot", thousands(rr["rules"]["certified"]["Spanish"]["n"]))
put("RerouteBenchChangeable", str(rr["benchmark"]["five"]["heavy"]["signature"]["n"] - rr["benchmark"]["five"]["heavy"]["signature"]["n"] // 5))
put("RerouteBenchInputs", thousands(rr["benchmark"]["five"]["heavy"]["signature"]["n"]))
put("RerouteIdenticalNearCert", "yes" if rr["rules"]["nearest"]["French+German"]["n_changed"] == rr["rules"]["certified"]["French+German"]["n_changed"] else "no")
assert rr["verdict"]["outcome"] == "adds little here" and rr["verdict_followup"]["outcome"] == "some"
pr_ = rr["protected_signature"]["French+German"]
put("RerouteCertOwn", thousands(pr_["n_certified_own"]))
put("RerouteSigLostOfCert", str(pr_["lost_by_signature"]))
put("RerouteSigLostOfCertPct", "%.2f" % (100 * pr_["share_lost_by_signature"]))
sg = rr["siblings_signature"]["French+German"]
put("RerouteSigToSibling", str(sg["n_to_sibling"]))
put("RerouteSigToSiblingOf", str(sg["n_changed_signature"]))
put("RerouteNearToSibling", str(rr["siblings"]["French+German"]["n_to_sibling"]))
put("RerouteLimitLow", "2")
put("RerouteExpectNearLow", "8")
put("RerouteExpectNearHigh", "20")
put("RerouteExpectSigLow", "5")
put("RerouteExpectSigHigh", "20")
put("RerouteUnchangedInZone", thousands(round(rr["rules"]["certified"]["French+German"]["certified_at_arrival"] * rr["rules"]["certified"]["French+German"]["n"])))

# table
ROWS = [(r"\signature{} (plain nearest signature)", "signature"), (r"\nearest{} (nearest zone)", "nearest"), (r"\cert{} (the rule of the paper)", "certified"),
        (r"\hybrid{} (zones, then nearest signature)", "hybrid"), (r"\scaled{} (scaled by the distance to unrelated text)", "background"),
        (r"\certscaled{} (zones, then \scaled{})", "certified4")]


def rrrow(label, rule):
    fg = rr["rules"][rule]["French+German"]
    it = rr["rules"][rule]["Italian"]
    inz = pct(fg["certified_at_arrival"]) if "certified_at_arrival" in fg else "--"
    return "    %s & %s & %s & %s (%s) & %s & %s \\\\" % (label, pct(fg["own_at_end"]), pct(it["own_at_end"]), "%.2f" % (100 * fg["changed"]), thousands(fg["n_changed"]),
                                                    inz, pct(rr["rules"][rule]["unedited"]["own_at_end"]))


lines = [r"\begin{tabular}{lrrrrr}", r"    \toprule",
         r"    Rule & Own, \% & Italian: own, \% & Changed, \% (n) & In zone, \% & Unedited, \% \\", r"    \midrule"]
for lab, rule in ROWS:
    lines.append(rrrow(lab, rule))
lines += [r"    \bottomrule", r"\end{tabular}", ""]
open(os.path.join(HERE, "tables", "reroute.tex"), "w", encoding="utf-8").write("\n".join(lines))


def rrbench(label, rule):
    cells = []
    for sq in ("five", "fifteen"):
        for wd in ("medium", "heavy"):
            b_ = rr["benchmark"][sq][wd][rule]
            cells.append("%s (%d)" % (pct(b_["own_at_end"]), b_["changed"]))
    return "    %s & %s \\\\" % (label, " & ".join(cells))


lines = [r"\begin{tabular}{lrrrr}", r"    \toprule", r"    & \multicolumn{2}{c}{Five tasks} & \multicolumn{2}{c}{Fifteen tasks} \\",
         r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5}", r"    Rule & Medium (n) & Heavy (n) & Medium (n) & Heavy (n) \\", r"    \midrule"]
for lab, rule in ROWS:
    lines.append(rrbench(lab.split(" (")[0], rule))
lines += [r"    \bottomrule", r"\end{tabular}", ""]
open(os.path.join(HERE, "tables", "reroute_bench.tex"), "w", encoding="utf-8").write("\n".join(lines))


# ------------------------------------------------------------------ the two fallback tests (scaled fallbacks)
vn, vb = rr["verdict_normalised"], rr["verdict_background"]
assert vn["outcome"] == "against" and vb["outcome"] == "success"
put("RerouteNormHeavyFive", pct(vn["heavy_five"]))
put("RerouteNormHeavyFifteen", pct(vn["heavy_fifteen"]))
put("RerouteNormMediumFive", pct(vn["medium_five"]))
put("RerouteNormMediumFifteen", pct(vn["medium_fifteen"]))
put("RerouteBgHeavyFive", pct(vb["heavy_five"]))
put("RerouteBgHeavyFifteen", pct(vb["heavy_fifteen"]))
put("RerouteBgMediumFive", pct(vb["medium_five"]))
put("RerouteBgMediumFifteen", pct(vb["medium_fifteen"]))
# the lines of the rule written before the runs (PLAN.md, 2026-11-18 01:28 and 02:18)
put("RerouteLineHeavy", "90")
put("RerouteLineMedium", "97")
put("RerouteLineItalian", "88")
put("RerouteLineHeavyAgainst", "80")
put("RerouteLineItalianAgainst", "85")
put("RerouteExpectBgHeavyLow", "88")
put("RerouteExpectBgHeavyHigh", "94")
put("RerouteExpectBgItLow", "86")
put("RerouteExpectBgItHigh", "92")
put("RerouteExpectCfItLow", "82")
put("RerouteExpectCfItHigh", "87")
put("RerouteExpectCfThreeItLow", "84")
put("RerouteExpectCfThreeItHigh", "88")
put("RerouteExpectNormItLow", "89")
put("RerouteExpectNormItHigh", "93")
put("RerouteExpectNormChangedLow", "1.5")
put("RerouteExpectNormChangedHigh", "2.5")
Bs = rr["B"]
put("RerouteBShare", "%.1f" % (100 * Bs["700_tasks"]["share_radius_equals_half_B"]))
put("RerouteBCorr", "%.2f" % Bs["700_tasks"]["corr_B_radius"])
put("RerouteBMin", "%.1f" % Bs["700_tasks"]["min"])
put("RerouteBMedian", "%.1f" % Bs["700_tasks"]["median"])
put("RerouteBMax", "%.1f" % Bs["700_tasks"]["max"])
put("RerouteBBenchFiveShare", "%.0f" % (100 * Bs["benchmark"]["five"]["share_radius_equals_half_B"]))
put("RerouteBBenchFifteenCount", str(round(Bs["benchmark"]["fifteen"]["share_radius_equals_half_B"] * 15)))
put("RerouteBBenchFiveMin", "%.1f" % Bs["benchmark"]["five"]["B_min"])
put("RerouteBBenchFiveMax", "%.1f" % Bs["benchmark"]["five"]["B_max"])
put("RerouteSigmaCorr", "%.2f" % rr["sigma"]["corr_sigma_radius_700_tasks"])
put("RerouteSigmaCorrSpearman", "%.2f" % rr["sigma"]["corr_sigma_radius_700_tasks_spearman"])
sl = rr["sigma"]["by_length"]
put("RerouteSigmaMin", "%.1f" % min(sl.values()))
put("RerouteSigmaMax", "%.1f" % max(sl.values()))
for rule, rn in (("certified", "Cert"), ("nearest", "Near"), ("normalised", "Norm"), ("certified3", "CertThree"), ("background", "Bg"), ("certified4", "CertFour")):
    st = rr["stress"]["light"][rule]["changed_per_wording"]
    put("RerouteStress%sMedium" % rn, str(st["medium"]))
    put("RerouteStress%sHeavy" % rn, str(st["heavy"]))
    put("RerouteStress%sCertified" % rn, str(st["original"] + (st["light"] if rule in ("certified", "certified3", "certified4") else 0)))
put("RerouteStressOld", str(rr["stress"]["light"]["certified4"]["n_old_inputs_per_wording"]))
put("RerouteStressTaken", str(rr["stress"]["light"]["certified4"]["changed_per_wording"]["medium"] + rr["stress"]["light"]["certified4"]["changed_per_wording"]["heavy"]))
put("RerouteZonesCost", "%.1f" % (round(100 * rr["rules"]["background"]["Italian"]["own_at_end"], 1) - round(100 * rr["rules"]["certified4"]["Italian"]["own_at_end"], 1)))
put("RerouteZonesCostFG", "%.1f" % (round(100 * rr["rules"]["background"]["French+German"]["own_at_end"], 1) - round(100 * rr["rules"]["certified4"]["French+German"]["own_at_end"], 1)))
put("RerouteItGainCert", "%.1f" % (100 * (rr["rules"]["certified4"]["Italian"]["own_at_end"] - rr["rules"]["certified"]["Italian"]["own_at_end"])))
pit = json.load(open(os.path.join(R, "paraphrases_it.json")))["stats"]["it"]
put("RerouteItKept", "%.0f" % (100 * pit["mean_share_original_words_kept"]))
put("RerouteItSame", str(pit["identical"]))
for rule, rn in (("background", "Bg"), ("certified4", "CertFour")):
    for g, gn in (("French+German", "FG"),):
        x = rr["rules"][rule][g]
        put("Reroute%sNLost" % rn, str(x["n_lost"]))
put("RerouteSigItOwn", pct(rr["rules"]["signature"]["Italian"]["own_at_end"]))
# the input-by-input check that the routes of certified4 on the benchmark equal those of the rule of the paper
bi = rr["benchmark_identity"]
for sq, sn in (("five", "Five"), ("fifteen", "Fifteen")):
    tot_n = sum(bi[sq][w]["certified_vs_certified4"]["n"] for w in ("original", "light", "medium", "heavy"))
    for key, kn in (("differ_after_last_task", "Last"), ("differ_at_any_arrival_from_own_task_on", "Scored"), ("differ_at_any_arrival_incl_before_own_task", "Before")):
        tot = sum(bi[sq][w]["certified_vs_certified4"][key] for w in ("original", "light", "medium", "heavy"))
        put("RerouteId%s%s" % (sn, kn), str(tot))
        assert all(bi[sq][w]["nearest_vs_background"][key] == bi[sq][w]["certified_vs_certified4"][key] for w in bi[sq])
    put("RerouteId%sComparisons" % sn, thousands(tot_n))
for w_, wn in (("original", "Original"), ("light", "Light"), ("medium", "Medium"), ("heavy", "Heavy")):
    put("RerouteIdFifteenBefore%s" % wn, str(bi["fifteen"][w_]["certified_vs_certified4"]["differ_at_any_arrival_incl_before_own_task"]))
nh = bi["fifteen_radius_not_half_B"]
assert len(nh) == 1
put("RerouteIdFifteenOriginalN", thousands(bi["fifteen"]["original"]["certified_vs_certified4"]["n"]))
put("RerouteIdTask", nh[0]["task"].split("_", 1)[1].replace("_", "\\_"))
put("RerouteIdTaskPos", str(nh[0]["position"]))
put("RerouteIdEarlier", nh[0]["nearest_earlier_task"].split("_", 1)[1].replace("_", "\\_"))
put("RerouteIdRadius", "%.1f" % nh[0]["radius"])
put("RerouteIdHalfB", "%.1f" % nh[0]["half_B"])

# the part of the benchmark rewordings: tasks where B_k is the nearest unrelated signature (the radius is half of it)

# ------------------------------------------------------------------ exploration on the calibration tasks (numbers from PLAN.md, 2026-11-18 02:05)
# NOT read from a json file: the script of the exploration (explore/lex.py) was kept in the scratch folder; these constants are copied from
# the entry "a look at the words, on the calibration tasks only (an exploration, not a test)" of silent_adapters/PLAN.md (German pivot,
# 175 calibration tasks, nothing from the 700 evaluation tasks), and from the coordinator's report of the round-trip exploration.
EXPLORE = {"DropShare": "13.7", "DropRare": "33", "DropCommon": "13", "SiblingPairs": "43", "SiblingSharedMin": "60",
           "SiblingRare": "31", "SiblingFew": "55", "KeepAllOne": "49", "KeepAllTwo": "37", "SiblingSameOne": "12", "SiblingSameTwo": "5",
           "RoundTripIdentical": "55", "RoundTripFarFrom": "42", "RoundTripFarTo": "73"}
for k, v in EXPLORE.items():
    put("Explore" + k, v)

# ------------------------------------------------------------------ sixth idea: a zone decided sentence by sentence
sj = load("sentence_judge_study")
sc = sj["calibration"]
put("SentR", "%.2f" % sc["r"])
put("SentTau", "%.4f" % sc["tau"])
put("SentTauAlpha", "0.05")
put("SentRAlpha", "0.10")
put("SentJudge", sc["chosen"].split("/")[-1])
for jn, jk in (("Xsmall", "cross-encoder/nli-deberta-v3-xsmall"), ("Base", "cross-encoder/nli-deberta-v3-base"), ("Mnli", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")):
    put("SentAuc" + jn, "%.3f" % sc["judges"][jk]["auc"])
    put("SentNegSame" + jn, pct(sc["judges"][jk]["share_negatives_ge_tau"]))
    put("SentPosSame" + jn, pct(sc["judges"][jk]["share_positives_ge_tau"]))
    put("SentTau" + jn, "%.4f" % sc["judges"][jk]["tau"])
put("SentNegatives", str(sc["judges"][sc["chosen"]]["n_negatives"]))
put("SentPositives", str(sc["judges"][sc["chosen"]]["n_positives"]))
for piv, pn in (("German", "De"), ("French", "Fr"), ("Spanish", "Es"), ("Italian", "It")):
    t_ = sj["table"][piv]
    for k in "CZPMR":
        put("Sent%s%s" % (pn, k), pct(t_[k]))
put("SentC", pct(sj["table"]["originals"]["C"]))
for pk, pn in (("de", "De"), ("fr", "Fr"), ("es", "Es"), ("it", "It")):
    c_ = sj["candidates"][pk]
    put("SentWithin%s" % pn, "%.1f" % c_["tasks_within_r_mean"])
    put("SentNone%s" % pn, str(c_["texts_with_none_within_r"]))
    put("SentMs%s" % pn, "%.0f" % c_["mean_wall_ms_per_text_cached_batch"])
    put("SentPairs%s" % pn, thousands(c_["unique_sentence_pairs_judged"]))
put("SentMax", str(max(sj["candidates"][k]["max"] for k in ("de", "fr", "es", "it"))))
assert sj["verdict"]["outcome"] == "against" and sj["verdict"]["pivot"] == "Italian"
cl = sj["collisions"]
put("SentCollPairs", str(cl["evaluated"]))
put("SentCollNotSame", str(cl["n_not_same_definition"]))
put("SentCollPass", str(cl["n_score_ge_tau_not_same_definition"]))
sb = sj["benchmark"]["summary"]
for lv, ln in (("light", "Light"), ("medium", "Medium"), ("heavy", "Heavy")):
    put("SentBench%sZone" % ln, str(sb[lv]["in_own_zone"]))
    assert sb[lv]["routed_right_five"] == sb[lv]["n"] and sb[lv]["certified_wrong_five"] == 0 and sb[lv]["certified_wrong_fifteen"] == 0
put("SentBenchN", str(sb["light"]["n"]))
put("SentSeconds", "%.2f" % (sj["cost"]["mean_ms_per_text"] / 1000))
put("SentThreads", "3")
put("SentWithinMean", "%.1f" % sj["candidates"]["it"]["tasks_within_r_mean"])
put("SentWrongCerts", str(sj["examples"]["n_wrong_certificates"]))
put("SentFirstPair", "")
del N["SentFirstPair"]
def _tn(t):
    return t.split("_", 1)[1].replace("_", "\\_")
for nm, i_ in (("One", 0), ("Two", 1), ("Three", 2)):
    ex = sj["examples"]["first_12"][i_]
    put("SentEx%sTask" % nm, _tn(ex["task"]))
    put("SentEx%sClaim" % nm, _tn(ex["claimed_by"]))
    put("SentEx%sScore" % nm, "%.4f" % ex["score"])
put("SentVerdictFrOutcome", "against" if (sj["table"]["French"]["M"] > 0.09 or sj["table"]["French"]["P"] < 0.55 or sj["table"]["French"]["C"] > 0.25) else "partial")

# ------------------------------------------------------------------ numbers file
open(os.path.join(HERE, "tables", "numbers_widen.tex"), "w", encoding="utf-8").write(
    "".join("\\newcommand{\\%s%s}{%s}\n" % ("" if k.startswith("Reroute") else "Widen", k, N[k]) for k in sorted(N)))


# ------------------------------------------------------------------ table: main
def w(name, text):
    open(os.path.join(HERE, "tables", name + ".tex"), "w", encoding="utf-8").write(text)


def row(label, feature, piv, r, verdict):
    return "    %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (label, feature, piv, pct(r["C"]), pct(r["P"]), pct(r["M"]), pct(r["R"]), verdict)


m5 = rt["t5-base"]["matched"]
ruler_new = dict(m5["NEW_beta0.1"]["fr"]["rule_A"], C=rt["t5-base"]["C_ref"])
lines = [
    r"\begin{tabular}{llcrrrrl}",
    r"    \toprule",
    r"    & Features and zone & Pivot & \textsf{C} & \textsf{P} & \textsf{M} & \textsf{R} & Verdict \\",
    r"    \midrule",
    row("Paper", "T5-base word-vectors, weighted mean", "both", ref, "reference"),
    row("Paper", "T5-base word-vectors, weighted mean", "French", dict(C=ref["C"], **{k: pr["French"][k] for k in "PMR"}), "ref. for (2)--(4)"),
    r"    \midrule",
    row("(1)", "SmolLM2 word-vectors", "both", fe["reference"], "ref. for (1)"),
    row("(1)", "SmolLM2 last layer", "both", fe["chosen"], "against"),
    row("(2)", "T5-base, paraphrase ruler", "French", ruler_new, "against"),
    row("(3)", "MiniLM, ruler, $\\alpha = 0.1$", "French", dict(primary), "against"),
    row("(4)", "MiniLM and NLI judge", "French",
        {"C": jv["C_U"]["C"], **{k: fr[k] for k in "PMR"}}, "against"),
    row("(5)", "MiniLM, sentence-wise NLI judge", "Italian",
        {"C": sj["table"]["Italian"]["C"], **{k: sj["table"]["Italian"][k] for k in "PMR"}}, "against"),
    r"    \midrule",
    "    Bar (3)--(5) & \\multicolumn{2}{l}{fixed in advance} & $\\le %d$ & $\\ge %d$ & $\\le %d$ & & \\\\" % (round(100 * BAR_C), round(100 * BAR_P), round(100 * BAR_M)),
    r"    \bottomrule",
    r"\end{tabular}", ""]
w("widen_main", "\n".join(lines))

# ------------------------------------------------------------------ table: encoder, six settings
lines = [r"\begin{tabular}{llrrrrrrr}", r"    \toprule",
         r"    Metric & $\alpha$ & Pivot & \textsf{C} & \textsf{U} & \textsf{P} & \textsf{M} & \textsf{R} & \textsf{Z} \\", r"    \midrule"]
for vi, (v, vn) in enumerate(VAR):
    for ai, (a, an) in enumerate(ALPHA):
        for pi, (pv, pn) in enumerate(PIV):
            r = mi["table"][v][a][pv]
            first = "%s" % ("plain" if v == "plain" else "ruler") if (ai == 0 and pi == 0) else ""
            alpha = a if pi == 0 else ""
            star = r"\textbf{" if (v == "ruler" and a == "0.1" and pv == "French") else None
            cells = [pct(r[k]) for k in "CUPMRZ"]
            if star:
                cells = [star + c + "}" for c in cells]
            lines.append("    %s & %s & %s & %s \\\\" % (first, alpha, pv, " & ".join(cells)))
        if not (vi == 1 and ai == 2):
            lines.append(r"    \addlinespace[2pt]")
    if vi == 0:
        lines.append(r"    \midrule")
lines += [r"    \bottomrule", r"\end{tabular}", ""]
w("widen_encoder", "\n".join(lines))

# ------------------------------------------------------------------ table: judge threshold curve
lines = [r"\begin{tabular}{rrrrrr}", r"    \toprule", r"    $\tau$ & \textsf{C} & \textsf{Z} & \textsf{P} & \textsf{M} & \textsf{R} \\", r"    \midrule"]
for r in sweep:
    lines.append("    %.2f & %s & %s & %s & %s & %s \\\\" % (r["tau"], pct(r["C"]), pct(r["Z"]), pct(r["P"]), pct(r["M"]), pct(r["R"])))
lines += [r"    \bottomrule", r"\end{tabular}", ""]
w("widen_judge", "\n".join(lines))


# ------------------------------------------------------------------ table: the Spanish check
def cell(d):
    return "%s (%s to %s)" % (pct(d["mean"]), pct(d["min"]), pct(d["max"]))


lines = [r"\begin{tabular}{lrrrrrrrrrr}", r"    \toprule",
         r"    & & \multicolumn{3}{c}{French} & \multicolumn{3}{c}{German} & \multicolumn{3}{c}{Spanish} \\",
         r"    \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}",
         r"    & \textsf{C} & \textsf{P} & \textsf{M} & \textsf{R} & \textsf{P} & \textsf{M} & \textsf{R} & \textsf{P} & \textsf{M} & \textsf{R} \\", r"    \midrule"]
lines.append("    Paper's rule & %s & %s \\\\" % (pct(pr["C"]), " & ".join(pct(pr[p][k]) for p in ("French", "German", "Spanish") for k in "PMR")))
lines.append("    Encoder & %s & %s \\\\" % (pct(er["C"]), " & ".join(pct(er[p][k]) for p in ("French", "German", "Spanish") for k in "PMR")))
lines += [r"    \bottomrule", r"\end{tabular}", ""]
w("widen_spanish", "\n".join(lines))
lines = [r"\begin{tabular}{lcc}", r"    \toprule", r"    Spanish pivot & Paper's rule & Encoder \\", r"    \midrule"]
for k in "CPMR":
    cols = [sm[key]["C"] if k == "C" else sm[key]["Spanish"][k] for key in ("paper_rule", "encoder_ruler_0.2")]
    lines.append("    \\textsf{%s} & %s & %s \\\\" % (k, cell(cols[0]), cell(cols[1])))
lines += [r"    \bottomrule", r"\end{tabular}", ""]
w("widen_splits", "\n".join(lines))

# ------------------------------------------------------------------ table: colliding siblings
PAIRS = [("task1403_check_validity_date_mmddyyyy", "task1333_check_validity_date_ddmmyyyy"),
         ("task1446_farthest_integers", "task1445_closest_integers"),
         ("task1201_atomic_classification_xintent", "task1203_atomic_classification_xreact"),
         ("task064_all_elements_except_first_i", "task063_first_i_elements"),
         ("task367_synthetic_remove_floats", "task369_synthetic_remove_odds"),
         ("task1308_amazonreview_category_classification", "task1313_amazonreview_polarity_classification")]
coll = {(r["task"], r["claimed_by_oldest"]): r for r in col["first_15"]}
mcase = {(r["task"], r["routed_to"]): r for r in judge["m_cases_5e"]["first_12"]}
tx = lambda s: "\\texttt{%s}" % s.replace("_", "\\_")
lines = [r"\scriptsize\begin{tabular}{llcc}", r"    \toprule",
         r"    Task that arrives & Task whose zone claims it & Listed under & Judge score \\", r"    \midrule"]
for new, old in PAIRS:
    c = coll.get((new, old))
    mc = mcase.get((new, old))
    assert c or mc, (new, old)
    where = ", ".join(x for x, y in (("(3)", c), ("(4)", mc)) if y)
    lines.append("    %s & %s & %s & %s \\\\" % (tx(new.split("_", 1)[1]), tx(old.split("_", 1)[1]), where,
                                              "%.2f" % mc["judge_score"] if mc else "--"))
lines += [r"    \bottomrule", r"\end{tabular}", ""]
w("widen_siblings", "\n".join(lines))

# ------------------------------------------------------------------ figure (shared look of figstyle.py)
import figstyle as fs
fs.apply()
INK, AXIS, GRID = fs.INK, fs.AXIS, fs.GRID
XMAX = 50
# the alternative recognisers are not methods of the paper: their own hues from the shared palette, plus the paper's rule in ink
C_JUDGE, C_PLAIN, C_RULER, C_RULE = fs.CHARCOAL, fs.LIGHTGREEN, fs.GREEN, fs.INK
C_WV, C_HID, C_SENT = fs.VERMILION, fs.GOLD, fs.CRIMSON
fig, ax = plt.subplots(figsize=(5.5, 3.0))
ax.grid(axis="x", visible=True)
ax.add_patch(Rectangle((0, 100 * BAR_P), 100 * BAR_M, 100 - 100 * BAR_P, facecolor=fs.GREEN, alpha=0.15, edgecolor="none", zorder=1))
ax.text(100 * BAR_M + 1.2, 100 - 1.5, "target region: $\\mathrm{\\mathsf{P}}$ $\\geq$ %d%% and $\\mathrm{\\mathsf{M}}$ $\\leq$ %d%% (no point reaches it)" % (round(100 * BAR_P), round(100 * BAR_M)),
        fontsize=7.5, color=fs.INK2, va="top", ha="left", zorder=2)
pc = lambda x: 100 * x
# judge curve (French): one point per threshold; points beyond XMAX are not drawn
jx = [pc(r["M"]) for r in sweep]
jy = [pc(r["P"]) for r in sweep]
off = sum(1 for x in jx if x > XMAX)
ax.plot(jx[:-1], jy[:-1], color=C_JUDGE, marker="D", ms=3.8, lw=1.3, label="(4) NLI judge, %d thresholds" % len(sweep), zorder=3)
ax.plot(jx[-1:], jy[-1:], color=C_JUDGE, marker="D", ms=3.8, mfc="white", mec=C_JUDGE, mew=1.0, ls="none", zorder=3)   # strictest threshold: almost nothing covered
# encoder, three alphas
for v, vn, color, marker, label, lw in (("plain", "Plain", C_PLAIN, "s", "(3) MiniLM encoder, plain metric", 1.3),
                                         ("ruler", "Ruler", C_RULER, "o", "(3) MiniLM encoder, paraphrase ruler", 2.0)):
    xs = [pc(mi["table"][v][a]["French"]["M"]) for a, _ in ALPHA]
    ys = [pc(mi["table"][v][a]["French"]["P"]) for a, _ in ALPHA]
    ax.plot(xs, ys, color=color, marker=marker, ms=4.5, lw=lw, label=label, zorder=4)
    offs_ = {("plain", "0.05"): (1.5, 6.0), ("plain", "0.1"): (0.0, 5.5), ("plain", "0.2"): (-4.0, 7.0),
             ("ruler", "0.05"): (1.0, -9.5), ("ruler", "0.1"): (3.0, -14.0), ("ruler", "0.2"): (-2.5, -7.5)}
    for (a, _), x, y in zip(ALPHA, xs, ys):
        dx, dy = offs_[(v, a)]
        ax.annotate("$\\alpha$=%s" % a, (x, y), xytext=(x + dx, y + dy), fontsize=7, color=fs.INK2, ha="center", va="center", zorder=5,
                    arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.5, shrinkA=0, shrinkB=2), bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
# the paper's rule and the other attempts, French pivot where there is one
ax.plot(pc(tab["fr"]["rule_B"]["M"]), pc(tab["fr"]["rule_B"]["P"]), marker="P", ms=6.5, color=C_RULE, ls="none", label="paper's rule (T5-base word-vectors)", zorder=6)
ax.plot(pc(m5["NEW_beta0.1"]["fr"]["rule_A"]["M"]), pc(m5["NEW_beta0.1"]["fr"]["rule_A"]["P"]), marker="^", ms=5.5, color=C_WV, ls="none",
        label="(2) paraphrase ruler on word-vectors", zorder=6)
ax.plot(pc(fe["chosen"]["M"]), pc(fe["chosen"]["P"]), marker="v", ms=5.5, color=C_HID, ls="none",
        label="(1) hidden states of SmolLM2 (both pivots)", zorder=6)
ax.plot(pc(sj["table"]["French"]["M"]), pc(sj["table"]["French"]["P"]), marker="*", ms=9, color=C_SENT, ls="none",
        label="(5) NLI judge, sentence by sentence", zorder=6)
ax.set_xlim(-1, XMAX)
ax.set_ylim(0, 100)
ax.set_xlabel("$\\mathrm{\\mathsf{M}}$: paraphrased inputs certified for another task (wrong certificate), %")
ax.set_ylabel("$\\mathrm{\\mathsf{P}}$: paraphrased inputs certified\nfor their own task, %")
ax.tick_params(axis="x", length=2.5)
h, l = ax.get_legend_handles_labels()
fs.legend_above(fig, h, l, ncol=2, y=0.92)
fig.subplots_adjust(top=0.9)
fs.save(fig, "widen_tradeoff")
assert off == 1, "the caption of the trade-off figure says one point"
put("FigOffPlot", str(off))
open(os.path.join(HERE, "tables", "numbers_widen.tex"), "w", encoding="utf-8").write(
    "".join("\\newcommand{\\%s%s}{%s}\n" % ("" if k.startswith("Reroute") else "Widen", k, N[k]) for k in sorted(N)))
# table: the stress test with the scaled fallbacks (the new task's definition is the light or the medium rewording of the definition of task 2)
STRESS = [(r"\cert{}", "certified"), (r"\nearest{}", "nearest"), ("Length-scaled fallback", "normalised"), (r"Zones, then length-scaled", "certified3"),
          (r"\scaled{}", "background"), (r"\certscaled{}", "certified4")]
lines = [r"\begin{tabular}{lrrrr}", r"    \toprule", r"    Rule & original & light & medium & heavy \\"]
for nd, lab in (("light", "New definition: the light rewording"), ("medium", "New definition: the medium rewording")):
    lines += [r"    \midrule", r"    \multicolumn{5}{l}{%s} \\" % lab]
    for lab_, rule in STRESS:
        cw = rr["stress"][nd][rule]["changed_per_wording"]
        lines.append("    %s & %d & %d & %d & %d \\\\" % (lab_, cw["original"], cw["light"], cw["medium"], cw["heavy"]))
lines += [r"    \bottomrule", r"\end{tabular}", ""]
put("RerouteStressMedCertHeavy", str(rr["stress"]["medium"]["certified"]["changed_per_wording"]["heavy"]))
put("RerouteStressMedCertFourHeavy", str(rr["stress"]["medium"]["certified4"]["changed_per_wording"]["heavy"]))
put("RerouteStressMedExtra", str(rr["stress"]["medium"]["certified4"]["changed_per_wording"]["heavy"] - rr["stress"]["medium"]["certified"]["changed_per_wording"]["heavy"]))
open(os.path.join(HERE, "tables", "reroute_stress.tex"), "w", encoding="utf-8").write("\n".join(lines))

open(os.path.join(HERE, "tables", "numbers_widen.tex"), "w", encoding="utf-8").write(
    "".join("\\newcommand{\\%s%s}{%s}\n" % ("" if k.startswith("Reroute") else "Widen", k, N[k]) for k in sorted(N)))
print("wrote 9 tables, 1 figure and %d number commands" % len(N))
