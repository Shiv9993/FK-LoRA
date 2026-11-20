"""Number commands and tables for the results of 2026-11-18 (audited recogniser, no-definition runs, routes). Sections of the paper
that read from it: the setting without instructions, the joint recogniser, the audit and the rules with a bound in advance.

    python3 audit_results.py

Reads   silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks/small_files.zip   (a: InfLoRA, GainLoRA without definitions)
        silent_adapters/results/2026-11-17_kaggle_five_tasks_seed42/small_files.zip                     (b: the same runs with definitions, wordings)
        routing/hidden_rules_study.json     (c: rules J, L, claims, H, H' on the frozen T5-base states; orders; fourth fresh batch if present)
        routing/audit_study.json            (d: joint recogniser on the sentence encoder, benchmarks and third fresh batch)
        routing/noinstruction_study.json    (e: zone rule with three features)
        ../silent_adapters/noinstruction_routes.json, noinstruction_routes_15.json   (f: routes and audits of the GPU rules)
        routing/conflict_count.json         (g: texts that occur in two tasks)
        optional (h): the GPU run of our method without definitions (zip of the same layout as (a)) and the small trained model
                      certified_switch/demo_results/no_instruction_t5-small_seed0_ridge.json; used when they exist.
Writes  tables/numbers_audit.tex            one \\newcommand per number (\\Aud...), a comment line above each group says where it comes from
        tables/noinst_published.tex, noinst_published_pertask.tex, recogniser_features.tex, recogniser_dev.tex, recogniser_pertask.tex,
        tables/recogniser_orders.tex, recogniser_fresh.tex, conflict_count.tex, routes_gpu.tex
At the end every macro is compared with the number in the source file, which is opened again (check_all). Nothing is typed by hand
except the names of the rows and the wording of the headers. Percentages: one decimal, no % sign; scores: two decimals.
"""
import functools
import json
import os
import re
import sys
import zipfile
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PATHS = {
    "pub_zip": os.path.join(REPO, "silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks/small_files.zip"),
    "pub_member": "silent_adapters/superni_noinstruction5_published_t5base_e10/results.json",
    "defs_zip": os.path.join(REPO, "silent_adapters/results/2026-11-17_kaggle_five_tasks_seed42/small_files.zip"),
    "defs_dir": "silent_adapters/superni_short5_t5base_e10/",
    "hidden": os.path.join(HERE, "routing/hidden_rules_study.json"),
    "audit": os.path.join(HERE, "routing/audit_study.json"),
    "noinst": os.path.join(HERE, "routing/noinstruction_study.json"),
    "boundary": os.path.join(HERE, "routing/boundary_study.json"),
    "routes5": os.path.join(REPO, "silent_adapters/noinstruction_routes.json"),
    "routes15": os.path.join(REPO, "silent_adapters/noinstruction_routes_15.json"),
    "conflict": os.path.join(HERE, "routing/conflict_count.json"),
    "ours_zip": os.path.join(REPO, "silent_adapters/results/2026-11-18_kaggle_noinstruction_ours_five_tasks/small_files.zip"),
    "ours_seeds_zip": os.path.join(REPO, "silent_adapters/results/2026-11-18_kaggle_noinstruction_ours_five_tasks_seeds43_44/small_files.zip"),
    "pub_seeds_zip": os.path.join(REPO, "silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks_seeds43_44/small_files.zip"),
    "scale_zip": os.path.join(REPO, "silent_adapters/results/2026-11-18_kaggle_hidden_states_scale/small_files.zip"),
    "scale_member": "hidden_states_scale/results.json",
    "runu_zip": os.path.join(REPO, "silent_adapters/results/2026-11-18_kaggle_instance_routes_with_definitions/small_files.zip"),
    "runu_member": "silent_adapters/superni_instance_routes_defs5_t5base_e10/results.json",
    "full_zip": os.path.join(REPO, "silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks_e100/small_files.zip"),
    "full_member": "silent_adapters/superni_noinstruction5_published_t5base_e100/results.json",
    "fifteen_zip": os.path.join(REPO, "silent_adapters/results/2026-11-18_kaggle_noinstruction_ours_fifteen_tasks/small_files.zip"),
    "fifteen_member": "silent_adapters/superni_noinstruction15_ours_t5base_e10/results.json",
    "fifteen_def_zip": os.path.join(REPO, "silent_adapters/results/2026-11-17_kaggle_fifteen_tasks_seed42/small_files.zip"),
    "fifteen_def_member": "silent_adapters/superni_o1_t5base_e10_ours/results.json",
    "fullours_zip": os.path.join(REPO, "silent_adapters/results/2026-11-18_kaggle_noinstruction_ours_five_tasks_e100/small_files.zip"),
    "fullours_member": "silent_adapters/superni_noinstruction5_ours_t5base_e100/results.json",
    "runu_s_zip": os.path.join(REPO, "silent_adapters/results/2026-11-18_kaggle_instance_routes_with_definitions_seeds43_44/small_files.zip"),
    "runu_s_member": "silent_adapters/superni_instance_routes_defs5_t5base_e10_s4344/results.json",
    "defs3_zip": os.path.join(REPO, "silent_adapters/results/2026-11-17_kaggle_five_tasks_seeds42-44/small_files.zip"),
    "defs3_member": "silent_adapters/superni_short5_t5base_e10/results.json",
    "fulldef_zip": os.path.join(REPO, "silent_adapters/results/2026-11-18_kaggle_published_five_tasks_e100_with_definitions/small_files.zip"),
    "fulldef_member": "silent_adapters/superni_published5_withdef_t5base_e100/results.json",
    "small": os.path.join(REPO, "certified_switch/demo_results/no_instruction_t5-small_seed0_ridge.json"),
    "out": os.path.join(HERE, "tables"),
}
if os.environ.get("AUD_PATHS"):          # only for testing the optional parts with other files: AUD_PATHS='{"hidden": "...", "out": "..."}'
    PATHS.update(json.loads(os.environ["AUD_PATHS"]))
OUT = PATHS["out"]
REQUIRED = ("pub_zip", "defs_zip", "hidden", "audit", "noinst", "boundary", "routes5", "routes15", "conflict")


def _zip_json(zpath, member):
    with zipfile.ZipFile(zpath) as z:
        return json.load(z.open(member))


def load_sources(verbose=False):
    """Opens every source file. A missing required file stops the script; a missing optional one is skipped and said."""
    for k in REQUIRED:
        if not os.path.exists(PATHS[k]):
            sys.exit("missing source file (%s): %s" % (k, PATHS[k]))
    S = {
        "pub": _zip_json(PATHS["pub_zip"], PATHS["pub_member"]),
        "defs_results": _zip_json(PATHS["defs_zip"], PATHS["defs_dir"] + "results.json"),
        "defs_retest": _zip_json(PATHS["defs_zip"], PATHS["defs_dir"] + "retest.json"),
    }
    for k in ("hidden", "audit", "noinst", "boundary", "routes5", "routes15", "conflict"):
        S[k] = json.load(open(PATHS[k]))
    S["ours"] = None
    if os.path.exists(PATHS["ours_zip"]):
        with zipfile.ZipFile(PATHS["ours_zip"]) as z:
            for n in z.namelist():
                if n.endswith("results.json"):
                    d = json.load(z.open(n))
                    if "by_seed" in d:
                        S["ours"] = d
                        break
    # by seed: seed 42 from (a) and (h), seeds 43 and 44 from the optional files of the later runs (same layout)
    S["pub_seeds"] = dict(S["pub"]["by_seed"])
    S["ours_seeds"] = dict(S["ours"]["by_seed"]) if S["ours"] else {}
    for key, dst in (("pub_seeds_zip", "pub_seeds"), ("ours_seeds_zip", "ours_seeds")):
        if os.path.exists(PATHS[key]):
            with zipfile.ZipFile(PATHS[key]) as z:
                for n in z.namelist():
                    if n.endswith("results.json"):
                        d = json.load(z.open(n))
                        if "by_seed" in d:
                            for sd, v in d["by_seed"].items():
                                S[dst].setdefault(sd, v)
                            break
    S["scale"] = _zip_json(PATHS["scale_zip"], PATHS["scale_member"]) if os.path.exists(PATHS["scale_zip"]) else None
    S["runu"] = _zip_json(PATHS["runu_zip"], PATHS["runu_member"]) if os.path.exists(PATHS["runu_zip"]) else None
    S["full"] = _zip_json(PATHS["full_zip"], PATHS["full_member"]) if os.path.exists(PATHS["full_zip"]) else None
    S["fifteen"] = _zip_json(PATHS["fifteen_zip"], PATHS["fifteen_member"]) if os.path.exists(PATHS["fifteen_zip"]) else None
    S["fifteen_def"] = _zip_json(PATHS["fifteen_def_zip"], PATHS["fifteen_def_member"]) if os.path.exists(PATHS["fifteen_def_zip"]) else None
    S["fullours"] = _zip_json(PATHS["fullours_zip"], PATHS["fullours_member"]) if os.path.exists(PATHS["fullours_zip"]) else None
    S["runu_seeds"] = dict(S["runu"]["by_seed"]) if S["runu"] else {}
    if S["runu"] and os.path.exists(PATHS["runu_s_zip"]):
        for sd, v in _zip_json(PATHS["runu_s_zip"], PATHS["runu_s_member"])["by_seed"].items():
            S["runu_seeds"].setdefault(sd, v)
    S["defs3"] = _zip_json(PATHS["defs3_zip"], PATHS["defs3_member"])["by_seed"] if os.path.exists(PATHS["defs3_zip"]) else {}
    S["fulldef"] = _zip_json(PATHS["fulldef_zip"], PATHS["fulldef_member"]) if os.path.exists(PATHS["fulldef_zip"]) else None
    S["small"] = json.load(open(PATHS["small"])) if os.path.exists(PATHS["small"]) else None
    if verbose:
        print("optional sources:")
        print("  (h) our GPU run without definitions   :", "FOUND" if S["ours"] else "not found")
        print("  seeds 43, 44 without definitions      : published %s, ours %s" % (
            "FOUND" if {"43", "44"} <= set(S["pub_seeds"]) else "not found", "FOUND" if {"43", "44"} <= set(S["ours_seeds"]) else "not found"))
        print("  (j) feature job at scale (GPU)        :", "FOUND" if S["scale"] else "not found")
        print("  run U (with definitions, own routes)  :", "FOUND" if S["runu"] else "not found")
        print("  run D (published methods, 100 epochs) :", "FOUND" if S["full"] else "not found")
        print("  15 tasks without definitions (ours)   :", "FOUND" if S["fifteen"] else "not found")
        print("  run F (our method, 100 epochs)        :", "FOUND" if S["fullours"] else "not found")
        print("  run U seeds 43, 44 (with definitions) :", "FOUND" if {"43", "44"} <= set(S["runu_seeds"]) else "not found", "; published three seeds with definitions:", "FOUND" if {"42", "43", "44"} <= set(S["defs3"]) else "not found")
        print("  run E (published, definitions, 100 ep):", "FOUND" if S["fulldef"] else "not found")
        print("  (h) small trained model, ridge        :", "FOUND" if S["small"] else "not found")
        print("  (c) verdict_fresh4 (fourth batch)     :", "FOUND" if "verdict_fresh4" in S["hidden"] else "not found")
    return S


# ------------------------------------------------------------------ formatting (one place)
def _clean(s):
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "$-$")


def fmt(kind, raw):
    """same convention as bound_results.py: "%.1f" % (100 * x) for shares, "%.2f" for scores"""
    if kind == "pct1":
        return _clean("%.1f" % (100 * raw))
    if kind == "pct0":
        return _clean("%.0f" % (100 * raw))
    if kind == "sc2":
        return _clean("%.2f" % raw)
    if kind == "sc0":
        return _clean("%.0f" % raw)
    if kind == "sc1":
        return _clean("%.1f" % raw)
    if kind == "int":
        assert float(raw) == int(raw), raw
        if abs(int(raw)) >= 1000:      # every count of four digits or more gets the thousands separator
            return "{:,}".format(int(raw)).replace(",", "{,}")
        return str(int(raw))
    if kind == "thou":
        return "{:,}".format(int(raw)).replace(",", "{,}")
    if kind == "str":
        return str(raw)
    raise ValueError(kind)


def decimal_tie(kind, raw):
    """True if exact decimal rounding of the printed source number (half up) would give another digit than the float formatting"""
    nd = {"pct1": (1, 100), "pct0": (0, 100), "sc2": (2, 1), "sc1": (1, 1)}.get(kind)
    if nd is None:
        return False
    d = (Decimal(repr(float(raw))) * nd[1]).quantize(Decimal(1).scaleb(-nd[0]), rounding=ROUND_HALF_UP)
    return _clean(str(d)) != fmt(kind, raw)


NUM = "Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen".split()


def G(src, *path):
    """accessor: a function of the sources that follows a path of keys in one source"""
    return lambda S: functools.reduce(lambda o, k: o[k], path, S[src])


REG = {}          # name -> (value, accessor, kind)
GROUPS = []       # (comment, [names])
_cur = None


def group(comment):
    global _cur
    _cur = (comment, [])
    GROUPS.append(_cur)


def put(name, acc, kind, S):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    val = fmt(kind, acc(S))
    assert REG.get(name, (val,))[0] == val, "two values for " + name
    if name not in REG:
        _cur[1].append(name)
    REG[name] = (val, acc, kind)
    return "\\%s{}" % name


def bold(x):
    return r"\textbf{%s}" % x


def short_task(t):
    return re.sub(r"^(SuperNI/)?task\d+_", "", t)


# the five tasks of the GPU runs: (keyword in the folder name, macro name, short name of the row)
FIVE = (("evalution", "Relation", "relation extr."), ("sst2", "Sst", "sst2"), ("glucose", "Glucose", "glucose"),
        ("sentiment140", "Sentiment", "sentiment140"), ("emotion", "Emotion", "emotion"))
# the fifteen SuperNI tasks of order 1 (row names)
NAMES15 = {"1572": "samsum", "363": "sst2", "1290": "xsum", "181": "outcome extr.", "002": "quoref", "1510": "relation extr.",
           "639": "multi-woz", "1729": "personachat", "073": "commonsenseqa", "1590": "diplomacy", "748": "glucose",
           "511": "reddit tifu", "591": "sciq", "1687": "sentiment140", "875": "emotion"}
METH = (("inflora", "Inflora", "InfLoRA"), ("gainlora_inflora", "Gainlora", "GainLoRA"))
WORDS = (("original", "Original"), ("light", "Light"), ("medium", "Medium"), ("heavy", "Heavy"))
RULES2 = (("certified", "Certified"), ("nearest", "Nearest"), ("centre", "Centre"), ("key", "Key"), ("all", "All"))
OURS_RULES = (("all", "All"), ("centre", "Centre"), ("table:ridge", "Ridge"), ("table:joint", "Joint"), ("table:claims", "Claims"),
              ("table:handicap", "Handicap"), ("table:own", "Own"))
FIELDS = (("learned", "Learned", "sc2"), ("AP", "AP", "sc2"), ("FT", "FT", "sc2"), ("changed", "Changed", "int"))


HARR = r"\shortstack[r]{Sent to\\own task\\at arrival,\\\% $\uparrow$}"
HEND = r"\shortstack[r]{Sent to\\own task\\at the end,\\\% $\uparrow$}"
HCHG = r"\shortstack[r]{Route\\changed,\\\% $\downarrow$}"
HARRn = r"\shortstack[r]{Sent\\to own\\task\\at\\arrival,\\\% $\uparrow$}"
HENDn = r"\shortstack[r]{Sent\\to own\\task\\at the\\end,\\\% $\uparrow$}"
HEST = r"\shortstack[r]{Audit\\$H/m$, \%}"
HBND = r"\shortstack[r]{95\% bound,\\\%}"
HADV = r"\shortstack[r]{Bound in\\advance, \%}"


HL = r"\shortstack[r]{Score right\\after\\learning $\uparrow$}"
HAP = r"\shortstack[r]{Final\\score $\uparrow$}"
HFT = r"\shortstack[r]{Forgetting\\(points) $\downarrow$}"
HOWN = r"\shortstack[r]{Sent to own\\task, \% $\uparrow$}"


def HCH(n):
    return r"\shortstack[r]{Answers\\changed\\of \%s{} $\downarrow$}" % n


def _macro_value(cell, which=0):
    ms = re.findall(r"\\(Aud[A-Za-z]+)\{\}", cell)
    if len(ms) <= which:
        return None, None
    return ms[which], float(REG[ms[which]][0].replace("$-$", "-").replace("{,}", ""))


def rowbold(cells, idxs, d, name, label, which=0):
    """bold, within one row, the better of the cells idxs (same measure side by side); all ties are bold; which = which macro of a cell"""
    if not BEST_BOLD:
        return
    vals = []
    for j in idxs:
        nm, v = _macro_value(cells[j], which)
        if nm is not None:
            vals.append((j, nm, v))
    if len(vals) < 2:
        return
    best = max(v for _, _, v in vals) if d > 0 else min(v for _, _, v in vals)
    for j, nm, v in vals:
        if abs(v - best) < 1e-9:
            cells[j] = cells[j].replace("\\%s{}" % nm, "\\textbf{\\%s{}}" % nm, 1)
            BOLD.append((name, "row", label, j, nm))


def bolden(rows, dirs, name=""):
    """rows: ("MID",) or (label, cells, group, include). Bold, in every column with a direction (+1 higher is better, -1 lower is better),
    the best displayed value among the rows of the same group that are included (the ceiling row is not a method and is left out);
    all tied values are bold; a group with fewer than two values in a column is not marked. Computed from the displayed numbers, never typed."""
    isrow = lambda r: r[0] not in ("MID", "RAW")
    if not BEST_BOLD:
        return [list(r) if isrow(r) else r for r in rows]
    out = [list(r) if isrow(r) else r for r in rows]
    for j, d in enumerate(dirs):
        if d is None:
            continue
        for g in {r[2] for r in rows if isrow(r)}:
            vals = []
            for i, r in enumerate(rows):
                if isrow(r) and r[2] == g and r[3]:
                    nm, v = _macro_value(r[1][j])
                    if nm is not None:
                        vals.append((i, nm, v))
            if len(vals) < 2:
                continue
            best = max(v for _, _, v in vals) if d > 0 else min(v for _, _, v in vals)
            for i, nm, v in vals:
                if abs(v - best) < 1e-9:
                    out[i][1][j] = out[i][1][j].replace("\\%s{}" % nm, "\\textbf{\\%s{}}" % nm, 1)
                    BOLD.append((name, str(g), rows[i][0], j, nm))
    return out


BOLD = []
BEST_BOLD = False        # no "best value" bold any more (the author dropped it); bold now marks only the chosen variant (recogniser_dev, recogniser_features, shapes_grid)


def render(rows, dirs, name=""):
    lines = []
    for r in bolden(rows, dirs, name):
        lines.append(r"    \midrule" if r[0] == "MID" else r[1] if r[0] == "RAW" else "    %s & %s \\\\" % (r[0], " & ".join(r[1])))
    return lines


def cp_upper(c, m, level=0.95):
    """One-sided Clopper-Pearson upper bound on a share, c events of m: the p with P(Binomial(m, p) <= c) = 1 - level, i.e.
    scipy.stats.beta.ppf(level, c + 1, m - c); 1.0 if c == m. Written out here because scipy is not installed; it is compared with
    the upper95 stored in the json files wherever they have one (see check_cp)."""
    from math import exp, lgamma, log
    if c >= m:
        return 1.0
    lc = [lgamma(m + 1) - lgamma(i + 1) - lgamma(m - i + 1) for i in range(c + 1)]

    def cdf(p):
        return sum(exp(l + i * log(p) + (m - i) * log(1 - p)) for i, l in enumerate(lc))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf(mid) > 1 - level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _follow(S_, path):
    return functools.reduce(lambda o, k: o[k], path, S_)


def est_acc(*path):
    """audit estimate: mean over the tasks of c/m (path leads to a per_task dict with c and m)"""
    def f(S_):
        pt = _follow(S_, path).values()
        return sum(v["c"] / v["m"] for v in pt) / len(pt)
    return f


def bound_acc(*path, computed=False):
    """95% bound: mean over the tasks of upper95 (stored), or of the Clopper-Pearson bound computed from c and m"""
    def f(S_):
        pt = _follow(S_, path).values()
        return sum((cp_upper(v["c"], v["m"]) if computed else v["upper95"]) for v in pt) / len(pt)
    return f


def check_cp(S):
    """the bound computed here equals every upper95 stored per task"""
    n = 0
    for src, path in (("hidden", ("dev", "full")), ):
        for key, v in S[src]["dev"]["full"].items():
            for t in v["per_task"].values():
                assert abs(cp_upper(t["c"], t["m"]) - t["upper95"]) < 1e-9, (key, t)
                n += 1
    for src in ("routes5", "routes15"):
        for rule, v in S[src]["measures"].items():
            for t in v["per_task"].values():
                assert abs(cp_upper(t["c"], t["m"]) - t["upper95"]) < 1e-9, (src, rule, t)
                n += 1
    return n


SEEDNAME = {"42": "FortyTwo", "43": "FortyThree", "44": "FortyFour"}


def seeds_complete(S_):
    """seeds for which both the published methods and our method (without definitions) have a re-test"""
    out = []
    for sd in sorted(set(S_["pub_seeds"]) & set(S_["ours_seeds"])):
        try:
            S_["pub_seeds"][sd]["retest"]["inflora"]["table"]["original|official"]
            S_["pub_seeds"][sd]["retest"]["gainlora_inflora"]["table"]["original|official"]
            S_["ours_seeds"][sd]["retest"]["certified2"]["table"]
        except KeyError:
            continue
        out.append(sd)
    return out


def seed_rows(S_, kind, key):
    """the re-test row of one method (kind pub: inflora, gainlora_inflora) or of one of our rules (kind ours: all, centre, table:ridge, ...) in every complete seed"""
    rows = []
    for sd in seeds_complete(S_):
        if kind == "pub":
            rows.append(S_["pub_seeds"][sd]["retest"][key]["table"]["original|official"])
        else:
            rows.append(S_["ours_seeds"][sd]["retest"]["certified2"]["table"]["original|" + key])
    return rows


def seeds_section(S):
    """three-seed numbers (seeds 42, 43, 44) without definitions; with the seed-42 files alone it is one seed (\\AudSeedsN = 1)"""
    files = {}
    if not S["ours"] or not seeds_complete(S):
        print("seeds: our run without definitions not found, no seed table")
        return files
    sds = seeds_complete(S)
    group("seeds without definitions: (a) results.json and (h) results.json by_seed 42, and the optional files of seeds 43, 44; retest table[original|<rule>]")
    put("AudSeedsN", lambda S_: len(seeds_complete(S_)), "int", S)
    ROWS = [("pub", "inflora", "Inflora", "InfLoRA"), ("pub", "gainlora_inflora", "Gainlora", "GainLoRA"),
            ("ours", "all", "Plain", r"\plain{}"), ("ours", "centre", "Centre", r"\centre{}"),
            ("ours", "table:ridge", "Ridge", r"\recog{} (ours)"), ("ours", "table:handicap", "Handicap", r"\handicap{} (ours)"),
            ("ours", "table:own", "Own", "own task (ceiling)")]
    mean = lambda xs: sum(xs) / len(xs)
    tab = []
    for kind, key, nm, label in ROWS:
        try:
            seed_rows(S, kind, key)
        except KeyError:
            continue
        pre = "AudSeeds" + nm
        cells = {}
        for f, fn_ in (("learned", "Learned"), ("AP", "AP"), ("FT", "FT")):
            acc = lambda S_, kind=kind, key=key, f=f: mean([r[f] for r in seed_rows(S_, kind, key)])
            cells[fn_] = put(pre + fn_, acc, "sc2", S)
            put(pre + fn_ + "One", acc, "sc1", S)
        lo = put(pre + "APLow", lambda S_, kind=kind, key=key: min(r["AP"] for r in seed_rows(S_, kind, key)), "sc2", S)
        hi = put(pre + "APHigh", lambda S_, kind=kind, key=key: max(r["AP"] for r in seed_rows(S_, kind, key)), "sc2", S)
        ch = put(pre + "Changed", lambda S_, kind=kind, key=key: sum(r["changed"] for r in seed_rows(S_, kind, key)), "int", S)
        cm = lambda S_, kind=kind, key=key: sum(r["compared"] for r in seed_rows(S_, kind, key))
        tab.append((label, cells, lo, hi, ch, cm))
    comps = {c[5](S) for c in tab}
    assert len(comps) == 1, comps
    put("AudSeedsCompared", tab[0][5], "thou", S)
    # per seed: ridge against the better published method
    have_ridge = any(t[0] == r"\recog{} (ours)" for t in tab)
    if have_ridge:
        for sd in sds:
            sn = SEEDNAME[sd]
            put("AudSeedsRidgeAPSeed" + sn, lambda S_, sd=sd: S_["ours_seeds"][sd]["retest"]["certified2"]["table"]["original|table:ridge"]["AP"], "sc2", S)
            put("AudSeedsBestPubAPSeed" + sn, lambda S_, sd=sd: max(S_["pub_seeds"][sd]["retest"][m]["table"]["original|official"]["AP"] for m in ("inflora", "gainlora_inflora")), "sc2", S)
        put("AudSeedsRidgeAboveInAll", lambda S_: sum(1 for sd in seeds_complete(S_) if S_["ours_seeds"][sd]["retest"]["certified2"]["table"]["original|table:ridge"]["AP"] >
                                                      max(S_["pub_seeds"][sd]["retest"][m]["table"]["original|official"]["AP"] for m in ("inflora", "gainlora_inflora"))), "int", S)
    head = [r"{\footnotesize\setlength{\tabcolsep}{2.5pt}\begin{tabular}{lrrrr}", r"    \toprule",
            r"    Method, no definition & \shortstack[r]{Score after\\learning $\uparrow$} & \shortstack[r]{Final\\score $\uparrow$} & \shortstack[r]{Forgetting\\(points) $\downarrow$} & \shortstack[r]{Answers changed\\of \AudSeedsCompared{} $\downarrow$} \\", r"    \midrule"]
    rows3, rows_e100 = [], []
    ceiling_labels = {lab_ for kind_, key_, nm_, lab_ in ROWS if key_ == "table:own"}
    for label, cells, lo, hi, ch, _ in tab:
        ceiling = label in ceiling_labels      # the ceiling row is marked by its rule key (table:own), not by its text
        if label == r"\plain{}":
            rows3.append(("MID",))
            rows_e100.append(("MID",))
        row = (label, [cells["Learned"], "%s {\\scriptsize (%s--%s)}" % (cells["AP"], lo, hi), cells["FT"], ch], "seeds3", not ceiling)
        rows3.append(row)
        rows_e100.append(row)
        if label == r"\handicap{} (ours)" and "AudFullOursRidgeLearned" in REG:
            for lab_, mn in ((r"\recog{} (ours), 100 epochs", "Ridge"), (r"\handicap{} (ours), 100 epochs", "Handicap")):
                rows_e100.append((lab_, ["\\AudFullOurs%sLearned{}" % mn, "\\AudFullOurs%sAP{}" % mn, "\\AudFullOurs%sFT{}" % mn, "\\AudFullOurs%sChanged{} of \\AudFullCompared{}" % mn], "seed42_e100", True))
        if label == "GainLoRA" and "AudFullInfloraLearned" in REG:      # run D: 100 epochs, seed 42, directly under the GainLoRA row
            for lab_, mn in (("InfLoRA, 100 epochs", "Inflora"), ("GainLoRA, 100 epochs", "Gainlora")):
                rows_e100.append((lab_, ["\\AudFull%sLearned{}" % mn, "\\AudFull%sAP{}" % mn, "\\AudFull%sFT{}" % mn, "\\AudFull%sChanged{} of \\AudFullCompared{}" % mn], "seed42_e100", True))
            if "AudFullDefOriginalInfloraLearned" in REG:        # run E: the same methods with definitions, 100 epochs
                for lab_, mn in (("InfLoRA, 100 epochs, with definitions", "Inflora"), ("GainLoRA, 100 epochs, with definitions", "Gainlora")):
                    rows_e100.append((lab_, ["\\AudFullDefOriginal%sLearned{}" % mn, "\\AudFullDefOriginal%sAP{}" % mn, "\\AudFullDefOriginal%sFT{}" % mn, "\\AudFullDefOriginal%sChanged{} of \\AudFullDefCompared{}" % mn], "seed42_e100", True))
    tail = [r"    \bottomrule", r"\end{tabular}}", ""]
    dirs = [1, 1, -1, -1]
    files["noinst_seeds.tex"] = "\n".join(head + render(rows3, dirs, "noinst_seeds") + tail)
    if "AudFullInfloraLearned" in REG:
        head_e100 = [head[0], head[1], head[2].replace("Method, no definition", "Method"), head[3]]
        files["noinst_seeds_e100.tex"] = "\n".join(head_e100 + render(rows_e100, dirs, "noinst_seeds_e100") + tail)
    return files


SCALE_SEQ = (("superni_order1", "SuperniOne", "SuperNI order 1"), ("superni_order2", "SuperniTwo", "SuperNI order 2"),
             ("long_order3", "LongThree", "Long Sequence order 3"), ("long_order4", "LongFour", "Long Sequence order 4"),
             ("fresh30", "FreshThirty", "fresh batch 1"), ("fresh60", "FreshSixty", "fresh batches 1+2"),
             ("fresh90", "FreshNinety", "fresh batches 1+2+3"), ("all120", "AllHundredTwenty", "all"))


def scale_section(S):
    """feature job on the GPU: frozen T5-base states, second benchmark and up to 120 tasks (hidden_states_scale/results.json)"""
    files = {}
    if not S["scale"]:
        print("scale: feature job on the GPU not found, no scale tables and no figure")
        return files
    R = S["scale"]
    SQ = ("scale", "sequences")
    group("(j) hidden_states_scale/results.json fingerprint_check")
    put("AudScaleFingerprintMatch", G("scale", "fingerprint_check", "matching"), "int", S)
    put("AudScaleFingerprintTotal", G("scale", "fingerprint_check", "of"), "int", S)
    assert not R["fingerprint_check"]["data_differs"] and not R["fingerprint_check"]["download_failed"]
    seqs = [(k, nm, lab) for k, nm, lab in SCALE_SEQ] + [(k + "_random", nm + "Rand", lab + ", random split") for k, nm, lab in SCALE_SEQ[4:]]
    for key, nm, lab in seqs:
        group("(j) hidden_states_scale/results.json sequences[%s]: rules L, H' (t = 2) and J (<= 30 tasks); means over tasks; audit estimate = mean of c/m" % key)
        pre = "AudScale" + nm
        sq = R["sequences"][key]
        put(pre + "Tasks", lambda S_, key=key: len(S_["scale"]["sequences"][key]["tasks"]), "int", S)
        for f, n_ in (("own_at_arrival", "Arrive"), ("own_at_end", "End"), ("changed", "Changed"), ("changed_ever", "ChangedEver"), ("certificate", "AuditLine")):
            put(pre + n_, G(*SQ, key, "L", "mean", f), "pct1", S)
        put(pre + "AuditEst", est_acc(*SQ, key, "L", "per_task"), "pct1", S)
        put(pre + "AuditBound", bound_acc(*SQ, key, "L", "per_task"), "pct1", S)
        assert abs(sum(v["upper95"] for v in sq["L"]["per_task"].values()) / len(sq["L"]["per_task"]) - sq["L"]["mean"]["upper95"]) < 1e-9
        for f, n_ in (("own_at_end", "HEnd"), ("changed", "HChanged"), ("handicap_bound", "HBound"), ("takes_of_own_heldout", "HTakes")):
            put(pre + n_, G(*SQ, key, "H_prime_t2", "mean", f), "pct1", S)
        if isinstance(sq["J"], dict):
            put(pre + "JEnd", G(*SQ, key, "J", "mean", "own_at_end"), "pct1", S)
    group("(j) sequences[long_order3|long_order4].reading_rule_L: reading of the second benchmark")
    for key, nm in (("long_order3", "Three"), ("long_order4", "Four")):
        if "reading_rule_L" in R["sequences"][key]:
            put("AudScaleLong%sUse" % nm, G(*SQ, key, "reading_rule_L", "use"), "str", S)
            put("AudScaleLong%sAudit" % nm, G(*SQ, key, "reading_rule_L", "audit"), "str", S)
    group("(j) sequences[long_order3]: rules L and H' per task (mnli, rte), and the number of tasks with own_at_end >= 99%")
    for t, tn in (("mnli", "Mnli"), ("rte", "Rte"), ("cb", "Cb")):
        k = "Long_Sequence/" + t
        for f, n_, kind in (("own_at_arrival", "Arrive", "pct1"), ("own_at_end", "End", "pct1"), ("changed", "Changed", "pct1"), ("c", "C", "int"), ("m", "M", "int")):
            put("AudScaleLong%s%s" % (tn, n_), G(*SQ, "long_order3", "L", "per_task", k, f), kind, S)
        put("AudScaleLong%sHEnd" % tn, G(*SQ, "long_order3", "H_prime_t2", "per_task", k, "own_at_end"), "pct1", S)
    put("AudScaleLongHigh", lambda S_: sum(1 for v in S_["scale"]["sequences"]["long_order3"]["L"]["per_task"].values() if v["own_at_end"] >= 0.99), "int", S)
    put("AudScaleLongTasks", lambda S_: len(S_["scale"]["sequences"]["long_order3"]["tasks"]), "int", S)
    group("(j) fresh_split_comparison[<sequence>_random - <sequence>]: largest absolute difference over the four pairs, percentage points")
    pairs = [k for k in R["fresh_split_comparison"] if k != "about"]
    assert len(pairs) == 4
    for rule, rn in (("L", "L"), ("H_prime_t2", "H")):
        for f, n_ in (("own_at_end", "End"), ("changed", "Changed"), ("changed_ever", "ChangedEver")):
            put("AudScaleSplitMaxDiff%s%s" % (rn, n_), lambda S_, rule=rule, f=f: max(abs(S_["scale"]["fresh_split_comparison"][k][rule][f]) for k in pairs), "pct1", S)
    for f, n_ in (("own_at_end", "End"), ("changed", "Changed"), ("changed_ever", "ChangedEver")):
        put("AudScaleSplitMaxDiff" + n_, lambda S_, f=f: max(abs(S_["scale"]["fresh_split_comparison"][k][rule][f]) for k in pairs for rule in ("L", "H_prime_t2")), "pct1", S)

    group("(j) sequences[all120] per task: the two tasks built on the same texts (circa), older = the one that arrives first in the sequence")
    PAIRT = ("task567_circa_text_generation", "task565_circa_answer_generation")

    def pair_task(S_, which):
        sq_ = S_["scale"]["sequences"]["all120"]["L"]["per_task"]
        idx = sorted((list(sq_).index([k for k in sq_ if k.endswith(t)][0]), [k for k in sq_ if k.endswith(t)][0]) for t in PAIRT)
        return idx[0][1] if which == "Older" else idx[1][1]
    for which in ("Older", "Newer"):
        for rule, rn in (("L", "L"), ("H_prime_t2", "H")):
            put("AudScalePair%s%sEnd" % (which, rn), lambda S_, w=which, rule=rule: S_["scale"]["sequences"]["all120"][rule]["per_task"][pair_task(S_, w)]["own_at_end"], "pct1", S)
    put("AudScalePairOlderLArrival", lambda S_: S_["scale"]["sequences"]["all120"]["L"]["per_task"][pair_task(S_, "Older")]["own_at_arrival"], "pct1", S)

    # ---- table: recogniser_scale.tex
    L = [r"{\scriptsize\setlength{\tabcolsep}{2.5pt}\begin{tabular}{p{0.85in}rrrrrrrr}", r"    \toprule",
         r"    & & \multicolumn{5}{c}{\recog{}} & \multicolumn{2}{c}{\handicap{}, $\nu = 2$} \\", r"    \cmidrule(lr){3-7}\cmidrule(l){8-9}",
         r"    Sequence & Tasks & " + HARRn + " & " + HENDn + " & " + HCHG + " & " + HEST + " & " + HBND + " & " + HENDn + " & " + HCHG + r" \\", r"    \midrule"]
    for key, nm, lab in SCALE_SEQ:
        pre = "\\AudScale" + nm
        c = [pre + x + "{}" for x in ("Tasks", "Arrive", "End", "Changed", "AuditEst", "AuditBound", "HEnd", "HChanged")]
        rowbold(c, (2, 6), 1, "recogniser_scale", lab)
        rowbold(c, (3, 7), -1, "recogniser_scale", lab)
        L.append("    %s & %s \\\\" % (lab, " & ".join(c)))
    L += [r"    \bottomrule", r"\end{tabular}}", ""]
    files["recogniser_scale.tex"] = "\n".join(L)

    # ---- table: recogniser_long_pertask.tex
    group("(j) sequences[long_order3] per task (order of arrival): rule L own_at_arrival, own_at_end, changed, c, m, c/m, upper95; rule H' own_at_end")
    L = [r"{\footnotesize\setlength{\tabcolsep}{2.5pt}\begin{tabular}{lrrrrrrr}", r"    \toprule",
         r"    & \multicolumn{6}{c}{\recog{}} & \handicap{} \\", r"    \cmidrule(lr){2-7}",
         r"    Task & " + HARR + " & " + HEND + " & " + HCHG + r" & Held-out & " + HEST + " & " + HBND + " & " + HEND + r" \\", r"    \midrule"]
    tasks3 = list(R["sequences"]["long_order3"]["L"]["per_task"])
    assert tasks3 == R["sequences"]["long_order3"]["tasks"] and len(tasks3) == 15
    for i, t in enumerate(tasks3):
        n_ = NUM[i + 1]
        pt = lambda f, i=i, rule="L": (lambda S_: S_["scale"]["sequences"]["long_order3"][rule]["per_task"][S_["scale"]["sequences"]["long_order3"]["tasks"][i]][f])
        c = [put("AudScaleLongPt%s%s" % (n_, nn), pt(f), "pct1", S) for f, nn in (("own_at_arrival", "Arrive"), ("own_at_end", "End"), ("changed", "Changed"))]
        cc = put("AudScaleLongPt%sC" % n_, pt("c"), "int", S)
        mm = put("AudScaleLongPt%sM" % n_, pt("m"), "int", S)
        c.append("%s of %s" % (cc, mm))
        c.append(put("AudScaleLongPt%sAuditEst" % n_, lambda S_, i=i: (lambda v: v["c"] / v["m"])(S_["scale"]["sequences"]["long_order3"]["L"]["per_task"][S_["scale"]["sequences"]["long_order3"]["tasks"][i]]), "pct1", S))
        c.append(put("AudScaleLongPt%sBound" % n_, pt("upper95"), "pct1", S))
        c.append(put("AudScaleLongPt%sHEnd" % n_, pt("own_at_end", rule="H_prime_t2"), "pct1", S))
        rowbold(c, (1, 6), 1, "recogniser_long_pertask", t.split("/")[1])
        L.append("    %s & %s \\\\" % (t.split("/")[1], " & ".join(c)))
    L += [r"    \bottomrule", r"\end{tabular}}", ""]
    files["recogniser_long_pertask.tex"] = "\n".join(L)

    # ---- figure: recogniser_scale.pdf (shared look of figstyle.py)
    import figstyle as fs
    fs.apply()
    # the 15-task point is a different collection (SuperNI order 1); the points at 30, 60, 90 and 120 tasks are the fresh batches
    # 1, 1+2, 1+2+3 and everything (15 + 15 + 90 tasks)
    order = ("superni_order1", "fresh30", "fresh60", "fresh90", "all120")
    xs = [len(R["sequences"][k]["tasks"]) for k in order]
    sq = lambda k, rule, f: 100 * R["sequences"][k][rule]["mean"][f]
    fig, (a1, a2) = fs.plt.subplots(1, 2, figsize=(5.5, 2.35))
    ends = {}
    for rule, name in (("L", "FinderLoRA"), ("H_prime_t2", "KeeperLoRA")):
        for a, f in ((a1, "own_at_end"), (a2, "changed")):
            ys = [sq(k, rule, f) for k in order]
            a.plot(xs, ys, color=fs.color(name), marker=fs.marker(name), ms=4.5, lw=2.0 if name == "FinderLoRA" else 1.3, zorder=3)
            ends[a, name] = ys[-1]
    bound = [sq(k, "L", "upper95") for k in order]
    a2.plot(xs, bound, color=fs.MUTED, lw=1.0, ls=(0, (4, 2)), zorder=2)
    a2.fill_between(xs, [sq(k, "L", "changed") for k in order], bound, color=fs.color("FinderLoRA"), alpha=0.15, lw=0, zorder=1)
    for a, ylo, yhi, gap in ((a1, 85, 100, 2.2), (a2, 0, 10, 1.4)):
        names = ["FinderLoRA", "KeeperLoRA"] + (["95% bound"] if a is a2 else [])
        yy = [ends[a, n] for n in names[:2]] + ([bound[-1]] if a is a2 else [])
        for n, ty in zip(names, fs.spread(yy, gap, lo=ylo + 0.4, hi=yhi - 0.4)):
            a.annotate(n, xy=(xs[-1], yy[names.index(n)]), xytext=(xs[-1] + 7, ty), ha="left", va="center", fontsize=7.5, color=fs.INK,
                       annotation_clip=False, arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.5, shrinkA=1, shrinkB=3))
        a.set_xlabel("number of tasks")
        a.set_xticks(xs)
        a.set_xlim(6, 178)
        a.spines["bottom"].set_bounds(6, 128)
        a.tick_params(axis="x", length=2.5)
    a1.set_ylabel("own task after the last task, %")
    a2.set_ylabel("routes changed, %")
    a1.set_ylim(85, 100)
    a2.set_ylim(0, 10)
    fs.panel_title(a1, "a", "own task kept")
    fs.panel_title(a2, "b", "routes changed")
    fig.subplots_adjust(wspace=0.5)
    fs.save(fig, "recogniser_scale")
    return files


def runu_section(S):
    """run U: five tasks with definitions, plain adapters, routes from the example alone (results.json by_seed 42, retest certified2)"""
    files = {}
    if not S["runu"]:
        print("run U not found, no instance_routes tables")
        return files
    T = ("runu", "by_seed", "42", "retest", "certified2", "table")
    RU = (("Cert", "certified"), ("Ridge", "table:ridge"), ("Handicap", "table:handicap"), ("Own", "table:own"),
          ("MixRidge", "zone+table:ridge"), ("MixHandicap", "zone+table:handicap"), ("Centre", "centre"), ("Plain", "all"))
    group("(k) run U: instance_routes_with_definitions results.json by_seed 42, retest[certified2].table[<wording>|<rule>]: five tasks with definitions, routes from the example alone")
    for w, wn in WORDS:
        for rn, rule in RU:
            key = "%s|%s" % (w, rule)
            pre = "AudU%s%s" % (wn, rn)
            put(pre + "AP", G(*T, key, "AP"), "sc2", S)
            put(pre + "APOne", G(*T, key, "AP"), "sc1", S)
            put(pre + "Changed", G(*T, key, "changed"), "int", S)
            put(pre + "FT", G(*T, key, "FT"), "sc2", S)
            if S["runu"]["by_seed"]["42"]["retest"]["certified2"]["table"][key].get("right_task") is not None:
                put(pre + "Right", G(*T, key, "right_task"), "pct1", S)
            assert S["runu"]["by_seed"]["42"]["retest"]["certified2"]["table"][key]["compared"] == 400
    put("AudUCompared", G(*T, "original|certified", "compared"), "int", S)
    gap = lambda rule, w: (lambda S_: S_["runu"]["by_seed"]["42"]["retest"]["certified2"]["table"][w + "|table:own"]["AP"] - S_["runu"]["by_seed"]["42"]["retest"]["certified2"]["table"][w + "|" + rule]["AP"])
    for rn, rule in (("Ridge", "table:ridge"), ("Handicap", "table:handicap")):
        put("AudU%sBelowOwnLow" % rn, lambda S_, rule=rule: min(gap(rule, w)(S_) for w, _ in WORDS), "sc1", S)
        put("AudU%sBelowOwnHigh" % rn, lambda S_, rule=rule: max(gap(rule, w)(S_) for w, _ in WORDS), "sc1", S)
    put("AudUHeavyCertBelowOwn", gap("certified", "heavy"), "sc1", S)
    # the certified rows of run U are the certified2 rows of the earlier run (source b), to two decimals

    def same(S_):
        for w, _ in WORDS:
            a = S_["runu"]["by_seed"]["42"]["retest"]["certified2"]["table"][w + "|certified"]
            b = S_["defs_retest"]["certified2"]["table"][w + "|certified"]
            if any(fmt("sc2", a[f]) != fmt("sc2", b[f]) for f in ("learned", "AP", "FT")) or a["changed"] != b["changed"]:
                return "no"
        return "yes"
    put("AudUCertSameAsEarlier", same, "str", S)
    print("run U: certified rows equal the earlier run (source b):", same(S))
    # table: the published methods are read from the existing \\AudWord... macros (same seed, earlier run)
    for w, wn in WORDS:
        for mn in ("Gainlora", "Inflora"):
            assert "AudWord%s%sAP" % (wn, mn) in REG and "AudWord%s%sChanged" % (wn, mn) in REG
    ROWS = ((r"\cert{} (zones on the instruction)", "Cert"), (r"\recog{} (example alone)", "Ridge"), (r"\handicap{}, $\nu = 2$ (example alone)", "Handicap"),
            ("every input to its own task (ceiling)", "Own"), (r"zone first, else \recog{}", "MixRidge"), (r"zone first, else \handicap{}", "MixHandicap"),
            (r"\centre{}", "Centre"), (r"\plain{}", "Plain"))
    PUB = (("GainLoRA", "Gainlora"), ("InfLoRA", "Inflora"))
    m_ = lambda n: "\\%s{}" % n

    # ---- three seeds (42, 43, 44) of run U, and of the published methods with definitions (2026-11-17 runs)
    seeds_u = lambda S_: sorted(S_["runu_seeds"])
    urow = lambda S_, w, rule: [S_["runu_seeds"][sd]["retest"]["certified2"]["table"][w + "|" + rule] for sd in seeds_u(S_)]
    mean_ = lambda xs: sum(xs) / len(xs)
    group("(k2) run U, seeds 42, 43, 44 (results.json by_seed of the three files), retest[certified2].table[<wording>|<rule>]: mean over the seeds, changed answers summed")
    put("AudUThreeSeeds", lambda S_: len(seeds_u(S_)), "int", S)
    for w, wn in WORDS:
        for rn, rule in RU:
            pre = "AudUThree%s%s" % (wn, rn)
            ap = lambda S_, w=w, rule=rule: mean_([r["AP"] for r in urow(S_, w, rule)])
            put(pre + "AP", ap, "sc2", S)
            put(pre + "APOne", ap, "sc1", S)
            put(pre + "Changed", lambda S_, w=w, rule=rule: sum(r["changed"] for r in urow(S_, w, rule)), "int", S)
            if rn in ("Cert", "Ridge", "Handicap"):
                put(pre + "APLow", lambda S_, w=w, rule=rule: min(r["AP"] for r in urow(S_, w, rule)), "sc2", S)
                put(pre + "APHigh", lambda S_, w=w, rule=rule: max(r["AP"] for r in urow(S_, w, rule)), "sc2", S)
            if all(r.get("right_task") is not None for r in urow(S, w, rule)):
                put(pre + "Right", lambda S_, w=w, rule=rule: mean_([r["right_task"] for r in urow(S_, w, rule)]), "pct1", S)
    put("AudUThreeCompared", lambda S_: sum(r["compared"] for r in urow(S_, "original", "certified")), "thou", S)
    mapx = lambda S_, w, rule: mean_([r["AP"] for r in urow(S_, w, rule)])
    put("AudUThreeHeavyRidgeAboveCert", lambda S_: mapx(S_, "heavy", "table:ridge") - mapx(S_, "heavy", "certified"), "sc1", S)
    put("AudUThreeOriginalRidgeBelowCert", lambda S_: mapx(S_, "original", "certified") - mapx(S_, "original", "table:ridge"), "sc1", S)
    put("AudUThreeMaxChangedPerSeed", lambda S_: max(r["changed"] for w, _ in WORDS for r in urow(S_, w, "table:ridge")), "int", S)
    gap3 = lambda rule, w: (lambda S_: mapx(S_, w, "table:own") - mapx(S_, w, rule))
    for rn, rule in (("Ridge", "table:ridge"), ("Handicap", "table:handicap")):
        put("AudUThree%sBelowOwnLow" % rn, lambda S_, rule=rule: min(gap3(rule, w)(S_) for w, _ in WORDS), "sc1", S)
        put("AudUThree%sBelowOwnHigh" % rn, lambda S_, rule=rule: max(gap3(rule, w)(S_) for w, _ in WORDS), "sc1", S)
    put("AudUThreeHeavyCertBelowOwn", gap3("certified", "heavy"), "sc1", S)
    pub3 = bool(S["defs3"]) and all(sd in S["defs3"] for sd in seeds_u(S))
    if pub3:
        group("(k2) published methods with definitions, seeds 42, 43, 44 (2026-11-17 runs, results.json by_seed retest[<method>].table[<wording>|official]): mean AP, changed summed")
        prow = lambda S_, m, w: [S_["defs3"][sd]["retest"][m]["table"][w + "|official"] for sd in seeds_u(S_)]
        for w, wn in WORDS:
            for m, mn, _ in METH:
                put("AudUThree%s%sAP" % (wn, mn), lambda S_, m=m, w=w: mean_([r["AP"] for r in prow(S_, m, w)]), "sc2", S)
                put("AudUThree%s%sChanged" % (wn, mn), lambda S_, m=m, w=w: sum(r["changed"] for r in prow(S_, m, w)), "int", S)
    else:
        print("published methods with definitions for all seeds not available: no published rows in the three-seed tables")

    def full_table(P, Q, cmp_macro, name):
        head = [r"{\scriptsize\setlength{\tabcolsep}{2pt}\begin{tabular}{p{1.2in}rrrrrrrrr}", r"    \toprule",
                r"    & \multicolumn{4}{c}{Final score $\uparrow$} & \multicolumn{4}{c}{\shortstack{Answers changed\\of @CMP@{} $\downarrow$}} & \shortstack[r]{Sent to own\\task, heavy, \% $\uparrow$} \\".replace("@CMP@", "\\" + cmp_macro),
                r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
                r"    & original & light & medium & heavy & original & light & medium & heavy & \\", r"    \midrule"]
        rows = []
        for lab, rn in ROWS:
            c = [m_(P(wn, rn, "AP")) for _, wn in WORDS] + [m_(P(wn, rn, "Changed")) for _, wn in WORDS]
            c.append(m_(P("Heavy", rn, "Right")) if P("Heavy", rn, "Right") in REG else "--")
            rows.append((lab, c, "all", rn != "Own"))
        if Q:
            rows.append(("MID",))
            for lab, mn in PUB:
                rows.append((lab, [m_(Q(wn, mn, "AP")) for _, wn in WORDS] + [m_(Q(wn, mn, "Changed")) for _, wn in WORDS] + ["--"], "all", True))
        return "\n".join(head + render(rows, [1] * 4 + [-1] * 4 + [1], name) + [r"    \bottomrule", r"\end{tabular}}", ""])

    def short_tab(P, Q, cmp_macro, name):
        head = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrr}", r"    \toprule",
                r"    & \multicolumn{2}{c}{Final score $\uparrow$} & \multicolumn{2}{c}{\shortstack{Answers changed\\of @CMP@{} $\downarrow$}} \\".replace("@CMP@", "\\" + cmp_macro), r"    \cmidrule(lr){2-3}\cmidrule(l){4-5}",
                r"    & original & heavy & original & heavy \\", r"    \midrule"]
        rows = [(lab, [m_(x) for x in (P("Original", rn, "AP"), P("Heavy", rn, "AP"), P("Original", rn, "Changed"), P("Heavy", rn, "Changed"))], "all", rn != "Own") for lab, rn in ROWS[:4]]
        if Q:
            rows.append(("MID",))
            for lab, mn in PUB:
                rows.append((lab, [m_(x) for x in (Q("Original", mn, "AP"), Q("Heavy", mn, "AP"), Q("Original", mn, "Changed"), Q("Heavy", mn, "Changed"))], "all", True))
        return "\n".join(head + render(rows, [1, 1, -1, -1], name) + [r"    \bottomrule", r"\end{tabular}}", ""])
    P1 = lambda wn, rn, f: "AudU%s%s%s" % (wn, rn, f)
    Q1 = lambda wn, mn, f: "AudWord%s%s%s" % (wn, mn, f)
    P3 = lambda wn, rn, f: "AudUThree%s%s%s" % (wn, rn, f)
    Q3 = (lambda wn, mn, f: "AudUThree%s%s%s" % (wn, mn, f)) if pub3 else None
    files["instance_routes.tex"] = full_table(P1, Q1, "AudUCompared", "instance_routes")
    files["instance_routes_seeds.tex"] = full_table(P3, Q3, "AudUThreeCompared", "instance_routes_seeds")
    files["instance_routes_short.tex"] = short_tab(P3, Q3, "AudUThreeCompared", "instance_routes_short")
    return files


def fullours_macros(S):
    """run F: our method without definitions at 100 epochs, five tasks, seed 42 (results.json by_seed 42, retest[certified2].table[original|<rule>])"""
    if not S["fullours"]:
        print("run F not found")
        return
    R = S["fullours"]
    assert R["cfg"]["epochs"] == 100 and list(R["by_seed"]) == ["42"]
    T = ("fullours", "by_seed", "42", "retest", "certified2", "table")
    group("(o) run F: noinstruction_ours_five_tasks_e100 results.json by_seed 42, retest[certified2].table[original|<rule>]: our method, 100 epochs, no definition")
    for rn, rule in (("Ridge", "table:ridge"), ("Handicap", "table:handicap"), ("Own", "table:own"), ("Centre", "centre"), ("Plain", "all")):
        key = "original|" + rule
        for f, fn_, k in FIELDS:
            put("AudFullOurs%s%s" % (rn, fn_), G(*T, key, f), k, S)
        put("AudFullOurs%sRight" % rn, G(*T, key, "right_task"), "pct1", S)
        assert R["by_seed"]["42"]["retest"]["certified2"]["table"][key]["compared"] == 400
    put("AudFullOursRidgeAPOne", G(*T, "original|table:ridge", "AP"), "sc1", S)
    put("AudFullOursHandicapAPOne", G(*T, "original|table:handicap", "AP"), "sc1", S)
    put("AudFullOursRidgeLearnedOne", G(*T, "original|table:ridge", "learned"), "sc1", S)
    put("AudFullOursOwnAPOne", G(*T, "original|table:own", "AP"), "sc1", S)
    ap = lambda S_, rule: S_["fullours"]["by_seed"]["42"]["retest"]["certified2"]["table"]["original|" + rule]["AP"]
    put("AudFullOursRidgeBelowOwn", lambda S_: ap(S_, "table:own") - ap(S_, "table:ridge"), "sc1", S)
    put("AudFullOursMinutes", G("fullours", "by_seed", "42", "summary", "certified2", "minutes"), "sc0", S)
    for j, (kw, tn, _) in enumerate(FIVE):
        put("AudFullOursPtLearned" + tn, lambda S_, j=j: S_["fullours"]["by_seed"]["42"]["summary"]["certified2"]["rows"][j][j], "sc2", S)
    if S["full"]:
        put("AudFullOursRidgeAboveGainlora", lambda S_: ap(S_, "table:ridge") - S_["full"]["by_seed"]["42"]["retest"]["gainlora_inflora"]["table"]["original|official"]["AP"], "sc1", S)
    if S["ours"]:
        put("AudFullOursLearnedGain", lambda S_: S_["fullours"]["by_seed"]["42"]["retest"]["certified2"]["table"]["original|table:ridge"]["learned"] -
            S_["ours"]["by_seed"]["42"]["retest"]["certified2"]["table"]["original|table:ridge"]["learned"], "sc1", S)


def full_section(S):
    """run D: the two published methods without definitions at their published 100 epochs (seed 42, re-test and official test)"""
    files = {}
    if not S["full"]:
        print("run D not found, no e100 tables")
        return files
    R = S["full"]
    assert R["cfg"]["epochs"] == 100 and list(R["by_seed"]) == ["42"]
    B = ("full", "by_seed", "42")
    group("(m) run D: noinstruction_published_five_tasks_e100 results.json by_seed 42, retest[<method>].table[original|official]: 100 epochs, no definition")
    for m, mn, label in METH:
        row = B + ("retest", m, "table", "original|official")
        for f, fn_, k in FIELDS:
            put("AudFull%s%s" % (mn, fn_), G(*row, f), k, S)
        put("AudFull%sLearnedOne" % mn, G(*row, "learned"), "sc1", S)
        put("AudFull%sAPOne" % mn, G(*row, "AP"), "sc1", S)
        put("AudFull%sFTInt" % mn, G(*row, "FT"), "sc0", S)
        assert R["by_seed"]["42"]["retest"][m]["table"]["original|official"]["compared"] == 400
    put("AudFullCompared", G(*B, "retest", "inflora", "table", "original|official", "compared"), "int", S)
    group("(m) run D results.json by_seed 42, summary[<method>]: official test, one random reference answer; rows[i][j] = score on task j after learning task i")
    for m, mn, label in METH:
        for f, fn_, k in FIELDS[:3]:
            put("AudFullOff%s%s" % (mn, fn_), G(*B, "summary", m, f), "sc2", S)
        for j, (kw, tn, _) in enumerate(FIVE):
            put("AudFull%sPtLearned%s" % (mn, tn), lambda S_, m=m, j=j: S_["full"]["by_seed"]["42"]["summary"][m]["rows"][j][j], "sc2", S)
            put("AudFull%sPtFinal%s" % (mn, tn), lambda S_, m=m, j=j: S_["full"]["by_seed"]["42"]["summary"][m]["rows"][-1][j], "sc2", S)
    put("AudFullEpochs", G("full", "cfg", "epochs"), "int", S)
    put("AudFullMinutes", lambda S_: max(v["minutes"] for v in S_["full"]["by_seed"]["42"]["summary"].values()), "sc0", S)
    # one-decimal forms of our seed-42 numbers at 10 epochs
    if S["ours"]:
        group("(h) GPU run of our method without definitions, results.json by_seed 42, retest[certified2].table: one-decimal forms")
        OB = ("ours", "by_seed", "42", "retest", "certified2", "table")
        for rn, rule, fs in (("Ridge", "table:ridge", (("AP", "AP"), ("learned", "Learned"))), ("Handicap", "table:handicap", (("AP", "AP"),))):
            for f, fn_ in fs:
                name = "AudOurs%s%sOne" % (rn, fn_)
                if name not in REG:
                    put(name, G(*OB, "original|" + rule, f), "sc1", S)
    # tables
    head = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{p{1.95in}rrrr}", r"    \toprule",
            r"    Method, epochs (seed 42, no definition) & " + HL + " & " + HAP + " & " + HFT + " & " + HCH("AudFullCompared") + r" \\", r"    \midrule"]
    mc = lambda pre: ["\\%s%s{}" % (pre, x) for x in ("Learned", "AP", "FT", "Changed")]
    rows = []
    if "AudFullOursRidgeLearned" in REG:
        rows += [(r"\recog{} (ours), 100 epochs", mc("AudFullOursRidge"), "e100", True), (r"\handicap{} (ours), 100 epochs", mc("AudFullOursHandicap"), "e100", True), ("MID",)]
    rows += [("%s, 100 epochs" % label, mc("AudFull" + mn), "e100", True) for m, mn, label in METH] + [("MID",)]
    rows += [("%s, 10 epochs" % label, mc("AudPub%sNoDef" % mn), "e10", True) for m, mn, label in METH]
    if "AudOursRidgeLearned" in REG:
        rows += [("MID",), (r"\recog{} (ours), 10 epochs", mc("AudOursRidge"), "e10", True), (r"\handicap{} (ours), 10 epochs", mc("AudOursHandicap"), "e10", True)]
    files["noinst_e100.tex"] = "\n".join(head + render(rows, [1, 1, -1, -1], "noinst_e100") + [r"    \bottomrule", r"\end{tabular}}", ""])
    L = [r"{\footnotesize\setlength{\tabcolsep}{2pt}\begin{tabular}{lrr}", r"    \toprule",
         r"    & \multicolumn{2}{c}{\shortstack{Score right after\\learning $\to$ at the end $\uparrow$}} \\",
         r"    \cmidrule(l){2-3}", r"    Task & InfLoRA & GainLoRA \\", r"    \midrule"]
    for kw, tn, tl in FIVE:
        c = ["\\AudFull%sPtLearned%s{}\\,$\\to$\\,\\AudFull%sPtFinal%s{}" % (mn, tn, mn, tn) for _, mn, _ in METH]
        rowbold(c, (0, 1), 1, "noinst_e100_pertask", tl, which=1)
        L.append("    %s & %s \\\\" % (tl, " & ".join(c)))
    L += [r"    \bottomrule", r"\end{tabular}}", ""]
    files["noinst_e100_pertask.tex"] = "\n".join(L)
    return files


def fulldef_section(S):
    """run E: the two published methods WITH definitions at their published 100 epochs, five tasks, seed 42 (results.json by_seed 42, retest[<method>].table[<wording>|official])"""
    files = {}
    if not S["fulldef"]:
        print("run E not found, no e100_definitions / e100_wordings tables")
        return files
    R = S["fulldef"]
    assert R["cfg"]["epochs"] == 100 and list(R["by_seed"]) == ["42"]
    B = ("fulldef", "by_seed", "42")
    group("(p) run E: published_five_tasks_e100_with_definitions results.json by_seed 42, retest[<method>].table[<wording>|official]: published methods with definitions, 100 epochs")
    for w, wn in WORDS:
        for m, mn, label in METH:
            row = B + ("retest", m, "table", w + "|official")
            for f, fn_, k in FIELDS:
                put("AudFullDef%s%s%s" % (wn, mn, fn_), G(*row, f), k, S)
            assert R["by_seed"]["42"]["retest"][m]["table"][w + "|official"]["compared"] == 400
            if w == "original":
                put("AudFullDef%s%sAPOne" % (wn, mn), G(*row, "AP"), "sc1", S)
                put("AudFullDef%s%sLearnedOne" % (wn, mn), G(*row, "learned"), "sc1", S)
                put("AudFullDef%s%sFTOne" % (wn, mn), G(*row, "FT"), "sc1", S)
    put("AudFullDefCompared", G(*B, "retest", "inflora", "table", "original|official", "compared"), "int", S)
    for m, mn, label in METH:
        for f, fn_, k in FIELDS[:3]:
            put("AudFullDefOff%s%s" % (mn, fn_), G(*B, "summary", m, f), "sc2", S)
    put("AudFullDefMinutes", lambda S_: max(v["minutes"] for v in S_["fulldef"]["by_seed"]["42"]["summary"].values()), "sc0", S)
    # table: the instruction as the only difference
    head = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{p{1.9in}lrrrr}", r"    \toprule",
            r"    Method (seed 42, 100 epochs) & Definitions & " + HL + " & " + HAP + " & " + HFT + " & " + HCH("AudFullDefCompared") + r" \\", r"    \midrule"]
    mc = lambda pre: ["\\%s%s{}" % (pre, x) for x in ("Learned", "AP", "FT", "Changed")]
    rows = []
    for m, mn, label in METH:
        rows.append((label, ["with"] + mc("AudFullDefOriginal" + mn), "x", True))
        rows.append((label, ["none"] + mc("AudFull" + mn), "x", True))
        rows.append(("MID",))
    if "AudFullOursRidgeLearned" in REG:
        rows.append((r"\recog{} (ours)", ["none"] + mc("AudFullOursRidge"), "x", True))
        rows.append((r"\handicap{} (ours)", ["none"] + mc("AudFullOursHandicap"), "x", True))
    else:
        rows.pop()
    files["e100_definitions.tex"] = "\n".join(head + render(rows, [None, 1, 1, -1, -1], "e100_definitions") + [r"    \bottomrule", r"\end{tabular}}", ""])
    # table: the four wordings
    head = [r"{\scriptsize\setlength{\tabcolsep}{2pt}\begin{tabular}{p{1.2in}rrrrrrrr}", r"    \toprule",
            r"    & \multicolumn{4}{c}{Final score $\uparrow$} & \multicolumn{4}{c}{\shortstack{Answers changed\\of \AudFullDefCompared{} $\downarrow$}} \\",
            r"    \cmidrule(lr){2-5}\cmidrule(l){6-9}", r"    & original & light & medium & heavy & original & light & medium & heavy \\", r"    \midrule"]
    m_ = lambda n: "\\%s{}" % n
    rows = []
    for m, mn, label in METH:
        rows.append(("%s, 100 epochs" % label, [m_("AudFullDef%s%sAP" % (wn, mn)) for _, wn in WORDS] + [m_("AudFullDef%s%sChanged" % (wn, mn)) for _, wn in WORDS], "w", True))
    if "AudUOriginalCertAP" in REG:
        rows.append(("MID",))
        for lab, rn in ((r"\cert{}, 10 epochs", "Cert"), (r"\recog{}, 10 epochs", "Ridge")):
            rows.append((lab, [m_("AudU%s%sAP" % (wn, rn)) for _, wn in WORDS] + [m_("AudU%s%sChanged" % (wn, rn)) for _, wn in WORDS], "w", True))
    files["e100_wordings.tex"] = "\n".join(head + render(rows, [None] * 8, "e100_wordings") + [r"    \bottomrule", r"\end{tabular}}", ""])
    return files


def fifteen_section(S):
    """the 15 SuperNI tasks of order 1: our method without definitions (GPU, seed 42, 10 epochs) and the run with definitions of 2026-11-17"""
    files = {}
    if not S["fifteen"]:
        print("15-task run without definitions not found, no noinst_fifteen table")
        return files
    T = ("fifteen", "by_seed", "42", "retest", "certified2", "table")
    RU = (("Ridge", "table:ridge"), ("Handicap", "table:handicap"), ("Own", "table:own"), ("Centre", "centre"), ("Plain", "all"))
    group("(n) 15 tasks, our method without definitions: noinstruction_ours_fifteen_tasks results.json by_seed 42, retest[certified2].table[original|<rule>]")
    for rn, rule in RU:
        key = "original|" + rule
        for f, fn_, k in FIELDS:
            put("AudFifteenNoDef%s%s" % (rn, fn_), G(*T, key, f), k, S)
        put("AudFifteenNoDef%sRight" % rn, G(*T, key, "right_task"), "pct1", S)
        if rn in ("Ridge", "Handicap"):
            put("AudFifteenNoDef%sAPOne" % rn, G(*T, key, "AP"), "sc1", S)
        assert S["fifteen"]["by_seed"]["42"]["retest"]["certified2"]["table"][key]["compared"] == 1097
    put("AudFifteenNoDefCompared", G(*T, "original|table:own", "compared"), "int", S)
    put("AudFifteenNoDefRidgeBelowOwn", lambda S_: S_["fifteen"]["by_seed"]["42"]["retest"]["certified2"]["table"]["original|table:own"]["AP"] -
        S_["fifteen"]["by_seed"]["42"]["retest"]["certified2"]["table"]["original|table:ridge"]["AP"], "sc1", S)
    put("AudFifteenNoDefMinutes", G("fifteen", "by_seed", "42", "summary", "certified2", "minutes"), "sc0", S)
    # the same 15 tasks with definitions (run of 2026-11-17), re-test, original wording, two decimals (the \\Fifteen... macros of numbers.tex have one)
    have_def = bool(S["fifteen_def"])
    if have_def:
        group("(n) 15 tasks with definitions: kaggle_fifteen_tasks_seed42 results.json by_seed 42, retest[certified2|gainlora_inflora|inflora].table[original|<rule>]")
        DB = ("fifteen_def", "by_seed", "42", "retest")
        for mn, m, key in (("Cert", "certified2", "original|certified"), ("Gainlora", "gainlora_inflora", "original|official"), ("Inflora", "inflora", "original|official")):
            for f, fn_, k in FIELDS:
                put("AudFifteenDef%s%s" % (mn, fn_), G(*DB, m, "table", key, f), k, S)
            assert S["fifteen_def"]["by_seed"]["42"]["retest"][m]["table"][key]["compared"] == 1097
        put("AudFifteenDefCertRight", G(*DB, "certified2", "table", "original|certified", "right_task"), "pct1", S)
        put("AudFifteenDefCompared", G(*DB, "certified2", "table", "original|certified", "compared"), "int", S)
    head = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{p{1.75in}rrrrr}", r"    \toprule",
            r"    Method (seed 42, 15 tasks) & " + HL + " & " + HAP + " & " + HFT + " & " + HCH("AudFifteenNoDefCompared") + " & " + HOWN + r" \\", r"    \midrule"]
    mc = lambda pre: ["\\%s%s{}" % (pre, x) for x in ("Learned", "AP", "FT", "Changed", "Right")]
    rows = []
    for lab, rn in ((r"\plain{}", "Plain"), (r"\centre{}", "Centre"), (r"\recog{} (ours)", "Ridge"), (r"\handicap{} (ours)", "Handicap"), ("every input to its own task (ceiling)", "Own")):
        rows.append((lab, mc("AudFifteenNoDef" + rn), "nodef", rn != "Own"))
    if have_def:
        rows.append(("MID",))
        rows.append((r"\cert{}, with definitions", ["\\AudFifteenDefCert%s{}" % x for x in ("Learned", "AP", "FT", "Changed", "Right")], "def", True))
        for lab, mn in (("GainLoRA", "Gainlora"), ("InfLoRA", "Inflora")):
            rows.append(("%s, with definitions" % lab, ["\\AudFifteenDef%s%s{}" % (mn, x) for x in ("Learned", "AP", "FT", "Changed")] + ["--"], "def", True))
    files["noinst_fifteen.tex"] = "\n".join(head + render(rows, [1, 1, -1, -1, 1], "noinst_fifteen") + [r"    \bottomrule", r"\end{tabular}}", ""])
    return files


def short_table():
    """tables/recogniser_short.tex for the short version: only macros that exist are used; a row whose source is missing is left out"""
    spec = (("15 SuperNI tasks (development)", ("AudDevLLamOne%s", ("Arrive", "End", "Changed", "AuditBound")), ("AudDevHPrimeTTwo%s", ("End", "Changed"))),
            ("15 Long Sequence tasks", ("AudScaleLongThree%s", ("Arrive", "End", "Changed", "AuditBound")), ("AudScaleLongThree%s", ("HEnd", "HChanged"))),
            ("30 unseen tasks (read once)", ("AudFreshFourth%s", ("LArrive", "LEnd", "LChanged", "AuditBound")), ("AudFreshFourth%s", ("HEnd", "HChanged"))),
            ("90 fresh tasks", ("AudScaleFreshNinety%s", ("Arrive", "End", "Changed", "AuditBound")), ("AudScaleFreshNinety%s", ("HEnd", "HChanged"))),
            ("120 tasks", ("AudScaleAllHundredTwenty%s", ("Arrive", "End", "Changed", "AuditBound")), ("AudScaleAllHundredTwenty%s", ("HEnd", "HChanged"))))
    L = [r"{\footnotesize\setlength{\tabcolsep}{1.2pt}\begin{tabular}{lrrrrrr}", r"    \toprule",
         r"    & \multicolumn{4}{c}{\recog{}, \%} & \multicolumn{2}{c}{\handicap{}, $\nu = 2$, \%} \\", r"    \cmidrule(lr){2-5}\cmidrule(l){6-7}",
         r"    & \shortstack[r]{own task\\at arrival $\uparrow$} & \shortstack[r]{own task\\at the end $\uparrow$} & \shortstack[r]{routes\\changed $\downarrow$} & \shortstack[r]{audit\\95\% bound} & \shortstack[r]{own task\\at the end $\uparrow$} & \shortstack[r]{routes\\changed $\downarrow$} \\", r"    \midrule"]
    for lab, (p1, f1), (p2, f2) in spec:
        names = [p1 % f for f in f1] + [p2 % f for f in f2]
        if not all(n in REG for n in names):
            print("short table: row left out (source missing):", lab)
            continue
        c = ["\\%s{}" % n for n in names]
        rowbold(c, (1, 4), 1, "recogniser_short", lab)
        rowbold(c, (2, 5), -1, "recogniser_short", lab)
        L.append("    %s & %s \\\\" % (lab, " & ".join(c)))
    L += [r"    \bottomrule", r"\end{tabular}}", ""]
    return {"recogniser_short.tex": "\n".join(L)}


def build(S):
    files = {}
    print("Clopper-Pearson bound checked against %d stored upper95 values" % check_cp(S))

    # ================================================================== (a), (b): published methods with and without definitions
    pub_tasks = S["pub"]["cfg"]["tasks"]
    assert len(pub_tasks) == 5 and all(k in t for (k, _, _), t in zip(FIVE, pub_tasks)), pub_tasks
    P_NO = ("pub", "by_seed", "42", "retest")
    comp = set()
    for m, mn, mlabel in METH:
        for dn, base in (("NoDef", P_NO + (m, "table", "original|official")), ("Def", ("defs_retest", m, "table", "original|official"))):
            src_, path = base[0], base[1:]
            if dn == "NoDef":
                group("(a) %s, results.json by_seed 42, retest[%s].table[original|official]: re-test of the first four tasks, no definition" % (PATHS["pub_member"].split("/")[1], m))
            else:
                group("(b) %sretest.json [%s].table[original|official]: same seed with definitions" % (PATHS["defs_dir"], m))
            for f, fn_, k in FIELDS:
                put("AudPub%s%s%s" % (mn, dn, fn_), G(src_, *path, f), k, S)
            put("AudPub%s%sCompared" % (mn, dn), G(src_, *path, "compared"), "int", S)
            comp.add(functools.reduce(lambda o, k_: o[k_], path, S[src_])["compared"])
            for (kw, tn, _), t in zip(FIVE, pub_tasks):
                put("AudPub%s%sPtLearned%s" % (mn, dn, tn), G(src_, *path, "learned_scores", t), "sc2", S)
                put("AudPub%s%sPtFinal%s" % (mn, dn, tn), G(src_, *path, "final_scores", t), "sc2", S)
    assert comp == {400}, comp
    put("AudPubCompared", G(*P_NO, "inflora", "table", "original|official", "compared"), "int", S)
    group("(a) results.json by_seed 42, summary[method]: official test, one reference answer per example, no definition")
    for m, mn, _ in METH:
        for f, fn_, k in FIELDS[:3]:
            put("AudOff%sNoDef%s" % (mn, fn_), G("pub", "by_seed", "42", "summary", m, f), "sc2", S)
    group("(b) results.json summary[method]: official test with definitions (seed 42)")
    for m, mn, _ in METH + (("certified2", "Certified", ""),):
        for f, fn_, k in FIELDS[:3]:
            put("AudOff%sDef%s" % (mn, fn_), G("defs_results", "summary", m, f), "sc2", S)
    group("(b) retest.json: re-test at the four wordings of the definition; certified2 under five rules, published methods under their own rule")
    for w, wn in WORDS:
        for r, rn in RULES2:
            for f, fn_, k in FIELDS + (("right_task", "RightTask", "pct1"),):
                put("AudWord%s%s%s" % (wn, rn, fn_), G("defs_retest", "certified2", "table", "%s|%s" % (w, r), f), k, S)
        for m, mn, _ in METH:
            for f, fn_, k in FIELDS:
                put("AudWord%s%s%s" % (wn, mn, fn_), G("defs_retest", m, "table", "%s|official" % w, f), k, S)

    # ---- (h) our method without definitions, if it exists
    ours = S["ours"]
    ours_rows = []
    if ours:
        OB = ("ours", "by_seed", "42", "retest", "certified2", "table")
        group("(h) GPU run of our method without definitions, results.json by_seed 42, retest[certified2].table[original|<rule>]")
        for r, rn in OURS_RULES:
            key = "original|" + r
            if key not in ours["by_seed"]["42"]["retest"]["certified2"]["table"]:
                continue
            for f, fn_, k in FIELDS:
                put("AudOurs%s%s" % (rn, fn_), G(*OB, key, f), k, S)
            put("AudOurs%sCompared" % rn, G(*OB, key, "compared"), "int", S)
            assert ours["by_seed"]["42"]["retest"]["certified2"]["table"][key]["compared"] == 400
            if r == "table:ridge":
                for (kw, tn, _), t in zip(FIVE, pub_tasks):
                    put("AudOursPtLearned" + tn, G(*OB, key, "learned_scores", t), "sc2", S)
                    put("AudOursPtFinal" + tn, G(*OB, key, "final_scores", t), "sc2", S)
            ours_rows.append((r, rn))
        group("(h) GPU run of our method without definitions, results.json by_seed 42, summary[certified2]: official test")
        for f, fn_, k in FIELDS[:3]:
            put("AudOffOursNoDef" + fn_, G("ours", "by_seed", "42", "summary", "certified2", f), "sc2", S)
    if S["small"]:
        group("(h) certified_switch/demo_results/no_instruction_t5-small_seed0_ridge.json rules[<rule>]: small trained model")
        for rule, v in S["small"]["rules"].items():
            rn = "".join(c for c in rule.title() if c.isalpha())
            for f, fn_, k in FIELDS:
                if f in v:
                    put("AudSmall%s%s" % (rn, fn_), G("small", "rules", rule, f), "sc1" if k == "sc2" else k, S)
            if "compared" in v:
                put("AudSmall%sCompared" % rn, G("small", "rules", rule, "compared"), "int", S)

    # ---- table: noinst_published.tex
    head = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{p{1.9in}lrrrr}", r"    \toprule",
            r"    Method & Definitions & " + HL + " & " + HAP + " & " + HFT + " & " + HCH("AudPubCompared") + r" \\", r"    \midrule"]
    rows = []
    for m, mn, label in METH:
        for dn, dl in (("Def", "with"), ("NoDef", "none")):
            rows.append((label, [dl] + ["\\AudPub%s%s%s{}" % (mn, dn, x) for x in ("Learned", "AP", "FT", "Changed")], dn, True))
    if ours_rows:
        rows.append(("MID",))
        lab = {"all": r"\plain{} (all adapters on)", "table:ridge": r"\recog{} (ours)",
               "table:handicap": r"\handicap{}, $\nu = 2$ (ours)", "table:own": "every input to its own task (ceiling)"}
        for r, rn in ours_rows:
            if r in lab:
                rows.append((lab[r], ["none"] + ["\\AudOurs%s%s{}" % (rn, x) for x in ("Learned", "AP", "FT", "Changed")], "NoDef", r != "table:own"))
    files["noinst_published.tex"] = "\n".join(head + render(rows, [None, 1, 1, -1, -1], "noinst_published") + [r"    \bottomrule", r"\end{tabular}}", ""])

    # ---- table: noinst_published_pertask.tex (learned -> final)
    has_ours = any(r == "table:ridge" for r, _ in ours_rows)
    cols = "l" + "r" * (5 if has_ours else 4)
    L = [r"{\footnotesize\setlength{\tabcolsep}{2pt}\begin{tabular}{%s}" % cols, r"    \toprule",
         r"    & \multicolumn{%d}{c}{\shortstack{Score right after learning\\$\to$ at the end $\uparrow$}} \\" % (5 if has_ours else 4), r"    \cmidrule(l){2-%d}" % (6 if has_ours else 5),
         r"    & \multicolumn{2}{c}{InfLoRA} & \multicolumn{2}{c}{GainLoRA}" + (r" & ours (L)" if has_ours else "") + r" \\",
         r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}" + (r"\cmidrule(l){6-6}" if has_ours else ""),
         r"    Task & with & none & with & none" + (r" & none" if has_ours else "") + r" \\", r"    \midrule"]
    for (kw, tn, tlabel) in FIVE:
        c = []
        for m, mn, _ in METH:
            for dn in ("Def", "NoDef"):
                c.append("\\AudPub%s%sPtLearned%s{}\\,$\\to$\\,\\AudPub%s%sPtFinal%s{}" % (mn, dn, tn, mn, dn, tn))
        if has_ours:
            c.append("\\AudOursPtLearned%s{}\\,$\\to$\\,\\AudOursPtFinal%s{}" % (tn, tn))
        rowbold(c, (0, 2), 1, "noinst_published_pertask", tlabel + " (with definitions)", which=1)
        rowbold(c, (1, 3, 4) if has_ours else (1, 3), 1, "noinst_published_pertask", tlabel + " (no definition)", which=1)
        L.append("    %s & %s \\\\" % (tlabel, " & ".join(c)))
    L += [r"    \bottomrule", r"\end{tabular}}", ""]
    files["noinst_published_pertask.tex"] = "\n".join(L)

    # ================================================================== (e) zone rule with three features
    group("(e) noinstruction_study.json results_reduced[superni_order1][<feature>][ratio10][0.05].mean: zone rule, ratio score, level 0.05")
    FEATS = (("t5static", "WordVec", "word vectors"), ("minilm", "Sentence", "sentence encoder"), ("t5hidden", "FrozenT", "frozen T5-base states"))
    ZF = (("coverage", "Inside"), ("blocked", "Blocked"), ("own_at_end", "Own"), ("changed", "Changed"))
    for fk, fn_, _ in FEATS:
        for f, n_ in ZF:
            put("AudZone%s%s" % (fn_, n_), G("noinst", "results_reduced", "superni_order1", fk, "ratio10", "0.05", "mean", f), "pct1", S)
    head_t = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrr}", r"    \toprule",
              r"    Feature & \shortstack[r]{Inside own\\zone, \%} & \shortstack[r]{Blocked,\\\% $\downarrow$} & " + HEND + " & " + HCHG + r" \\", r"    \midrule"]
    rows_t = [(flabel, ["\\AudZone%s%s{}" % (fn_, n_) for _, n_ in ZF], "feat", True) for fk, fn_, flabel in FEATS]
    top = head_t + render(rows_t, [None, -1, 1, -1], "recogniser_features_top") + [r"    \bottomrule", r"\end{tabular}}"]

    # ================================================================== (d) joint recogniser on the sentence encoder
    A = S["audit"]
    dev_s = A["dev"]["superni"]

    def best_key(S_, fam, feat):
        ks = [k for k in S_["audit"]["dev"]["superni"] if k.startswith("%s|%s|" % (fam, feat))]
        return max(ks, key=lambda k: S_["audit"]["dev"]["superni"][k]["mean"]["own_at_end"])
    AREC = (("lr", "minilm", "LrMini", "logistic reg., sentence enc., $C=$"),
            ("lr", "minilm+shape", "LrShape", "\\quad + shape numbers, $C=$"),
            ("lr", "minilm+char", "LrChar", "\\quad + character n-grams, $C=$"),
            ("analytic", "minilm", "CosMini", "cosine features, sentence enc., $\\lambda=$"),
            ("analytic", "minilm+shape", "CosShape", "\\quad + shape numbers, $\\lambda=$"))
    group("(d) audit_study.json dev[superni|long][<recogniser>|<feature>|<best parameter>].mean (best parameter by SuperNI own_at_end)")
    chosen_variant = A["chosen"]["variant"]
    head_b = [r"{\footnotesize\setlength{\tabcolsep}{1.8pt}\begin{tabular}{lrrrrrr}", r"    \toprule",
              r"    & \multicolumn{2}{c}{\shortstack{Sent to own task\\at the end, \% $\uparrow$}} & \multicolumn{2}{c}{\shortstack{Route changed,\\\% $\downarrow$}} & \multicolumn{2}{c}{\shortstack{Audit\\$(H{+}1)/(m{+}1)$, \%}} \\",
              r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
              r"    Recogniser, feature & SuperNI & Long & SuperNI & Long & SuperNI & Long \\", r"    \midrule"]
    rows_b = []
    chosen_seen = False
    for fam, feat, nm, lab in AREC:
        bk = best_key(S, fam, feat)
        put("AudRec%sParam" % nm, lambda S_, fam=fam, feat=feat: best_key(S_, fam, feat).split("|")[2], "str", S)
        cells = []
        for f, fn_ in (("own_at_end", "End"), ("changed", "Changed"), ("certificate", "AuditLine")):
            for seq, sn in (("superni", "Superni"), ("long", "Long")):
                cells.append(put("AudRec%s%s%s" % (nm, fn_, sn), lambda S_, fam=fam, feat=feat, f=f, seq=seq: S_["audit"]["dev"][seq][best_key(S_, fam, feat)]["mean"][f], "pct1", S))
        chosen_seen |= bk == chosen_variant
        row_lab = lab + "\\AudRec%sParam{}" % nm
        if bk == chosen_variant:
            row_lab, cells = bold(row_lab), [bold(c) for c in cells]
        rows_b.append((row_lab, cells, "var", True))
    assert chosen_seen, "the chosen variant %s is not one of the rows" % chosen_variant
    chosen_m = A["chosen"]["superni"]["mean"]
    assert abs(chosen_m["own_at_end"] - A["dev"]["superni"][chosen_variant]["mean"]["own_at_end"]) < 1e-12
    bot = head_b + render(rows_b, [1, 1, -1, -1, None, None], "recogniser_features_bottom") + [r"    \bottomrule", r"\end{tabular}}"]
    # the chosen variant: audit estimate c/m and 95% bound from its per-task counts (the other variants store only the mean of (c+1)/(m+1))
    chosen_nm = [nm for fam, feat, nm, lab in AREC if best_key(S, fam, feat) == chosen_variant][0]
    group("(d) audit_study.json chosen[superni|long].per_task[<task>] c, m: audit estimate c/m and 95% bound (Clopper-Pearson computed here: no upper95 is stored)")
    ce = []
    for f_, n_ in ((est_acc, "Est"), (lambda *p: bound_acc(*p, computed=True), "Bound")):
        for seq, sn in (("superni", "Superni"), ("long", "Long")):
            ce.append(put("AudRecChosenAudit%s%s" % (n_, sn), f_("audit", "chosen", seq, "per_task"), "pct1", S))
    for seq in ("superni", "long"):
        for t in A["chosen"][seq]["per_task"].values():
            assert abs((t["c"] + 1) / (t["m"] + 1) - t["certificate"]) < 1e-12
    fin = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrr}", r"    \toprule",
           r"    & \multicolumn{2}{c}{Audit $H/m$, \%} & \multicolumn{2}{c}{95\% bound, \%} \\",
           r"    \cmidrule(lr){2-3}\cmidrule(l){4-5}",
           r"    Chosen variant & SuperNI & Long & SuperNI & Long \\", r"    \midrule",
           "    logistic reg. + shape numbers, $C=$\\AudRec%sParam{} & %s \\\\" % (chosen_nm, " & ".join(ce)),
           r"    \bottomrule", r"\end{tabular}}"]
    files["recogniser_features.tex"] = "\n".join(top) + "\n\\par\\medskip\n" + "\n".join(bot) + "\n\\par\\medskip\n" + "\n".join(fin) + "\n"

    # ================================================================== (c) rules on the frozen T5-base states
    H = S["hidden"]

    def dpath(key):
        return ("hidden", "dev", "full", key, "mean") if key.split("|")[0] in ("J", "L", "claims") else ("hidden", "dev_rule_H", key, "mean")
    DEV = (("JCOne", r"\recog{} with logistic regression ($C = 1$)", "J|C1", False), ("LLamOne", r"\recog{}, ridge $1\times$", "L|lam1", True),
           ("LLamTen", r"\recog{}, ridge $10\times$", "L|lam10", False), ("LLamHundred", r"\recog{}, ridge $100\times$", "L|lam100", False),
           ("ClaimsROne", r"zones, newest wins, $r = 1$", "claims|r1", False),
           ("HPrimeTZero", r"\handicap{}, $\nu = 0$", "H'|t0", False), ("HPrimeTOne", r"\handicap{}, $\nu = 1$", "H'|t1", False),
           ("HPrimeTTwo", r"\handicap{}, $\nu = 2$", "H'|t2", True), ("HPrimeTFive", r"\handicap{}, $\nu = 5$", "H'|t5", False), ("HPrimeTNone", r"\handicap{} without protection ($\nu = \infty$)", "H'|t1000000", False),
           ("HTZero", r"\handicap{} without the floor (first form), $\nu = 0$", "H|t0", False), ("HTOne", r"\handicap{} without the floor (first form), $\nu = 1$", "H|t1", False),
           ("HTTwo", r"\handicap{} without the floor (first form), $\nu = 2$", "H|t2", False))
    group("(c) hidden_rules_study.json dev.full[<rule>].mean (rules J, L, claims) and dev_rule_H[<rule>].mean (rules H, H'): 15 SuperNI tasks")
    rows = []
    for nm, lab, key, is_b in DEV:
        valid = functools.reduce(lambda o, k: o[k], dpath(key)[:-1], S)["certificate_valid"]
        c = [put("AudDev%s%s" % (nm, n_), G(*dpath(key), f), "pct1", S) for f, n_ in (("own_at_arrival", "Arrive"), ("own_at_end", "End"), ("changed", "Changed"))]
        if valid:
            put("AudDev%sAuditLine" % nm, G(*dpath(key), "certificate"), "pct1", S)
            pt_path = dpath(key)[:-1] + ("per_task",)
            assert abs(sum(v["upper95"] for v in functools.reduce(lambda o, k: o[k], pt_path[1:], H).values()) / 15 - functools.reduce(lambda o, k: o[k], dpath(key)[1:], H)["upper95"]) < 1e-9
            c.append(put("AudDev%sAuditEst" % nm, est_acc(*pt_path), "pct1", S))
            c.append(put("AudDev%sAuditBound" % nm, bound_acc(*pt_path), "pct1", S))
        else:
            c += ["--", "--"]
        bf = "claims_bound" if key.startswith("claims") else "handicap_bound" if key.startswith("H") else None
        if bf and key != "H'|t1000000":
            c.append(put("AudDev%sBound" % nm, G(*dpath(key), bf), "pct1", S))
        else:
            c.append("--")
        if is_b:
            c = [bold(x) for x in c]
            lab = bold(lab)
        rows.append((nm, lab, c))
    head = [r"{\scriptsize\setlength{\tabcolsep}{2.5pt}\begin{tabular}{p{1.6in}rrrrrr}", r"    \toprule",
            r"    Rule & " + HARR + " & " + HEND + " & " + HCHG + " & " + HEST + " & " + HBND + " & " + HADV + r" \\", r"    \midrule"]
    rows_d = []
    for nm, lab, c in rows:
        if nm == "HTZero":
            rows_d.append(("MID",))
        rows_d.append((lab, c, "rules", True))
    files["recogniser_dev.tex"] = "\n".join(head + render(rows_d, [1, 1, -1, None, None, None], "recogniser_dev") + [r"    \bottomrule", r"\end{tabular}}", ""])

    # ---- per task, chosen rule L
    PT = H["chosen"]["per_task"]
    assert H["chosen"]["variant"] == "L|lam1"
    group("(c) hidden_rules_study.json chosen.per_task[<task>] (rule L, ridge 1): own_at_arrival, own_at_end, changed, c, m, certificate, upper95")
    L = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrrrr}", r"    \toprule",
         r"    Task & " + HARR + " & " + HEND + " & " + HCHG + r" & Held-out & " + HEST + " & " + HBND + r" \\", r"    \midrule"]
    for i, t in enumerate(PT):
        idn = re.search(r"task(\d+)_", t).group(1)
        n_ = NUM[i + 1]
        acc = lambda f, i=i: (lambda S_: S_["hidden"]["chosen"]["per_task"][list(S_["hidden"]["chosen"]["per_task"])[i]][f])
        c = [put("AudPt%s%s" % (n_, nn), acc(f), "pct1", S) for f, nn in (("own_at_arrival", "Arrive"), ("own_at_end", "End"), ("changed", "Changed"))]
        cc = put("AudPt%sC" % n_, acc("c"), "int", S)
        mm = put("AudPt%sM" % n_, acc("m"), "int", S)
        c.append("%s of %s" % (cc, mm))
        put("AudPt%sAuditLine" % n_, acc("certificate"), "pct1", S)
        c.append(put("AudPt%sAuditEst" % n_, lambda S_, i=i: (lambda v: v["c"] / v["m"])(S_["hidden"]["chosen"]["per_task"][list(S_["hidden"]["chosen"]["per_task"])[i]]), "pct1", S))
        c.append(put("AudPt%sBound" % n_, acc("upper95"), "pct1", S))
        v = PT[t]
        assert abs((v["c"] + 1) / (v["m"] + 1) - v["certificate"]) < 1e-12, t
        L.append("    %s & %s \\\\" % (NAMES15[idn.lstrip("0") if idn.lstrip("0") in NAMES15 else idn], " & ".join(c)))
    L += [r"    \bottomrule", r"\end{tabular}}", ""]
    files["recogniser_pertask.tex"] = "\n".join(L)

    # ---- orders
    # only means are stored for the other orders (no per-task c), so c/m cannot be recovered: the audit column stays the mean of (c+1)/(m+1) and is labelled so
    group("(c) hidden_rules_study.json chosen.mean (order 1, rule L), dev_rule_H[H'|t2].mean (order 1, H'), dev_other_orders[<order>][L | H_prime_t2].mean")
    ORD = [("One", "order 1", ("hidden", "chosen", "mean"), ("hidden", "dev_rule_H", "H'|t2", "mean"))]
    ORD.append(("Two", "order 2", ("hidden", "dev_other_orders", "order2", "L", "mean"), ("hidden", "dev_other_orders", "order2", "H_prime_t2", "mean")))
    for r_ in range(5):
        ORD.append(("Rand" + NUM[r_], "random %d" % r_, ("hidden", "dev_other_orders", "random%d" % r_, "L", "mean"),
                    ("hidden", "dev_other_orders", "random%d" % r_, "H_prime_t2", "mean")))
    L = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrrrrr}", r"    \toprule",
         r"    & \multicolumn{5}{c}{\recog{}} & \multicolumn{2}{c}{\handicap{}, $\nu = 2$} \\", r"    \cmidrule(lr){2-6}\cmidrule(l){7-8}",
         r"    Order & " + HARR + " & " + HEND + " & " + HCHG + r" & \shortstack[r]{$(H{+}1)/$\\$(m{+}1)$, \%} & " + HBND + " & " + HEND + " & " + HCHG + r" \\", r"    \midrule"]
    for nm, lab, lp, hp in ORD:
        c = [put("AudOrd%sL%s" % (nm, n_), G(*lp, f), "pct1", S) for f, n_ in (("own_at_arrival", "Arrive"), ("own_at_end", "End"), ("changed", "Changed"), ("certificate", "AuditLine"), ("upper95", "AuditBound"))]
        c += [put("AudOrd%sH%s" % (nm, n_), G(*hp, f), "pct1", S) for f, n_ in (("own_at_end", "End"), ("changed", "Changed"))]
        rowbold(c, (1, 5), 1, "recogniser_orders", lab)
        rowbold(c, (2, 6), -1, "recogniser_orders", lab)
        L.append("    %s & %s \\\\" % (lab, " & ".join(c)))
    L += [r"    \bottomrule", r"\end{tabular}}", ""]
    files["recogniser_orders.tex"] = "\n".join(L)

    # ---- fresh tasks
    group("(d) audit_study.json verdict_fresh3 and fresh3.chosen.per_task: third batch of fresh tasks, chosen variant")
    V3 = ("audit", "verdict_fresh3")
    c = [put("AudFreshThird%s" % n_, G(*V3, f), "pct1", S) for f, n_ in (("mean_own_at_arrival", "Arrive"), ("mean_own_at_end", "End"), ("mean_changed", "Changed"), ("mean_certificate", "AuditLine"))]
    c = c[:3] + [put("AudFreshThirdAuditEst", est_acc("audit", "fresh3", "chosen", "per_task"), "pct1", S),
                 put("AudFreshThirdAuditBound", bound_acc("audit", "fresh3", "chosen", "per_task", computed=True), "pct1", S)]   # no upper95 stored here: computed
    assert abs(sum((v["c"] + 1) / (v["m"] + 1) for v in A["fresh3"]["chosen"]["per_task"].values()) / 30 - A["verdict_fresh3"]["mean_certificate"]) < 1e-9
    put("AudFreshThirdUse", G(*V3, "use"), "str", S)
    put("AudFreshThirdCert", G(*V3, "certificate"), "str", S)
    pt3 = lambda S_: S_["audit"]["fresh3"]["chosen"]["per_task"]
    put("AudFreshThirdTasks", lambda S_: len(pt3(S_)), "int", S)
    put("AudFreshThirdLow", lambda S_: sum(1 for v in pt3(S_).values() if v["own_at_end"] < 0.70), "int", S)
    PAIR = ("task567_circa_text_generation", "task565_circa_answer_generation")
    put("AudFreshThirdPairLow", lambda S_: min(S_["audit"]["fresh3"]["chosen"]["per_task"][t]["own_at_end"] for t in PAIR), "pct1", S)
    put("AudFreshThirdPairHigh", lambda S_: max(S_["audit"]["fresh3"]["chosen"]["per_task"][t]["own_at_end"] for t in PAIR), "pct1", S)
    assert A["fresh3"]["chosen"]["variant"] == chosen_variant
    rows = [("third batch: sentence enc. + shape numbers, logistic reg.", c, "\\AudFreshThirdUse{} / \\AudFreshThirdCert{}", "\\AudFreshThirdLow{}")]
    if "verdict_fresh4" in H:
        V4 = H["verdict_fresh4"]
        group("(c) hidden_rules_study.json verdict_fresh4 and second_rule_H_prime_t2: fourth batch of fresh tasks, frozen T5-base states")
        mk = lambda d: d["mean"] if "mean" in d else d
        c4 = [put("AudFreshFourthL%s" % n_, G("hidden", "verdict_fresh4", "mean", f), "pct1", S) for f, n_ in (("own_at_arrival", "Arrive"), ("own_at_end", "End"), ("changed", "Changed"), ("certificate", "AuditLine")) if f in mk(V4)]
        put("AudFreshFourthUse", G("hidden", "verdict_fresh4", "use"), "str", S)
        put("AudFreshFourthCert", G("hidden", "verdict_fresh4", "certificate"), "str", S)
        pt4 = V4.get("per_task", {})
        put("AudFreshFourthTasks", lambda S_: len(S_["hidden"]["verdict_fresh4"]["per_task"]), "int", S)
        put("AudFreshFourthLow", lambda S_: sum(1 for v in S_["hidden"]["verdict_fresh4"]["per_task"].values() if v["own_at_end"] < 0.70), "int", S)
        ok4 = bool(pt4) and all("c" in v and "m" in v for v in pt4.values())
        if ok4:
            hasub = all("upper95" in v for v in pt4.values())
            c4 = c4[:3] + [put("AudFreshFourthAuditEst", est_acc("hidden", "verdict_fresh4", "per_task"), "pct1", S),
                           put("AudFreshFourthAuditBound", bound_acc("hidden", "verdict_fresh4", "per_task", computed=not hasub), "pct1", S)]
        else:
            c4 = c4[:3] + ["--", "--"]
        rows.append((r"fourth batch: frozen T5-base states, \recog{}", (c4 + ["--"] * 5)[:5], "\\AudFreshFourthUse{} / \\AudFreshFourthCert{}", "\\AudFreshFourthLow{}"))
        r2 = V4.get("second_rule_H_prime_t2")
        if r2:
            m2 = mk(r2)
            c5 = [put("AudFreshFourthH%s" % n_, G("hidden", "verdict_fresh4", "second_rule_H_prime_t2", *(("mean",) if "mean" in r2 else ()), f), "pct1", S) if f in m2 else "--"
                  for f, n_ in (("own_at_arrival", "Arrive"), ("own_at_end", "End"), ("changed", "Changed"))]
            rows.append((r"fourth batch: frozen T5-base states, \handicap{}, $\nu = 2$", c5 + ["--", "--"], "--", "--"))
    # ---- first and second fresh batch (boundary_study.json), sentence encoder
    group("(b2) boundary_study.json verdict_fresh2 (second batch, chosen variant), verdict_fresh (first batch, oldest wins), verdict_claims_fresh (first batch, newest wins)")
    BD = "boundary"
    pre = []
    V2 = (BD, "verdict_fresh2")
    assert S[BD]["verdict_fresh2"]["variant"] == S[BD]["chosen"]["variant"]
    c2 = [put("AudShapeFreshSecond%s" % n_, G(*V2, f), "pct1", S) for f, n_ in (("mean_own_at_arrival", "Arrive"), ("mean_own_at_end", "End"), ("mean_changed", "Changed"), ("mean_bound", "Bound"))]
    put("AudShapeFreshSecondUse", G(*V2, "use"), "str", S)
    put("AudShapeFreshSecondGuarantee", G(*V2, "guarantee"), "str", S)
    pre.append(("second batch: zones, newest wins, one boundary per older task, sentence encoder + shape numbers",
                [c2[0], c2[1], c2[2], "--", "--"], "\\AudShapeFreshSecondUse{} / \\AudShapeFreshSecondGuarantee{}", "--"))
    V1 = (BD, "verdict_fresh")
    V1c = (BD, "verdict_claims_fresh")
    c1 = [put("AudShapeFreshFirstOldest%s" % n_, G(*V1, f), "pct1", S) for f, n_ in (("mean_own_at_end", "End"), ("mean_blocked", "Blocked"), ("mean_changed", "Changed"))]
    put("AudShapeFreshFirstOldestUse", G(*V1, "use"), "str", S)
    put("AudShapeFreshFirstOldestClaim", G(*V1, "claim"), "str", S)
    c1c = [put("AudShapeFreshFirstNewest%s" % n_, G(*V1c, f), "pct1", S) for f, n_ in (("mean_own_at_arrival", "Arrive"), ("mean_own_at_end", "End"), ("mean_changed", "Changed"), ("mean_bound", "Bound"))]
    put("AudShapeFreshFirstNewestUse", G(*V1c, "use"), "str", S)
    put("AudShapeFreshFirstNewestGuarantee", G(*V1c, "guarantee"), "str", S)
    pre.insert(0, ("first batch: zones, newest wins, sentence encoder", [c1c[0], c1c[1], c1c[2], "--", "--"], "\\AudShapeFreshFirstNewestUse{} / \\AudShapeFreshFirstNewestGuarantee{}", "--"))
    pre.insert(0, ("first batch: zones, oldest wins, sentence encoder", ["--", c1[0], c1[2], "--", "--"], "\\AudShapeFreshFirstOldestUse{} / \\AudShapeFreshFirstOldestClaim{}", "--"))
    rows = pre + rows
    L = [r"{\scriptsize\setlength{\tabcolsep}{2pt}\begin{tabular}{p{0.95in}rrrrrcr}", r"    \toprule",
         r"    Fresh tasks & " + HARR + " & " + HEND + " & " + HCHG + " & " + HEST + " & " + HBND + r" & Verdict & \shortstack[r]{Tasks below\\70\% $\downarrow$} \\", r"    \midrule"]
    for lab, c, v, lo in rows:
        L.append("    %s & %s & %s & %s \\\\" % (lab, " & ".join(c), v, lo))
    L += [r"    \bottomrule", r"\end{tabular}}", ""]
    files["recogniser_fresh.tex"] = "\n".join(L)

    # ================================================================== boundary study: the four shapes of boundary
    group("(b2) boundary_study.json dev_grid[<feature>|<kind>|r1|<C>][superni_order1].mean: best variant at r = 1 by own_at_end, per kind of boundary")
    BG = S["boundary"]["dev_grid"]
    KINDS = (("ovr", "Ovr", "one boundary against everything"), ("pair", "Pair", "one boundary per older task"),
             ("pair-knn", "PairKnn", "balls around the task's inputs"), ("pair-rff", "PairRff", "curved boundaries"))
    chosen_key = S["boundary"]["chosen"]["variant"]

    def best_g(S_, feat_prefix, kind):
        g = S_["boundary"]["dev_grid"]
        ks = [k for k in g if k.split("|")[1] == kind and k.split("|")[2] == "r1" and
              (k.split("|")[0] == "minilm" if feat_prefix == "minilm" else k.split("|")[0].startswith("minilm+shape"))]
        return max(ks, key=lambda k: g[k]["superni_order1"]["mean"]["own_at_end"])
    assert len(BG) == 44
    L = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrrr}", r"    \toprule",
         r"    & \multicolumn{2}{c}{Sentence encoder} & \multicolumn{3}{c}{+ shape numbers} \\", r"    \cmidrule(lr){2-3}\cmidrule(l){4-6}",
         r"    Boundary & " + HEND + " & " + HCHG + " & " + HEND + " & " + HCHG + r" & weight \\", r"    \midrule"]
    rows_s = []
    for kind, kn, klab in KINDS:
        cells = []
        for fp, fn_ in (("minilm", "Mini"), ("shape", "Shape")):
            bk = best_g(S, fp, kind)
            isb = bk == chosen_key
            for f, n_ in (("own_at_end", "End"), ("changed", "Changed"), ("own_at_arrival", "Arrive"), ("bound", "Bound")):
                m = put("AudShape%s%s%s" % (kn, fn_, n_), lambda S_, fp=fp, kind=kind, f=f: S_["boundary"]["dev_grid"][best_g(S_, fp, kind)]["superni_order1"]["mean"][f], "pct1", S)
                if n_ in ("End", "Changed"):
                    cells.append(bold(m) if isb else m)
            if fp == "shape":
                w = put("AudShape%sShapeWeight" % kn, lambda S_, kind=kind: best_g(S_, "shape", kind).split("|")[0][len("minilm+shape"):], "str", S)
                cells.append(bold(w) if isb else w)
        rows_s.append((klab, cells, "shapes", True))
    assert any(best_g(S, "shape", k) == chosen_key for k, _, _ in KINDS), "the chosen variant is not the best of its kind"
    L += render(rows_s, [1, -1, 1, -1, None], "shapes_grid") + [r"    \bottomrule", r"\end{tabular}}", ""]
    files["shapes_grid.tex"] = "\n".join(L)

    # ================================================================== (g) conflict count
    group("(g) conflict_count.json [<collection>]: tasks, inputs, inputs_that_also_occur_in_another_task, share")
    COL = (("superni_15", "One", "SuperNI, 15 tasks"), ("long_sequence_15", "Two", "Long Sequence, 15 tasks"),
           ("fresh_batch_1", "Three", "fresh batch 1"), ("fresh_batch_2", "Four", "fresh batch 2"), ("fresh_batch_3", "Five", "fresh batch 3"))
    L = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrr}", r"    \toprule",
         r"    Collection & Tasks & Inputs & \shortstack[r]{Occur also\\in another\\task} & \shortstack[r]{Share,\\\%} \\", r"    \midrule"]
    for key, nm, lab in COL:
        d = S["conflict"][key]
        assert abs(d["share"] - d["inputs_that_also_occur_in_another_task"] / d["inputs"]) < 5e-6, key
        c = [put("AudConf%sTasks" % nm, G("conflict", key, "tasks"), "int", S), put("AudConf%sInputs" % nm, G("conflict", key, "inputs"), "thou", S),
             put("AudConf%sShared" % nm, G("conflict", key, "inputs_that_also_occur_in_another_task"), "thou", S),
             put("AudConf%sShare" % nm, G("conflict", key, "share"), "pct1", S)]
        L.append("    %s & %s \\\\" % (lab, " & ".join(c)))
    group("(g) conflict_count.json fresh_batch_3.pairs_with_at_least_1_percent: share of the inputs of each of the two tasks that occur in the other")

    def pair_share(S_):
        pr = S_["conflict"]["fresh_batch_3"]["pairs_with_at_least_1_percent"]
        assert len(pr) == 2 and pr[0]["share_of_inputs"] == pr[1]["share_of_inputs"]
        return pr[0]["share_of_inputs"]
    put("AudConfPairShare", pair_share, "pct1", S)
    L += [r"    \bottomrule", r"\end{tabular}}", ""]
    files["conflict_count.tex"] = "\n".join(L)

    # ================================================================== (f) routes of the GPU rules
    def first_key(S_, src, prefix, rule):
        return [k for k in S_[src]["measures"][rule] if k.startswith(prefix)][0]

    def n_test(S_, src, rule, last=True):
        pt = S_[src]["measures"][rule]["per_task"]
        return sum(v["n_test"] for v in pt.values()) - (0 if last else list(pt.values())[-1]["n_test"])
    ROUTE_RULES = (("ridge", "Ridge", r"\recog{}"), ("handicap", "Handicap", r"\handicap{}, $\nu = 2$"),
                   ("joint", "Joint", r"\recog{} with logistic regression"), ("own", "Own", "every input to its own task (ceiling)"))
    head = [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrr}", r"    \toprule",
            r"    Rule & \shortstack[r]{Not sent\\to own task\\at the end $\downarrow$} & \shortstack[r]{Route\\changed $\downarrow$} & " + HEST + " & " + HBND + r" \\", r"    \midrule"]
    rows_r = []
    for si, (src, sn, slab) in enumerate((("routes5", "Five", "five-task sequence"), ("routes15", "Fifteen", "15 SuperNI tasks"))):
        group("(f) %s measures[<rule>]: %s (not_own_at_end, routes_changed, per_task.n_test, mean.certificate)" % (os.path.basename(PATHS[src]), slab))
        if si:
            rows_r.append(("MID",))
        rows_r.append(("RAW", r"    \multicolumn{5}{l}{\textit{%s}} \\" % slab))
        for rule, rn, lab in ROUTE_RULES:
            pre = "AudRoute%s%s" % (sn, rn)
            a = lambda S_, src=src, rule=rule: S_[src]["measures"][rule][first_key(S_, src, "not_own_at_end", rule)]
            b = lambda S_, src=src, rule=rule: S_[src]["measures"][rule][first_key(S_, src, "routes_changed", rule)]
            no = put(pre + "NotOwn", a, "int", S)
            nn = put(pre + "N", lambda S_, src=src, rule=rule: n_test(S_, src, rule), "int", S)
            ch = put(pre + "Changed", b, "int", S)
            cm = put(pre + "Compared", lambda S_, src=src, rule=rule: n_test(S_, src, rule, last=False), "int", S)
            if src == "routes15":
                assert S[src]["measures"][rule]["compared"] == n_test(S, src, rule, last=False) and S[src]["measures"][rule]["test_inputs"] == n_test(S, src, rule)
            else:
                assert "%s_of_%d" % ("not_own_at_end", n_test(S, src, rule)) in S[src]["measures"][rule]
                assert ("_%d_compared" % n_test(S, src, rule, last=False)) in first_key(S, src, "routes_changed", rule)
            valid = S[src]["measures"][rule]["certificate_valid"]
            if valid:
                put(pre + "AuditLine", G(src, "measures", rule, "mean", "certificate"), "pct1", S)
                au = put(pre + "AuditEst", est_acc(src, "measures", rule, "per_task"), "pct1", S)
                ab = put(pre + "AuditBound", bound_acc(src, "measures", rule, "per_task"), "pct1", S)
            else:
                au = ab = "--"
            rows_r.append((lab, ["%s of %s" % (no, nn), "%s of %s" % (ch, cm), au, ab], "routes" + sn, rule != "own"))
    files["routes_gpu.tex"] = "\n".join(head + render(rows_r, [-1, -1, None, None], "routes_gpu") + [r"    \bottomrule", r"\end{tabular}}", ""])
    group("(f) noinstruction_routes.json measures.ridge.per_task: first four tasks (the compared ones) and the share of changed routes")
    rp = lambda S_: list(S_["routes5"]["measures"]["ridge"]["per_task"].values())
    put("AudRouteFiveRidgeFirstFourAuditEst", lambda S_: sum(v["c"] / v["m"] for v in rp(S_)[:4]) / 4, "pct1", S)
    put("AudRouteFiveRidgeFirstFourBoundLow", lambda S_: min(v["upper95"] for v in rp(S_)[:4]), "pct1", S)
    put("AudRouteFiveRidgeFirstFourBoundHigh", lambda S_: max(v["upper95"] for v in rp(S_)[:4]), "pct1", S)
    put("AudRouteFiveRidgeChangedShare", lambda S_: S_["routes5"]["measures"]["ridge"][first_key(S_, "routes5", "routes_changed", "ridge")] / n_test(S_, "routes5", "ridge", last=False), "pct1", S)
    fullours_macros(S)
    files.update(runu_section(S))
    files.update(full_section(S))
    files.update(fulldef_section(S))
    files.update(fifteen_section(S))
    files.update(seeds_section(S))
    files.update(scale_section(S))
    files.update(short_table())
    return files


# ------------------------------------------------------------------ the check
def parse_macros(path):
    txt = open(path, encoding="utf-8").read()
    return dict(re.findall(r"^\\newcommand\{\\(\w+)\}\{(.*)\}$", txt, flags=re.M))


def check_all(files):
    """Opens every source again and compares each macro of the written file with the rounded source value."""
    S2 = load_sources()
    written = parse_macros(os.path.join(OUT, "numbers_audit.tex"))
    bad, n = [], 0
    for name, (val, acc, kind) in REG.items():
        n += 1
        if name not in written:
            bad.append((name, "not written", val))
            continue
        expect = fmt(kind, acc(S2))
        if written[name] != expect:
            bad.append((name, written[name], expect))
    extra = set(written) - set(REG)
    for e in sorted(extra):
        bad.append((e, "written but not registered", ""))
    # every macro used by a table is defined
    for fname, txt in files.items():
        txt2 = open(os.path.join(OUT, fname), encoding="utf-8").read()
        assert txt2 == txt
        for used in set(re.findall(r"\\(Aud[A-Za-z]+)\{\}", txt2)):
            if used not in written:
                bad.append((used, "used in " + fname + " but not defined", ""))
    if bad:
        print("MISMATCHES (%d):" % len(bad))
        for b in bad:
            print("  %s: written %r, source %r" % b)
        return False
    print("all %d numbers verified" % n)
    return True


def main():
    S = load_sources(verbose=True)
    files = build(S)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "numbers_audit.tex"), "w", encoding="utf-8") as f:
        f.write("% written by audit_results.py; do not edit\n")
        for comment, names in GROUPS:
            if not names:
                continue
            f.write("% " + comment + "\n")
            for nme in names:
                f.write("\\newcommand{\\%s}{%s}\n" % (nme, REG[nme][0]))
    for fname, txt in files.items():
        open(os.path.join(OUT, fname), "w", encoding="utf-8").write(txt)
    print("wrote numbers_audit.tex (%d macros) and %d tables to %s" % (len(REG), len(files), OUT))
    if os.environ.get("AUD_BOLD"):
        for t, g, lab, j, nm in BOLD:
            print("bold", t, g, lab, "col", j, nm)
    ok = check_all(files)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
