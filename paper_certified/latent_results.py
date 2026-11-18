"""Tables and number commands for the re-implementation of Latent-LoRA (appendix app:latent).

    python3 latent_results.py

Reads ../silent_adapters/results/2026-11-17_kaggle_latent_lora_five_tasks/small_files.zip
(summary.json and results/{ours,latent_lora}_seed{42,43,44}.json inside) and writes
  tables/latent_main.tex      scores, original instructions, mean over three seeds
  tables/latent_wordings.tex  AP, answers changed and right task for the four wordings
  tables/latent_pertask.tex   score of each task right after it was learned, and last-epoch loss
  tables/numbers_latent.tex   one \\newcommand per number used in the text (\\Latent...)
Every number is read from the zip; nothing is typed by hand. Run it again if the archive changes.
"""
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "..", "silent_adapters", "results", "2026-11-17_kaggle_latent_lora_five_tasks", "small_files.zip")
SEEDS = (42, 43, 44)
WORDINGS = ("original", "light", "medium", "heavy")
TASKS = [("Relation", "relation extr."), ("Sst", "sst2"), ("Glucose", "glucose"),
         ("Senti", "sentiment140"), ("Emotion", "emotion")]

z = zipfile.ZipFile(ZIP)
summary = json.loads(z.read("latent_lora_five/summary.json"))["table"]
runs = {(m, s): json.loads(z.read("latent_lora_five/results/%s_seed%d.json" % (m, s)))
        for m in ("ours", "latent_lora") for s in SEEDS}
settings = runs[("ours", 42)]["settings"]


def row(method, rule, wording):
    """Mean over the three seeds (the summary file's own aggregate); changed and compared are sums."""
    r = summary["%s|%s|%s" % (method, rule, wording)]
    assert r["seeds"] == list(SEEDS) and r["tasks_done"] == [5, 5, 5], "run not complete"
    return r


def thousands(n):
    return "{:,}".format(int(n)).replace(",", "{,}")


def f2(x):
    return "%.2f" % x


def f1(x):
    return "%.1f" % x


def pct(x):
    return "%.1f" % (100 * x)


def changed(r):
    return "%s of %s" % (thousands(r["changed"]), thousands(r["compared"]))


def diag(method, rule, wording, seed):
    sm = runs[(method, seed)]["metrics"]["%s|%s" % (wording, rule)]["score_matrix"]
    return [sm[i][i] for i in range(5)]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def loss(method, i):
    return mean(runs[(method, s)]["steps"][i]["loss_last_epoch"] for s in SEEDS)


# ---------------------------------------------------------------- table 1
cert, allon = row("ours", "certified", "original"), row("ours", "all", "original")
soft, sumr = row("latent_lora", "soft", "original"), row("latent_lora", "sum", "original")
main_rows = [(r"\cert{} (ours)", cert), (r"\plain{} (same adapters, all on)", allon),
             ("Latent-LoRA, soft rule", soft),
             ("Latent-LoRA, sum rule", sumr)]
t = [r"\begin{tabular}{lccccc}", r"    \toprule",
     r"    Method & Learned & AP $\uparrow$ & FT $\downarrow$ & Changed (of 1{,}200) & Right task (\%) \\",
     r"    \midrule"]
for i, (name, r) in enumerate(main_rows):
    if i == 2:
        t.append(r"    \midrule")
    t.append("    %s & %s & %s & %s & %s & %s \\\\" % (name, f2(r["learned"]), f2(r["AP"]), f2(r["FT"]), thousands(r["changed"]),
                                                     "--" if r in (allon, sumr) else pct(r["right_task"])))  # no router: nothing to score
t += [r"    \bottomrule", r"\end{tabular}", ""]
open(os.path.join(HERE, "tables", "latent_main.tex"), "w", encoding="utf-8").write("\n".join(t))

# ---------------------------------------------------------------- table 2
t = [r"\begin{tabular}{lcccccc}", r"    \toprule",
     r"    & \multicolumn{3}{c}{\cert{} (ours)} & \multicolumn{3}{c}{Latent-LoRA, soft rule} \\",
     r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
     r"    Wording & AP $\uparrow$ & Changed & Right (\%) & AP $\uparrow$ & Changed & Right (\%) \\",
     r"    \midrule"]
for w in WORDINGS:
    a, b = row("ours", "certified", w), row("latent_lora", "soft", w)
    t.append("    %s & %s & %s & %s & %s & %s & %s \\\\" % (w, f2(a["AP"]), thousands(a["changed"]), pct(a["right_task"]),
                                                         f2(b["AP"]), thousands(b["changed"]), pct(b["right_task"])))
t += [r"    \bottomrule", r"\end{tabular}", ""]
open(os.path.join(HERE, "tables", "latent_wordings.tex"), "w", encoding="utf-8").write("\n".join(t))

# ---------------------------------------------------------------- table 3
ours_sc = [mean(diag("ours", "certified", "original", s)[i] for s in SEEDS) for i in range(5)]
soft_sc = [mean(diag("latent_lora", "soft", "original", s)[i] for s in SEEDS) for i in range(5)]
t = [r"\begin{tabular}{lcccc}", r"    \toprule",
     r"    & \multicolumn{2}{c}{Score right after learning} & \multicolumn{2}{c}{Training loss, last epoch} \\",
     r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5}",
     r"    Task (in order) & \cert{} (ours) & Latent-LoRA & \cert{} (ours) & Latent-LoRA \\", r"    \midrule"]
for i, (_, label) in enumerate(TASKS):
    t.append("    %s & %s & %s & %s & %s \\\\" % (label, f1(ours_sc[i]), f1(soft_sc[i]), f2(loss("ours", i)), f2(loss("latent_lora", i))))
t += [r"    \bottomrule", r"\end{tabular}", ""]
open(os.path.join(HERE, "tables", "latent_pertask.tex"), "w", encoding="utf-8").write("\n".join(t))

# ---------------------------------------------------------------- numbers
N = {}
for key, r in (("Cert", cert), ("All", allon), ("Soft", soft), ("Sum", sumr)):
    N[key + "Learned"], N[key + "AP"], N[key + "FT"] = f2(r["learned"]), f2(r["AP"]), f2(r["FT"])
    N[key + "Changed"], N[key + "Right"] = thousands(r["changed"]), pct(r["right_task"])
N["Compared"] = thousands(cert["compared"])
for key, method, rule in (("Cert", "ours", "certified"), ("Soft", "latent_lora", "soft")):
    for w, W in zip(WORDINGS, ("Original", "Light", "Medium", "Heavy")):
        r = row(method, rule, w)
        N["%sAP%s" % (key, W)], N["%sChanged%s" % (key, W)], N["%sRight%s" % (key, W)] = f2(r["AP"]), thousands(r["changed"]), pct(r["right_task"])
N["AllAPLight"] = f2(row("ours", "all", "light")["AP"])
# seed 42, soft rule, original instructions: score right after learning each task; mean last-epoch loss
d42 = diag("latent_lora", "soft", "original", 42)
for i, (name, _) in enumerate(TASKS):
    N["SeedSoft" + name] = f1(d42[i])
N["LossSoftGlucose"], N["LossCertGlucose"] = f1(loss("latent_lora", 2)), f1(loss("ours", 2))
# Sentiment140 under the light rewording, range over seeds
sl = [diag("latent_lora", "soft", "light", s)[3] for s in SEEDS]
N["SoftSentOriginal"] = f1(mean(diag("latent_lora", "soft", "original", s)[3] for s in SEEDS))
N["SoftSentLightLow"], N["SoftSentLightHigh"] = "%d" % round(min(sl)), "%d" % round(max(sl))
N["SoftSentMediumMean"] = f1(mean(diag("latent_lora", "soft", "medium", s)[3] for s in SEEDS))
# settings
N["Epochs"], N["TrainInputs"], N["TestInputs"] = str(settings["epochs"]), str(settings["train"]), str(settings["test"])
N["Seeds"] = {3: "three"}[len(SEEDS)]
N["Rank"], N["LLRank"] = str(settings["ours_rank"]), str(settings["ll_rank"])

for k in N:
    assert k.isalpha(), k
open(os.path.join(HERE, "tables", "numbers_latent.tex"), "w", encoding="utf-8").write(
    "".join("\\newcommand{\\Latent%s}{%s}\n" % (k, N[k]) for k in sorted(N)))
print("wrote 3 tables and %d number commands" % len(N))
