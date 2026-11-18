"""Texts that occur word for word in the inputs of two tasks (a description, not a test): the part of the 'price of having no instruction'
that can be counted directly. For every collection: for each ordered pair (A, B), the share of the inputs of A that also occur among the
inputs of B (after stripping leading and trailing white space). The fourth fresh batch is not opened.

    python3 conflict_count.py     # writes routing/conflict_count.json
"""
import json, os
S = os.environ.get("FK_LORA_SCRATCH", os.path.join(os.path.expanduser("~"), ".cache", "fk_lora"))   # scratch folder: models, data, the Python environment
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(S, "ext_gainlora", "repo", "CL_Benchmark")
ORDER1 = ["task1572_samsum_summary", "task363_sst2_polarity_classification", "task1290_xsum_summarization", "task181_outcome_extraction", "task002_quoref_answer_generation",
          "task1510_evalution_relation_extraction", "task639_multi_woz_user_utterance_generation", "task1729_personachat_generate_next", "task073_commonsenseqa_answer_generation",
          "task1590_diplomacy_text_generation", "task748_glucose_reverse_cause_event_detection", "task511_reddit_tifu_long_text_summarization", "task591_sciq_answer_generation",
          "task1687_sentiment140_classification", "task875_emotion_classification"]
LONG = "yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic".split(",")
def bench(folder, names):
    out = {}
    for n in names:
        x = []
        for sp in ("train", "test"):
            x += [str(i["input"]).strip() for i in json.load(open(os.path.join(B, folder, n, sp + ".json")))["Instances"]]
        out[n] = x
    return out
def fresh(f):
    return {t["name"]: [str(i["input"]).strip() for i in t["instances"]] for t in json.load(open(os.path.join(S, "ni_fresh", f)))["tasks"]}
cols = {"superni_15": bench("SuperNI", ORDER1), "long_sequence_15": bench("Long_Sequence", LONG),
        "fresh_batch_1": fresh("fresh_tasks.json"), "fresh_batch_2": fresh("fresh_tasks_2.json"), "fresh_batch_3": fresh("fresh_tasks_3.json")}
res = {"note": __doc__.split("\n\n")[0]}
for c, tasks in cols.items():
    sets = {n: set(x) for n, x in tasks.items()}; pairs = []; tot = 0; shared_any = 0
    for a, xa in tasks.items():
        others = set().union(*[sets[b] for b in sets if b != a]) if len(sets) > 1 else set()
        tot += len(xa); shared_any += sum(x in others for x in xa)
        for b in tasks:
            if a == b: continue
            sh = sum(x in sets[b] for x in xa) / len(xa)
            if sh >= 0.01: pairs.append({"task": a, "also_in": b, "share_of_inputs": round(sh, 4), "inputs": len(xa)})
    res[c] = {"tasks": len(tasks), "inputs": tot, "inputs_that_also_occur_in_another_task": shared_any, "share": round(shared_any / tot, 5),
              "pairs_with_at_least_1_percent": sorted(pairs, key=lambda p: -p["share_of_inputs"])}
    print(f"{c:18s} tasks {len(tasks):3d} inputs {tot:7d} also in another task {shared_any:6d} ({100 * shared_any / tot:.2f}%)")
    for p in res[c]["pairs_with_at_least_1_percent"][:8]: print(f"      {p['task'][:44]:44s} in {p['also_in'][:44]:44s} {100 * p['share_of_inputs']:.1f}%")
json.dump(res, open(os.path.join(HERE, "routing", "conflict_count.json"), "w"), indent=1)
