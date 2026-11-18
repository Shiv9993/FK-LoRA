"""Step 1: machine paraphrases of every task definition by back-translation (MarianMT, beam 4, no sampling).
en->de->en and en->fr->en, sentence by sentence. Output: routing/paraphrases.json (used as they come).
Other pivots, kept apart as an untouched check: `python paraphrase_make.py es paraphrases_es.json`."""
import json, os, re, sys, time, collections
import torch
from transformers import MarianMTModel, MarianTokenizer
S = os.environ.get("FK_LORA_SCRATCH", os.path.join(os.path.expanduser("~"), ".cache", "fk_lora"))   # scratch folder: models, data, the Python environment
HERE = os.path.dirname(os.path.abspath(__file__))
PIVOTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["de", "fr"]
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "paraphrases.json"
tasks = json.load(open(f"{S}/ni/ni_tasks.json"))["tasks"]
split = lambda d: [s for s in re.split(r"(?<=[.!?])\s+", d.strip()) if s]
CACHE = f"{S}/feature_study/para_cache.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def translate(texts, name):
    key = lambda t: name + "|" + t
    todo = sorted({t for t in texts if key(t) not in cache}, key=len)
    if not todo: return
    tok = MarianTokenizer.from_pretrained(name); m = MarianMTModel.from_pretrained(name).eval()
    t0 = time.time()
    for s in range(0, len(todo), 24):
        b = todo[s:s + 24]
        enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            out = m.generate(**enc, num_beams=4, do_sample=False, max_length=256)
        for t, o in zip(b, tok.batch_decode(out, skip_special_tokens=True)): cache[key(t)] = o
        if (s // 24) % 20 == 0:
            print(name, s, len(todo), f"{time.time()-t0:.0f}s", flush=True)
            json.dump(cache, open(CACHE, "w"))
    json.dump(cache, open(CACHE, "w"))

res = {t["name"]: {} for t in tasks}
for p in PIVOTS:
    sents = sorted({s for t in tasks for s in split(t["definition"])})
    print(p, "sentences", len(sents), flush=True)
    translate(sents, f"Helsinki-NLP/opus-mt-en-{p}")
    mid = {s: cache[f"Helsinki-NLP/opus-mt-en-{p}|{s}"] for s in sents}
    translate(sorted(set(mid.values())), f"Helsinki-NLP/opus-mt-{p}-en")
    for t in tasks:
        res[t["name"]][p] = " ".join(cache[f"Helsinki-NLP/opus-mt-{p}-en|{mid[s]}"] for s in split(t["definition"]))
norm = lambda s: " ".join(s.lower().split())
words = lambda s: re.findall(r"\w+", s.lower())
stats = {}
for p in PIVOTS:
    ident = sum(norm(res[t["name"]][p]) == norm(t["definition"]) for t in tasks)
    kept = [len(collections.Counter(words(t["definition"])) & collections.Counter(words(res[t["name"]][p]))) and
            sum((collections.Counter(words(t["definition"])) & collections.Counter(words(res[t["name"]][p]))).values()) / max(1, len(words(t["definition"]))) for t in tasks]
    stats[p] = {"identical": ident, "of": len(tasks), "mean_share_original_words_kept": sum(kept) / len(kept)}
    print(p, stats[p])
json.dump({"tasks": res, "stats": stats}, open(f"{HERE}/routing/{OUT_NAME}", "w"), indent=1)
print("done")
