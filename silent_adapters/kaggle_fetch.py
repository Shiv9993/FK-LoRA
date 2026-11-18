"""Check a Kaggle notebook run and, when it has finished, fetch its log and its small result files.

    python3 kaggle_fetch.py <kaggle user> <notebook slug> [name for the results folder]

It reads Kaggle's public API with `curl`. No token is written here or read from a file: the
session this was made in adds the login to requests for www.kaggle.com by itself. Anywhere else,
set KAGGLE_API_TOKEN and the script sends it as a Bearer token.

What is fetched: the run's log (as plain text) and every .json, .jsonl, .log and .txt file of its
output, except tokenizer files. Weights are not fetched. Everything goes into
results/<name>/ as `log.txt` and `small_files.zip`.
"""
import json
import os
import subprocess
import sys
import urllib.parse
import zipfile

API = "https://www.kaggle.com/api/v1/kernels/"
HERE = os.path.dirname(os.path.abspath(__file__))


def curl(url, out=None):
    cmd = ["curl", "-sS", "-L", "--max-time", "180"]
    if os.environ.get("KAGGLE_API_TOKEN") and url.startswith(API):
        cmd += ["-H", "Authorization: Bearer " + os.environ["KAGGLE_API_TOKEN"]]
    if out:
        cmd += ["-o", out, "-w", "%{http_code}"]
    r = subprocess.run(cmd + [url], capture_output=True, text=True)
    return r.stdout


def status(user, slug):
    return json.loads(curl(f"{API}status?userName={user}&kernelSlug={slug}"))


def fetch(user, slug, name):
    files, token, log = [], None, ""
    while True:
        page = json.loads(curl(f"{API}output?userName={user}&kernelSlug={slug}"
                               + (f"&pageToken={urllib.parse.quote(token)}" if token else "")))
        log = log or page.get("log") or ""
        files += page.get("files", [])
        token = page.get("nextPageToken")
        if not token:
            break
    dest = os.path.join(HERE, "results", name)
    os.makedirs(dest, exist_ok=True)
    try:                                               # the log is a list of pieces with a time stamp each
        text = "".join(piece.get("data", "") for piece in json.loads(log))
    except ValueError:
        text = log
    open(os.path.join(dest, "log.txt"), "w").write(text)
    want = [f for f in files if f["fileName"].endswith((".json", ".jsonl", ".log", ".txt"))
            and not os.path.basename(f["fileName"]).startswith(("tokenizer", "special_tokens", "spiece", "added_tokens"))]
    tmp = os.path.join(dest, "_tmp")
    failed = 0
    with zipfile.ZipFile(os.path.join(dest, "small_files.zip"), "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in want:
            code = curl(f["url"], out=tmp)
            if code.strip() == "200":
                z.write(tmp, f["fileName"])
            else:
                failed += 1
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"{len(files)} output files listed; {len(want) - failed} small files saved, {failed} failed; log: {len(text)} characters")
    print("saved in", dest)
    return dest


if __name__ == "__main__":
    user, slug = sys.argv[1], sys.argv[2]
    s = status(user, slug)
    print(f"{user}/{slug}: {s.get('status')}", s.get("failureMessage") or "")
    if s.get("status") in ("complete", "error", "cancelAcknowledged"):
        fetch(user, slug, sys.argv[3] if len(sys.argv) > 3 else slug)
