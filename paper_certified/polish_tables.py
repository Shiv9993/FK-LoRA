#!/usr/bin/env python3
"""Post-process the generated tables/*.tex: plain-word column headers (no bold is added).

Run it AFTER the generator scripts (they overwrite the tables):  python3 polish_tables.py

* Driven only by the explicit per-table configuration TABLES below; nothing is guessed.
* Idempotent: a header is replaced only when its old text is still there, a cell is
  bolded only when it is not bold yet.  Running it twice changes nothing more.
* Never changes a number: headers are text, bold wraps the leading number of a cell.
* Tables written by audit_results.py are not in the configuration and are never touched.

Header configuration: {old header text: new header}.  A new header is a string or a list of
lines; a list becomes \\shortstack[a]{line\\\\line} (a = alignment of the column, taken from
the tabular column specification) or, inside a \\multicolumn, one line.  Only the rows above
the first \\midrule are header rows.

Bold configuration: a list of rules {cols, mode, block}.  mode "max"/"min" says which value is
best; block "all" compares all rows of the table, "midrule" compares only the rows between
two \\midrule lines.  Ties are all bold.  The leading number of a cell is compared and wrapped.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tables")

class Stack(list):
    """Lines that stay stacked (\\shortstack) also inside a \\multicolumn."""


UP = r"$\uparrow$"
DOWN = r"$\downarrow$"

# ---- shared header texts ------------------------------------------------------------------
LEARNED = ["Score", "right after", "learning " + UP]
LEARNED_NARROW = ["Score", "right", "after", "learning", UP]
FINAL = "Final score " + UP
FINAL_STACK = ["Final", "score " + UP]
FINAL_NARROW = ["Final", "score", UP]
FT = ["Forgetting", "(points) " + DOWN]
FT_NARROW = ["Forgetting", "(points)", DOWN]
CHG = ["Answers", "changed " + DOWN]
RCHG = ["Routes", "changed " + DOWN]
OWN = ["Sent to own", "task, \\% " + UP]
OWN_NOPCT = ["Sent to own", "task " + UP]
CERTIN = "Certified inputs"

# widen_*: the letters are defined in the captions (Section "paraphrase", appendix widen)
W_C = ["tasks that", "collide", r"(\textsf{C}), \% " + DOWN]
W_U = ["own task,", "unedited", r"(\textsf{U}), \% " + UP]
W_P = ["right", "certificate", r"(\textsf{P}), \% " + UP]
W_M = ["wrong", "certificate", r"(\textsf{M}), \% " + DOWN]
W_R = ["own task,", "paraphrased", r"(\textsf{R}), \% " + UP]
W_Z = ["inside own", "zone", r"(\textsf{Z}), \% " + UP]
_W = {"C": W_C, "U": W_U, "P": W_P, "M": W_M, "R": W_R, "Z": W_Z}


def widen(letters):
    """Headers for the letters that the table has as column headers."""
    return {"\\textsf{%s}" % c: _W[c] for c in letters}


TABLES = {
    # ---------------------------------------------------------------- main comparisons
    "five_main": dict(
        headers={"Learned": LEARNED, "AP $\\uparrow$": FINAL_STACK, "FT $\\downarrow$": FT,
                 "Changed": CHG, "Certified": CERTIN},
        bold=[]),
    "fifteen_main": dict(
        headers={"Learned": LEARNED, "AP $\\uparrow$": FINAL_STACK, "FT $\\downarrow$": FT,
                 "Changed": CHG, "Certified": CERTIN},
        bold=[]),
    "five_official": dict(
        headers={"Learned": LEARNED, "AP $\\uparrow$": FINAL_STACK, "FT $\\downarrow$": FT},
        bold=[]),
    "fifteen_official": dict(
        headers={"Learned": LEARNED, "AP $\\uparrow$": FINAL_STACK, "FT $\\downarrow$": FT},
        bold=[]),
    "five_wordings": dict(
        headers={"AP $\\uparrow$": FINAL,
                 "Answers changed (of 1{,}200)": "Answers changed (of 1{,}200) " + DOWN},
        bold=[]),
    "fifteen_wordings": dict(
        headers={"AP $\\uparrow$": FINAL,
                 "Answers changed": "Answers changed " + DOWN},
        bold=[]),
    "five_wordings_more": dict(
        headers={"Learned": LEARNED_NARROW, "FT": FT_NARROW}, tabcolsep="3pt",
        bold=[]),
    "fifteen_wordings_more": dict(
        headers={"Learned": LEARNED_NARROW, "FT": FT_NARROW}, tabcolsep="3pt",
        bold=[]),
    # rows are wordings, the two rules are column groups: no bold
    "five_routes": dict(headers={"Right task": OWN_NOPCT, "Answers changed": CHG}, bold=[]),
    "fifteen_routes": dict(headers={"Right task": OWN_NOPCT, "Answers changed": CHG}, bold=[]),
    # per-task listings "learned -> final": no bold, headers are already words
    "five_pertask": dict(headers={}, bold=[]),
    "fifteen_pertask": dict(headers={}, bold=[]),
    # ---------------------------------------------------------------- latent adapters
    "latent_main": dict(
        headers={"Learned": LEARNED, "AP $\\uparrow$": FINAL_STACK, "FT $\\downarrow$": FT,
                 "Changed (of 1{,}200)": ["Answers", "changed (of", "1{,}200) " + DOWN],
                 "Right task (\\%)": OWN},
        bold=[]),
    "latent_wordings": dict(
        headers={"AP $\\uparrow$": FINAL_STACK, "Changed": CHG,
                 "Right (\\%)": OWN}, bold=[]),
    "latent_pertask": dict(
        headers={"Score right after learning": "Score right after learning " + UP,
                 "Training loss, last epoch": "Training loss, last epoch " + DOWN},
        bold=[]),
    # ---------------------------------------------------------------- models, demo, orders
    "demo": dict(
        headers={"Learned": LEARNED_NARROW, "AP": ["Final", "score", UP], "Changed": CHG,
                 "Certified": ["Certified", "inputs"]},
        bold=[]),
    "models": dict(
        headers={"$d$": ["Embedding", "dimension", "$d$"],
                 "certified, \\%": ["certified", "inputs, \\%"],
                 "right task, \\%": ["sent to", "own task,", "\\% " + UP],
                 "Changed": ["Routes", "changed", DOWN]},
        bold=[]),
    "orders": dict(
        headers={"Right task, \\%": OWN, "Certified, \\%": ["Certified", "inputs, \\%"],
                 "Changed": RCHG},
        bold=[]),
    "kappa": dict(
        headers={"Five: certified, \\%": ["Five tasks:", "certified inputs, \\%"],
                 "Five: right task, \\%": ["Five tasks: sent to", "own task, \\% " + UP],
                 "Fifteen: right task, \\%": ["Fifteen tasks: sent to", "own task, \\% " + UP]},
        bold=[]),
    "background": dict(
        headers={"certified, \\%": ["certified", "inputs, \\%"],
                 "right task, \\%": ["sent to own", "task, \\% " + UP],
                 "right task, heavy, \\%": ["sent to own task,", "heavy wording, \\% " + UP]},
        bold=[]),
    "stress": dict(
        headers={"orig.": "original", "med.": "medium",
                 "Routes changed, \\cert{}": "Routes changed, \\cert{} " + DOWN,
                 "Routes changed, \\nearest{}": "Routes changed, \\nearest{} " + DOWN},
        bold=[]),
    "distances": dict(headers={}, bold=[]),
    "geometry": dict(
        headers={"$n_k$": ["Opening", "length", "$n_k$"], "$R_k$": ["Radius", "$R_k$"]},
        tabcolsep="2pt", bold=[]),
    # ---------------------------------------------------------------- edit tests
    "edits": dict(
        headers={"Certified, \\%": "Certified inputs, \\%", "Right task, \\%": OWN},
        bold=[]),
    "edits_full": dict(
        headers={"Certified, \\%": "Certified inputs, \\%",
                 "Right task, \\%": "Sent to own task, \\% " + UP},
        bold=[]),
    "edits_pertask": dict(
        headers={"Certified, \\%": "Certified inputs, \\%",
                 "Right task, \\%": "Sent to own task, \\% " + UP,
                 "$n_k$": ["Opening", "length", "$n_k$"],
                 "$100\\,R_k/D$": ["Safe edit", "share, \\%", "$100\\,R_k/D$"],
                 "16 drop.": ["16", "dropped"], "16 repl.": ["16", "replaced"]},
        tabcolsep="3pt", bold=[]),
    # ---------------------------------------------------------------- many tasks
    "many": dict(headers={}, bold=[]),
    "many_examples": dict(headers={}, bold=[]),
    "many_kappa": dict(headers={"in \\%": ["Share of", "tasks, \\%"]}, tabcolsep="3pt", bold=[]),
    "many_orders": dict(headers={"Changed later": "Routes changed later " + DOWN}, bold=[]),
    # ---------------------------------------------------------------- no-instruction bound
    "bound_pilot": dict(
        headers={"Changed, \\%": ["Routes", "changed, \\% " + DOWN],
                 "Blocked, \\%": ["Blocked inputs,", "\\% " + DOWN],
                 "Own task, \\%": OWN},
        bold=[]),
    "bound_pilotb": dict(
        headers={"Learned": LEARNED, "End": FINAL, "Forgetting": FT, "Changed": CHG,
                 "Own, \\%": OWN},
        bold=[]),
    "bound_minilm": dict(headers={}, bold=[]),      # prose, not a table
    "rewordings": dict(headers={}, bold=[]),        # list of paragraphs, not a table
    # ---------------------------------------------------------------- re-routing rules
    "reroute": dict(
        headers={"Own, \\%": OWN,
                 "Italian: own, \\%": ["Italian pivot: sent", "to own task, \\% " + UP],
                 "Changed, \\% (n)": ["Routes changed,", "\\% (n) " + DOWN],
                 "In zone, \\%": ["Inside a zone", "at arrival, \\%"],
                 "Unedited, \\%": ["Unedited, sent to", "own task, \\% " + UP]},
        bold=[]),
    "reroute_bench": dict(
        headers={"Medium (n)": ["Medium wording:", "own task, \\% " + UP, "(routes changed)"],
                 "Heavy (n)": ["Heavy wording:", "own task, \\% " + UP, "(routes changed)"]},
        bold=[]),
    "reroute_stress": dict(
        headers={w: ["Routes", "changed,", w + " " + DOWN]
                 for w in ("original", "light", "medium", "heavy")},
        bold=[]),
    # ---------------------------------------------------------------- widen the zones
    # letters kept: with the words the table is wider than the page
    "widen_main": dict(headers={}, bold=[]),
    "widen_encoder": dict(headers=widen("CUPMRZ"), bold=[]),
    "widen_judge": dict(headers=widen("CZPMR"), bold=[]),
    "widen_siblings": dict(headers={}, bold=[]),
    "widen_spanish": dict(
        headers={},                 # letters kept: with the words the table does not fit
        bold=[]),
    # rows are the measures (the letters are written out in the row labels)
    "widen_splits": dict(
        headers={},
        cells={r"\textsf{C}": r"Tasks that collide (\textsf{C}), \% " + DOWN,
               r"\textsf{P}": r"Right certificate (\textsf{P}), \% " + UP,
               r"\textsf{M}": r"Wrong certificate (\textsf{M}), \% " + DOWN,
               r"\textsf{R}": r"Sent to own task (\textsf{R}), \% " + UP}),
}

# ------------------------------------------------------------------------------------------
NUM = re.compile(r"^(\\textbf\{)?(\$-\$)?(\d+(?:(?:\{,\}|,)\d{3})*(?:\.\d+)?)")
SPLIT = re.compile(r"(?<!\\)&")
MULTI = re.compile(r"^\\multicolumn\{(\d+)\}\{([^}]*)\}\{(.*)\}$", re.S)


def is_row(line):
    return line.rstrip().endswith("\\\\")


def split_row(line):
    return SPLIT.split(line.rstrip()[:-2])


def join_row(parts):
    return "&".join(parts) + "\\\\"


def colspec(text):
    m = re.search(r"\\begin\{tabular\}\{([^}]*)\}", text)
    spec = m.group(1)
    assert set(spec) <= set("lcr"), spec
    return spec


def make_header(lines, align, in_multicolumn):
    if isinstance(lines, str):
        return lines
    if isinstance(lines, Stack):
        return "\\shortstack[c]{%s}" % "\\\\".join(lines)
    if in_multicolumn:
        return " ".join(lines)
    return "\\shortstack[%s]{%s}" % (align, "\\\\".join(lines))


def lead(cell):
    """Leading number of a cell, or None for '' and '--'."""
    c = cell.strip()
    if c in ("", "--"):
        return None
    m = NUM.match(c)
    if not m:
        raise ValueError("cannot read a number from %r" % cell)
    v = float(m.group(3).replace("{,}", "").replace(",", ""))
    return -v if m.group(2) else v


def bold_cell(cell):
    """Wrap the leading number (with its minus sign) of a cell in \\textbf."""
    pre = cell[:len(cell) - len(cell.lstrip())]
    post = cell[len(cell.rstrip()):]
    c = cell.strip()
    m = NUM.match(c)
    assert m and not m.group(1)
    start = m.start(2) if m.group(2) else m.start(3)
    return pre + c[:start] + "\\textbf{" + c[start:m.end()] + "}" + c[m.end():] + post


def is_bold(cell):
    return cell.strip().startswith("\\textbf{")


def pick(vals, mode):
    vals = [v for v in vals if v is not None]
    if mode == "min_nonneg":
        vals = [v for v in vals if v >= 0]
        mode = "min"
    if not vals:
        return None
    return max(vals) if mode == "max" else min(vals)


def process(name, cfg):
    path = os.path.join(TAB, name + ".tex")
    if not os.path.exists(path):
        return "%-22s MISSING" % name
    text = open(path, encoding="utf-8").read()
    if "\\begin{tabular}" not in text:
        return "%-22s not a tabular (prose/list): nothing to do" % name
    lines = text.split("\n")
    spec = colspec(text)
    first_mid = next(i for i, l in enumerate(lines) if l.strip() == "\\midrule")
    hdr_changed = 0
    seen = set()

    # ---- headers (rows above the first \midrule)
    for i in range(first_mid):
        if not is_row(lines[i]):
            continue
        parts = split_row(lines[i])
        col = 0
        changed = False
        for j, raw in enumerate(parts):
            cell = raw.strip()
            span, align, inner, m = 1, spec[col] if col < len(spec) else "l", cell, MULTI.match(cell)
            if m:
                span, align, inner = int(m.group(1)), m.group(2), m.group(3)
            if inner in cfg["headers"]:
                new = make_header(cfg["headers"][inner], align, bool(m))
                if m:
                    new = "\\multicolumn{%d}{%s}{%s}" % (span, align, new)
                parts[j] = raw[:len(raw) - len(raw.lstrip())] + new + raw[len(raw.rstrip()):]
                changed = True
                hdr_changed += 1
                seen.add(inner)
            col += span
        if changed:
            lines[i] = join_row(parts)
    joined = "\n".join(lines)
    warn = []
    for old, new in cfg["headers"].items():
        if old not in seen:
            probe = new if isinstance(new, str) else "\\\\".join(new)
            if probe not in joined and " ".join(new if not isinstance(new, str) else [new]) not in joined:
                warn.append("header %r not found" % old)

    # ---- row labels (first column of the body)
    for i in range(first_mid, len(lines)):
        if cfg.get("cells") and is_row(lines[i]):
            parts = split_row(lines[i])
            raw = parts[0]
            if raw.strip() in cfg["cells"]:
                parts[0] = (raw[:len(raw) - len(raw.lstrip())] + cfg["cells"][raw.strip()]
                            + raw[len(raw.rstrip()):])
                lines[i] = join_row(parts)
                hdr_changed += 1

    # ---- bold
    n_bold = 0
    body = []          # (line index, block id)
    block = 0
    for i in range(first_mid, len(lines)):
        if lines[i].strip() == "\\midrule":
            block += 1
        elif is_row(lines[i]) and not split_row(lines[i])[0].strip().startswith("\\multicolumn"):
            body.append((i, block))

    for rule in cfg.get("bold", []):
        groups = {}
        for i, b in body:
            groups.setdefault(b if rule["block"] == "midrule" else 0, []).append(i)
        for idxs in groups.values():
            for c in rule["cols"]:
                rows = [split_row(lines[i]) for i in idxs]
                vals = [lead(p[c]) for p in rows]
                best = pick(vals, rule["mode"])
                if best is None or len({v for v in vals if v is not None}) < 2:
                    continue            # nothing to compare, or every row ties: no best
                for i, p, v in zip(idxs, rows, vals):
                    if v == best and not is_bold(p[c]):
                        p[c] = bold_cell(p[c])
                        lines[i] = join_row(p)
                        n_bold += 1
    # ---- column padding, where the words make the table wider than the page
    pad = cfg.get("tabcolsep")
    if pad:
        want = "\\setlength{\\tabcolsep}{%s}%%" % pad
        if lines[0].startswith("\\setlength{\\tabcolsep}"):
            lines[0] = want
        else:
            lines.insert(0, want)

    new_text = "\n".join(lines)
    if new_text != text:
        open(path, "w", encoding="utf-8").write(new_text)
    msg = "%-22s headers changed: %2d, cells bolded: %2d" % (name, hdr_changed, n_bold)
    if warn:
        msg += "   WARNING: " + "; ".join(warn)
    return msg


def main():
    for name, cfg in TABLES.items():
        print(process(name, cfg))


if __name__ == "__main__":
    sys.exit(main())
