"""Build main.docx from the LaTeX sources so that it follows the layout of main.pdf.

    python3 make_docx.py [base name] [--keep]

The base name is main by default; `python3 make_docx.py short` builds short.docx from short.tex,
short.aux and short.bbl. --keep keeps the _docx_build folder.

Needs: pandoc, pdftoppm (poppler), python-docx, Pillow. Run `python3 make_results.py` and
`pdflatex main && bibtex main && pdflatex main && pdflatex main` first: the numbers of sections,
equations, tables, figures and theorems are read from main.aux, the reference list and the citation
labels from main.bbl, and the numbers quoted in the text from tables/numbers.tex.

What it does
  1. Rewrites the LaTeX into plain LaTeX that pandoc understands: all files in one, the commands for
     numbers and method names written out, theorems, proofs and the algorithm as ordinary text and
     tables, all numbers filled in, citations written out in the style of the PDF.
  2. Runs pandoc to get a Word file with real Word equations, tables and pictures.
  3. Restyles that file to the NeurIPS layout: letter page with a 5.5 inch text column, Times 10 pt,
     title between two rules, authors side by side, indented abstract, numbered equations, tables
     with horizontal rules only, centred figures, page numbers.

Word and LaTeX break lines and pages differently, and LaTeX moves tables and figures to the top of a
page while Word keeps them where they are written, so the page breaks are not identical.
"""
import copy
import os
import re
import shutil
import subprocess
import sys

from docx import Document
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches
from PIL import ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "_docx_build")
NAME = "main"
OUT = os.path.join(HERE, "main.docx")

TEXT_W = 7920          # 5.5 in, in twentieths of a point (6.5 in for main_compact, set in main())
TEXT_IN = 5.5          # the same width in inches
LINK_COLOR = "000073"  # blue!45!black, as in the PDF
SERIF = "Times New Roman"
MONO = "Courier New"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


# ----------------------------------------------------------------------------------------------
# 1. LaTeX -> plain LaTeX
# ----------------------------------------------------------------------------------------------
def braced(text, start):
    """text[start] is '{'. Returns (content, index after the closing brace)."""
    assert text[start] == "{", text[start:start + 20]
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError("unbalanced braces")


def replace_command(text, name, fn):
    """Replace every \\name{arg} with fn(arg)."""
    out, pos = [], 0
    pat = re.compile(r"\\" + name + r"\s*\{")
    while True:
        m = pat.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        arg, end = braced(text, m.end() - 1)
        out.append(text[pos:m.start()])
        out.append(fn(arg))
        pos = end


def read_bbl(path):
    """Returns the entries of the reference list: key, short label, year, LaTeX body."""
    raw = open(path, encoding="utf-8").read()
    raw = raw.split("\\end{thebibliography}")[0]
    parts = re.split(r"\\bibitem\[", raw)[1:]
    entries = []
    for part in parts:
        depth, i = 1, 0
        while depth:
            depth += {"[": 1, "]": -1}.get(part[i], 0)
            i += 1
        label = " ".join(part[:i - 1].split())
        label = re.sub(r"\{\\natexlab\{([a-z])\}\}", r"\1", label)       # 2022{\natexlab{a}} -> 2022a
        key, end = braced(part, i)
        body = " ".join(part[end:].split())
        m = re.match(r"(.*?)\((\d{4}[a-z]?)\)", label)
        short, year = m.group(1), m.group(2)
        body = body.replace("\\newblock ", "").replace("\\penalty0 ", "").replace("\\penalty0", "")
        body = re.sub(r"\\natexlab\{([^}]*)\}", r"\1", body)
        entries.append({"key": key, "short": short, "year": year, "body": body})
    return entries


_GREEK = {"theta": "\u03b8", "kappa": "\u03ba", "lambda": "\u03bb", "alpha": "\u03b1"}
_SYMBOL = {"le": "\u2264", "ge": "\u2265", "times": "\u00d7", "dots": "\u2026", "in": "\u2208",
           "uparrow": "\u2191", "downarrow": "\u2193", "Delta": "\u0394"}
_RELATION = set("=<>\u2264\u2265\u00d7\u2208")


def simple_math(tex, script=False):
    """Inline maths made of letters, numbers and a few signs is written as ordinary text
    (italic letters, real subscripts), which looks the same in every program that opens the
    file. Returns None for anything else; that stays a Word equation."""
    out, i, n = [], 0, len(tex)

    def group(j):
        """The token or {group} that starts at j, converted. Returns (text, next index)."""
        if j < n and tex[j] == "{":
            inner, end = braced(tex, j)
        elif j < n and tex[j] == "\\":
            m = re.match(r"\\[A-Za-z]+", tex[j:])
            if not m:
                return None, j
            inner, end = m.group(0), j + m.end()
        elif j < n:
            inner, end = tex[j], j + 1
        else:
            return None, j
        return simple_math(inner, script=True), end

    while i < n:
        ch = tex[i]
        if ch.isspace():
            i += 1
        elif ch.isalpha():
            out.append("\\textit{%s}" % ch)
            i += 1
        elif ch.isdigit() or ch == ".":
            out.append(ch)
            i += 1
        elif ch in "()[]":
            out.append(ch)
            i += 1
        elif ch == ",":
            out.append("," if script else ", ")
            i += 1
        elif ch in "+-=<>":
            sign = "\u2212" if ch == "-" else ch
            unary = not out
            out.append(sign if (script or unary or ch in "+-" and unary) else " %s " % sign)
            i += 1
        elif ch in "_^":
            inner, i = group(i + 1)
            if inner is None or "textsu" in inner:
                return None
            out.append("\\text%s{%s}" % ("subscript" if ch == "_" else "superscript", inner))
        elif ch == "\\":
            m = re.match(r"\\([A-Za-z]+|,)", tex[i:])
            if not m:
                return None
            name = m.group(1)
            i += m.end()
            if name == ",":
                continue
            if name in _GREEK:
                out.append("\\textit{%s}" % _GREEK[name])
            elif name in _SYMBOL:
                sym = _SYMBOL[name]
                if sym == "\u2026":
                    out.append(sym)
                elif sym in _RELATION and not script:
                    out.append(" %s " % sym)
                else:
                    out.append(sym)
            elif name == "tilde":
                while i < n and tex[i].isspace():
                    i += 1
                if i >= n or tex[i] != "{":
                    return None
                inner, i = braced(tex, i)
                if not (len(inner) == 1 and inner.isalpha()):
                    return None
                out.append("\\textit{%s\u0303}" % inner)
            else:
                return None
        else:
            return None
    return "".join(out)


def flatten(path):
    """One file: comments removed, every input command replaced by the file it names."""
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"(?<!\\)%.*", "", text)

    def sub(m):
        name = m.group(1)
        return "" if name in ("tables/numbers", "tables/numbers_latent", "tables/numbers_widen", "tables/numbers_bound", "tables/numbers_audit") else flatten(os.path.join(HERE, name + ".tex"))
    return re.sub(r"\\input\{([^}]*)\}", sub, text)


THEOREMS = {"theorem": ("Theorem", True), "proposition": ("Proposition", True), "lemma": ("Lemma", True), "corollary": ("Corollary", True),
            "definition": ("Definition", False), "remark": ("Remark", False)}
DISPLAY = r"(\\begin\{equation\}.*?\\end\{equation\}|(?<!\\)\\\[.*?(?<!\\)\\\])"


def italic_text(body):
    """The statement of a theorem is set in italics; display formulas stay as they are."""
    parts = re.split(DISPLAY, body, flags=re.S)
    for i in range(0, len(parts), 2):
        chunks = [c.strip() for c in re.split(r"\n\s*\n", parts[i])]
        parts[i] = "\n\n".join("\\emph{%s}" % c for c in chunks if c)
    return "\n".join(parts)


def prepare_tex():
    src = flatten(os.path.join(HERE, NAME + ".tex"))
    refs = read_bbl(os.path.join(HERE, NAME + ".bbl"))
    by_key = {r["key"]: r for r in refs}
    labels = {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}",
                                                           open(os.path.join(HERE, NAME + ".aux"), encoding="utf-8").read())}

    # ---- the commands for numbers and for the names of the methods ----
    numbers = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", "".join(open(os.path.join(HERE, "tables", f), encoding="utf-8").read()
                   for f in ("numbers.tex", "numbers_latent.tex", "numbers_widen.tex", "numbers_bound.tex", "numbers_audit.tex") if os.path.exists(os.path.join(HERE, "tables", f)))))
    names = {"plain": "\\textsc{Plain}", "keyswitch": "\\textsc{Key}", "cert": "ClaimLoRA", "nearest": "\\textsc{Nearest}",
             "exactfirst": "\\textsc{Exact-first}", "centre": "\\textsc{Centre}", "signature": "\\textsc{Signature}", "fivefull": ("Appendix~\\ref{sec:fivefull}" if NAME == "short" else "Section~\\ref{sec:five}"), "hybrid": "\\textsc{Hybrid}", "scaled": "\\textsc{Scaled}", "certscaled": "\\textsc{Cert-scaled}", "recog": "FinderLoRA", "handicap": "KeeperLoRA", "R": "\\mathbb{R}"}
    head, body = src.split("\\begin{document}")
    known = dict(numbers, **names)
    body = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda m: known[m.group(1)].replace("{,}", ",") if m.group(1) in known else m.group(0), body)

    # ---- title and authors (the author block is rebuilt later as a table) ----
    m = re.search(r"\\title\s*\{", head)
    title, _ = braced(head, m.end() - 1)
    m = re.search(r"\\author\s*\{", head)
    author_raw, _ = braced(head, m.end() - 1)
    authors = []
    for block in re.split(r"\\And|\\AND|\\and", author_raw):
        lines = [l.strip() for l in block.split("\\\\") if l.strip()]
        lines = [re.sub(r"\\texttt\{([^}]*)\}", r"\1", l) for l in lines]
        if lines:
            authors.append(lines)
    src = ("\\documentclass{article}\n\\usepackage{booktabs,amsmath,amssymb,graphicx}\n\\title{%s}\n\\author{AUTHORBLOCK}\n\\begin{document}" % title) + body

    # ---- the two-line formula with two numbers becomes two formulas ----
    def align(m):
        lines, depth, last = [], 0, 0
        body = m.group(1)
        for t in re.finditer(r"\\begin\{\w+\}|\\end\{\w+\}|\\\\(?:\[[^\]]*\])?", body):
            if t.group(0).startswith("\\begin"):
                depth += 1
            elif t.group(0).startswith("\\end"):
                depth -= 1
            elif depth == 0:
                lines.append(body[last:t.start()])
                last = t.end()
        lines.append(body[last:])
        return "".join("\\begin{equation}%s\\end{equation}\n" % part.replace("&", "", 1) for part in lines if part.strip())
    src = re.sub(r"\\begin\{align\}(.*?)\\end\{align\}", align, src, flags=re.S)
    assert "\\begin{align" not in src

    # ---- theorems, definitions, remarks and proofs as ordinary text ----
    count = {k: 0 for k in THEOREMS}

    def theorem(m):
        kind, note, text = m.group(1), m.group(2), m.group(3)
        count[kind] += 1
        name, italic = THEOREMS[kind]
        label = re.match(r"\s*\\label\{([^}]*)\}", text)                      # the label of the statement itself, if it has one
        if label:
            assert labels[label.group(1)] == str(count[kind]), (label.group(1), count[kind])
            text = text[label.end():]
        text = text.strip()
        head = ("\\textit{%s %d}" if kind == "remark" else "\\textbf{%s %d}") % (name, count[kind])
        head += (" (%s)." % note[1:-1]) if note else "."
        return "\n\n%s %s\n\n" % (head, italic_text(text) if italic else text)
    src = re.sub(r"\\begin\{(%s)\}(\[[^\]]*\])?(.*?)\\end\{\1\}" % "|".join(THEOREMS), theorem, src, flags=re.S)
    src = re.sub(r"\\begin\{proof\}(.*?)\\end\{proof\}", lambda m: "\n\n\\textit{Proof.} %s \u25a1\n\n" % m.group(1).strip(), src, flags=re.S)

    # ---- the algorithm as a table without a number of its own ----
    def algorithm(m):
        text = m.group(1)
        cap = re.search(r"\\caption\{(.*?)\}", text).group(1)
        number = labels[re.search(r"\\label\{([^}]*)\}", text).group(1)]
        rows = re.search(r"\\begin\{tabular\}\{.*?\}\n(.*?)\\end\{tabular\}", text, re.S).group(1)
        rows = re.sub(r"\\multicolumn\{2\}\{@\{\}l\}", r"\\multicolumn{2}{l}", rows)
        rows = re.sub(r"\\\\\[\d+pt\]", r"\\\\", rows)
        rows = re.sub(r"\s*\\hfill\s*\\emph\{(.*?)\}", lambda k: " \\quad (\\emph{%s})" % k.group(1), rows)
        return ("\\begin{table}\n\\caption{ALGORITHM %s: %s}\n\\small\n\\begin{tabular}{rl}\n\\toprule\n%s\\bottomrule\n\\end{tabular}\n\\end{table}"
                % (number, cap, rows))
    src = re.sub(r"\\begin\{algorithm\}(.*?)\\end\{algorithm\}", algorithm, src, flags=re.S)
    src = re.sub(r"(?<!\\)\\\[(.*?)(?<!\\)\\\]", lambda m: "\n\nCENTERLINE $%s$\n\n" % " ".join(m.group(1).split()), src, flags=re.S)

    # ---- numbers of sections; captions of tables and figures; equations are numbered in order ----
    sec = sub = eq = 0
    in_appendix = False
    out, pos = [], 0
    token = re.compile(r"\\appendix\b|\\section\*?\s*\{|\\subsection\s*\{|\\begin\{equation\}|\\begin\{(table|figure)\}(\[[^\]]*\])?")
    while True:
        m = token.search(src, pos)
        if not m:
            out.append(src[pos:])
            break
        out.append(src[pos:m.start()])
        t = m.group(0)
        pos = m.end()
        if t.startswith("\\appendix"):
            in_appendix, sec = True, 0
        elif t.startswith("\\section"):
            name, pos = braced(src, m.end() - 1)
            if "*" in t:
                out.append("\\section{%s}" % name)
            else:
                sec, sub = sec + 1, 0
                out.append("\\section{%s\u2003%s}" % (chr(64 + sec) if in_appendix else str(sec), name))
        elif t.startswith("\\subsection"):
            name, pos = braced(src, m.end() - 1)
            sub += 1
            out.append("\\subsection{%s.%d\u2003%s}" % (chr(64 + sec) if in_appendix else str(sec), sub, name))
        elif t.startswith("\\begin{equation"):
            eq += 1
            out.append(t)
        else:
            kind = m.group(1)
            end = src.index("\\end{%s}" % kind, pos)
            block = src[pos:end]
            label = re.search(r"\\label\{([^}]*)\}", block)
            cm = re.search(r"\\caption\s*\{", block)
            if cm:
                cap, cap_end = braced(block, cm.end() - 1)
                if cap.startswith("ALGORITHM "):
                    cap = "Algorithm " + cap[len("ALGORITHM "):]
                else:
                    cap = "%s %s: %s" % ("Table" if kind == "table" else "Figure", labels[label.group(1)], cap)
                block = block[:cm.start()] + "\\caption{%s}" % cap + block[cap_end:]
            out.append("\\begin{%s}" % kind + block)
            pos = end
    src = "".join(out)
    src = re.sub(r"\\label\{[^}]*\}", "", src)
    src = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(%s)" % labels.get(m.group(1), "??"), src)
    src = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "??"), src)

    # ---- a table with several tabulars: the caption stays with the first, the others follow as tables of their own ----
    def split_table(m):
        block = m.group(1)
        parts = re.split(r"(?<=\\end{tabular})\}?\s*\\par\\medskip\s*", block)
        if len(parts) < 2:
            return m.group(0)
        parts[0] = parts[0].rstrip()
        return "\\begin{table}%s\n\\end{table}\n" % parts[0] + "".join("\n\\begin{table}\n%s\n\\end{table}\n" % p.strip() for p in parts[1:])
    src = re.sub(r"\\begin\{table\}(.*?)\\end\{table\}", split_table, src, flags=re.S)

    # ---- a stacked header cell becomes its parts joined by a space ----
    src = re.sub(r"\\shortstack(?:\[[a-z]\])?\s*\{", "\\\\shortstack{", src)
    src = replace_command(src, "shortstack", lambda a: " ".join(x.strip() for x in a.split("\\\\")))

    # ---- a column of fixed width in inches (pandoc does not read it) is a left-aligned text column ----
    src = re.sub(r"(\\begin\{tabular\}\{[^}\n]*?)p\{[\d.]+in\}", r"\1l", src)

    # ---- what pandoc drops from the tables: font size, rules inside the body, rules under spanning heads ----
    table_info = []
    for m in re.finditer(r"\\begin\{table\}(.*?)\\end\{table\}", src, re.S):
        body = m.group(1)
        last_end = 0
        for tm in re.finditer(r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\}", body, re.S):
            pre = body[last_end:tm.start()]            # what stands before this tabular: its font size and column padding
            last_end = tm.end()
            ctx = pre if re.search(r"\\(?:footnotesize|scriptsize|small)\b", pre) else body
            size = 7 if re.search(r"\\scriptsize(?:\\setlength|\n)", ctx) else 8 if "\\footnotesize" in ctx else 9 if "\\small" in ctx else 10
            pm = re.search(r"\\tabcolsep\}\{([\d.]+)pt\}", pre) or re.search(r"\\tabcolsep\}\{([\d.]+)pt\}", body)
            pad = float(pm.group(1)) if pm else 6
            tabular = tm.group(1)
            rules, cmid, row = [], [], 0
            for piece in re.split(r"(\\\\(?:\[[^\]]*\])?|\\midrule|\\toprule|\\bottomrule|\\cmidrule(?:\([^)]*\))?\{\d+-\d+\})", tabular):
                if piece.startswith("\\\\"):
                    row += 1
                elif piece == "\\midrule":
                    rules.append(row)          # a rule below row number `row` (1 = the first row)
                elif piece.startswith("\\cmidrule"):
                    a, b = re.search(r"\{(\d+)-(\d+)\}", piece).groups()
                    cmid.append((row, int(a), int(b)))
            table_info.append({"size": size, "pad": pad, "rules": rules, "cmid": cmid, "algorithm": "\\caption{Algorithm " in body})
    # a font-size group round a tabular, and the space between two tabulars of one table
    src = re.sub(r"\{\\(?:footnotesize|scriptsize)(?:\\setlength\{\\tabcolsep\}\{[^}]*\})?(?=\\begin\{tabular\})", "", src)
    src = re.sub(r"(\\end\{tabular\})\}", r"\1", src)
    src = src.replace("\\par\\medskip", "")
    src = re.sub(r"\\cmidrule(?:\([^)]*\))?\{\d+-\d+\}", "", src)
    src = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", src)

    # ---- citations, written as in the PDF ----
    def citep(arg):
        return "[" + ", ".join("\\hyperlink{ref-%s}{%s, %s}" % (k, by_key[k]["short"], by_key[k]["year"]) for k in [x.strip() for x in arg.split(",")]) + "]"

    def citet(arg):
        return ", ".join("%s [\\hyperlink{ref-%s}{%s}]" % (by_key[k]["short"], k, by_key[k]["year"]) for k in [x.strip() for x in arg.split(",")])
    src = replace_command(src, "citep", citep)
    src = replace_command(src, "citet", citet)
    src = replace_command(src, "citealp", lambda arg: ", ".join("\\hyperlink{ref-%s}{%s %s}" % (k, by_key[k]["short"], by_key[k]["year"]) for k in [x.strip() for x in arg.split(",")]))
    assert "\\cite" not in src, "a citation command that is not handled"

    # ---- reference list, copied from main.bbl ----
    ref_tex = ["\\section{References}", ""]
    for r in refs:
        ref_tex.append("\\hypertarget{ref-%s}{}REFITEM %s" % (r["key"], r["body"]))
        ref_tex.append("")
    src = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", lambda m: "\n".join(ref_tex), src)

    # ---- words inside a display formula: no brackets (some programs show them with a backslash) ----
    def words(m):
        def text(k):
            inner = k.group(1).replace("(", "\u2014 ").replace(")", "")
            return "\\text{%s}" % inner if inner.strip() else ""
        return re.sub(r"\\text\{([^{}]*)\}", text, m.group(0))
    src = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", words, src, flags=re.S)
    src = src.replace("the oldest zone that contains } x", "the oldest zone with } x")       # keeps formula (11) inside the column

    # ---- simple inline maths as text; the rest stays a Word equation ----
    src = src.replace("\\tfrac", "\\frac")
    pieces = re.split(r"(\\begin\{equation\}.*?\\end\{equation\})", src, flags=re.S)
    for i in range(0, len(pieces), 2):
        pieces[i] = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$",
                           lambda m: simple_math(m.group(1)) or m.group(0), pieces[i], flags=re.S)
    src = "".join(pieces)

    # ---- small things pandoc does not know ----
    src = replace_command(src, "paragraph", lambda a: "\\textbf{%s}\u2003" % a)
    for a, b in [("\\Bigl", "\\left"), ("\\Bigr", "\\right"), ("\\bigl", ""), ("\\bigr", ""), ("\\big\\|", "\\|"),
                 ("\\maketitle", "\\maketitle"), ("\\appendix", "")]:
        src = src.replace(a, b)
    src = src.replace("$\\langle$", "\u27e8").replace("$\\rangle$", "\u27e9")

    # ---- figures as PNG at 300 dpi, with the width they have in the PDF ----
    widths = []

    def figure(m):
        opt, path = m.group(1) or "", m.group(2)
        w = re.search(r"width=([\d.]*)\\linewidth", opt)
        widths.append(float(w.group(1) or 1) if w else 1.0)
        name = os.path.basename(path)
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", os.path.join(HERE, path + ".pdf"), os.path.join(BUILD, name)], check=True)
        return "\\includegraphics{%s.png}" % name
    src = re.sub(r"\\includegraphics(\[[^\]]*\])?\{(figures/[^}]*)\}", figure, src)

    open(os.path.join(BUILD, NAME + "_docx.tex"), "w", encoding="utf-8").write(src)
    return {"title": title, "authors": authors, "tables": table_info, "n_eq": eq, "figures": widths}


# ----------------------------------------------------------------------------------------------
# 2. helpers for the Word XML
# ----------------------------------------------------------------------------------------------
def el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k.replace("_", ":", 1)), str(v))
    return e


def get_or_add(parent, tag, first=False):
    child = parent.find(qn(tag))
    if child is None:
        child = OxmlElement(tag)
        if first:
            parent.insert(0, child)
        else:
            parent.append(child)
    return child


def set_ppr(p, **kw):
    """Direct paragraph formatting. Children are written in schema order."""
    old = p.find(qn("w:pPr"))
    style = None
    keep = {}
    if old is not None:
        s = old.find(qn("w:pStyle"))
        style = s.get(qn("w:val")) if s is not None else None
        for tag in ("w:numPr", "w:keepNext", "w:pBdr"):
            c = old.find(qn(tag))
            if c is not None:
                keep[tag] = c
        p.remove(old)
    style = kw.get("style", style)
    ppr = OxmlElement("w:pPr")
    if style:
        ppr.append(el("w:pStyle", w_val=style))
    if kw.get("keep_next") or "w:keepNext" in keep:
        if kw.get("keep_next", True):
            ppr.append(el("w:keepNext"))
    if kw.get("keep_lines"):
        ppr.append(el("w:keepLines"))
    if "w:numPr" in keep:
        ppr.append(keep["w:numPr"])
    if "borders" in kw:
        bdr = OxmlElement("w:pBdr")
        for side, (size, space) in kw["borders"].items():
            bdr.append(el("w:" + side, w_val="single", w_sz=size, w_space=space, w_color="000000"))
        ppr.append(bdr)
    elif "w:pBdr" in keep:
        ppr.append(keep["w:pBdr"])
    sp = {}
    if "before" in kw:
        sp["w_before"] = int(kw["before"] * 20)
    if "after" in kw:
        sp["w_after"] = int(kw["after"] * 20)
    if "line" in kw:
        sp["w_line"] = kw["line"]
        sp["w_lineRule"] = "auto"
    if sp:
        ppr.append(el("w:spacing", **sp))
    ind = {}
    for k in ("left", "right", "hanging", "firstLine"):
        if k in kw:
            ind["w_" + k] = kw[k]
    if ind:
        ppr.append(el("w:ind", **ind))
    if "align" in kw:
        ppr.append(el("w:jc", w_val=kw["align"]))
    p.insert(0, ppr)
    return ppr


def run(text, font=SERIF, size=10, bold=False, italic=False, color=None):
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    rpr.append(el("w:rFonts", w_ascii=font, w_hAnsi=font, w_cs=font, w_eastAsia=font))
    if bold:
        rpr.append(el("w:b"))
    if italic:
        rpr.append(el("w:i"))
    if color:
        rpr.append(el("w:color", w_val=color))
    rpr.append(el("w:sz", w_val=int(size * 2)))
    rpr.append(el("w:szCs", w_val=int(size * 2)))
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    return r


def para(*runs, **kw):
    p = OxmlElement("w:p")
    for r in runs:
        p.append(r)
    set_ppr(p, **kw)
    return p


def plain_table(widths, align="center"):
    """A table without borders. Returns (tbl, list of cells)."""
    tbl = OxmlElement("w:tbl")
    pr = OxmlElement("w:tblPr")
    pr.append(el("w:tblW", w_w=sum(widths), w_type="dxa"))
    pr.append(el("w:jc", w_val=align))
    pr.append(el("w:tblLayout", w_type="fixed"))
    mar = OxmlElement("w:tblCellMar")
    for side in ("top", "left", "bottom", "right"):
        mar.append(el("w:" + side, w_w=0, w_type="dxa"))
    pr.append(mar)
    pr.append(el("w:tblLook", w_val="0000"))
    tbl.append(pr)
    grid = OxmlElement("w:tblGrid")
    for w in widths:
        grid.append(el("w:gridCol", w_w=w))
    tbl.append(grid)
    tr = OxmlElement("w:tr")
    trpr = OxmlElement("w:trPr")
    trpr.append(el("w:cantSplit"))
    tr.append(trpr)
    cells = []
    for w in widths:
        tc = OxmlElement("w:tc")
        tcpr = OxmlElement("w:tcPr")
        tcpr.append(el("w:tcW", w_w=w, w_type="dxa"))
        tcpr.append(el("w:vAlign", w_val="center"))
        tc.append(tcpr)
        tr.append(tc)
        cells.append(tc)
    tbl.append(tr)
    return tbl, cells


def style_xml(style_id, name, kind="paragraph", based_on=None, next_style=None, ppr="", rpr="",
              default=False, extra=""):
    return (
        '<w:style %s w:type="%s"%s w:styleId="%s"><w:name w:val="%s"/>%s%s%s<w:qFormat/>%s%s</w:style>'
        % (nsdecls("w"), kind, ' w:default="1"' if default else "", style_id, name,
           '<w:basedOn w:val="%s"/>' % based_on if based_on else "",
           '<w:next w:val="%s"/>' % next_style if next_style else "", extra,
           "<w:pPr>%s</w:pPr>" % ppr if ppr else "",
           "<w:rPr>%s</w:rPr>" % rpr if rpr else ""))


def fonts(name):
    return '<w:rFonts w:ascii="%s" w:hAnsi="%s" w:cs="%s" w:eastAsia="%s"/>' % (name, name, name, name)


def size(pt):
    return '<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (pt * 2, pt * 2)


BODY_LINE = 228          # 0.95 of single spacing: close to 10 pt type on 11 pt lines
STYLES = [
    style_xml("Normal", "Normal", default=True),
    style_xml("BodyText", "Body Text", based_on="Normal",
              ppr='<w:spacing w:before="0" w:after="110" w:line="%d" w:lineRule="auto"/><w:jc w:val="both"/>' % BODY_LINE),
    style_xml("FirstParagraph", "First Paragraph", based_on="BodyText", next_style="BodyText"),
    style_xml("Compact", "Compact", based_on="BodyText"),
    style_xml("Title", "Title", based_on="Normal", next_style="BodyText",
              ppr='<w:keepNext/><w:spacing w:before="0" w:after="0"/><w:jc w:val="center"/>',
              rpr="<w:b/><w:bCs/>" + size(17)),
    style_xml("Author", "Author", based_on="Normal", ppr='<w:jc w:val="center"/>'),
    style_xml("Date", "Date", based_on="Normal", ppr='<w:jc w:val="center"/>'),
    style_xml("AbstractTitle", "Abstract Title", based_on="Normal", next_style="Abstract",
              ppr='<w:keepNext/><w:spacing w:before="420" w:after="330"/><w:jc w:val="center"/>',
              rpr="<w:b/><w:bCs/>" + size(12)),
    style_xml("Abstract", "Abstract", based_on="Normal",
              ppr='<w:spacing w:before="0" w:after="110" w:line="%d" w:lineRule="auto"/>'
                  '<w:ind w:left="720" w:right="720"/><w:jc w:val="both"/>' % BODY_LINE),
    style_xml("Heading1", "heading 1", based_on="Normal", next_style="BodyText",
              ppr='<w:keepNext/><w:keepLines/><w:spacing w:before="300" w:after="200"/><w:outlineLvl w:val="0"/>',
              rpr="<w:b/><w:bCs/>" + size(12)),
    style_xml("Heading2", "heading 2", based_on="Normal", next_style="BodyText",
              ppr='<w:keepNext/><w:keepLines/><w:spacing w:before="240" w:after="150"/><w:outlineLvl w:val="1"/>',
              rpr="<w:b/><w:bCs/>" + size(10)),
    style_xml("Caption", "caption", based_on="Normal",
              ppr='<w:spacing w:before="0" w:after="120" w:line="%d" w:lineRule="auto"/><w:jc w:val="both"/>' % BODY_LINE),
    style_xml("TableCaption", "Table Caption", based_on="Caption", ppr="<w:keepNext/>"),
    style_xml("ImageCaption", "Image Caption", based_on="Caption"),
    style_xml("Figure", "Figure", based_on="Normal", ppr='<w:jc w:val="center"/>'),
    style_xml("CaptionedFigure", "Captioned Figure", based_on="Figure", ppr="<w:keepNext/>"),
    style_xml("Bibliography", "Bibliography", based_on="Normal",
              ppr='<w:spacing w:before="0" w:after="90" w:line="%d" w:lineRule="auto"/>'
                  '<w:ind w:left="200" w:hanging="200"/><w:jc w:val="both"/>' % BODY_LINE),
    style_xml("Footer", "footer", based_on="Normal", ppr='<w:jc w:val="center"/>'),
    style_xml("FootnoteText", "footnote text", based_on="Normal", rpr=size(9)),
    style_xml("DefaultParagraphFont", "Default Paragraph Font", kind="character", default=True,
              extra="<w:uiPriority w:val=\"1\"/><w:semiHidden/>"),
    style_xml("VerbatimChar", "Verbatim Char", kind="character", based_on="DefaultParagraphFont",
              rpr=fonts(MONO) + size(9)),
    style_xml("Hyperlink", "Hyperlink", kind="character", based_on="DefaultParagraphFont",
              rpr='<w:color w:val="%s"/>' % LINK_COLOR),
    style_xml("FootnoteReference", "footnote reference", kind="character",
              based_on="DefaultParagraphFont", rpr='<w:vertAlign w:val="superscript"/>'),
    '<w:style %s w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/>'
    '<w:uiPriority w:val="99"/><w:semiHidden/><w:unhideWhenUsed/><w:tblPr><w:tblInd w:w="0" w:type="dxa"/>'
    '<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/>'
    '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>'
    % nsdecls("w"),
    '<w:style %s w:type="table" w:styleId="Table"><w:name w:val="Table"/><w:basedOn w:val="TableNormal"/>'
    '<w:qFormat/></w:style>' % nsdecls("w"),
    '<w:style %s w:type="numbering" w:default="1" w:styleId="NoList"><w:name w:val="No List"/>'
    '<w:uiPriority w:val="99"/><w:semiHidden/><w:unhideWhenUsed/></w:style>' % nsdecls("w"),
]


def rebuild_styles(doc):
    styles = doc.styles.element
    for child in list(styles):
        styles.remove(child)
    styles.append(parse_xml(
        '<w:docDefaults %s><w:rPrDefault><w:rPr>%s%s<w:lang w:val="en-US" w:eastAsia="en-US" w:bidi="ar-SA"/>'
        '</w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/>'
        '</w:pPr></w:pPrDefault></w:docDefaults>' % (nsdecls("w"), fonts(SERIF), size(10))))
    for xml in STYLES:
        styles.append(parse_xml(xml))


# ---- text width, for the column widths of the tables ----
_FONT_FILES = {
    ("serif", False): "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ("serif", True): "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    ("mono", False): "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ("mono", True): "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
}
_font_cache = {}


def text_width(text, pt, mono=False, bold=False):
    """Width in twentieths of a point. Liberation has the same widths as Times / Courier New."""
    key = ("mono" if mono else "serif", bold)
    if key not in _font_cache:
        path = _FONT_FILES[key]
        _font_cache[key] = ImageFont.truetype(path, 200) if os.path.exists(path) else None
    font = _font_cache[key]
    if font is None:
        return len(text) * pt * (0.6 if mono else 0.5) * 20
    return font.getlength(text) / 200.0 * pt * 20


def cell_width(tc, pt, words=False):
    """Natural width of the widest paragraph in a cell (with words=True: of its widest word)."""
    if words:
        best = 0
        for p in tc.iter(qn("w:p")):
            for node in p.iter(qn("w:r")):
                rpr = node.find(qn("w:rPr"))
                caps = rpr is not None and rpr.find(qn("w:smallCaps")) is not None
                bold = rpr is not None and rpr.find(qn("w:b")) is not None
                for word in "".join(t.text or "" for t in node.findall(qn("w:t"))).split():
                    if caps:
                        w = sum(text_width(ch.upper(), pt * 0.8 if ch.islower() else pt, bold=bold) for ch in word)
                    else:
                        w = text_width(word, pt, bold=bold)
                    best = max(best, w)
        return best
    best = 0
    for p in tc.iter(qn("w:p")):
        w = 0
        for node in p.iter():
            if node.tag == qn("w:r"):
                rpr = node.find(qn("w:rPr"))
                mono = bold = caps = False
                if rpr is not None:
                    st = rpr.find(qn("w:rStyle"))
                    mono = st is not None and st.get(qn("w:val")) == "VerbatimChar"
                    bold = rpr.find(qn("w:b")) is not None
                    caps = rpr.find(qn("w:smallCaps")) is not None
                text = "".join(t.text or "" for t in node.findall(qn("w:t")))
                if caps:        # small capitals: lower-case letters are drawn as smaller capitals
                    w += sum(text_width(ch.upper(), pt * 0.8 if ch.islower() else pt, bold=bold)
                             for ch in text)
                else:
                    w += text_width(text, pt - 1 if mono else pt, mono=mono, bold=bold)
            elif node.tag == "{%s}t" % M_NS:
                w += text_width(node.text or "", pt) * 1.15
        best = max(best, w)
    return best


# ----------------------------------------------------------------------------------------------
# 3. restyle the pandoc output
# ----------------------------------------------------------------------------------------------
def style_of(p):
    ppr = p.find(qn("w:pPr"))
    if ppr is None:
        return None
    s = ppr.find(qn("w:pStyle"))
    return s.get(qn("w:val")) if s is not None else None


def text_of(node):
    return "".join(t.text or "" for t in node.iter(qn("w:t")))


def restyle(info):
    doc = Document(os.path.join(BUILD, "raw.docx"))
    rebuild_styles(doc)
    body = doc.element.body

    # ---- page ----
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches((8.5 - TEXT_IN) / 2)
    sec.top_margin, sec.bottom_margin = (Inches(1.0), Inches(1.0)) if TEXT_IN == 5.5 else (Inches(0.9), Inches(0.85))
    sec.footer_distance = Inches(0.55)
    sec.header_distance = Inches(0.5)
    sec.gutter = Inches(0)
    sec.different_first_page_header_footer = True
    fp = sec.footer.paragraphs[0]._p
    for c in list(fp):
        fp.remove(c)
    set_ppr(fp, style="Footer", align="center")
    r = run("")
    r.remove(r.find(qn("w:t")))
    r.append(el("w:fldChar", w_fldCharType="begin"))
    fp.append(r)
    r = run("")
    r.remove(r.find(qn("w:t")))
    it = OxmlElement("w:instrText")
    it.text = " PAGE "
    it.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(it)
    fp.append(r)
    r = run("")
    r.remove(r.find(qn("w:t")))
    r.append(el("w:fldChar", w_fldCharType="separate"))
    fp.append(r)
    fp.append(run("1"))
    r = run("")
    r.remove(r.find(qn("w:t")))
    r.append(el("w:fldChar", w_fldCharType="end"))
    fp.append(r)
    fp1 = sec.first_page_footer.paragraphs[0]._p
    for c in list(fp1):
        fp1.remove(c)
    set_ppr(fp1, style="Footer", align="left")
    fp1.append(run("", size=9))            # first page: no page number, as in the PDF

    # ---- title block ----
    for p in list(body.iter(qn("w:p"))):
        st = style_of(p)
        if st == "Title":
            for c in list(p):
                if c.tag != qn("w:pPr"):
                    p.remove(c)
            lines = [" ".join(l.split()) for l in info["title"].split("\\\\")]
            for i, line in enumerate(lines):
                r = run(line, size=17, bold=True)
                if i:
                    r.insert(1, el("w:br"))
                p.append(r)
            set_ppr(p, style="Title", align="center", before=7, after=22, line=252,
                    borders={"top": (32, 18), "bottom": (8, 21)})
        elif st == "Author":
            n = len(info["authors"])
            col = 3740
            tbl, cells = plain_table([col] * n)
            for tc, lines in zip(cells, info["authors"]):
                for i, line in enumerate(lines):
                    if i == 0:
                        tc.append(para(run(line, bold=True), align="center", line=BODY_LINE))
                    elif "@" in line:
                        tc.append(para(run(line, font=MONO, size=9), align="center", line=BODY_LINE))
                    else:
                        tc.append(para(run(line), align="center", line=BODY_LINE))
            p.addprevious(tbl)
            body.remove(p)
        elif st == "Date":
            body.remove(p)

    # ---- display equations: equation in the middle, number at the right ----
    n_eq = 0
    for p in list(body.iter(qn("w:p"))):
        if p.getparent() is not body:
            continue
        maths = [c for c in p if c.tag == "{%s}oMathPara" % M_NS]
        if not maths:
            continue
        ppr = p.find(qn("w:pPr"))
        segments, cur = [], []
        for c in list(p):
            if c.tag == qn("w:pPr"):
                continue
            if c.tag == "{%s}oMathPara" % M_NS:
                segments.append(("text", cur))
                segments.append(("math", c))
                cur = []
            else:
                cur.append(c)
        segments.append(("text", cur))
        anchor = p
        for i, (kind, content) in enumerate(segments):
            if kind == "text":
                if not "".join(text_of(c) for c in content).strip() and \
                        not any(c.tag == "{%s}oMath" % M_NS or c.find(".//{%s}oMath" % M_NS) is not None
                                for c in content):
                    continue
                newp = OxmlElement("w:p")
                if ppr is not None:
                    newp.append(copy.deepcopy(ppr))
                for c in content:
                    newp.append(c)
                first = next(newp.iter(qn("w:t")), None)
                if first is not None and i > 0:
                    first.text = (first.text or "").lstrip()
                followed = i + 1 < len(segments)
                set_ppr(newp, style="BodyText", **({"after": 0, "keep_next": True} if followed else {}))
                anchor.addnext(newp)
                anchor = newp
            else:
                n_eq += 1
                tbl, cells = plain_table([700, TEXT_W - 1400, 700])
                cells[0].append(para(before=5, after=5))
                mp = OxmlElement("w:p")
                mp.append(content)
                set_ppr(mp, before=5, after=7, align="center")
                cells[1].append(mp)
                cells[2].append(para(run("(%d)" % n_eq), before=5, after=7, align="right"))
                anchor.addnext(tbl)
                anchor = tbl
        body.remove(p)
    assert n_eq == info["n_eq"], (n_eq, info["n_eq"])

    # ---- tables ----
    def span(tc):
        pr = tc.find(qn("w:tcPr"))
        g = pr.find(qn("w:gridSpan")) if pr is not None else None
        return int(g.get(qn("w:val"))) if g is not None else 1

    tables = [t for t in body.iter(qn("w:tbl"))
              if t.find(qn("w:tblPr")).find(qn("w:tblStyle")) is not None]
    assert len(tables) == len(info["tables"]), (len(tables), len(info["tables"]))
    for tbl, ti in zip(tables, info["tables"]):
        pt, pad = ti["size"], int(ti["pad"] * 20)
        rows = tbl.findall(qn("w:tr"))
        spans = [[span(tc) for tc in r.findall(qn("w:tc"))] for r in rows]
        ncol = max(sum(s) for s in spans)

        def natural(pt):
            widths = [0] * ncol
            for r, ss in zip(rows, spans):
                j = 0
                for tc, s in zip(r.findall(qn("w:tc")), ss):
                    if s == 1:
                        widths[j] = max(widths[j], cell_width(tc, pt))
                    j += s
            widths = [int(w * 1.04) + 2 * pad for w in widths]
            for r, ss in zip(rows, spans):                  # a spanning cell that needs more room than its columns have
                j = 0
                for tc, s in zip(r.findall(qn("w:tc")), ss):
                    if s > 1 and not ti["algorithm"]:
                        need = int(cell_width(tc, pt) * 1.04) + 2 * pad
                        have = sum(widths[j:j + s])
                        if need > have:
                            for q in range(j, j + s):
                                widths[q] += (need - have) // s + 1
                    j += s
            return widths

        widths = natural(pt)
        while sum(widths) > TEXT_W and pt > 7 and not ti["algorithm"]:      # smaller type before the columns are squeezed
            pt -= 0.5
            widths = natural(pt)
        if ti["algorithm"]:
            widths = [widths[0], TEXT_W - widths[0]]
        if sum(widths) > TEXT_W:
            # shrink only the room above the widest word of each column, so that no word is split
            least = [0] * ncol
            for r, ss in zip(rows, spans):
                j = 0
                for tc, s_ in zip(r.findall(qn("w:tc")), ss):
                    if s_ == 1:
                        least[j] = max(least[j], int(cell_width(tc, pt, words=True) * 1.06) + 2 * pad)
                    j += s_
            least = [min(l, w) for l, w in zip(least, widths)]
            room = sum(widths) - sum(least)
            if sum(least) < TEXT_W and room > 0:
                scale = (TEXT_W - sum(least)) / float(room)
                widths = [int(l + (w - l) * scale) for l, w in zip(least, widths)]
            else:
                scale = TEXT_W / float(sum(widths))
                widths = [int(w * scale) for w in widths]
        pr = tbl.find(qn("w:tblPr"))
        for c in list(pr):
            pr.remove(c)
        pr.append(el("w:tblStyle", w_val="Table"))
        pr.append(el("w:tblW", w_w=sum(widths), w_type="dxa"))
        pr.append(el("w:jc", w_val="center"))
        pr.append(el("w:tblLayout", w_type="fixed"))
        mar = OxmlElement("w:tblCellMar")
        for side, w in (("top", 0), ("left", pad), ("bottom", 0), ("right", pad)):
            mar.append(el("w:" + side, w_w=w, w_type="dxa"))
        pr.append(mar)
        pr.append(el("w:tblLook", w_val="0000"))
        grid = tbl.find(qn("w:tblGrid"))
        for c in list(grid):
            grid.remove(c)
        for w in widths:
            grid.append(el("w:gridCol", w_w=w))
        for i, (r, ss) in enumerate(zip(rows, spans)):
            trpr = r.find(qn("w:trPr"))
            if trpr is not None:
                r.remove(trpr)
            trpr = OxmlElement("w:trPr")
            trpr.append(el("w:cantSplit"))
            r.insert(0, trpr)
            last = i == len(rows) - 1
            rule_below = (i + 1) in ti["rules"]
            j = 0
            for tc, s in zip(r.findall(qn("w:tc")), ss):
                old = tc.find(qn("w:tcPr"))
                if old is not None:
                    tc.remove(old)
                tcpr = OxmlElement("w:tcPr")
                tcpr.append(el("w:tcW", w_w=sum(widths[j:j + s]), w_type="dxa"))
                if s > 1:
                    tcpr.append(el("w:gridSpan", w_val=s))
                part = any(r0 == i + 1 and a - 1 <= j and j + s <= b for r0, a, b in ti["cmid"])
                bd = OxmlElement("w:tcBorders")
                if i == 0:
                    bd.append(el("w:top", w_val="single", w_sz=8, w_space=0, w_color="000000"))
                if last:
                    bd.append(el("w:bottom", w_val="single", w_sz=8, w_space=0, w_color="000000"))
                elif rule_below or part:
                    bd.append(el("w:bottom", w_val="single", w_sz=4, w_space=0, w_color="000000"))
                if len(bd):
                    tcpr.append(bd)
                tc.insert(0, tcpr)
                for p in tc.findall(qn("w:p")):
                    ppr = p.find(qn("w:pPr"))
                    jc = ppr.find(qn("w:jc")) if ppr is not None else None
                    align = jc.get(qn("w:val")) if jc is not None else "left"
                    if ppr is not None:
                        p.remove(ppr)
                    edge_top = i == 0 or (i in ti["rules"])
                    set_ppr(p, align=align, before=3 if edge_top else 0.5,
                            after=3 if (last or rule_below) else 0.5,
                            keep_next=not last)
                    for rr in p.iter(qn("w:r")):
                        rpr = get_or_add(rr, "w:rPr", first=True)
                        mono = rpr.find(qn("w:rStyle")) is not None and \
                            rpr.find(qn("w:rStyle")).get(qn("w:val")) == "VerbatimChar"
                        for tag in ("w:sz", "w:szCs"):
                            old = rpr.find(qn(tag))
                            if old is not None:
                                rpr.remove(old)
                        sz = int((pt - 1 if mono else pt) * 2)
                        rpr.append(el("w:sz", w_val=sz))
                        rpr.append(el("w:szCs", w_val=sz))
                j += s
        nxt = tbl.getnext()
        if nxt is not None and nxt.tag == qn("w:p"):
            ppr = nxt.find(qn("w:pPr"))
            if ppr is not None and style_of(nxt) in ("BodyText", "FirstParagraph", None):
                sp = ppr.find(qn("w:spacing"))
                if sp is None:
                    set_ppr(nxt, before=12)
        prev = tbl.getprevious()
        if prev is not None and style_of(prev) == "TableCaption":
            long = text_width(text_of(prev), 10) > TEXT_W
            set_ppr(prev, style="TableCaption", before=10, after=5, align="both" if long else "center")

    # ---- figures ----
    n_fig = 0
    for p in list(body.iter(qn("w:p"))):
        if p.find(".//" + qn("w:drawing")) is None:
            continue
        share = info["figures"][n_fig]
        n_fig += 1
        for ext in p.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent",
                          "{http://schemas.openxmlformats.org/drawingml/2006/main}ext"):
            cx, cy = int(ext.get("cx")), int(ext.get("cy"))
            new_cx = int(Inches(TEXT_IN) * share)
            ext.set("cx", str(new_cx))
            ext.set("cy", str(int(cy * new_cx / float(cx))))
        set_ppr(p, style="CaptionedFigure", align="center", before=8, after=4, keep_next=True)
    for p in body.iter(qn("w:p")):
        if style_of(p) == "ImageCaption":
            long = text_width(text_of(p), 10) > TEXT_W
            set_ppr(p, style="ImageCaption", after=14, align="both" if long else "center")

    # ---- body text ----
    in_refs = False
    for p in list(body.iterchildren(qn("w:p"))):
        st = style_of(p)
        txt = text_of(p)
        if st == "Heading1":
            in_refs = txt.strip() == "References"
            continue
        if in_refs and txt.startswith("REFITEM"):
            for t in p.iter(qn("w:t")):
                if (t.text or "").startswith("REFITEM"):
                    t.text = t.text[len("REFITEM"):].lstrip()
                    break
            set_ppr(p, style="Bibliography")
            continue
        if txt.startswith("CENTERLINE"):
            for t in p.iter(qn("w:t")):
                if (t.text or "").startswith("CENTERLINE"):
                    t.text = t.text[len("CENTERLINE"):].lstrip()
                    break
            set_ppr(p, style="BodyText", align="center")
            continue
        ppr = p.find(qn("w:pPr"))
        if ppr is not None and ppr.find(qn("w:numPr")) is not None:
            set_ppr(p, style="Compact", left=500, hanging=220, after=3)
            continue
        runs = [r for r in p.iter(qn("w:r")) if text_of(r).strip()]
        if runs and st in ("BodyText", "FirstParagraph") and all(
                (r.find(qn("w:rPr")) is not None and r.find(qn("w:rPr")).find(qn("w:rStyle")) is not None
                 and r.find(qn("w:rPr")).find(qn("w:rStyle")).get(qn("w:val")) == "VerbatimChar")
                or text_of(r).strip().startswith("\u27e8") for r in runs[:1]) and txt.startswith("Definition:"):
            set_ppr(p, style="BodyText", align="center")

    # last list item of each list: normal space after it
    items = [p for p in body.iterchildren(qn("w:p"))
             if p.find(qn("w:pPr")) is not None and p.find(qn("w:pPr")).find(qn("w:numPr")) is not None]
    for p in items:
        nxt = p.getnext()
        if nxt is None or nxt.find(qn("w:pPr")) is None or nxt.find(qn("w:pPr")).find(qn("w:numPr")) is None:
            set_ppr(p, style="Compact", left=500, hanging=220, after=5.5)

    # ---- Word needs a paragraph between two tables, and this keeps them apart ----
    for tbl in list(body.iterchildren(qn("w:tbl"))):
        nxt = tbl.getnext()
        if nxt is not None and nxt.tag == qn("w:tbl"):
            tbl.addnext(para(after=0, line=120))

    # ---- math: an empty separator inside brackets makes LibreOffice draw bars instead of the brackets; with one
    #      element between the brackets no separator is needed, so it is left out (Word does not need it) ----
    for dnode in doc.element.iter("{%s}d" % M_NS):
        if len(dnode.findall("{%s}e" % M_NS)) == 1:
            dpr = dnode.find("{%s}dPr" % M_NS)
            for sep in ([] if dpr is None else dpr.findall("{%s}sepChr" % M_NS)):
                if sep.get("{%s}val" % M_NS) == "":
                    dpr.remove(sep)

    # ---- math: element order inside delimiters (pandoc writes them in the wrong order) ----
    order = ["begChr", "sepChr", "endChr", "grow", "shp", "ctrlPr"]
    for dpr in doc.element.iter("{%s}dPr" % M_NS):
        kids = sorted(list(dpr), key=lambda c: order.index(c.tag.split("}")[1]))
        for c in list(dpr):
            dpr.remove(c)
        for c in kids:
            dpr.append(c)

    for parent_tag, order in (("rPr", ["lit", "nor", "scr", "sty", "brk", "aln"]),
                              ("mcPr", ["count", "mcJc"])):
        for node in doc.element.iter("{%s}%s" % (M_NS, parent_tag)):
            kids = sorted(list(node), key=lambda c: order.index(c.tag.split("}")[1]))
            if any(c.tag.endswith("}nor") for c in kids):       # "normal text" excludes a style
                kids = [c for c in kids if c.tag.split("}")[1] not in ("scr", "sty")]
            for c in list(node):
                node.remove(c)
            for c in kids:
                node.append(c)
    numbering = doc.part.numbering_part.element
    for nsid in numbering.iter(qn("w:nsid")):
        nsid.set(qn("w:val"), nsid.get(qn("w:val")).rjust(8, "0"))

    # ---- math runs: the font Word itself writes ----
    for mr in doc.element.iter("{%s}r" % M_NS):
        if mr.find(qn("w:rPr")) is not None:
            continue
        rpr = OxmlElement("w:rPr")
        rpr.append(el("w:rFonts", w_ascii="Cambria Math", w_hAnsi="Cambria Math"))
        mrpr = mr.find("{%s}rPr" % M_NS)
        if mrpr is not None:
            mrpr.addnext(rpr)
        else:
            mr.insert(0, rpr)

    # ---- hyperlinks: colour only, as in the PDF ----
    for r in doc.element.iter(qn("w:r")):
        rpr = r.find(qn("w:rPr"))
        if rpr is not None:
            st = rpr.find(qn("w:rStyle"))
            if st is not None and st.get(qn("w:val")) == "Hyperlink":
                for u in rpr.findall(qn("w:u")):
                    rpr.remove(u)

    # ---- settings: hyphenate like LaTeX does, modern layout mode ----
    settings = doc.settings.element
    for c in list(settings):
        settings.remove(c)
    for xml in (
        '<w:defaultTabStop %s w:val="720"/>' % nsdecls("w"),
        '<w:autoHyphenation %s/>' % nsdecls("w"),
        '<w:characterSpacingControl %s w:val="doNotCompress"/>' % nsdecls("w"),
        '<w:footnotePr %s><w:footnote w:id="-1"/><w:footnote w:id="0"/></w:footnotePr>' % nsdecls("w"),
        '<w:compat %s><w:compatSetting w:name="compatibilityMode" '
        'w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>' % nsdecls("w"),
        '<m:mathPr %s><m:mathFont m:val="Cambria Math"/><m:brkBin m:val="before"/>'
        '<m:brkBinSub m:val="--"/><m:smallFrac m:val="0"/><m:dispDef/><m:lMargin m:val="0"/>'
        '<m:rMargin m:val="0"/><m:defJc m:val="centerGroup"/><m:wrapIndent m:val="1440"/>'
        '<m:intLim m:val="subSup"/><m:naryLim m:val="undOvr"/></m:mathPr>' % nsdecls("m"),
        '<w:themeFontLang %s w:val="en-US"/>' % nsdecls("w"),
        '<w:decimalSymbol %s w:val="."/>' % nsdecls("w"),
        '<w:listSeparator %s w:val=","/>' % nsdecls("w"),
    ):
        settings.append(parse_xml(xml))

    doc.core_properties.title = " ".join(info["title"].replace("\\\\", " ").split())
    doc.core_properties.author = ", ".join(a[0] for a in info["authors"])
    doc.save(OUT)


def main():
    global NAME, OUT, TEXT_W, TEXT_IN
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        NAME = args[0]
        OUT = os.path.join(HERE, NAME + ".docx")
    if NAME == "main_compact":          # the PDF of this version has a 6.5 in text block
        TEXT_W, TEXT_IN = 9360, 6.5
    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(BUILD)
    info = prepare_tex()
    subprocess.run(["pandoc", NAME + "_docx.tex", "-f", "latex", "-t", "docx", "-o", "raw.docx"],
                   cwd=BUILD, check=True)
    restyle(info)
    if "--keep" not in sys.argv:
        shutil.rmtree(BUILD, ignore_errors=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
