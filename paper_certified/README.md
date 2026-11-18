# Paper draft: the certified switch ("Finders Keepers")

Status file for whoever continues this draft.

* `main.tex` includes `sec_abstract.tex`, `sec_method.tex`, `sec_results.tex`, `sec_intro.tex`,
  `sec_related.tex`, `sec_discussion.tex` (discussion, limitations, conclusion, acknowledgment)
  and `sec_appendix.tex` (which includes `sec_appendix_fifteen.tex`, `sec_appendix_widen.tex` and
  `sec_appendix_latent.tex`).
  Section order, as asked for: methodology, results, introduction, then the rest.
* `make_results.py` makes every table (`tables/`), every figure (`figures/`) and `facts.txt` (the
  numbers to quote in the text) from the result archives in `../silent_adapters/results/`, the
  routing checks in `routing/` and the demos in `../certified_switch/demo_results/`. Run it again
  whenever a result arrives; do not type numbers by hand.
* `sec_results.tex` also includes `sec_results_fifteen.tex` and, after the subsection on the 875 tasks,
  `sec_results_paraphrase.tex` (machine paraphrases of 700 real tasks, label `sec:paraphrase`). The short
  version puts that file into its appendix (`short_appendix.tex`), like `sec_results_fifteen.tex`.
* `sec_results.tex` includes `sec_results_five.tex`.

Build:

```bash
python3 make_results.py
```

```bash
python3 latent_results.py
```

`latent_results.py` (run it after `make_results.py`) makes `tables/latent_main.tex`,
`tables/latent_wordings.tex`, `tables/latent_pertask.tex` and `tables/numbers_latent.tex` (the
`\Latent...` commands) from the Kaggle archive of the Latent-LoRA run. Both `main.tex` and
`short.tex` input the numbers file, and `make_docx.py` reads it too.

```bash
python3 widen_results.py
```

`widen_results.py` (run it after `latent_results.py`) reads `routing/{feature,ruler,encoder,judge}_study.json`
and `routing/paraphrases.json` and writes `tables/widen_main.tex`, `tables/widen_encoder.tex`,
`tables/widen_judge.tex`, `tables/widen_siblings.tex`, `figures/widen_tradeoff.{pdf,png}` and
`tables/numbers_widen.tex` (the `\Widen...` commands). `main.tex` and `short.tex` input the numbers file and
`make_docx.py` reads it too. The thresholds of the rules fixed before the runs (75%, 5%, 19%) are typed in the
script; every other number is read from the json files.

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

```bash
python3 make_docx.py
```

The last command builds `main.docx` in the layout of the PDF (it reads the numbers of sections,
equations, tables and theorems from `main.aux`, so the PDF has to be compiled first). Check a new
Word file with the validator of the docx skill and by converting it to PDF; brackets in formulas
can look like bars in LibreOffice although the file is correct.

**Two versions.** `main.tex` is the long version (method first, as asked for; 62 pages).
`short.tex` is a 9-page conference version in the usual order (introduction, related work, method,
experiments, limitations, conclusion), made on 2026-11-17 by a smaller model from the long version
and checked afterwards; everything that did not fit is in its appendix (`short_*.tex`, then
`sec_appendix.tex` unchanged). Both use the same tables, figures and number commands. Build the
short one with `pdflatex short && bibtex short && pdflatex short && pdflatex short`. Its main text
fits the nine pages only just: after any edit check that the Conclusion still ends on page 9.

**The short version as two files.** `short.pdf` holds everything (9 pages of paper, then the
acknowledgments and references up to page 12, then 25 pages of appendix). For readers who want a
short document, `short_main_only.pdf` is pages 1 to 12 and `short_appendix_only.pdf` is the rest
(the supplementary material; on 2026-11-18: 13 and 48 pages). After every new build of `short.pdf` make the two again with
`pdfseparate` and `pdfunite`, and check on which page the appendix starts (it begins on a new page).

Rules for the text: plain words; say what was tested and what was not; every comparison with the
published methods must say whether a gap comes from learning or from forgetting; the reworded
instructions are a development set written by machine during development (the paper calls them
the development rewordings and must not suggest that people wrote them); no claim about the published setting (a larger model;
100 epochs were run only for the published methods, one seed).

## Status on 2026-11-17, 22:00 UTC (written for a later check-in)

Done: every section of both versions, including the 15-task run (`sec_results_fifteen.tex`,
`sec_appendix_fifteen.tex`; in the short version the same file is included in the appendix). The
long version compiles to 38 pages, the short one to 37 (9 pages of paper, references up to page 12,
appendix from page 13). `main.docx` and `short.docx` are built. No `%%NAME%%` marker is left.

How numbers get into the text: `make_results.py` also writes `tables/numbers.tex`, one LaTeX command
per number (for example `\FiveCertAP`, `\ManyCollidingPct`, `\FifteenCertChanged`). The prose
uses these commands, so it follows the result files by itself. After new results: run
`make_results.py`, compile, and **read the sentences again**: the numbers update, the words
around them (for example "well below", "do not forget") do not.

CPU analyses of the routes (run inside the folder of the official GainLoRA code, see the top of each
file): `extra_analysis.py` (distances from the zones, cost), `order_analysis.py` (task order),
`edit_analysis.py` (mechanical edit test), `background_analysis.py` (other unrelated inputs),
`many_tasks_fetch.py` and `many_tasks_analysis.py` (875 real tasks), `exact_first_analysis.py`
(the variant), `bound_check.py` (the edit bound), `paraphrase_make.py` (machine paraphrases of the
875 definitions), `feature_study.py` and `ruler_study.py` (two ideas for wider zones, see the plan).
The rules of each test were written in `../silent_adapters/PLAN.md` before it was run, and its
reading after. Their numbers are in `routing/*.json`.

Open:

* ~~The comparison with our re-implementation of Latent-LoRA~~ — done 2026-11-17 (night): appendix
  `sec_appendix_latent.tex` (label `app:latent`), numbers in `tables/numbers_latent.tex`, tables
  from `latent_results.py`. It learned little (10 epochs, no public code), so only its router is
  compared; the run was not pre-registered.
* A last careful read of the changed parts of both PDFs (Section `sec:paraphrase`, Appendices `app:widen` and `app:reroute`, the abstract, introduction, discussion and the short version's limitations).
* A fresh machine has no LaTeX: `apt-get install -y texlive-latex-extra texlive-fonts-recommended
  texlive-fonts-extra` first.

## Status on 2026-11-18

Done: the test with machine paraphrases of 700 real tasks and the four attempts to widen the zones are in
both versions (long: Section `sec:paraphrase` after the 875-task subsection, Appendix `app:widen`; short: one
sentence each in abstract, intro, experiments and limitations, the section itself and `app:widen` in the
appendix). Abstract, introduction ("What we do not claim", contribution on the cost), discussion (section on
word-vectors, limitations, conclusion) and the sentence that introduces the mechanical edit test were changed so
that they agree with it. Pages: `main.pdf` 46; `short.pdf` 44 (9 pages of paper, the Conclusion ends on page 9,
acknowledgments and references to page 12, appendix from page 13); `short_main_only.pdf` 12 and
`short_appendix_only.pdf` 32 pages. No overfull box, no undefined reference or citation in either log;
`main.docx` and `short.docx` built and validated.

The Spanish check (`routing/spanish_check.json`, script `spanish_check.py`; rule and reading in the last two sections of
`../silent_adapters/PLAN.md`) is in both versions: verdict "not confirmed" (gain in P below the 15 points asked for). It is
read by `widen_results.py` (table `tables/widen_spanish.tex`, `\WidenSp...` and `\WidenSplit...` commands). The main
table has a row for the rule of the paper on the French pivot, for like-with-like comparison of the French rows. The
short version keeps the conference style untouched (no change of line spacing or captions); to fit nine pages the edit
figure, the paragraphs on reproducibility and on the guaranteed tolerance and the text on other models were moved to
its appendix (they are there in full, with pointers).

References added for this work: `junczys2018marian`, `tiedemann2020opusmt`, `reimers2019sbert`, `wang2020minilm`,
`he2023debertav3`, `vovk2005algorithmic`, `angelopoulos2023conformal`. Checked on the ACL Anthology / arXiv on
2026-11-18: Marian, OPUS-MT, Sentence-BERT (authors, venue, pages), DeBERTaV3 (ICLR 2023, from the arXiv comment).
Also checked on 2026-11-18: MiniLM (title, authors, NeurIPS 2020, on the proceedings page; the page shows no
page numbers), Angelopoulos and Bates (title, volume 16, issue 4, pages 494-591, 2023, in the Crossref record of
DOI 10.1561/2200000101; the arXiv version has the older title) and Vovk, Gammerman and Shafer (title, Springer,
2005, in the Crossref record of DOI 10.1007/b106715; that record lists no authors).

### Update later on 2026-11-18

* **Second reader's corrections** (items 1 to 19) applied to `sec_results_paraphrase.tex`, `sec_appendix_widen.tex`, the abstract,
  introduction and discussion; new remark "Other zones" (`rem:otherzones`) after Theorem 1. Figure and Spanish table redone (the Spanish check now
  has two small tables, `widen_spanish` and `widen_splits`). Referees of the first round: see the earlier report (items applied in the long
  and short versions).
* **Re-routing study** (`routing/reroute_study.json`, script `reroute_study.py`; rules and readings in the last five sections of `../silent_adapters/PLAN.md`):
  Section `sec:paraphrase` (paragraph "Do later tasks re-route reworded inputs?", Table `tab:reroute`) and Appendix `app:reroute`. `widen_results.py` reads
  it and writes `tables/reroute.tex` and the `\Reroute...` commands (in `tables/numbers_widen.tex`). The first reading went against the paper's claim at scale
  (nearest zone and certified switch changed the same 13 of 4,198 routes); the plain nearest signature serves more machine-paraphrased inputs, and the
  fallbacks trade places on the benchmark. Stated in the abstract, introduction, discussion and the short version.
* New macros `\signature` and `\hybrid` (main.tex, short.tex, make_docx.py).
* Short version: to keep the Conclusion on page 9 without any change of style, the stress-test table and the 875-task table were moved to its appendix.
* References added: `song2020mpnet` (confirmed on the NeurIPS proceedings page) and `xiao2023cpack` (authors and title on arXiv; the venue SIGIR 2024
  is taken from the arXiv comment and was not confirmed on the publisher's page); Chung et al. now has number 70 and pages 1-53 (confirmed on jmlr.org).
* Pages: `main.pdf` 52; `short.pdf` 51 (the Conclusion ends on page 9; acknowledgments from page 10; references to page 13; the appendix starts on
  page 14); `short_main_only.pdf` 13 and `short_appendix_only.pdf` 38 pages.

### Final state on 2026-11-18

* Four careful reads (two of the long version's method, results and the new parts, one of the short version, one of the last changes) were done on
  2026-11-18 and their corrections applied; the reports are not in the repository. The short version's Discussion in the appendix now reads the same file as the long
  version (`sec_discussion_paras.tex`), so the two cannot drift apart.
* Scripts and results added: `reroute_study.py` with `routing/reroute_study.json` (read by `widen_results.py`, which writes `tables/reroute.tex`, `tables/widen_*.tex`,
  `figures/widen_tradeoff.pdf` and `tables/numbers_widen.tex`).
* The short version keeps the conference style untouched (no float spacing, font size or line spacing changes; only the empty notice string, because the authors asked for no "Preprint." line).
* Pages: `main.pdf` 53; `short.pdf` 52 (the Conclusion ends on page 9; acknowledgments and references pages 10 to 13; the appendix starts on page 14);
  `short_main_only.pdf` 13 and `short_appendix_only.pdf` 39 pages. No overfull box, no undefined or multiply defined reference, no "??" in either PDF; both docx validated.
* The stand-alone code implements the zero-radius convention (test in `certified_switch/test_zero_radius.py`); the experiment code does not (appendix `app:impl`).

**Open:** (1) rewordings written by people (ten are enough, two per task); (2) before a conference submission: the conference checklist and anonymisation
(author names, acknowledgments, the repository link); (3) the order of the sections of the long version (method, results, introduction) is the authors' choice;
reviewers may expect the introduction first; (4) the 2026 preprint references (`latentlora2026`, `spacelora2026`, `lorauter2026`, `read2026`,
`maskthetarget2026`, and the other 2026 entries) should be opened once more; (5) a last read of the final PDFs.

### Last state on 2026-11-18 (after the late additions)

* **New files:** `sec_bound.tex` (subsection `sec:bound`: zones from the inputs of a task, Theorem `thm:bound`, Corollary, Remark), `sec_results_noinstruction.tex`
  (`sec:noinstruction`: Pilot A on routes, Pilot B with a small trained model), `bound_results.py` (writes `tables/numbers_bound.tex`, `tables/bound_pilot.tex`,
  `tables/bound_pilotb.tex`, `tables/bound_minilm.tex`; run it after `widen_results.py`), `noinstruction_study.py` and its result `routing/noinstruction_study.json`
  (Pilot B: `../certified_switch/demo_results/no_instruction_t5-small_seed0*.json`), `routing/sentence_judge_study.json` (attempt 5 to widen the zones; the diary calls it the sixth idea),
  `routing/paraphrases_it.json` (Italian paraphrases), `routing/reroute_study.json` with the scaled fallbacks and the identity check on the benchmark.
  `tables/bound_minilm.tex` is one sentence with the sentence-encoder numbers (or, if the feature is missing from the file, a note that it was not finished).
* **Statements added:** the two fallback tests and their success (`app:reroute`), the identity of the routes of the variant with those of the rule of the paper on the benchmark sequences (so every GPU score is also
  the score of the variant), attempt 5 and its verdict (against, Italian pivot), the bound without instructions (met in every run; no method that works without instructions was reached), and the benchmark point
  (with unedited instructions a test cannot tell retention from recognition; nothing is claimed about the published methods without instructions).
* Pages: `main.pdf` 62; `short.pdf` 61 (the Conclusion ends on page 9; acknowledgments and references pages 10 to 13; the appendix starts on page 14); `short_main_only.pdf` 13 and `short_appendix_only.pdf` 48 pages.
  No overfull box, no undefined or multiply defined reference, no "??"; both docx validated.
* Six careful reads were made on 2026-11-18 (four before 02:00 UTC on the whole paper, two after 03:30 UTC on the parts added later) and their corrections were applied; the reports are not in the repository. The sentence-encoder rows of the no-instruction test are in (they change no verdict).
* **Open:** (1) rewordings written by people, and the published methods run without instructions (the experiment that would show whether they depend on the instruction); (2) a frozen feature in which the inputs of different
  tasks are apart without an instruction (hidden states of the frozen model, a sentence encoder), then a GPU run of the version without instructions; (3) a GPU run of the fallback variant is not needed for the benchmark, because its routes there are
  identical to those of the rule of the paper, but it is untested with trained answers at scale; (4) the conference checklist and anonymisation (author names, acknowledgments, repository link) before a conference submission;
  (5) the order of the sections of the long version (method, results, introduction) is the authors' choice, and reviewers may expect the introduction first; (6) the 2026 preprint references should be opened once more;
  (7) a final read of the parts added on 2026-11-18 after 01:30 UTC (fallback tests, attempt 5, the bound and its tests).

## Before this paper is shared

* References: the ten that had been written from memory and the thirteen added afterwards were
  checked against the web on 2026-11-17 (all correct). The other twenty had been looked up when the
  first paper was written; the 2026 preprints among them (`latentlora2026`, `spacelora2026`,
  `lorauter2026`, `read2026`, `maskthetarget2026`) should be opened once more before submission.
* The reworded instructions are a development set written by machine. A claim about
  rewording needs rewordings written by a person who has not seen the method (ten are enough: two
  per task). The mechanical edit test does not replace them.
* Every claim is limited to the reduced setting (T5-base, 10 epochs, free GPUs).

## Restructured on 2026-11-18 (best method first, as asked for)

This section is newer than the file list and the section-order sentence at the top of this file, which describe the earlier order (method, results, introduction). The earlier "authors' choice" of putting the introduction third is given up.

* **New order of the long version** (`main.tex`, 78 pages): abstract (`sec_abstract.tex`), 1 Introduction, 2 Related work, 3 Method, 4 Results, 5 Discussion, 6 Limitations, 7 Conclusion, appendices.
  * 3 Method: 3.1 the setting and the lemma "an answer changes only with its route"; 3.2 without instructions (recogniser, audit, handicap; `sec_audit.tex`); then word-vector features, protected zones, the certified switch, what the zones guarantee, the variant for sibling tasks, zones from the inputs of a task, the compared methods, the protocol.
  * 4 Results: 4.1 published methods without instructions (`sec_results_nodef_published.tex`); 4.2 recognition without instructions (`sec_results_recognition.tex`); 4.3 trained answers without instructions (`sec_results_nodef_trained.tex`); 4.4 five tasks with instructions, including the new subsection "Routes from the example alone" (tables `tables/instance_routes.tex` and `tables/instance_routes_short.tex`); then reworded instructions, the gap, fifteen tasks, stress test, 875 tasks, edits, orders, unrelated inputs, other models, rewording at scale, zones from inputs.
* **Short version** (`short.tex`, 77 pages, main text 9 pages): the same order. 3.2 recogniser, audit, handicap; 3.3 zones and the certified switch; 3.4 zones from inputs; 4.2 without instructions; 4.3 with instructions. The word-vector feature details and the full statements of four propositions moved to the appendix. Title and authors unchanged.
* **`sec_results_recogniser.tex` is now an empty stub.** Its content was moved into the files above.
* **Results used:** `../silent_adapters/results/2026-11-18_kaggle_instance_routes_with_definitions/` (run U), `../silent_adapters/results/2026-11-18_kaggle_hidden_states_scale/` (feature job), and the earlier 2026-11-18 folders of the runs without instructions.
* **Status (about 12:32 UTC):** both versions were corrected after two independent readings (about 45 points in the long version, about 35 in the short one: leftovers of the old order, caveats for the newest results, exact wording for tasks that share texts, a symbol clash). Both now contain run D (the published methods without definitions at 100 epochs). Long version 80 pages; short version 78 pages, main text 9 pages (the Conclusion ends on page 9). The Word files and the split PDFs were rebuilt (the split PDFs are now small files, made with pypdf). The files were sent at about 12:32 UTC. The page counts were not checked here. The PDFs sent on 2026-11-18 at 03:56 UTC are older than these results.
* **New results order of the long version:** 4.1 published methods without instructions; 4.2 recognition without instructions; 4.3 trained answers without instructions (with the 100-epoch run); 4.4 five tasks with instructions; 4.5 reworded instructions; 4.6 routes from the example alone; 4.7 fifteen tasks; 4.8 where the gap comes from (`sec_results_gap.tex`, moved behind the fifteen tasks); then stress test, 875 tasks, edits, orders, unrelated inputs, other models, rewording at scale, zones from inputs.
* **New table files for run D:** `tables/noinst_e100.tex`, `tables/noinst_e100_pertask.tex`, `tables/noinst_seeds_e100.tex`. Results: `../silent_adapters/results/2026-11-18_kaggle_noinstruction_published_five_tasks_e100/`.
* **Not in the paper yet:** run E (the control with definitions at 100 epochs; not run), the 15-task runs without definitions.

**Shortened copy of the long version.** `main_compact.tex` reads the files `k_*.tex` (`k_abstract`, `k_method`, `k_results_noinst`, `k_results_inst`, `k_results_scale`, `k_intro`, `k_related`, `k_discussion`, and for the appendix `k_app_method`, `k_app_noinst`, `k_app_inst`, `k_appendix`, `k_app_widen`) and the same tables and number files as `main.tex`. Build: `pdflatex main_compact && bibtex main_compact && pdflatex main_compact && pdflatex main_compact` (66 pages); Word copy: `python3 make_docx.py main_compact`. It says the same as the long version with less repetition; a change of content must be made in both (`sec_*.tex` and `k_*.tex`).

**Figures.** `figstyle.py` holds the colours, fonts and helpers of every figure; a script that draws a figure imports it. `make_figures.py` draws `answer_grid`, `stage_heatmap`, `route_matrix` and `learn_keep` from the run archives (run it after `audit_results.py`; it checks its numbers against `tables/numbers_audit.tex`). `figures_src/overview.tex` and `figures_src/switch.tex` are stand-alone TikZ files: compile with `pdflatex` and copy the PDF to `figures/`. The files `figblock_*.tex` hold the figure environments with their captions; the long, the shortened and the 9-page version all input them.
