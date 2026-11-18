# Latent-LoRA as a comparison method (our re-implementation)

Paper: R. R. Azghan, G. K. Gudur, G. Pedrielli, P. Turaga, H. Ghasemzadeh, "Latent-LoRA: Compact Latent-Space Adapters with
Gradient-Free Routing for Continual Learning", arXiv:2607.23837 (v1, 26 Jul 2026); Findings of EMNLP 2026
(https://aclanthology.org/2026.findings-emnlp.1390/).

**There is no official code.** On 2026-11-17 the arXiv PDF (`https://arxiv.org/pdf/2607.23837`), the ACL Anthology PDF
(`https://aclanthology.org/2026.findings-emnlp.1390.pdf`) and both abstract pages were downloaded and searched (text of both PDFs,
HTML of both pages): none contains a link to code (no github/gitlab/huggingface/zenodo/supplement link; the arXiv "Code, Data, Media" tab only offers third-party code
finders). Nothing else was searched. **Everything below is written from the paper's text only.** The files are therefore a
*re-implementation*, and section 2 lists every detail that the paper does not give.

Files

| File | What it is |
|---|---|
| `latent_lora.py` | `LatentLoRAModel` (`learn`, `answer`), `GMMRouter`, `LatentLoRALayer`; and `CertifiedOnDevice`, a thin subclass of the project's `ContinualModel` (see section 4) |
| `run_five.py` | the five-task runner for both methods (self-test, tiny mode, answers after every task, scores, summary) |
| `make_notebook.py`, `kaggle_cells.py`, `latent_lora_kaggle.ipynb` | the Kaggle notebook (generated; all code is embedded in it) |
| `test_equivalence.py` | CPU checks (section 5) |
| `rewordings.json` | copy of `certified_switch/rewordings.json` (not edited) |

## 1. The method, from the paper's text

**Setting (Sec. 2.1, Eq. 1).** Tasks arrive one at a time; exemplar-free (no stored examples), task-agnostic (no task identity at test time).

**Adapter (Sec. 3.2, LoRA-XS, Eq. 5).** For a frozen weight `W (m x n)` the rank-r truncated SVD `W ~ U_r Sigma_r V_r^T` is computed once and all three
factors stay frozen. The only trainable object of task t in this layer is a square matrix `R_t (r x r)`, and
`Delta W_t = U_r Sigma_r R_t V_r^T`  (Eq. 5). Trainable numbers per module: r^2 (r = 32: 1,024; the paper reports 147K per task on T5-Large = 144 modules x 1,024, Sec. 4.5, Table 8).
Adapted layers: query and value projections of every attention layer (Sec. 4.1, App. C.3: T5-Large has 144 target modules: 24 encoder self-attention + 24 decoder
self-attention + 24 decoder cross-attention, each with q and v). The base model, including its embedding layer, is frozen (Sec. 3.1).

**Interference and regularisation (Sec. 3.2, Eq. 6 to 10, Proposition 1, App. A).** With `R~_t = Sigma_r R_t`, the inner product of the output perturbations of tasks i and j
is `psi(h)^T R~_i^T R~_j psi(h)`, `psi(h) = V_r^T h` (Eq. 7), bounded by `||R~_i^T R~_j||_2 ||psi(h)||^2` (Eq. 9).
The regulariser is `L_ortho = sum_{i<j} ||R~_i^T R~_j||_F^2` (Eq. 8). Training objective of task t (Eq. 10):
`L_t = - sum_{(x,y) in D_t} log p(y|x) + lambda * sum_{i<t} ||R~_i^T R~_t||_F^2`. Only `R_t` gets gradients; older `R_i` are frozen snapshots, never modified (Sec. 3.2 "Snapshot isolation").

**Task recognition (Sec. 3.3, Eq. 11 to 15), no gradient, no trained parameter.**
- Features: `phi(x) = MeanPool(Emb(x))`, the mean of the base model's input-embedding vectors of the tokens of x (Eq. 11). Stationary, because adapters do not touch the embedding layer.
- After the adapter of task t is trained, a Gaussian mixture with K components is fitted on `{phi(x) : x in D_t}` (Eq. 12): weights `pi_{t,k}` and means `mu_{t,k}` "initialized via K-means";
  one covariance C shared by all tasks and all components.
- `C = (sum_t S_t) / (sum_t n_t) + eps I`, `S_t` the scatter matrix of task t around its mean `phi_bar_t` (Eq. 13). When a task arrives, C is updated without the data of older tasks.
- At inference: log-likelihood of `phi(x)` under every task's mixture; posterior under a uniform prior `p(t|x) = p(phi|t) / sum_t' p(phi|t')` (Eq. 14).
- Combination: `R(x) = sum_t p(t|x) R_t` (Eq. 15, "soft routing"): one blended adapter per input, inserted in each attention projection (Sec. 4.5: the same overhead as one LoRA forward pass).
  The "sum-at-inference" ablation (Table 4) is `R(x) = sum_t R_t`.

**Hyper-parameters the paper gives for T5 (Sec. 4.1 and App. C.3).**

| Item | Paper | Where |
|---|---|---|
| rank r | 32 (baselines: 8) | Sec. 4.1, C.3 |
| alpha | 16 | C.3 ("our method uses rank r=32 and alpha=16") |
| layers | q and v of all attention blocks | Sec. 4.1, C.3 |
| optimiser | AdamW, constant learning rate | Sec. 4.1, C.3 |
| learning rate (T5) | 3e-4 | C.3 |
| batch size (T5) | 8 | C.3 |
| epochs (T5) | 30 | C.3 |
| lambda | 0.05 for SuperNI (0.02 for Long Sequence) | C.3 ("selected based on the relative magnitude of the Sigma-weighted ortho loss to the task loss on a held-out validation split of T5-Large") |
| GMM | K = 5 components per task, K-means with 30 iterations, eps = 0.01, soft routing | C.3 |
| decoding | greedy, maximum 50 new tokens | C.3 |
| seeds | 3 | Sec. 4.1 |
| metrics | AP (Eq. 16), FM (Eq. 17); Rouge-L with `rouge_score` and stemming for generation tasks, exact-match accuracy for classification | Sec. 4.1, C.3 |
| hardware | A100 | C.3 |

Paper's result for orientation (T5-Large, SuperNI Order 1, 15 tasks): AP 48.60, FM 0.01 (Table 1). Not comparable to the five-task t5-base setting here.

## 2. What the paper does not give, and the choice made here

| # | Detail missing in the paper | Choice made | Reason |
|---|---|---|---|
| 1 | How alpha enters the layer | output scaled by `alpha / r` = 16/32 = 0.5 | the usual LoRA convention; alpha is listed next to r |
| 2 | Initial value of `R_t` | zeros | the new adapter then changes nothing at the start; its gradient is non-zero because the update is linear in R |
| 3 | What is switched on while task t is trained | only `R_t` (the router of task t does not exist yet, it is fitted after training; older adapters are frozen and not used) | Eq. 10 trains only R_t; the router that could blend older adapters is fitted afterwards. Older adapters still enter through the Eq. 8 term |
| 4 | Text the router sees | the whole instruction-formatted input (definition + example) as tokenised for the model (EOS included, padding left out, cut at 512 tokens); at test time the same, so reworded definitions go through the same router | "the input text x"; instruction tuning is used (Sec. 4.1) |
| 5 | Pooling details | plain mean over the non-padding tokens | Eq. 11 says MeanPool |
| 6 | K-means seeding; use of EM | k-means++ seeding, 30 Lloyd steps, no EM refinement; `pi_k` = share of points in cluster k; `mu_k` = centroid; empty clusters dropped | App. C.3 says K-means with 30 iterations; Sec. 4.5 says means and weights are "set analytically from K-means" |
| 7 | Covariance | exactly Eq. 13 (within-task scatter around the task mean, divided by the total n, plus 0.01 I); kept as running sums, so no old data is needed; recomputed (Cholesky) after each task and used for all tasks | Eq. 13 and the text below it |
| 8 | Normal-density constants | omitted (shared by all tasks, cancel in Eq. 14) | exact |
| 9 | Meaning of the sum in Eq. 8/10 over many modules | the Eq. 8 term is computed per adapted module and **summed over the 72 modules of t5-base**; the language-model loss is the usual mean over answer tokens of the batch (not the sum over the data set of Eq. 10) | the paper only writes one module; a mean over modules is available as `--ll_ortho_reduce mean` (not used). Because lambda was tuned by the authors against *their* loss scale, 0.05 here is a transfer, not a tuned value |
| 10 | Weight decay of AdamW | PyTorch default 0.01, the same as for "ours" | not given |
| 11 | Dropout, precision | T5's default dropout 0.1 in training for both methods; float32 | not given |
| 12 | Epochs | **10, not the paper's 30** (as asked, same as "ours") | project's small setting. Batch size 8 with 800 examples = 1,000 optimiser steps per task instead of the paper's 3,000 |
| 13 | Batch size | Latent-LoRA: the paper's **8**; "ours": effective **32** (pieces of 16, accumulated twice) | the paper gives 8; the project's benchmark setting is 32. This is a difference between the two methods, in the paper's favour |
| 14 | Prompt template, data | the project's template (`demo.py`) and the SuperNI files of the GainLoRA repository | the same inputs for both methods |
| 15 | Score | Rouge-L of `demo.py` (lower-case words, no stemming, best of all reference answers) for all five tasks | same function as "ours". The paper uses `rouge_score` with stemming and accuracy for classification tasks; for one-word answers the two agree except for case-folding |
| 16 | Forgetting measure | `FT` = mean over tasks 1..T-1 of (best score after learning it and before the last task) minus (final score); also stored: `FT_vs_learned` (score right after learning minus final) | the paper's Eq. 17 takes the maximum over `i >= j`, which includes the last step (never below 0 in the paper's definition); the project's earlier notebooks use the definition given here |
| 17 | SVD | full SVD in float64 on the CPU, top r kept, converted to float32 | exact, once per layer |
| 18 | Routing at the very first task | one task: posterior is 1 | trivial |
| 19 | Task order | the five tasks of the project (not the paper's 15-task orders) | the comparison runs the project's setting |
| 20 | Cross-attention | the router weights of an input are used for its encoder self-attention, decoder self-attention and cross-attention adapters alike | one blended adapter per input (Sec. 3.3) |

Rules recorded for Latent-LoRA from the same trained adapters: `soft` (the paper, Eq. 15) and `sum` (all adapters added; the paper's no-router ablation, Table 4). `answer` also accepts `hard` (most probable task only), not used in the run.
`answer` returns, like `ContinualModel.answer`, (answers, most probable task of every input, "certified" flags = all False; there is no guarantee).

## 3. How both methods are run (`run_five.py`)

For every seed (42, 43, 44) and method (`ours`, `latent_lora`): a fresh `t5-base`; the five tasks task1510, task363, task748, task1687, task875, in this order; 800 training examples
(all of `train.json`), the 100 test examples; 10 epochs; input length 512, answer length 50; greedy decoding; AdamW with constant learning rate.
- `ours`: `ContinualModel` (`certified_switch.py`, not edited), rule `certified`, rank 4, alpha 32, learning rate 3e-4; from the same adapters the answers under `nearest` and `all` (plain adapters) are also recorded.
- `latent_lora`: this folder's model, rank 32, alpha 16, lambda 0.05, K = 5, eps = 0.01, learning rate 3e-4, batch 8; rules `soft` (the method) and `sum`.
After each task the test inputs of every task learned so far are answered with the original and the three reworded definitions; every answer is saved in `results/<method>_seed<seed>.json` (rewritten after every task, so a crash keeps the finished tasks; an interrupted job restarts from task 1 because no weights are saved).
Reported per method, rule and wording: **Learned**, **AP**, **FT**, **answers changed** (tasks 1 to 4, end versus right after learning), and the share of test inputs given to their own task.
For `ours`/`certified`/original, answers changed is 0 by construction (a run that shows otherwise is a bug).

Self-test (always on, first thing in every `run_five.py` process): for each method, two tasks with 8 training examples and one optimiser step each (in two pieces of 4: gradient accumulation is exercised),
`answer` under every rule, one JSON save and read-back, and a check that every parameter and buffer of the model is on the device. On failure the process prints a boxed message and exits with code 2; the notebook then starts no further job.

Environment variables / options: `LL_TEST=1` (tiny mode: `t5-small`, 24 training and 8 test examples per task, 1 epoch, one seed), `LL_DEVICE`, `--methods`, `--seeds`, `--summary`.

## 4. Devices (what happens on a GPU)

- The base model is loaded on the CPU, wrapped, and moved to the device **once**, inside the constructors (`to_device_once`). `self.device` is set after that.
- `ContinualModel` builds `Zones` from `model.get_input_embeddings().weight` at construction and reads `self.device` at construction. A model already on a GPU would put the zone computations on the GPU in float64.
  `certified_switch.py` is not edited; `CertifiedOnDevice` (a subclass here) builds it on the CPU, so zones, whitening, signatures and routes are float64 CPU tensors, and then moves the model. Adapters made later by `new_task` are created on the device of the weight they wrap.
  Its `learn` has the same steps as `ContinualModel.learn` but goes through the shared loop with gradient accumulation and an out-of-memory fallback (halves the piece size and repeats the step). With piece size >= batch the adapters are bit-identical to those of `ContinualModel.learn` (`test_equivalence.py`).
- Latent-LoRA: the SVD, the embedding table copy, k-means, covariance, Cholesky and posterior are float64 CPU tensors. Only the per-batch router weights (`weights[s:s+batch].to(device, dtype)`) cross to the device, and token ids (`_pad`) are created on the device.
- Crossings (CPU -> device only): token-id and attention-mask batches (`_pad`), the per-input route (ours: long tensor; Latent-LoRA: float weights matrix), the frozen SVD buffers `U, S, V` (moved with the model). Device -> CPU: generated token ids (decoded on the CPU), `loss.item()`, route tensors are never on the device.

## 5. Tests done (CPU, 2 threads)

`test_equivalence.py` (t5-small): (1) `CertifiedOnDevice.learn` == `ContinualModel.learn` exactly when pieces >= batch; (2) with accumulation the adapters differ by 2e-5 (no dropout; with dropout the random masks differ);
(3) the out-of-memory fallback; (4) the Latent-LoRA layer equals `base + (alpha/r) U S R(x) V^T x` for one-hot and blended weights, older adapters are not changed by a later task, routes of two tasks' inputs are right, answers do not depend on the batch size; (5) `rouge_l` in `run_five.py` is the text of `demo.py`.
Tiny-mode end-to-end run (`LL_TEST=1`, t5-small, 24 training / 8 test examples, 1 epoch) of both methods and all five tasks, once directly and once through the notebook's code (two jobs in parallel): the saved answers and `summary.json` are in `results_tiny/` (scores around 0.3 of 100 mean nothing at this size). For `ours`, rule `certified`, all four wordings: 0 of 32 answers changed (original wording checked from the saved JSON). The notebook test (`LL_NB_TEST=1`, paths set by `LL_NB_OUT`, `LL_NB_DATA`, `LL_NB_WORK`, `LL_NB_WORKERS=2`) runs the concatenated code cells as a script.

## 6. Known limits

- It cannot be said that this is the authors' implementation. The paper's numbers cannot be used to validate it: the setting (5 tasks, t5-base, 10 epochs) differs.
- 10 epochs instead of 30 leaves a 74K-parameter adapter (r^2 x 72 modules) with fewer steps than the paper used; the learning scores of Latent-LoRA may be lower than in the paper for that reason alone.
- ~~Nothing was run on a GPU.~~ It was run on Kaggle on 2026-11-17 (see "Result of the Kaggle run" below). Time and memory in the CPU report were estimates.

## 7. Result of the Kaggle run (2026-11-17)

Kaggle T4 x2, t5-base, five tasks, 800 training and 100 test inputs, 10 epochs, seeds 42 to 44, both methods through `run_five.py`.
Raw files: `silent_adapters/results/2026-11-17_kaggle_latent_lora_five_tasks/`. Not pre-registered (no rule was written in `PLAN.md` first).
Original instructions, mean of three seeds: the certified switch has Learned = AP = 76.49, FT 0.00, 0 of 1,200 answers changed (all adapters on: AP 36.04, 970 changed).
Latent-LoRA, soft rule: Learned = AP = 42.55, FT 0.00, 0 changed; sum rule: AP 47.66, FT 1.31, 429 changed.
Our re-implementation learned the first task (90.8 for seed 42) and almost nothing after it (Glucose 2.2; last-epoch loss about 2.0 against about 0.6 for ours).
We do not know if the reason is the reduced budget (10 of 30 epochs, smaller model) or a mistake in this re-implementation, so **the run says nothing about the method's learning or retention.**
Its router gave every test input to the right task in all four wordings and no answer changed (soft rule); it was not run in the stress test with a look-alike task.
See `RESULTS_LOG.md` of the repository (section 4L) and the appendix of `paper_certified/`.
