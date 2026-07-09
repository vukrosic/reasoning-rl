# Literature review — reasoning-rl

Annotated, technique-extraction bibliography for this RL reasoning post-training repo. Each entry:
what it is, the **implementable** technique(s) to borrow, and the source URL. Rule: **no number or
mechanism from this list may be cited in a writeup until the source PDF is read directly** — the
`read?` flag tracks that.

Goal: make a ≤1B model reason better with RL (GRPO — see `grpo_train.py` / `grpo_train_base.py`) and
beat the official tiny reasoning models (Qwen3.5-0.8B, MiniCPM5-1B) on GSM8K + MATH-500 from the same
base. The concrete failure this targets: the target base in thinking mode **does not terminate** —
GSM8K traces hit the token cap without closing `</think>` (see `eval_gsm8k.py`). So the read-list is
ranked against two problems: **(1) termination / length control**, **(2) the GRPO algorithm itself
collapsing on a small model**. Cluster C is the go/no-go sanity check before a paid GPU run.

Status: pass 1 (2026-07-09) — recent (2026) papers surfaced via search. **All UNREAD ✗** except
where noted; 2602.09591 abstract confirmed by direct fetch.

---

## A. Termination / length control — the `</think>` non-termination failure  ★ top priority

### A1. On the Optimal Reasoning Length for RL-Trained Language Models  ★ read first
- **What:** trains length-control methods on multiple bases incl. **Qwen3-1.7B-Base** and
  **DeepSeek-R1-Distill-Qwen-1.5B** (our size class); math + code.
- **Borrow / warning:** accuracy is **non-monotonic in length** — peaks at an intermediate value,
  not "more is better". Load-bearing for us: a naive length penalty **hurts** reasoning acquisition on
  a weak base and helps only once the model has a strong prior. So do **not** bolt a brevity penalty
  onto the first GRPO run — characterise the length–accuracy curve first, then target the intermediate
  optimum. Named failure modes: long→dispersion, short→under-thinking.
- **Source:** https://arxiv.org/abs/2602.09591  — read? **abstract only ✓ (fetched 2026-07-09), full PDF ✗**

### A2. APR — Anchor-based Process Rewards (penalize structural redundancy)
- **What:** process-reward shaping that cuts redundant reasoning structure without penalizing correct
  chains — an alternative to a flat brevity reward (which A1 warns against).
- **Borrow:** reward-shaping recipe for teaching crisp termination without collapsing correctness.
- **Source:** https://arxiv.org/pdf/2602.00760  — read? ✗

### A3. Self-Compression of Chain-of-Thought via Multi-Agent RL
- **What:** RL setup that compresses CoT to shorter *terminating* traces.
- **Borrow:** target-length / compression signal; compare against A1's intermediate-optimum finding.
- **Source:** https://arxiv.org/pdf/2601.21919  — read? ✗
- Prior-art also named in this space (from A1's related work), read if we go the pruning route:
  O1-Pruner, ThinkPrune, Kimi k1.5 length penalty.

---

## B. The GRPO algorithm — vanilla GRPO collapses on a ≤1B model

We are on stock GRPO (`grpo_train.py`). At 0.8–1B, within-group rollouts homogenize →
advantage/entropy collapse. Fixes native to a `GRPOConfig`-style trainer:

### B1. Advantage Collapse in GRPO: Diagnosis and Mitigation  ★
- **What:** diagnoses GRPO's advantage/entropy collapse (within-group homogenization → std→0 →
  degenerate advantages) and proposes a mitigation. Most on-target for a small-model GRPO run.
- **Borrow:** the diagnostic (watch group-advantage variance over steps) + the fix.
- **Source:** https://arxiv.org/html/2605.21125v1  — read? ✗

### B2. STARE — Surprisal-Guided Token-Level Advantage Reweighting for Policy Entropy Stability
- **What:** reweights token advantages by surprisal to hold policy entropy up — a GRPO drop-in
  against premature convergence / mode collapse.
- **Borrow:** entropy-stability reweighting; pairs with B1.
- **Source:** https://arxiv.org/html/2606.19236  — read? ✗

### B3. Comparative Analysis & Parametric Tuning of PPO, GRPO, and DAPO  ★ practical
- **What:** head-to-head + hyperparameter sweep for LLM-reasoning RL.
- **Borrow:** use it to set the FIRST GRPO config (clip, KL, group size, lr) from measured tuning
  instead of guessing before a paid run.
- **Source:** https://arxiv.org/html/2512.07611v1  — read? ✗
- Pre-cutoff foundations these build on (already known, skim only): DAPO (clip-higher, dynamic
  sampling, soft-overlong) https://arxiv.org/html/2503.14476v2 ; Dr. GRPO (kill length-norm +
  std-difficulty bias) https://arxiv.org/html/2503.20783v2 ; GSPO (sequence-level ratios, Qwen team).

---

## C. Go/no-go sanity check — is a "win" real capability or reranking?  (read BEFORE the GPU run)

### C1. The Debate on RLVR Reasoning Capability Boundary: Shrinkage, Expansion, or Both?
- **What:** two-stage dynamic view of whether RLVR expands the base model's reasoning set or just
  reranks what it already samples (the 2026 update to the pass@k debate).
- **Borrow:** decides our headline metric. Because the eval is *our GRPO vs official Qwen
  post-training on the same base*, a pass@1 gain could be pure reranking. Fix pass@1-vs-pass@k
  reporting from this **before** spending budget.
- **Source:** https://arxiv.org/html/2510.04028v1  — read? ✗
- Companion: Limits of Generalization in RLVR (math case studies) https://arxiv.org/pdf/2510.27044 — read? ✗

---

## Read order (if only three)
1. **2602.09591** (A1) — length/termination, tested at our size.
2. **2605.21125** (B1) — why our GRPO traces collapse.
3. **2512.07611** (B3) — how to set the config for the first run.

## Provenance
Surfaced via web search on 2026-07-09. arXiv IDs are from search results; only 2602.09591's abstract
was fetched directly. Verify each PDF before citing any number or mechanism.
