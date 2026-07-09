# bench — Qwen3.5-0.8B reasoning scoreboard

Establish the **baseline to beat**, then score our own GRPO-trained model on
the exact same board.

## Apples-to-apples

Take the same public base Qwen post-trained from, run *our* GRPO on it, compare
our version vs Qwen's official version on identical benchmarks. Only the
post-training recipe differs.

| role | HF id | params |
|------|-------|--------|
| base we post-train | `Qwen/Qwen3.5-0.8B-Base` | 0.8B |
| official target (the bar) | `Qwen/Qwen3.5-0.8B` | 0.8B |

Benchmarks: **GSM8K** (test) and **MATH-500** — held-out, `\boxed{}`
extraction, `<think>` trace stripped before grading so reasoning numbers can't
leak a match. Metric = **pass@1**.

## Known issue this project exists to fix

Qwen3.5-0.8B in **thinking mode does not terminate** — its own card warns it is
"more prone to entering thinking loops." On-box, **100% of GSM8K traces hit the
token cap without closing `</think>`**, so thinking-mode pass@1 is an
under-count. The eval reports the truncation rate rather than hiding it, and
supports `--no-think` to get a clean terminating number. Teaching crisp
termination is exactly what the GRPO run targets.

## Clone and run (on a CUDA GPU box, torch preinstalled)

```bash
bash run_all.sh                 # download + eval, 200 problems/bench
N=500 bash run_all.sh           # headline numbers

# single mode / bench
python eval.py --model qwen3.5-0.8b --bench gsm8k --n 200            # thinking on
python eval.py --model qwen3.5-0.8b --bench gsm8k --n 200 --no-think # terminating
python eval.py --model qwen3.5-0.8b --bench all   --n 500

# score our own trained checkpoint on the same board
python eval.py --model ./outputs/my_run --bench all --n 500
```

`run_all.sh` guards for CUDA torch, installs deps **without touching torch**
(so the box's CUDA build isn't clobbered), downloads, evals, prints the board.

## Files

- `models.py` — registry: HF ids, thinking toggle, card sampling
- `benchmarks.py` — GSM8K + MATH-500 loaders, `\boxed{}` extraction, grading
- `download_models.py` — `snapshot_download` into `./hf_models/`
- `eval.py` — vLLM runner → `RESULT` line + JSON in `results/`
- `run_all.sh` — download + full scoreboard
- `requirements.txt` — vllm>=0.17 (needed for the Gated-DeltaNet arch),
  transformers>=5.13; torch is assumed preinstalled

## Targets to beat

**Our GRPO-trained `Qwen3.5-0.8B-Base` must exceed the official model:**

- **GSM8K: > 48.0%** (the reliable bar)
- **MATH-500: > 33.0%** (a floor; raise the token budget for a firmer target)

## Baseline (filled from the on-box run)

Official `Qwen/Qwen3.5-0.8B`, n=100, single sample, vLLM 0.24 on an
RTX 5060 Ti (2026-07-09):

| bench | mode | pass@1 | truncated | note |
|-------|------|--------|-----------|------|
| GSM8K | `--no-think` | **48.0%** | 19% | real capability baseline |
| GSM8K | thinking on | **11.0%** | 100% | thinking-loop: never closes `</think>` — an artifact, not capability |
| MATH-500 | `--no-think` | **33.0%** | 46% | floor — 46% still hit the 6144-tok cap, so under-counted |

Two findings:
1. Thinking mode *lowers* GSM8K (48% → 11%) because the model loops instead of
   terminating. Teaching it to reason **and** stop is what GRPO targets.
2. Even without thinking, long outputs truncate (19% GSM8K, 46% MATH-500), so
   these are **floors**. Raise `--max-tokens` / `--max-model-len` for a tighter
   number before treating MATH-500 as the final bar.

_Reproduce:_ `bash run_all.sh` (add `--no-think` per the commands above).
