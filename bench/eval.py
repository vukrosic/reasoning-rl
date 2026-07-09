"""Evaluate a small reasoning model on GSM8K / MATH-500 with vLLM.

Run on the GPU box, AFTER download_models.py:

    # both targets, both benchmarks, 200 problems each
    python eval.py --model qwen3.5-0.8b --bench all --n 200
    python eval.py --model minicpm5-1b  --bench all --n 200

    # our own trained checkpoint, scored on the same board
    python eval.py --model ./outputs/my_run --bench all --n 200

Prints one machine-greppable RESULT line per (model, bench) and writes a
JSON with the full config + per-item records to results/.

Design choices that keep the number honest:
  - held-out TEST splits only
  - thinking trace stripped before the answer is extracted
  - sampling params default to each model's card-recommended THINKING values
    (temperature > 0), so pass '--n-samples k' to average pass@1 over k draws;
    override with --greedy for a deterministic single-shot sanity number
  - exact config (model path, sampling, thinking on/off) is saved next to the
    score so a run is never a mystery later
"""
import argparse
import json
import os
import time

from vllm import LLM, SamplingParams

from models import resolve
from benchmarks import ANSWER_INSTRUCTION, BENCHMARKS, extract_answer, grade

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def build_sampling(cfg, args, max_tokens):
    if args.greedy:
        base = {"temperature": 0.0}
    else:
        base = dict(cfg["sampling"])
        if args.temperature is not None:
            base["temperature"] = args.temperature
    return SamplingParams(max_tokens=max_tokens, n=args.n_samples,
                          seed=args.seed, **base)


def run_bench(llm, cfg, bench_name, args):
    spec = BENCHMARKS[bench_name]
    items = spec["loader"](args.n)
    kind = spec["kind"]
    max_tokens = args.max_tokens or spec["max_tokens"]
    # --no-think overrides the model's default so we can measure a TERMINATING
    # number even when thinking mode loops (Qwen3.5-0.8B does, 100% of the time).
    think = cfg["enable_thinking"] and not args.no_think

    conversations = [
        [{"role": "user",
          "content": f"{it['question']}\n\n{ANSWER_INSTRUCTION}"}]
        for it in items
    ]

    sp = build_sampling(cfg, args, max_tokens)
    t0 = time.time()
    outputs = llm.chat(
        conversations,
        sampling_params=sp,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": think},
    )
    dt = time.time() - t0

    records, hits, total = [], 0, 0
    trunc, unclosed = 0, 0  # health signals: did the trace run out of budget?
    for it, out in zip(items, outputs):
        sample_ok = []
        for comp in out.outputs:  # n_samples completions
            pred = extract_answer(comp.text, kind)
            ok = grade(pred, it["gold"], kind)
            sample_ok.append(ok)
            total += 1
            hits += int(ok)
            if comp.finish_reason == "length":
                trunc += 1
            if think and "</think>" not in comp.text:
                unclosed += 1
        c0 = out.outputs[0]
        records.append({
            "question": it["question"][:200],
            "gold": it["gold"],
            "pred": extract_answer(c0.text, kind),
            "correct": sample_ok,
            "finish": c0.finish_reason,
            "gen_tokens": len(c0.token_ids),
            "raw_tail": c0.text[-300:],  # keep the ending for offline debug
        })

    acc = hits / total if total else 0.0
    trunc_rate = trunc / total if total else 0.0
    unclosed_rate = unclosed / total if total else 0.0
    label = cfg["key"]
    print(f"RESULT model={label} bench={bench_name} "
          f"n={len(items)} samples={args.n_samples} "
          f"pass@1={acc:.4f} thinking={think} "
          f"greedy={args.greedy} max_tok={max_tokens} "
          f"trunc={trunc_rate:.2f} unclosed_think={unclosed_rate:.2f} "
          f"time={dt:.1f}s")
    if trunc_rate > 0.05 or unclosed_rate > 0.05:
        print(f"WARN model={label} bench={bench_name}: "
              f"{trunc_rate:.0%} truncated / {unclosed_rate:.0%} unclosed "
              f"thinking -> raise --max-tokens; scores are UNDER-counted")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = label.replace("/", "_").strip("._")
    mode = "think" if think else "nothink"
    path = os.path.join(RESULTS_DIR, f"{safe}__{bench_name}__{mode}.json")
    with open(path, "w") as f:
        json.dump({
            "model": label,
            "hf_id": cfg["hf_id"],
            "bench": bench_name,
            "n": len(items),
            "n_samples": args.n_samples,
            "greedy": args.greedy,
            "enable_thinking": think,
            "sampling": ("greedy" if args.greedy else cfg["sampling"]),
            "max_tokens": max_tokens,
            "pass@1": acc,
            "trunc_rate": round(trunc_rate, 4),
            "unclosed_think_rate": round(unclosed_rate, 4),
            "seconds": round(dt, 1),
            "records": records,
        }, f, indent=2)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="registry key (qwen3.5-0.8b | minicpm5-1b) or a path")
    ap.add_argument("--bench", default="all",
                    help="gsm8k | math500 | all")
    ap.add_argument("--n", type=int, default=200,
                    help="problems per benchmark (None-like: pass a big number)")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="completions per problem for pass@1 averaging")
    ap.add_argument("--greedy", action="store_true",
                    help="temperature 0, single deterministic pass")
    ap.add_argument("--no-think", action="store_true",
                    help="force thinking mode OFF (get a terminating number "
                         "when the model's thinking mode loops)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the model's recommended temperature")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override per-bench generation budget (reasoning "
                         "traces can need more than the default)")
    ap.add_argument("--seed", type=int, default=0)
    # vLLM engine knobs -- tuned for a single 16GB GPU by default
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    cfg = resolve(args.model)
    # prefer a local snapshot if download_models.py already fetched it
    local = os.path.join(os.environ.get("HF_MODELS_DIR", "./hf_models"),
                         cfg["key"])
    model_path = local if os.path.isdir(local) else cfg["hf_id"]
    print(f"[load] {cfg['key']} from {model_path}")

    llm = LLM(
        model=model_path,
        dtype=args.dtype,
        trust_remote_code=cfg["trust_remote_code"],
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    benches = list(BENCHMARKS) if args.bench == "all" else [args.bench]
    scores = {b: run_bench(llm, cfg, b, args) for b in benches}
    print("\n=== SUMMARY ===")
    for b, s in scores.items():
        print(f"{cfg['key']:16s} {b:10s} pass@1={s:.4f}")


if __name__ == "__main__":
    main()
