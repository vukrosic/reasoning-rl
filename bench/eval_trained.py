"""Template-matched GSM8K eval for the R1-Zero GRPO models (plain vLLM).

Our trained model speaks the R1-Zero format (<think>...</think>
<answer>N</answer>), not chat + \\boxed{}, so it is scored in its native
format. Same GSM8K test questions, same pass@1 metric as the bench board, so
"our model vs official Qwen3.5-0.8B (48.0%)" is a fair comparison.

Run on the box (after training frees the GPU):
  python eval_trained.py --model ./hf_models/qwen3.5-0.8b-base --n 200   # before
  python eval_trained.py --model ./outputs/grpo_qwen35_base_merged --n 200  # after
"""
import argparse
import re
import time

from datasets import load_dataset
from vllm import LLM, SamplingParams

TEMPLATE = (
    "A conversation between User and Assistant. The User asks a math "
    "question; the Assistant solves it step by step inside <think> </think> "
    "tags and then gives ONLY the final number inside <answer> </answer> "
    "tags.\n"
    "User: {q}\n"
    "Assistant: <think>"
)


def extract_pred(text: str) -> str:
    text = text.split("User:")[0].split("Question:")[0]
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    src = m.group(1) if m else text
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", src.replace("$", ""))
    if not nums:
        return ""
    s = nums[-1].replace(",", "")
    try:
        f = float(s)
    except (ValueError, OverflowError):
        return ""
    if f != f or f in (float("inf"), float("-inf")):
        return ""
    return str(int(f)) if f == int(f) else str(f)


def gold(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--max-model-len", type=int, default=1024)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main")["test"].select(range(args.n))
    prompts = [TEMPLATE.format(q=ex["question"]) for ex in ds]
    golds = [gold(ex["answer"]) for ex in ds]

    llm = LLM(model=args.model, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        stop=["</answer>", "\nUser:", "User:"],
                        include_stop_str_in_output=True)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0

    hits = 0
    for out, g in zip(outs, golds):
        if extract_pred(out.outputs[0].text) == g:
            hits += 1
    acc = hits / len(golds)
    print(f"RESULT_TRAINED model={args.model} bench=gsm8k n={len(golds)} "
          f"pass@1={acc:.4f} time={dt:.1f}s")


if __name__ == "__main__":
    main()
