"""Held-out GSM8K eval for the BASE-model GRPO run (Qwen3-1.7B-Base).

Fairness contract:
- identical prompt template, stop sequences, greedy decoding, precision
  (bf16) and answer extraction for base and trained checkpoints;
- test split only, never touched during prompt/reward design;
- --oneshot gives the base model's best-shot reference line (1-shot
  prompting), so format-anchoring gains aren't sold as pure RL gains.

Run on the GPU box:
  python eval_gsm8k_base.py --n 200                # base, R1-Zero template
  python eval_gsm8k_base.py --n 200 --oneshot      # base, 1-shot reference
  python eval_gsm8k_base.py --n 200 --adapter outputs_base/grpo_lora_base
"""
import argparse
import re
import time

from datasets import load_dataset
from unsloth import FastLanguageModel
from vllm import SamplingParams

TEMPLATE = (
    "A conversation between User and Assistant. The User asks a math "
    "question; the Assistant solves it step by step inside <think> </think> "
    "tags and then gives ONLY the final number inside <answer> </answer> "
    "tags.\n"
    "User: {q}\n"
    "Assistant: <think>"
)

ONESHOT = (
    "Solve each math problem step by step. End with 'Answer: <number>'.\n\n"
    "Question: Tom has 3 boxes with 4 apples each. He eats 2 apples. "
    "How many apples are left?\n"
    "Solution: Tom has 3 * 4 = 12 apples. After eating 2, he has "
    "12 - 2 = 10 apples.\n"
    "Answer: 10\n\n"
    "Question: {q}\n"
    "Solution:"
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
    if f != f or f in (float("inf"), float("-inf")):  # nan/inf guard:
        return ""            # base models sometimes emit absurdly long digit runs
    return str(int(f)) if f == int(f) else str(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--oneshot", action="store_true")
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main")["test"].select(range(args.n))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-1.7B-Base",
        max_seq_length=1024,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.55,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    lora_request = model.load_lora(args.adapter) if args.adapter else None

    fmt = ONESHOT if args.oneshot else TEMPLATE
    stops = ["User:", "Question:"]
    prompts = [fmt.format(q=ex["question"]) for ex in ds]
    golds = [ex["answer"].split("####")[-1].strip().replace(",", "")
             for ex in ds]

    sp = SamplingParams(temperature=0.0, max_tokens=512, stop=stops)
    t0 = time.time()
    outputs = model.fast_generate(prompts, sampling_params=sp,
                                   lora_request=lora_request)
    dt = time.time() - t0

    correct = sum(1 for out, gold in zip(outputs, golds)
                  if extract_pred(out.outputs[0].text) == gold)

    acc = correct / len(golds)
    label = (f"LoRA({args.adapter})" if args.adapter
             else ("BASE-1shot" if args.oneshot else "BASE"))
    print(f"RESULT model={label} n={len(golds)} correct={correct} "
          f"acc={acc:.4f} time={dt:.1f}s")


if __name__ == "__main__":
    main()
