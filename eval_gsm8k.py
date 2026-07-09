"""Held-out GSM8K eval: base model vs the GRPO-trained LoRA adapter.

Uses the TEST split (never seen during training) with greedy decoding
(temperature 0) - a real benchmark number, not training-rollout accuracy.

Run on the GPU box:
  python eval_gsm8k.py --n 200                        # base model
  python eval_gsm8k.py --n 200 --adapter outputs/grpo_lora   # trained
"""
import argparse
import re
import time

from datasets import load_dataset
from unsloth import FastLanguageModel
from vllm import SamplingParams

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_last_number(text: str) -> str:
    src = text
    if "<answer>" in text and "</answer>" in text:
        src = text.split("<answer>")[-1].split("</answer>")[0]
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", src.replace("$", ""))
    if not nums:
        return ""
    s = nums[-1].replace(",", "")
    f = float(s)
    return str(int(f)) if f == int(f) else str(f)


def gsm8k_gold(example) -> str:
    return example["answer"].split("####")[-1].strip().replace(",", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--adapter", type=str, default=None,
                     help="path to LoRA adapter; omit to eval base model")
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main")["test"].select(range(args.n))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=1024,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.6,
    )
    # load_lora only exists once the model has LoRA hooks attached -
    # must call get_peft_model with the SAME config used in training,
    # even to just load weights into it for eval.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    lora_request = None
    if args.adapter:
        lora_request = model.load_lora(args.adapter)

    prompts = []
    golds = []
    for ex in ds:
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(text)
        golds.append(gsm8k_gold(ex))

    sp = SamplingParams(temperature=0.0, max_tokens=400)
    t0 = time.time()
    outputs = model.fast_generate(prompts, sampling_params=sp,
                                   lora_request=lora_request)
    dt = time.time() - t0

    correct = 0
    for out, gold in zip(outputs, golds):
        pred = extract_last_number(out.outputs[0].text)
        if pred == gold:
            correct += 1

    acc = correct / len(golds)
    label = f"LoRA({args.adapter})" if args.adapter else "BASE"
    print(f"RESULT model={label} n={len(golds)} correct={correct} "
          f"acc={acc:.4f} time={dt:.1f}s")


if __name__ == "__main__":
    main()
