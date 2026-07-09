"""Probe raw rollouts from Qwen3-1.7B-Base (no chat template) on GSM8K.

v2 after first probe showed near-degenerate output with the 4-bit quant:
- load 16-bit (1.7B bf16 = 3.4 GB, fits; small base models degrade hard in 4-bit)
- stop sequences so it can't ramble into hallucinated "User:" turns
- add a 1-shot format (classic base-model anchor)
"""
from datasets import load_dataset
from unsloth import FastLanguageModel
from vllm import SamplingParams

FMT_B = (
    "A conversation between User and Assistant. The User asks a math "
    "question; the Assistant solves it step by step inside <think> </think> "
    "tags and then gives ONLY the final number inside <answer> </answer> "
    "tags.\n"
    "User: {q}\n"
    "Assistant: <think>"
)

FMT_C = (
    "Solve each math problem step by step. End with 'Answer: <number>'.\n\n"
    "Question: Tom has 3 boxes with 4 apples each. He eats 2 apples. "
    "How many apples are left?\n"
    "Solution: Tom has 3 * 4 = 12 apples. After eating 2, he has "
    "12 - 2 = 10 apples.\n"
    "Answer: 10\n\n"
    "Question: {q}\n"
    "Solution:"
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.7B-Base",
    max_seq_length=1024,
    load_in_4bit=False,          # 16-bit: small base models degrade in 4-bit
    fast_inference=True,
    max_lora_rank=16,
    gpu_memory_utilization=0.6,
)

ds = load_dataset("openai/gsm8k", "main")["train"].select(range(3))

for name, fmt, stops in [
    ("B r1zero", FMT_B, ["User:", "\nA conversation"]),
    ("C 1shot", FMT_C, ["Question:", "\n\nQuestion"]),
]:
    for temp, n in [(0.0, 1), (1.0, 2)]:
        sp = SamplingParams(temperature=temp, max_tokens=512, n=n, stop=stops)
        prompts = [fmt.format(q=ex["question"]) for ex in ds]
        outs = model.fast_generate(prompts, sampling_params=sp)
        for ex, out in zip(ds, outs):
            gold = ex["answer"].split("####")[-1].strip()
            for i, o in enumerate(out.outputs):
                print("=" * 70)
                print(f"FORMAT {name} | temp {temp} | GOLD {gold} | sample {i}")
                print(o.text[:700])
