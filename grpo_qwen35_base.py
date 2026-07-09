"""GRPO on Qwen3.5-0.8B-Base — R1-Zero style, plain TRL + PEFT (no Unsloth).

Why not Unsloth: Qwen3.5-0.8B-Base is a Gated-DeltaNet hybrid
(Qwen3_5GatedDeltaNet); Unsloth's fast path doesn't know that module and
fights the cu130 torch stack. Plain transformers loads it, so we use
transformers + PEFT + TRL GRPO directly.

LoRA targets (from on-box probe 2026-07-09): the model has 24 layers =
18 linear-attention (in_proj_qkv/z/b/a, out_proj) + 6 full-attention
(q/k/v/o_proj), all with MLP gate/up/down_proj. Targeting only q/k/v/o_proj
(the old recipe) would adapt just the 6 full-attn layers and leave 3/4 of the
attention untrained -- so we target every linear leaf except lm_head.

Goal: beat the official Qwen3.5-0.8B baseline on GSM8K (48.0% --no-think).

plain: take the model that only autocompletes, reward it when the final number
is right and it stops cleanly, and let it teach itself to answer math.
"""
import re
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen3.5-0.8B-Base"
MAX_SEQ = 1024
LORA_RANK = 16

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
)
model.config.use_cache = False

# every linear leaf except lm_head -> covers both attention types + MLP
LORA_TARGETS = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

peft_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_RANK, lora_dropout=0.0, bias="none",
    target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
)

# R1-Zero template; completion starts INSIDE <think> (pre-seeded).
TEMPLATE = (
    "A conversation between User and Assistant. The User asks a math "
    "question; the Assistant solves it step by step inside <think> </think> "
    "tags and then gives ONLY the final number inside <answer> </answer> "
    "tags.\n"
    "User: {q}\n"
    "Assistant: <think>"
)


def truncate_hallucinated_turns(text: str) -> str:
    return text.split("User:")[0]


def extract_pred(text: str) -> str:
    text = truncate_hallucinated_turns(text)
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


def extract_gsm8k_answer(text: str) -> str:
    return text.split("####")[-1].strip().replace(",", "")


def get_gsm8k(split="train"):
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(lambda x: {
        "prompt": TEMPLATE.format(q=x["question"]),
        "answer": extract_gsm8k_answer(x["answer"]),
    })


dataset = get_gsm8k()
print(f"dataset size: {len(dataset)}")

_DBG = [0]


def correctness_reward_func(prompts, completions, answer, **kwargs):
    if _DBG[0] < 3:
        _DBG[0] += 1
        print(f"\n===== DEBUG batch {_DBG[0]} =====")
        for c, a in list(zip(completions, answer))[:2]:
            print(f"--- gold={a} pred={extract_pred(c)}")
            print(repr(c)[:400])
    return [2.0 if extract_pred(c) == a else 0.0
            for c, a in zip(completions, answer)]


def tag_reward_func(completions, **kwargs):
    return [0.5 if ("</think>" in truncate_hallucinated_turns(c)
                    and re.search(r"<answer>.*?</answer>",
                                  truncate_hallucinated_turns(c), re.DOTALL))
            else 0.0 for c in completions]


def clean_end_reward_func(completions, **kwargs):
    out = []
    for c in completions:
        t = truncate_hallucinated_turns(c)
        m = re.search(r"</answer>", t)
        out.append(0.5 if m and len(t[m.end():].strip()) <= 10 else 0.0)
    return out


import os
STEPS = int(os.environ.get("STEPS", "300"))

args = GRPOConfig(
    use_vllm=False,               # HF generate: robust on the Gated-DeltaNet arch
    learning_rate=2e-5,
    adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1,
    warmup_ratio=0.1, lr_scheduler_type="cosine", optim="adamw_torch",
    logging_steps=1, bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    per_device_train_batch_size=8, gradient_accumulation_steps=1,
    num_generations=8, max_completion_length=320,
    temperature=1.0,
    max_steps=STEPS, save_steps=STEPS, max_grad_norm=0.1, report_to="none",
    output_dir="/root/reasoning-rl-bench/outputs/grpo_qwen35_base",
    log_completions=False,
)

trainer = GRPOTrainer(
    model=model, processing_class=tokenizer,
    reward_funcs=[correctness_reward_func, tag_reward_func,
                  clean_end_reward_func],
    args=args, train_dataset=dataset, peft_config=peft_config,
)
trainer.train()
trainer.save_model(args.output_dir)  # LoRA adapter
print("SAVED_ADAPTER ->", args.output_dir)

# merge LoRA into the base and save a full model so eval can load it directly
merged_dir = args.output_dir + "_merged"
merged = trainer.model.merge_and_unload()
merged.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print("DONE_MERGED ->", merged_dir)
