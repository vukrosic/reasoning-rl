"""GRPO post-training of Qwen2.5-1.5B-Instruct on GSM8K, via Unsloth + TRL.

Reward = format reward (did it wrap reasoning in <reasoning>...</reasoning> and
the final answer in <answer>...</answer>) + correctness reward (does the
extracted answer match GSM8K's ground truth number).
"""
import re
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

MAX_SEQ_LENGTH = 1024
LORA_RANK = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,          # vLLM backend for fast rollouts
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.6,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

SYSTEM_PROMPT = (
    "Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
)


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_last_number(text: str) -> str:
    # Base Qwen2.5-Instruct ignores the tag format cold (verified by probing
    # raw rollouts): it does correct math but ends with prose like
    # "Therefore, the answer is 72 clips." Scoring only inside <answer> tags
    # gives every group all-zero reward -> zero advantage -> no gradient.
    # Grading the last number in the text lets correctness fire from step 1;
    # the tag rewards below stay as pure shaping.
    src = extract_answer(text) or text
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", src.replace("$", ""))
    if not nums:
        return ""
    s = nums[-1].replace(",", "")
    f = float(s)
    return str(int(f)) if f == int(f) else str(f)


def extract_gsm8k_answer(text: str) -> str:
    return text.split("####")[-1].strip().replace(",", "")


def get_gsm8k_questions(split="train"):
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_gsm8k_answer(x["answer"]),
    })
    return data


dataset = get_gsm8k_questions()
print(f"dataset size: {len(dataset)}")


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_last_number(r) for r in responses]
    return [2.0 if e == a else 0.0 for e, a in zip(extracted, answer)]


def format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    # partial credit for using both tag pairs anywhere; this is the ladder
    # rung between "no tags at all" (probe-verified starting state) and the
    # strict exact layout above
    responses = [c[0]["content"] for c in completions]
    return [0.5 if ("<reasoning>" in r and "</reasoning>" in r
                    and "<answer>" in r and "</answer>" in r) else 0.0
            for r in responses]


training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_generations=8,
    max_prompt_length=256,
    # 200 was too tight: GSM8K reasoning + the <reasoning>/<answer> wrapper
    # clipped before the answer tag on most rollouts -> all rewards 0, no
    # learning signal (observed in run 3). 400 lets completions terminate.
    max_completion_length=400,
    max_steps=100,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="/workspace/grpo-unsloth/outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[soft_format_reward_func, format_reward_func, correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
)

# Unsloth's compiled GRPOTrainer assumes trl's multimodal-era __init__ (which
# sets these token-id attrs even for text-only models). With an old trl they
# are missing and the first step crashes on `self.image_token_id`. Harmless
# no-op on trl>=0.24 where they already exist (None for text-only models).
for attr in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
    if not hasattr(trainer, attr):
        setattr(trainer, attr, None)

trainer.train()

model.save_lora("/workspace/grpo-unsloth/outputs/grpo_lora")
print("DONE")
