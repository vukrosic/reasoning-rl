"""GRPO on a raw BASE model (Qwen3-1.7B-Base) — R1-Zero-style, no SFT.

Differences from grpo_train.py (the instruct-model run), each probe-driven:
- 16-bit, not 4-bit: probe showed the 1.7B base model degenerates badly
  under the bnb-4bit quant (multilingual token soup); bf16 output is sane.
- R1-Zero-style plain-text template (base model has no chat template),
  assistant turn pre-seeded with "<think>" so the tag rung is reachable.
- Rewards truncate the completion at any hallucinated "User:" turn before
  extraction — base models invent follow-up questions, and grading numbers
  from those would be reward noise.
- LR 2e-5 (~10x the instruct run): LoRA needs ~10x the full-finetune LR to
  match it ("LoRA Without Regret", Thinking Machines 2025). The instruct
  run's 5e-6 barely moved the policy (KL ~ 0.0005).

plain: take a model that only autocompletes, reward it when the final
number is right, and watch it teach itself to answer math cleanly.
"""
import re
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

MAX_SEQ_LENGTH = 1024
LORA_RANK = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.7B-Base",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.55,
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

# R1-Zero-style template. The completion starts INSIDE <think> (pre-seeded),
# so the model's job is: reason, close the tag, emit <answer>N</answer>.
TEMPLATE = (
    "A conversation between User and Assistant. The User asks a math "
    "question; the Assistant solves it step by step inside <think> </think> "
    "tags and then gives ONLY the final number inside <answer> </answer> "
    "tags.\n"
    "User: {q}\n"
    "Assistant: <think>"
)


def truncate_hallucinated_turns(text: str) -> str:
    # base models invent "User: ..." follow-ups; never grade past them
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
    if f != f or f in (float("inf"), float("-inf")):  # nan/inf guard:
        return ""            # base models sometimes emit absurdly long digit runs
    return str(int(f)) if f == int(f) else str(f)


def extract_gsm8k_answer(text: str) -> str:
    return text.split("####")[-1].strip().replace(",", "")


def get_gsm8k_questions(split="train"):
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(lambda x: {
        "prompt": TEMPLATE.format(q=x["question"]),
        "answer": extract_gsm8k_answer(x["answer"]),
    })
    return data


dataset = get_gsm8k_questions()
print(f"dataset size: {len(dataset)}")


_DEBUG_CALLS = [0]


def correctness_reward_func(prompts, completions, answer, **kwargs):
    # rule 4: when reward is flat, print completions — not hyperparameters.
    # Dump the first batches so a zero-reward run is diagnosable from the log.
    if _DEBUG_CALLS[0] < 3:
        _DEBUG_CALLS[0] += 1
        print(f"\n===== DEBUG batch {_DEBUG_CALLS[0]} =====")
        print("completion type:", type(completions[0]))
        for c, a in list(zip(completions, answer))[:2]:
            print(f"--- gold={a} pred={extract_pred(c) if isinstance(c, str) else '?'}")
            print(repr(c)[:500])
    return [2.0 if extract_pred(c) == a else 0.0
            for c, a in zip(completions, answer)]


def tag_reward_func(completions, **kwargs):
    # probe-verified reachable rung: the base model already emits
    # <answer> tags on some rollouts
    return [0.5 if ("</think>" in truncate_hallucinated_turns(c)
                    and re.search(r"<answer>.*?</answer>",
                                  truncate_hallucinated_turns(c), re.DOTALL))
            else 0.0 for c in completions]


def clean_end_reward_func(completions, **kwargs):
    # shaping against the probe's failure mode: rambling past the answer
    # into hallucinated turns / code-token junk
    out = []
    for c in completions:
        t = truncate_hallucinated_turns(c)
        m = re.search(r"</answer>", t)
        out.append(0.5 if m and len(t[m.end():].strip()) <= 10 else 0.0)
    return out


training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=2e-5,
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
    max_prompt_length=320,
    max_completion_length=512,
    max_steps=100,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="/workspace/grpo-unsloth/outputs_base",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward_func, tag_reward_func,
                  clean_end_reward_func],
    args=training_args,
    train_dataset=dataset,
)

for attr in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
    if not hasattr(trainer, attr):
        setattr(trainer, attr, None)

trainer.train()

model.save_lora("/workspace/grpo-unsloth/outputs_base/grpo_lora_base")
print("DONE")
