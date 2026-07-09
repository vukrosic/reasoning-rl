"""Print raw completions for 2 GSM8K prompts — see what the model actually emits."""
from unsloth import FastLanguageModel
from datasets import load_dataset
from vllm import SamplingParams

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length=1024, load_in_4bit=True, fast_inference=True,
    max_lora_rank=16, gpu_memory_utilization=0.6,
)
SYSTEM_PROMPT = (
    "Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
)
ds = load_dataset("openai/gsm8k", "main")["train"]
for i in range(2):
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": ds[i]["question"]}],
        tokenize=False, add_generation_prompt=True)
    out = model.fast_generate([text], SamplingParams(temperature=1.0, max_tokens=400, n=2))
    for j, o in enumerate(out[0].outputs):
        print(f"===== PROMPT {i} SAMPLE {j} =====")
        print(repr(o.text[:600]))
print("GT:", ds[0]["answer"].split("####")[-1].strip())
