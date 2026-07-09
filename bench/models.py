"""Model registry for the Qwen3.5-0.8B reasoning scoreboard.

APPLES-TO-APPLES DESIGN
-----------------------
We take the SAME public base Qwen post-trained from, run our own GRPO
post-training on it, and compare our version vs Qwen's official version on
identical benchmarks. Only the post-training recipe differs.

  base (we post-train this)   ->   official target (the bar)
  Qwen/Qwen3.5-0.8B-Base      ->   Qwen/Qwen3.5-0.8B

Architecture / tractability (verified 2026-07-09):
  - qwen3.5-0.8b is a Gated-DeltaNet hybrid (3:1 linear-attn : full-attn).
    Unsloth supports it (dedicated 0.8B fine-tune notebook; GRPO on the
    Qwen3.5 family). vLLM >= 0.17 loads it (we run 0.24). So it is trainable.
  - Known Qwen3.5-0.8B quirk (from its own card): in thinking mode it is
    "more prone to entering thinking loops ... may prevent it from terminating
    generation properly." Confirmed on-box: 100% of GSM8K traces hit the token
    cap without closing </think>. Eval caps max_tokens and REPORTS the
    truncation rate; do NOT swap in nicer sampling to hide it. Teaching crisp
    termination is exactly what our GRPO run is for.

Each entry carries what the eval needs: HF repo id, whether to switch on the
reasoning trace, and the card-recommended THINKING sampling params. `is_base`
marks pretrain-only checkpoints (no chat template -> eval few-shot, not chat).
All values lifted from the model card 2026-07-09; re-check before the GPU run.
"""

# Sampling params come straight from the model cards. `extra` maps to
# vllm.SamplingParams kwargs so we can pass through top_k / min_p / penalties
# that the reasoning recipes care about.
# Qwen3.5-0.8B thinking-mode sampling -- EXACT card values (2026-07-09):
#   temperature=1.0 top_p=0.95 top_k=20 min_p=0.0 presence_penalty=1.5
#   repetition_penalty=1.0
# presence_penalty=1.5 is what the card recommends; it does NOT fix the
# documented thinking-loop, so eval caps max_tokens and reports trunc rate.
_QWEN_THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
               "presence_penalty": 1.5, "repetition_penalty": 1.0}
MODELS = {
    # ---- official target (the bar to beat) ----
    "qwen3.5-0.8b": {
        "hf_id": "Qwen/Qwen3.5-0.8B",
        "trust_remote_code": False,
        "enable_thinking": True,
        "is_base": False,
        "compare_base": "qwen3.5-0.8b-base",
        "sampling": _QWEN_THINK,
    },

    # ---- the shared base we post-train (apples-to-apples) ----
    "qwen3.5-0.8b-base": {
        "hf_id": "Qwen/Qwen3.5-0.8B-Base",
        "trust_remote_code": False,
        "enable_thinking": False,   # pretrain-only: no chat template
        "is_base": True,
        "sampling": _QWEN_THINK,
    },
}


def resolve(name_or_path: str) -> dict:
    """Return a config dict for a registry key, or a bare local/HF path.

    Passing a path we don't know about is allowed (e.g. our own trained
    checkpoint) -- it just gets sane thinking defaults so we can score it on
    the same board.
    """
    if name_or_path in MODELS:
        cfg = dict(MODELS[name_or_path])
        cfg["key"] = name_or_path
        return cfg
    return {
        "key": name_or_path,
        "hf_id": name_or_path,
        "trust_remote_code": False,
        "enable_thinking": True,
        "is_base": False,
        "sampling": {"temperature": 0.7, "top_p": 0.95},
    }
