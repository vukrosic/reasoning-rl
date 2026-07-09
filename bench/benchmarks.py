"""Benchmark loaders + answer extraction + grading.

Two math-reasoning benchmarks, both held-out test splits:
  - gsm8k    grade-school word problems, gold is the number after '####'
  - math500  the standard 500-problem MATH subset, gold is a \\boxed answer

We prompt every model to end with \\boxed{...} so extraction is uniform.
Reasoning models emit a <think>...</think> trace first; we strip it before
extracting the answer so numbers inside the reasoning can't leak a match.

Grading:
  - MATH: uses `math_verify` (symbolic equivalence) when installed, so
    1/2 == 0.5 == 0.50 etc. Falls back to a normalized string/number compare.
  - GSM8K: numeric equality of the extracted final number.
"""
import re

# Standard instruction so both models put the final answer somewhere we can
# find it regardless of their native format.
ANSWER_INSTRUCTION = (
    "Solve the problem. Put ONLY your final answer inside \\boxed{}."
)


# ---------------------------------------------------------------------------
# loaders  -> list of {"question": str, "gold": str}
# ---------------------------------------------------------------------------
def load_gsm8k(n=None):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")["test"]
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return [
        {"question": ex["question"],
         "gold": ex["answer"].split("####")[-1].strip().replace(",", "")}
        for ex in ds
    ]


def load_math500(n=None):
    from datasets import load_dataset
    # HuggingFaceH4/MATH-500 is the canonical 500-problem eval subset.
    ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return [
        {"question": ex["problem"], "gold": str(ex["answer"]).strip()}
        for ex in ds
    ]


BENCHMARKS = {
    "gsm8k": {"loader": load_gsm8k, "kind": "gsm8k", "max_tokens": 3072},
    "math500": {"loader": load_math500, "kind": "math", "max_tokens": 6144},
}


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def strip_thinking(text: str) -> str:
    """Drop everything up to and including the reasoning trace."""
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text


def extract_boxed(text: str):
    """Return the content of the LAST \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, out = 0, []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def extract_last_number(text: str):
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace("$", ""))
    if not nums:
        return None
    s = nums[-1].replace(",", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def extract_answer(raw: str, kind: str):
    body = strip_thinking(raw)
    boxed = extract_boxed(body)
    if kind == "gsm8k":
        # prefer a boxed number, else the last number in the answer body
        if boxed is not None:
            n = extract_last_number(boxed)
            if n is not None:
                return n
        return extract_last_number(body)
    # math: keep the boxed expression verbatim for symbolic grading
    return boxed if boxed is not None else body.strip()


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _HAVE_MATH_VERIFY = True
except Exception:
    _HAVE_MATH_VERIFY = False


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).replace("\\left", "").replace(
        "\\right", "").replace("$", "").strip().rstrip(".")


def grade(pred: str, gold: str, kind: str) -> bool:
    if pred is None:
        return False
    if kind == "gsm8k":
        g = extract_last_number(gold) or gold.strip()
        return _norm(pred) == _norm(g)
    # math: try symbolic equivalence first
    if _HAVE_MATH_VERIFY:
        try:
            return bool(_mv_verify(_mv_parse(gold), _mv_parse(pred)))
        except Exception:
            pass
    if _norm(pred) == _norm(gold):
        return True
    pn, gn = extract_last_number(pred), extract_last_number(gold)
    return pn is not None and pn == gn
