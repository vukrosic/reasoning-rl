#!/usr/bin/env bash
# Full scoreboard run on the GPU box. Downloads both targets, evals both on
# GSM8K + MATH-500, leaves JSON + RESULT lines in results/.
#
#   bash run_all.sh          # 200 problems/bench (fast, ~cheap)
#   N=500 bash run_all.sh    # full MATH-500 + 500 GSM8K
set -euo pipefail
cd "$(dirname "$0")"

N="${N:-200}"

# Prefer a ready GPU venv (vast.ai / most CUDA images ship torch at /venv/main).
if [ -x /venv/main/bin/python ]; then
  export PATH=/venv/main/bin:$PATH
fi
PY="$(command -v python3 || command -v python)"
echo "== python: $($PY -c 'import sys;print(sys.executable)') =="

# torch+CUDA must already be present; we never install torch (would clobber it).
if ! $PY -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "ERROR: no CUDA torch found. Run on a GPU box with torch preinstalled." >&2
  exit 1
fi
$PY -c 'import torch;print("== torch",torch.__version__,"cuda",torch.version.cuda,"==")'

echo "== installing deps (torch left untouched) =="
$PY -m pip install -q -r requirements.txt

echo "== downloading target models =="
$PY download_models.py

echo "== eval qwen3.5-0.8b =="
$PY eval.py --model qwen3.5-0.8b --bench all --n "$N"

echo "== done. baseline scoreboard: =="
$PY - <<'PY'
import glob, json
rows = []
for p in sorted(glob.glob("results/*.json")):
    d = json.load(open(p))
    rows.append((d["model"], d["bench"], d["pass@1"]))
print(f'{"model":16s} {"bench":10s} pass@1')
for m, b, s in rows:
    print(f"{m:16s} {b:10s} {s:.4f}")
PY
