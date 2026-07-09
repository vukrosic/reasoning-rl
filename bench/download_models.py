"""Download models into ./hf_models/ (run on the GPU box, later).

Nothing is fetched at prepare-time. On the box:
    python download_models.py                       # official target (the bar)
    python download_models.py qwen3.5-0.8b-base     # the base we post-train
    python download_models.py --all                 # everything in the registry
The eval loads straight from the local snapshot dir it prints.
"""
import argparse
import os

from huggingface_hub import snapshot_download

from models import MODELS

DEST_ROOT = os.environ.get("HF_MODELS_DIR", "./hf_models")

# default = just the official target, so the first baseline run stays cheap
DEFAULT_KEYS = ["qwen3.5-0.8b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", choices=list(MODELS),
                    help="registry keys to fetch (default: the 2 targets)")
    ap.add_argument("--all", action="store_true",
                    help="download every model in the registry")
    args = ap.parse_args()

    keys = list(MODELS) if args.all else (args.keys or DEFAULT_KEYS)
    for key in keys:
        hf_id = MODELS[key]["hf_id"]
        dest = os.path.join(DEST_ROOT, key)
        print(f"[download] {hf_id} -> {dest}")
        snapshot_download(
            repo_id=hf_id,
            local_dir=dest,
            # weights + config + tokenizer only; skip duplicate formats
            ignore_patterns=["*.pth", "*.onnx", "original/*"],
        )
        print(f"[done] {key} at {dest}")


if __name__ == "__main__":
    main()
