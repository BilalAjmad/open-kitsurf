"""Download real small-model weights (GGUF) for the agentic loop.

These are real, ready-to-use 0.5B and 1B instruct models. They run on CPU
through llama.cpp. Nothing is trained here; nothing is faked.

    PYTHONPATH=.pylibs python3 scripts/fetch_model.py
"""
from __future__ import annotations

import os

from huggingface_hub import hf_hub_download

JOBS = [
    ("bartowski/Qwen2.5-0.5B-Instruct-GGUF", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"),
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
]


def main() -> None:
    os.makedirs("models", exist_ok=True)
    for repo, fn in JOBS:
        try:
            path = hf_hub_download(repo, fn, local_dir="models")
            print(f"OK   {path}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {repo}/{fn}: {e}")


if __name__ == "__main__":
    main()
