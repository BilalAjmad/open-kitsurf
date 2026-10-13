"""Interactive chat with a real small model on your own computer.

Two modes:
    chat  : plain conversation (fast, direct)
    agent : full agentic loop with tools, observation grounding, verification

Usage:
    PYTHONPATH=.pylibs:src python3 scripts/aichat.py
    PYTHONPATH=.pylibs:src python3 scripts/aichat.py --mode agent
    PYTHONPATH=.pylibs:src python3 scripts/aichat.py --gguf models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smallm.agent import (AgentLoop, RealBrain, VirtualComputer, default_registry,
                          load_llama)

DEFAULT_GGUF = "models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

BANNER = """SmallM - a small, real, agentic AI on your own computer.
Type a message. Type 'exit' to quit. Type '/mode' to toggle chat/agent.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--mode", choices=["chat", "agent"], default="chat")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    if not os.path.exists(args.gguf):
        sys.exit(f"missing GGUF: {args.gguf}  (run scripts/fetch_model.py)")

    print(f"loading {args.gguf} ...", flush=True)
    llm = load_llama(args.gguf, n_ctx=args.n_ctx, n_threads=args.threads)
    brain = RealBrain(llm, max_new_tokens=args.max_tokens)
    workdir = tempfile.mkdtemp(prefix="smallm_sandbox_")
    loop = AgentLoop(brain, default_registry(VirtualComputer(workdir)))

    mode = args.mode
    history = []
    print(BANNER + f"[mode: {mode}]\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("exit", "quit"):
            break
        if user == "/mode":
            mode = "agent" if mode == "chat" else "chat"
            print(f"[mode: {mode}]\n")
            continue

        if mode == "agent":
            res = loop.run(user)
            for s in res.steps:
                print(f"  [{s.kind}] {s.content}")
            print(f"smallm> {res.answer}  (passed={res.passed})\n")
        else:
            history.append({"role": "user", "content": user})
            out = llm.create_chat_completion(messages=history,
                                             max_tokens=args.max_tokens,
                                             temperature=0.6)
            reply = out["choices"][0]["message"]["content"].strip()
            history.append({"role": "assistant", "content": reply})
            print(f"smallm> {reply}\n")


if __name__ == "__main__":
    main()
