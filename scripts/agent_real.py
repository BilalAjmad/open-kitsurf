"""Run the full agentic loop with a REAL small model (GGUF) on your computer.

The model is small (0.5B-1B). The loop gives it tools, observation grounding,
and strict-English enforcement, so it behaves like an agent far larger than it
is. Works on CPU.

Usage:
    PYTHONPATH=.pylibs:src python3 scripts/agent_real.py \
        --gguf models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
        --prompt "Compute 13+12."

    PYTHONPATH=.pylibs:src python3 scripts/agent_real.py --eval
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

TASKS = [
    "Compute 13+12.",
    "Compute 7*6.",
    "Compute 4839*27.",
    "Compute 100000-1.",
    "Compute 98765/5.",
    "Create note.txt with the content hello. Then check it.",
]


def run_task(loop, task: str, show: bool = False) -> bool:
    res = loop.run(task)
    tag = "OK " if res.passed else "BAD"
    print(f"[{tag}] {task!r} -> {res.answer!r}  (passed={res.passed}, iters={res.iterations})")
    if show:
        for s in res.steps:
            print(f"  [{s.kind}] {s.content[:200]}")
    return res.passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--max-iterations", type=int, default=8)
    ap.add_argument("--vision", action="store_true",
                    help="add a real `see` tool (loads SmolVLM-256M)")
    args = ap.parse_args()

    if not os.path.exists(args.gguf):
        sys.exit(f"missing GGUF: {args.gguf}  (run scripts/fetch_model.py first)")

    print(f"loading {args.gguf} ...", flush=True)
    llm = load_llama(args.gguf, n_ctx=args.n_ctx, n_threads=args.threads)
    brain = RealBrain(llm)

    vlm = None
    if args.vision:
        from smallm.agent import VLMBrain, VisionAnswer  # noqa: F401
        from smallm.agent.tools import with_vision

        print("loading SmolVLM-256M (vision) ...", flush=True)
        vlm = VLMBrain()

    def fresh_loop():
        workdir = tempfile.mkdtemp(prefix="smallm_sandbox_")
        c = VirtualComputer(workdir)
        reg = default_registry(c)
        if vlm is not None:
            reg = with_vision(c, reg, vlm)
        return AgentLoop(brain, reg, max_iterations=args.max_iterations)

    if args.prompt:
        res = fresh_loop().run(args.prompt)
        for s in res.steps:
            print(f"[{s.kind:<12}] {s.content}")
        print(f"\nanswer: {res.answer}  (passed={res.passed})")
        return

    tasks = TASKS if args.eval else TASKS[:1]
    passed = sum(run_task(fresh_loop(), t) for t in tasks)
    print(f"\n{passed}/{len(tasks)} tasks passed")


if __name__ == "__main__":
    main()
