#!/usr/bin/env python3
"""SmallM-Nova: run the model with one command. No setup needed.

    python3 run.py                  # menu: pick a model, then chat
    python3 run.py chat             # chat with the base model
    python3 run.py chat --model soup
    python3 run.py agent            # agentic loop (think / tool / verify)
    python3 run.py vision           # test that the model sees images

Everything (torch, source paths) is set up inside this file, so you can run it
straight from the project folder.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# torch and the smallm package live in the project, not in the system Python.
for p in (os.path.join(ROOT, ".pylibs"), os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

MODELS = {
    "base": "checkpoints/nova_ready",        # distilled, coherent English
    "soup": "checkpoints/nova_ready_soup",   # model soup, slightly better
    "agent": "checkpoints/nova_ready_agent", # think / tool / verify
    "vision": "checkpoints/nova_vision",     # sees images
}


def _load(model_dir: str, threads: int = 4):
    import torch
    from smallm.config import ModelConfig
    from smallm.model import build_model
    from smallm.tokenizer import BPETokenizer

    torch.set_num_threads(threads)
    ck = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    cfg = ModelConfig.from_dict(ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BPETokenizer.load(os.path.join(model_dir, "tokenizer.json"))
    return model, tok


def chat(model_name: str, prompt: str | None, max_new: int, temperature: float):
    import torch

    model, tok = _load(MODELS[model_name])
    print(f"[{model_name}] model loaded. Ctrl-D to exit.\n")

    def one(text: str) -> None:
        ids = torch.tensor([tok.encode("<bos>" + text, allow_special=True)])
        out = model.generate(ids, max_new_tokens=max_new, eos_token_id=tok.eos_id,
                             temperature=temperature, repetition_penalty=1.3,
                             no_repeat_ngram_size=3, top_k=50)
        gen = out[0, ids.shape[1]:].tolist()
        print(tok.decode(gen, skip_special=False).replace("<eos>", "").strip(), "\n")

    if prompt:
        one(prompt)
        return
    for line in sys.stdin:
        line = line.strip()
        if line:
            one(line)


def agent(model_name: str, prompt: str | None, max_new: int, temperature: float):
    import tempfile

    from smallm.agent import (AgentLoop, ModelBrain, Verifier, VirtualComputer,
                              default_registry)

    model, tok = _load(MODELS[model_name])
    workdir = tempfile.mkdtemp(prefix="smallm_sandbox_")
    computer = VirtualComputer(workdir)
    loop = AgentLoop(ModelBrain(model, tok, max_new_tokens=max_new,
                                temperature=temperature),
                     default_registry(computer), Verifier())
    print(f"[{model_name}] agent ready (sandbox: {workdir}). Ctrl-D to exit.\n")

    def one(task: str) -> None:
        res = loop.run(task)
        for s in res.steps:
            print(f"[{s.kind}] {s.content[:120]}")
        print(f"--- answer (passed={res.passed}) ---\n{res.answer}\n")

    if prompt:
        one(prompt)
        return
    for line in sys.stdin:
        line = line.strip()
        if line:
            one(line)


def vision(_prompt, _max_new, _temperature):
    # Real image grounding: coloured shapes the model has never seen.
    script = os.path.join(ROOT, "scripts", "eval_vision.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, ".pylibs") + os.pathsep + os.path.join(ROOT, "src")
    os.execve(sys.executable, [sys.executable, script,
                               os.path.join(MODELS["vision"], "model.pt")], env)


# Real off-the-shelf models (GGUF via llama.cpp). These are the 0.5B / 1B
# models from the README; they are the strongest agents in this repo.
REAL = {
    "qwen": "models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    "llama": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "qwen3": "models/Qwen3-0.6B-Q8_0.gguf",
    "gemma3": "models/gemma-3-1b-it-Q4_K_M.gguf",
    "deepseek": "models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
}


def real(name: str, mode: str) -> None:
    """Launch the interactive chat/agent for a real GGUF model."""
    script = os.path.join(ROOT, "scripts", "aichat.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, ".pylibs") + os.pathsep + os.path.join(ROOT, "src")
    os.execve(sys.executable,
              [sys.executable, script, "--gguf", REAL[name], "--mode", mode], env)


def benchmark(which: str) -> None:
    """Run the strict, real benchmark across all models."""
    script = os.path.join(ROOT, "scripts", "benchmark.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, ".pylibs") + os.pathsep + os.path.join(ROOT, "src")
    flag = {"lm": "--lm", "agent": "--agent", "report": "--report"}[which]
    os.execve(sys.executable, [sys.executable, script, flag], env)


def menu():
    print("SmallM - pick what to run:\n")
    print("  --- real models (2026), strongest agents ---")
    print("  1. qwen3    Qwen3-0.6B     chat/agent   (agent 11/11)")
    print("  2. qwen     Qwen2.5-0.5B   chat/agent   (agent 11/11)")
    print("  3. deepseek DeepSeek-R1-1.5B agent       (agent  8/11)")
    print("  4. gemma3   Gemma-3-1B     chat/agent   (agent  5/11)")
    print("  5. llama    Llama-3.2-1B   chat/agent   (agent  2/11)")
    print("  --- our own trained Nova (already in checkpoints/) ---")
    for i, (name, path) in enumerate(MODELS.items(), 6):
        ok = "ready" if os.path.exists(os.path.join(path, "model.pt")) else "MISSING"
        print(f"  {i}. {name:7} chat      ({ok})")
    choice = input("\nChoice [1]: ").strip().lower() or "1"
    if choice in ("1", "qwen3"):
        real("qwen3", "agent" if input("agent mode? [y/N] ").strip().lower() == "y" else "chat")
    elif choice in ("2", "qwen"):
        real("qwen", "agent" if input("agent mode? [y/N] ").strip().lower() == "y" else "chat")
    elif choice in ("3", "deepseek"):
        real("deepseek", "agent")
    elif choice in ("4", "gemma3"):
        real("gemma3", "agent" if input("agent mode? [y/N] ").strip().lower() == "y" else "chat")
    elif choice in ("5", "llama"):
        real("llama", "agent" if input("agent mode? [y/N] ").strip().lower() == "y" else "chat")
    elif choice in ("6", "base"):
        chat("base", None, 120, 0.7)
    elif choice in ("7", "soup"):
        chat("soup", None, 120, 0.7)
    elif choice in ("8", "agent"):
        agent("agent", None, 256, 0.0)
    elif choice in ("9", "vision"):
        vision(None, None, None)
    else:
        print("unknown choice")


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args and not args[0].startswith("-") else "menu"
    rest = args[1:] if args and not args[0].startswith("-") else args

    # Real GGUF models: `run.py qwen`, `run.py llama --agent`.
    if cmd in REAL:
        real(cmd, "agent" if "--agent" in rest else "chat")
        return

    # Strict benchmark: `run.py bench` (report), `run.py bench --lm`, `--agent`.
    if cmd == "bench":
        which = "lm" if "--lm" in rest else "agent" if "--agent" in rest else "report"
        benchmark(which)
        return

    # Each mode has a sensible default model; --model can still override it.
    default_model = {"chat": "base", "agent": "agent", "vision": "vision"}.get(cmd, "base")
    model_name, prompt, max_new, temperature = default_model, None, 120, 0.7
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--model":
            model_name = rest[i + 1]; i += 2
        elif a == "--prompt":
            prompt = rest[i + 1]; i += 2
        elif a == "--max-new":
            max_new = int(rest[i + 1]); i += 2
        elif a == "--temperature":
            temperature = float(rest[i + 1]); i += 2
        else:
            i += 1

    if cmd == "chat":
        chat(model_name, prompt, max_new, temperature)
    elif cmd == "agent":
        agent(model_name, prompt, max_new, temperature)
    elif cmd == "vision":
        vision(None, None, None)
    else:
        menu()


if __name__ == "__main__":
    main()
