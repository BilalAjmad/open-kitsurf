"""Ready-to-use CLI for a trained SmallM checkpoint.

Two modes:
  --mode chat   : plain text generation
  --mode agent  : full agentic loop (think, tools, verify) in a sandbox

Usage:
    PYTHONPATH=src python3 scripts/chat.py --model checkpoints/smallm_30m_agent
    PYTHONPATH=src python3 scripts/chat.py --model checkpoints/smallm_30m_agent --mode agent
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from smallm.config import ModelConfig, TextConfig, VisionConfig
from smallm.model import build_model
from smallm.tokenizer import BPETokenizer


def load_model(model_dir: str, threads: int = 4):
    torch.set_num_threads(threads)
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    cfg = ModelConfig.from_dict(sd["config"])
    model = build_model(cfg)
    model.load_state_dict(sd["model"])
    model.eval()
    tok = BPETokenizer.load(os.path.join(model_dir, "tokenizer.json"))
    return model, tok


def chat_once(model, tok, user: str, max_new: int = 200, temperature: float = 0.7,
              instruct: bool = False, repetition_penalty: float = 1.3,
              no_repeat_ngram_size: int = 3, top_k: int = 50) -> str:
    if instruct:
        # use after agentic SFT, which teaches this exact <system>/<user> format
        from smallm.data.agentic import SYSTEM_PROMPT

        prompt = f"<bos><system> {SYSTEM_PROMPT} </system><user> {user} </user><assistant>"
    else:
        # base mode: continue the user's text directly (docs begin with <bos>)
        prompt = "<bos>" + user
    ids = torch.tensor([tok.encode(prompt, allow_special=True)])
    out = model.generate(ids, max_new_tokens=max_new, eos_token_id=tok.eos_id,
                         temperature=temperature, repetition_penalty=repetition_penalty,
                         no_repeat_ngram_size=no_repeat_ngram_size, top_k=top_k)
    gen = out[0, ids.shape[1]:].tolist()
    return tok.decode(gen, skip_special=False).replace("<eos>", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["chat", "agent"], default="chat")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prompt", default=None, help="run one prompt and exit")
    ap.add_argument("--instruct", action="store_true",
                    help="use after agentic SFT (system/user prompt format)")
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    model, tok = load_model(args.model, args.threads)

    if args.mode == "agent":
        from smallm.agent import (AgentLoop, ModelBrain, Verifier, VirtualComputer,
                                  default_registry)
        workdir = tempfile.mkdtemp(prefix="smallm_sandbox_")
        computer = VirtualComputer(workdir)
        registry = default_registry(computer)
        brain = ModelBrain(model, tok, max_new_tokens=args.max_new, temperature=args.temperature)
        loop = AgentLoop(brain, registry, Verifier())

        def respond(text):
            res = loop.run(text)
            print("--- steps ---")
            for s in res.steps:
                print(f"[{s.kind}] {s.content[:120]}")
            print(f"--- answer (passed={res.passed}) ---")
            print(res.answer)

        if args.prompt:
            respond(args.prompt)
            return
        print("SmallM agent ready. Type a task. Ctrl-D to exit.")
        for line in sys.stdin:
            line = line.strip()
            if line:
                respond(line)
        return

    if args.prompt:
        print(chat_once(model, tok, args.prompt, args.max_new, args.temperature, args.instruct))
        return
    print("SmallM ready. Type a message. Ctrl-D to exit.")
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(chat_once(model, tok, line, args.max_new, args.temperature, args.instruct))


if __name__ == "__main__":
    main()
