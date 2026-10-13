"""Strict, real benchmark across every model in this repo.

Two independent measurements, no faked numbers:

1. LM quality (`--lm`): bits-per-byte on fixed evaluation text. BPB is
   tokenizer-independent, so a 4096-vocab Nova and a 151k-vocab Qwen are
   compared fairly. Lower is better.

2. Agentic ability (`--agent --model NAME`): the same task suite is run
   through the agent loop. Every answer is graded twice:
     - `passed`  : the loop's own self-verifier
     - `correct` : an INDEPENDENT ground-truth check done here
   A task where `passed` is True but `correct` is False is a false pass, and
   is reported as such. That is the honest part.

Usage:
    python3 scripts/benchmark.py --lm
    python3 scripts/benchmark.py --agent --model qwen-0.5b
    python3 scripts/benchmark.py --report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BENCH_DIR = "benchmarks"

TORCH_MODELS = {
    "teacher-11m": "checkpoints/smallm_11m",
    "nova-base": "checkpoints/nova_ready",
    "nova-soup": "checkpoints/nova_ready_soup",
}
GGUF_MODELS = {
    "qwen2.5-0.5b": "models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    "llama3.2-1b": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "qwen3-0.6b": "models/Qwen3-0.6B-Q8_0.gguf",
    "gemma3-1b": "models/gemma-3-1b-it-Q4_K_M.gguf",
    "deepseek-r1-1.5b": "models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
}

# ---- evaluation text (fixed, so every model sees the same bytes) -----------
EVAL_TEXT = (
    "Once upon a time there was a little girl named Lily. She lived in a small "
    "house near a big green forest. Every morning she walked to the forest to "
    "pick flowers for her mother. One day she found a small bird with a broken "
    "wing. Lily picked up the bird and carried it home. She made a soft bed for "
    "it and gave it water and seeds. After many days the bird was strong again. "
    "Lily opened the window and the bird flew away. The next spring the bird "
    "came back with a bright song for Lily. She smiled and knew that kindness "
    "always returns. "
)

# ---- agentic suite with independent ground truth ---------------------------
CALC_TASKS = [
    ("Compute 13+12.", 25),
    ("Compute 7*6.", 42),
    ("Compute 4839*27.", 130653),
    ("Compute 100000-1.", 99999),
    ("Compute 98765/5.", 19753),
    ("Compute 2+2.", 4),
    ("Compute 15*4.", 60),
    ("Compute 100-37.", 63),
    ("Compute 81/9.", 9),
    ("Compute 12+30.", 42),
]
FILE_TASKS = [
    ("Create note.txt with the content hello. Then read it back.", "note.txt", "hello"),
]


def _last_int(text: str):
    import re

    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# 1. LM quality: bits per byte
# ---------------------------------------------------------------------------
def lm_bpb_torch(model_dir: str) -> dict:
    import torch

    from smallm.config import ModelConfig
    from smallm.model import build_model
    from smallm.tokenizer import BPETokenizer

    torch.set_num_threads(4)
    ck = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    cfg = ModelConfig.from_dict(ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BPETokenizer.load(os.path.join(model_dir, "tokenizer.json"))

    ids = tok.encode(EVAL_TEXT, allow_special=False)
    n_bytes = len(EVAL_TEXT.encode("utf-8"))
    x = torch.tensor([ids[:-1]])
    y = torch.tensor([ids[1:]])
    with torch.no_grad():
        loss = float(model(input_ids=x, labels=y)["loss"])
    return {"ce_nats": loss, "n_tokens": len(ids), "n_bytes": n_bytes,
            "bpb": loss / math.log(2) * (len(ids) / n_bytes)}


def lm_bpb_gguf(gguf_path: str) -> dict:
    import numpy as np
    from llama_cpp import Llama

    llm = Llama(model_path=gguf_path, n_ctx=1024, n_threads=4,
                verbose=False, logits_all=True)
    # Score with the low-level token/eval API. `create_completion(logprobs=...)`
    # mis-reports these GGUFs (it gives CE ~12 for a trivial sentence), so it is
    # not used. This reads the raw logits and computes the true NLL.
    ids = llm.tokenize(EVAL_TEXT.encode("utf-8"), add_bos=True)
    llm.reset()
    llm.eval(ids)
    scores = llm.scores  # (n_tokens, vocab), logits seen when predicting each token
    total, n = 0.0, 0
    for i in range(1, len(ids)):
        logits = scores[i - 1].astype("float64")
        m = logits.max()
        logsumexp = m + np.log(np.exp(logits - m).sum())
        total += logsumexp - logits[ids[i]]
        n += 1
    mean_nll = total / n
    n_bytes = len(EVAL_TEXT.encode("utf-8"))
    return {"ce_nats": mean_nll, "n_tokens": n, "n_bytes": n_bytes,
            "bpb": mean_nll / math.log(2) * (n / n_bytes)}


def run_lm() -> None:
    os.makedirs(BENCH_DIR, exist_ok=True)
    results = {}
    for name, path in TORCH_MODELS.items():
        try:
            results[name] = lm_bpb_torch(path)
            print(f"{name:14} bpb={results[name]['bpb']:.3f}  ce={results[name]['ce_nats']:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:14} FAILED: {e}")
    for name, path in GGUF_MODELS.items():
        try:
            results[name] = lm_bpb_gguf(path)
            print(f"{name:14} bpb={results[name]['bpb']:.3f}  ce={results[name]['ce_nats']:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:14} FAILED: {e}")
    with open(os.path.join(BENCH_DIR, "lm.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", os.path.join(BENCH_DIR, "lm.json"))


# ---------------------------------------------------------------------------
# 2. Agentic ability, graded by the loop AND by independent ground truth
# ---------------------------------------------------------------------------
def run_agent(model_name: str) -> None:
    from smallm.agent import (AgentLoop, ModelBrain, RealBrain, Verifier,
                              VirtualComputer, default_registry, load_llama)

    os.makedirs(BENCH_DIR, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="bench_")

    if model_name in GGUF_MODELS:
        llm = load_llama(GGUF_MODELS[model_name])
        brain = RealBrain(llm, max_new_tokens=400, temperature=0.0)
    elif model_name in ("nova-agent", "nova-base"):
        import torch

        from smallm.config import ModelConfig
        from smallm.model import build_model
        from smallm.tokenizer import BPETokenizer

        folder = "checkpoints/nova_ready_agent" if model_name == "nova-agent" else "checkpoints/nova_ready"
        torch.set_num_threads(4)
        ck = torch.load(os.path.join(folder, "model.pt"), map_location="cpu")
        cfg = ModelConfig.from_dict(ck["config"])
        model = build_model(cfg)
        model.load_state_dict(ck["model"])
        model.eval()
        tok = BPETokenizer.load(os.path.join(folder, "tokenizer.json"))
        brain = ModelBrain(model, tok, max_new_tokens=256, temperature=0.0)
    else:
        raise SystemExit(f"unknown model {model_name}")

    computer = VirtualComputer(workdir)
    loop = AgentLoop(brain, default_registry(computer), Verifier(), max_iterations=8)

    rows = []
    for task, expected in CALC_TASKS:
        res = loop.run(task)
        got = _last_int(res.answer)
        correct = got == expected
        rows.append({"task": task, "expected": expected, "got": got,
                     "passed": res.passed, "correct": correct,
                     "false_pass": bool(res.passed and not correct),
                     "iters": res.iterations})
        print(f"[{'OK ' if correct else 'BAD'}] {task!r} -> {got} "
              f"(passed={res.passed} correct={correct})")

    for task, fname, content in FILE_TASKS:
        res = loop.run(task)
        p = os.path.join(workdir, fname)
        on_disk = os.path.exists(p) and content in open(p).read()
        rows.append({"task": task, "expected": f"{fname}={content}", "got": on_disk,
                     "passed": res.passed, "correct": bool(on_disk),
                     "false_pass": bool(res.passed and not on_disk),
                     "iters": res.iterations})
        print(f"[{'OK ' if on_disk else 'BAD'}] {task!r} -> file_written={on_disk} "
              f"(passed={res.passed})")

    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    false_pass = sum(r["false_pass"] for r in rows)
    summary = {"model": model_name, "n_tasks": n, "correct": correct,
               "accuracy": correct / n, "false_passes": false_pass, "rows": rows}
    out = os.path.join(BENCH_DIR, f"agent_{model_name}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{model_name}: {correct}/{n} correct, {false_pass} false passes")
    print("wrote", out)


# ---------------------------------------------------------------------------
# 3. Report
# ---------------------------------------------------------------------------
def run_report() -> None:
    lines = ["# Benchmark results (strict, real)", ""]

    lm_path = os.path.join(BENCH_DIR, "lm.json")
    if os.path.exists(lm_path):
        lm = json.load(open(lm_path))
        lines += ["## Language-model quality on fixed evaluation text", "",
                  "Lower bits-per-byte is better. BPB is tokenizer-independent, so",
                  "models with different vocabularies are compared fairly.", "",
                  "| Model | bits/byte | CE (nats) | tokens |", "|---|---|---|---|"]
        for name, r in sorted(lm.items(), key=lambda kv: kv[1]["bpb"]):
            lines.append(f"| {name} | {r['bpb']:.3f} | {r['ce_nats']:.3f} | {r['n_tokens']} |")
        lines.append("")

    agent_rows = []
    for fn in sorted(os.listdir(BENCH_DIR)):
        if fn.startswith("agent_") and fn.endswith(".json"):
            agent_rows.append(json.load(open(os.path.join(BENCH_DIR, fn))))
    if agent_rows:
        lines += ["## Agentic ability (same task suite, graded by ground truth)", "",
                  "`correct` is an independent check, not the model's own opinion.",
                  "`false pass` counts answers the loop accepted but that were wrong.", "",
                  "| Model | correct | accuracy | false passes |", "|---|---|---|---|"]
        for r in sorted(agent_rows, key=lambda x: -x["accuracy"]):
            lines.append(f"| {r['model']} | {r['correct']}/{r['n_tasks']} | "
                         f"{r['accuracy']*100:.0f}% | {r['false_passes']} |")
        lines.append("")

    md = "\n".join(lines) + "\n"
    with open(os.path.join(BENCH_DIR, "results.md"), "w") as f:
        f.write(md)
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", action="store_true")
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.lm:
        run_lm()
    elif args.agent:
        run_agent(args.model)
    elif args.report:
        run_report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
