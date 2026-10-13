"""The full stack on CPU: a real text agent that uses a real vision model.

A 0.5B text model drives the loop; when it needs to look at an image it calls
the `see` tool, which is answered by a real 256M vision-language model.

    PYTHONPATH=.pylibs:src python3 scripts/agent_vision_demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PIL import Image, ImageDraw

from smallm.agent import (AgentLoop, RealBrain, VLMBrain, VirtualComputer,
                          default_registry, load_llama)
from smallm.agent.tools import with_vision

GGUF = "models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"


def make_image(path, colour, shape):
    img = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(img)
    fill = {"red": (220, 30, 30), "green": (30, 180, 60), "blue": (40, 80, 220)}[colour]
    m = 64
    box = [m, m, 256 - m, 256 - m]
    if shape == "circle":
        d.ellipse(box, fill=fill)
    elif shape == "square":
        d.rectangle(box, fill=fill)
    else:
        d.polygon([(128, m), (m, 256 - m), (256 - m, 256 - m)], fill=fill)
    img.save(path)


def main():
    print("loading text model ...", flush=True)
    llm = load_llama(GGUF, n_ctx=4096, n_threads=4)
    brain = RealBrain(llm)

    print("loading vision model ...", flush=True)
    vlm = VLMBrain()

    workdir = tempfile.mkdtemp(prefix="smallm_vision_")
    make_image(os.path.join(workdir, "image.png"), "red", "circle")

    c = VirtualComputer(workdir)
    reg = with_vision(c, default_registry(c), vlm)
    loop = AgentLoop(brain, reg, max_iterations=8)

    task = ("Look at image.png and tell me the colour of the shape. "
            "Use the see tool on image.png.")
    res = loop.run(task)
    for s in res.steps:
        print(f"[{s.kind:<12}] {s.content}")
    print(f"\nanswer: {res.answer}  (passed={res.passed})")
    # The sandbox image is a red circle. The answer is grounded if it names
    # what the vision model actually saw in the pixels.
    a = res.answer.lower()
    print("grounded in pixels:", "circle" in a or "red" in a)


if __name__ == "__main__":
    main()
