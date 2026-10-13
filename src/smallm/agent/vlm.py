"""Real vision: a genuine 256M vision-language model on CPU.

The from-scratch vision tower in ``smallm.model.vision`` proves the image path
works on a synthetic shape task. This module gives the agent *real* eyes with a
real model: SmolVLM-256M-Instruct, which answers questions about arbitrary
images. It runs on CPU in a few seconds per answer.

Optional dependency: transformers, torch, pillow. Import is lazy so the rest of
the package works without them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

DEFAULT_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


@dataclass
class VisionAnswer:
    question: str
    answer: str
    seconds: float


class VLMBrain:
    """Wraps SmolVLM for image question answering."""

    def __init__(self, model_id: str = DEFAULT_VLM, max_new_tokens: int = 64):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        self.model.eval()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def ask(self, image, question: str) -> VisionAnswer:
        torch = self._torch
        msgs = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }]
        prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        t = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return VisionAnswer(question, text, time.time() - t)

    def describe(self, image, questions: Optional[List[str]] = None) -> List[VisionAnswer]:
        questions = questions or [
            "Describe this image in one short sentence.",
            "What is the main colour? Answer in one word.",
        ]
        return [self.ask(image, q) for q in questions]
