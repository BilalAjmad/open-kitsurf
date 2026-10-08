import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import numpy as np
import torch

from smallm.train import uniform_soup, greedy_soup, save_soup
from smallm.data import TokenCorpus, build_agentic_sft_pairs, build_agentic_corpus
from smallm.tokenizer import BPETokenizer


def test_uniform_soup_averages():
    a = {"w": torch.zeros(4), "n": torch.tensor([5])}
    b = {"w": torch.full((4,), 2.0), "n": torch.tensor([7])}
    s = uniform_soup([a, b])
    assert torch.allclose(s["w"], torch.ones(4))
    assert s["n"].item() == 5  # integer buffer taken from first


def test_greedy_soup_selects_lower_loss():
    def sd(v):
        return {"w": torch.full((4,), float(v))}
    # loss is distance from target 0.5 -> soup of 0 and 1 is best
    def ev_loss(state):
        return float((state["w"].mean() - 0.5).abs())
    soup, chosen = greedy_soup([sd(0), sd(1), sd(9)], ev_loss)
    assert 0 in chosen and 1 in chosen
    assert abs(float(soup["w"].mean()) - 0.5) < 1e-6


def test_save_soup_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "soup.pt")
        save_soup({"w": torch.ones(3)}, {"x": 1}, p)
        sd = torch.load(p)
        assert sd["config"]["x"] == 1


def test_token_corpus_memmap():
    with tempfile.TemporaryDirectory() as d:
        arr = np.arange(0, 2000, dtype=np.uint16)
        arr.tofile(os.path.join(d, "tokens.bin"))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"n_tokens": len(arr)}, f)
        c = TokenCorpus(os.path.join(d, "tokens.bin"), os.path.join(d, "meta.json"))
        b = c.sample_batch(4, 32)
        assert b["input_ids"].shape == (4, 32)
        assert b["labels"].shape == (4, 32)
        # labels are inputs shifted by one
        assert torch.equal(b["labels"][:, :-1], b["input_ids"][:, 1:])


def test_agentic_sft_pairs_format():
    pairs = build_agentic_sft_pairs(repeats=1)
    assert len(pairs) > 5
    for prompt, resp in pairs:
        assert prompt.endswith("<assistant>")
        assert "<eos>" in resp
    corpus = build_agentic_corpus(repeats=1)
    joined = " ".join(corpus)
    assert " thinking" in joined and "<tool_call>" in joined and "<verify>" in joined
