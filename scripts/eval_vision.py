"""Measure vision grounding: can the model caption unseen images?

    PYTHONPATH=src python3 scripts/eval_vision.py [checkpoint]

Reports accuracy with the real image versus a blank image. The gap is the
proof that the vision tower is doing the work, not the language prior.
"""
import sys, torch
sys.path.insert(0, "src")
from smallm.config import ModelConfig
from smallm.model import build_model
from smallm.tokenizer import BPETokenizer
sys.path.insert(0, "scripts")
from train_vision import make_batch, COLOURS, SHAPES

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/smallm_11m_vision/model.pt"
sd = torch.load(ckpt, map_location="cpu")
cfg = ModelConfig.from_dict(sd["config"])
model = build_model(cfg); model.load_state_dict(sd["model"]); model.eval()
tok = BPETokenizer.load("data/processed/tokenizer.json")
n_img = model.vision.num_visual_tokens

rng = torch.Generator().manual_seed(999)
imgs, caps = make_batch(60, cfg.vision.image_size, rng)

# baseline: caption with a blank image (vision removed)
blank = torch.zeros_like(imgs)
correct = 0
blank_correct = 0
for i in range(len(caps)):
    prompt = [tok.bos_id] + [3] * n_img
    ids = torch.tensor([prompt])
    out = model.generate(ids, images=imgs[i:i+1], max_new_tokens=14,
                         eos_token_id=tok.eos_id, temperature=0.0)
    gen = tok.decode(out[0, len(prompt):].tolist(), skip_special=False).replace("<eos>", "").strip()
    ob = model.generate(ids, images=blank[i:i+1], max_new_tokens=14,
                        eos_token_id=tok.eos_id, temperature=0.0)
    genb = tok.decode(ob[0, len(prompt):].tolist(), skip_special=False).replace("<eos>", "").strip()
    ok = gen == caps[i]
    okb = genb == caps[i]
    correct += ok
    blank_correct += okb
    if i < 12:
        print(f"[{'OK ' if ok else 'BAD'}] want={caps[i]!r} got={gen!r}  (blank={genb!r})")

print(f"\nimage-conditioned accuracy: {correct}/{len(caps)}")
print(f"blank-image baseline:       {blank_correct}/{len(caps)}")
