#!/usr/bin/env python3
"""Write the Colab notebook that tests a saved hybrid checkpoint."""

import json
from pathlib import Path

TRAIN = json.loads(Path("src/lib/ncm/ncm_hybrid_colab.ipynb").read_text())
MODEL_SRC = "".join(TRAIN["cells"][3]["source"])
if "class MultiScaleResidualVQ" not in MODEL_SRC:
    raise SystemExit("training notebook cell 3 is not the model cell")

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip() + "\n"))


def code(source: str) -> None:
    CELLS.append(("code", source.strip() + "\n"))


md(
    """
# Test a hybrid next-concept checkpoint

Loads `hybrid.pt` from the training notebook and tests it. Nothing here updates weights.

The checkpoint was trained on text sampled from a teacher such as Qwen2.5-0.5B. This notebook samples a new batch with a different seed and a different prompt list, then asks three questions:

1. Can the residual quantizer reconstruct those tokens?
2. If the coarser codes are right, can the generator predict the next level?
3. Does one teacher repair pass make a free draft more like the teacher, and how many forwards does that cost?

A short training run can fail the quality checks. That is a result, not a notebook error. The repair-rule check should pass even then: it only tests that low-confidence tokens are the ones that change.

Runtime → Change runtime type → **T4 GPU**. The teacher is loaded in fp16. A 0.5B checkpoint stays well under 16 GB. A 1.5B checkpoint fits. 3B and larger are refused.
"""
)

code(
    """
%pip install -q "transformers>=4.46,<5" accelerate sentencepiece protobuf
"""
)

code(
    """
import gc
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_CANDIDATES = [
    Path("/content/ncm-hybrid/hybrid.pt"),
    Path("/content/drive/MyDrive/ncm-hybrid/hybrid.pt"),
    Path("hybrid.pt"),
]
N_TEST = 32
BENCH_BATCH = 4
SEED = 99

TEST_PROMPTS = [
    "The creek was low",
    "Write a field note about",
    "At the market",
    "Three boats left",
    "Under the bridge",
    "The last train",
    "A green bottle",
    "Before the rain",
]

if not torch.cuda.is_available():
    raise SystemExit("No GPU. In Colab: Runtime → Change runtime type → T4 GPU, then rerun.")
props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
print(f"GPU {props.name}  {vram_gb:.1f} GB")
if vram_gb < 12:
    raise SystemExit(f"This test wants a 16 GB T4. This GPU reports {vram_gb:.1f} GB.")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda")
"""
)

CELLS.append(("code", MODEL_SRC if MODEL_SRC.endswith("\n") else MODEL_SRC + "\n"))

md(
    """
## Load the checkpoint

Looks in `/content/ncm-hybrid/hybrid.pt`, then Drive, then the working directory. If none of those exist, the next cell opens a file picker. Mount Drive yourself before running if the file lives there. This cell does not mount anything.
"""
)

code(
    """
checkpoint_path = next((path for path in CHECKPOINT_CANDIDATES if path.exists()), None)
if checkpoint_path is None:
    try:
        from google.colab import files
        print("No hybrid.pt found. Upload the file saved by the training notebook.")
        uploaded = files.upload()
        if not uploaded:
            raise SystemExit("No file uploaded.")
        name = next(iter(uploaded))
        checkpoint_path = Path("/content") / name
    except ImportError:
        raise SystemExit("Put hybrid.pt at /content/ncm-hybrid/hybrid.pt and rerun this cell.")

ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
needed = ["model", "levels", "codebook", "vq_dim", "gen_dim", "gen_layers", "gen_heads", "tau", "vq", "generator", "unigram"]
missing = [key for key in needed if key not in ckpt]
if missing:
    raise SystemExit(f"{checkpoint_path} is missing {missing}. It is not a training-notebook checkpoint.")

model_id = ckpt["model"]
low = model_id.lower()
if any(tag in low for tag in ["7b", "8b", "9b", "14b", "32b", "70b", "72b", "4b", "3b"]):
    raise SystemExit(f"{model_id} does not fit a free 16 GB T4. Retrain the checkpoint from a 0.5B or 1.5B teacher.")

levels = [int(x) for x in ckpt["levels"]]
vocab = int(ckpt["vq"]["embed.weight"].shape[0])
codebook = int(ckpt["codebook"])
if int(ckpt["vq"]["codebooks.0"].shape[0]) != codebook:
    raise SystemExit("codebook size in the file does not match the codebook tensor")

vq = MultiScaleResidualVQ(vocab, int(ckpt["vq_dim"]), codebook, levels).to(device)
gen = NextConcept(vq, int(ckpt["gen_dim"]), int(ckpt["gen_layers"]), int(ckpt["gen_heads"])).to(device)
incompatible = gen.load_state_dict(ckpt["generator"], strict=True)
unigram = ckpt["unigram"].to(device).float()
unigram = unigram / unigram.sum()
vq.eval()
gen.eval()
tau = float(ckpt["tau"])
context = levels[-1]
print(f"loaded {checkpoint_path}")
print(f"teacher id {model_id} | context {context} | levels {len(levels)} | tau {tau}")
print(f"allocated {torch.cuda.memory_allocated()/1024**3:.2f} GB")
del incompatible
"""
)

md(
    """
## Held-out text

These prompts are not the ones the training notebook used, and the seed is not the training seed. The concept model itself is not conditioned on the prompt. The prompt only steers the fresh teacher text used as the reconstruction target, and the teacher continuations in the speed table.
"""
)

code(
    """
try:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
if int(tokenizer.vocab_size) != vocab:
    raise SystemExit(f"tokenizer vocab {tokenizer.vocab_size} does not match checkpoint vocab {vocab}")

teacher = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
teacher.to(device)
teacher.eval()
print(f"teacher resident {torch.cuda.memory_allocated()/1024**3:.2f} GB")


def take_context(out, enc):
    width = enc["input_ids"].shape[1]
    rows = []
    for i in range(out.shape[0]):
        start = width - int(enc["attention_mask"][i].sum())
        content = out[i, start : start + context]
        if content.shape[0] < context:
            content = torch.nn.functional.pad(content, (0, context - content.shape[0]), value=tokenizer.pad_token_id)
        rows.append(content[:context].to(torch.long))
    return torch.stack(rows)


@torch.no_grad()
def sample_teacher_text(n, prompts, sample):
    rows = []
    cursor = 0
    while len(rows) < n:
        batch_prompts = [prompts[(cursor + i) % len(prompts)] for i in range(min(4, n - len(rows)))]
        cursor += len(batch_prompts)
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        out = teacher.generate(
            **enc,
            max_new_tokens=context,
            do_sample=sample,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
        rows.extend(list(take_context(out, enc)))
    return torch.stack(rows[:n])
"""
)

code(
    """
@torch.no_grad()
def reconstruction_accuracy(tokens):
    correct = 0
    total = 0
    for start in range(0, tokens.shape[0], 4):
        batch = tokens[start : start + 4]
        pred = vq.logits_from_codes(vq.encode(batch)).argmax(-1)
        correct += (pred == batch).sum().item()
        total += batch.numel()
    return correct / max(1, total)


@torch.no_grad()
def concept_accuracy(tokens):
    per_level = []
    correct = 0
    total = 0
    for start in range(0, tokens.shape[0], 4):
        batch = tokens[start : start + 4]
        codes = vq.encode(batch)
        for li in range(1, len(levels)):
            pred = gen.forward_level(codes, li).argmax(-1)
            hit = (pred == codes[li]).sum().item()
            if len(per_level) < li:
                per_level.append([0, 0])
            per_level[li - 1][0] += hit
            per_level[li - 1][1] += codes[li].numel()
            correct += hit
            total += codes[li].numel()
    return correct / max(1, total), [hit / count for hit, count in per_level]


@torch.no_grad()
def cascade_accuracy(tokens):
    correct = 0
    total = 0
    for start in range(0, tokens.shape[0], 4):
        batch = tokens[start : start + 4]
        true = vq.encode(batch)
        codes = [true[0]]
        for li in range(1, len(levels)):
            codes.append(gen.forward_level(codes, li).argmax(-1))
        pred = vq.logits_from_codes(codes).argmax(-1)
        correct += (pred == batch).sum().item()
        total += batch.numel()
    return correct / max(1, total)


def repair_rule_holds(draft, hybrid, teacher_model, tau_value):
    logits = teacher_model(draft).logits
    pred = logits.argmax(-1)
    conf = logits.softmax(-1)[:, :-1].gather(-1, draft[:, 1:].unsqueeze(-1)).squeeze(-1)
    replace = conf < tau_value
    kept = hybrid[:, 1:][~replace]
    kept_draft = draft[:, 1:][~replace]
    changed = hybrid[:, 1:][replace]
    changed_teacher = pred[:, :-1][replace]
    same_prefix = torch.equal(hybrid[:, :1], draft[:, :1])
    kept_ok = kept.numel() == 0 or torch.equal(kept, kept_draft)
    changed_ok = changed.numel() == 0 or torch.equal(changed, changed_teacher)
    return same_prefix and kept_ok and changed_ok, replace.float().mean().item()
"""
)

md(
    """
## Run the tests

Reconstruction and concept accuracy use the fresh teacher sample. The speed table uses a greedy teacher continuation, an unconditional concept draft, and that draft after one repair pass. Negative log-likelihood is the teacher's own score of the finished text. Lower is closer to the teacher. It is not a human quality rating.
"""
)

code(
    """
held = sample_teacher_text(N_TEST, TEST_PROMPTS, sample=True)
print("held-out", tuple(held.shape))
print(tokenizer.decode(held[0], skip_special_tokens=True)[:300])

recon = reconstruction_accuracy(held)
concept, per_level = concept_accuracy(held)
cascade = cascade_accuracy(held)
chance = 1.0 / vocab
print(f"tokenizer reconstruction {recon:.3%}   chance {chance:.4%}")
print(f"next-level code accuracy {concept:.3%}   (teacher-forced coarser codes)")
print(f"cascade from the coarsest code {cascade:.3%}")
print("per level:", " ".join(f"L{i+1}:{acc:.2f}" for i, acc in enumerate(per_level)))

with torch.no_grad():
    teacher_tokens, teacher_ms, teacher_peak = cuda_ms(lambda: sample_teacher_text(BENCH_BATCH, TEST_PROMPTS, sample=False))
    teacher_nll_value = teacher_nll(teacher, teacher_tokens)
    draft, ncm_ms, ncm_peak = cuda_ms(lambda: generate_ncm(gen, vq, unigram, BENCH_BATCH))
    ncm_nll_value = teacher_nll(teacher, draft)

    calls = {"n": 0}
    original_forward = teacher.forward

    def counting_forward(*args, **kwargs):
        calls["n"] += 1
        return original_forward(*args, **kwargs)

    teacher.forward = counting_forward
    try:
        (hybrid, reported_rate), repair_ms, repair_peak = cuda_ms(lambda: hybrid_repair(draft, teacher, tau))
        forward_calls = calls["n"]
        rule_ok, rate = repair_rule_holds(draft, hybrid, teacher, tau)
    finally:
        teacher.forward = original_forward
    hybrid_nll_value = teacher_nll(teacher, hybrid)

rows = [
    ("teacher", teacher_nll_value, context, teacher_ms / BENCH_BATCH, teacher_peak, "greedy continuation"),
    ("next-concept", ncm_nll_value, len(levels), ncm_ms / BENCH_BATCH, ncm_peak, "unconditional draft"),
    ("hybrid", hybrid_nll_value, len(levels) + 1, (ncm_ms + repair_ms) / BENCH_BATCH, max(ncm_peak, repair_peak), f"repair rate {rate:.0%}"),
]
print()
print(f"{'writer':16} {'NLL':>8} {'forwards':>10} {'ms/seq':>10} {'peak GB':>8}  note")
for name, nll, forwards, ms, peak, note in rows:
    print(f"{name:16} {nll:8.3f} {forwards:10d} {ms:10.1f} {peak:8.2f}  {note}")

checks = [
    ("repair leaves token 0 unchanged and only replaces low-confidence tokens", rule_ok),
    ("repair is a single teacher forward", forward_calls == 1),
    ("hybrid uses fewer forwards than the teacher", len(levels) + 1 < context),
    ("hybrid NLL is lower than the raw draft", hybrid_nll_value < ncm_nll_value),
]
print()
failed = 0
for label, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + label)
    failed += not ok
print()
if recon < 0.05:
    print(f"NOTE  tokenizer reconstruction is {recon:.3%}. The quantizer has not learned token identity on this checkpoint.")
else:
    print(f"NOTE  tokenizer reconstruction is {recon:.3%}. Chance is {chance:.4%}.")
if rate > 0.9:
    print(f"NOTE  repair rate is {rate:.0%}. The teacher is rewriting almost the whole draft, so the hybrid is mostly the teacher.")
elif rate < 0.02:
    print(f"NOTE  repair rate is {rate:.0%}. The threshold barely fired. The hybrid is the draft.")
print("failed checks:", failed)

print("\\n--- teacher ---")
print(tokenizer.decode(teacher_tokens[0], skip_special_tokens=True)[:400])
print("\\n--- next-concept draft ---")
print(tokenizer.decode(draft[0], skip_special_tokens=True)[:400])
print("\\n--- hybrid ---")
print(tokenizer.decode(hybrid[0], skip_special_tokens=True)[:400])
"""
)

code(
    """
import matplotlib.pyplot as plt

names = [row[0] for row in rows]
nlls = [row[1] for row in rows]
forwards = [row[2] for row in rows]
colors = ["#1d1a16", "#9c4221", "#c47a4a"]

fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
axes[0].bar(names, nlls, color=colors)
axes[0].set_ylabel("teacher NLL")
axes[0].set_title("Closer to the teacher")
axes[1].bar(names, forwards, color=colors)
axes[1].set_ylabel("forwards")
axes[1].set_title("Work per sequence")
axes[2].plot(range(1, len(per_level) + 1), per_level, color="#9c4221", marker="o")
axes[2].set_ylim(0, 1)
axes[2].set_xlabel("level")
axes[2].set_ylabel("code accuracy")
axes[2].set_title("Teacher-forced codes")
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
plt.show()
"""
)

md(
    """
## How to read a failure

The repair-rule check and the forward count are tests of the hybrid, not of training quality. They should pass on any checkpoint this training notebook wrote.

The NLL check fails when the repair did not pull the draft toward the teacher. A repair rate near 100% means the threshold rejected the draft and the hybrid number is mostly one teacher rewrite of a bad sequence. A reconstruction near chance means the quantizer did not learn. Train longer, or lower `TAU` in the training notebook, before treating the quality numbers as a comparison.
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
    },
    "cells": [],
}
for kind, source in CELLS:
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    nb["cells"].append(cell)

path = Path("src/lib/ncm/ncm_hybrid_test.ipynb")
path.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {path} ({path.stat().st_size} bytes, {len(nb['cells'])} cells)")
