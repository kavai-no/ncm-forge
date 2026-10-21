#!/usr/bin/env python3
"""Write the Colab notebook. Run from the repo root."""

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip() + "\n"))


def code(source: str) -> None:
    CELLS.append(("code", source.strip() + "\n"))


md(
    """
# Hybrid next-concept model on a free Colab T4

Trains the same hybrid as NCM Forge, sized for **Google Colab's free T4 (16 GB)**.

A next-concept model writes a whole level of discrete codes in one forward. The teacher (a normal Hugging Face model such as Qwen) does two jobs: it supplies the training text and the input embedding, and at inference it rewrites only the draft tokens it does not trust. That repair is **one** teacher forward, not one forward per token.

## What fits

| | weights | fits this notebook |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct | ~1 GB fp16 | yes, the default |
| Qwen2.5-1.5B-Instruct or Qwen3-1.7B | ~3 GB fp16 | yes, change `MODEL_ID` |
| 3B and above | 6 GB+ | no — the free GPU runs out once sampling, the KV cache, and the concept model overlap |

The teacher is loaded in fp16, its embedding table is copied, then it is **deleted** before the concept model trains. It is loaded again only for the benchmark. Peak on a 0.5B run stays under 8 GB. A 1.5B run stays under 12 GB.

Runtime → Change runtime type → **T4 GPU**. Then run the cells in order. Training text is Python functions from CodeSearchNet (real GitHub functions, split by repository), not a list of prompts and not Wikipedia. A default run uses a few thousand windows and a few hundred steps, on the order of 15–25 minutes. It does not see the whole corpus, and it will not learn to write a working program.
"""
)

code(
    """
%pip install -q "transformers>=4.46,<5" accelerate sentencepiece protobuf datasets
"""
)

code(
    """
import gc
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Free Colab T4. Do not point this at a 7B checkpoint.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

CONTEXT = 128
LEVELS = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
CODEBOOK = 2048
VQ_DIM = 256
GEN_DIM = 384
GEN_LAYERS = 6
GEN_HEADS = 6
N_TRAIN = 4096
N_VAL = 256
PREFIX = 32
VQ_STEPS = 500
GEN_STEPS = 700
BATCH = 8
LR = 3e-4
TAU = 0.45
SEED = 7
OUTPUT = Path("/content/ncm-hybrid")

# CodeSearchNet Python. Public, no HF gate. Train / validation / test are
# split by repository. Each row is one function and its docstring.
# The test split is left for the test notebook.
DATASET_ID = "code-search-net/code_search_net"
DATASET_CONFIG = "python"

assert GEN_DIM % GEN_HEADS == 0, "GEN_DIM must divide evenly across heads"
assert LEVELS[-1] == CONTEXT, "the finest level has to be the token length"

if not torch.cuda.is_available():
    raise SystemExit("No GPU. In Colab: Runtime → Change runtime type → T4 GPU, then rerun.")

props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
print(f"GPU {props.name}  {vram_gb:.1f} GB")
if vram_gb < 12:
    raise SystemExit(f"This notebook wants a 16 GB T4. This GPU reports {vram_gb:.1f} GB.")

low = MODEL_ID.lower()
if any(tag in low for tag in ["7b", "8b", "9b", "14b", "32b", "70b", "72b", "4b", "3b"]):
    raise SystemExit(
        f"{MODEL_ID} does not fit a free 16 GB T4 together with sampling and the concept model. "
        "Use Qwen/Qwen2.5-0.5B-Instruct or Qwen/Qwen2.5-1.5B-Instruct."
    )

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda")
print(
    f"teacher {MODEL_ID} | context {CONTEXT} | levels {len(LEVELS)} | "
    f"codebook {CODEBOOK} | vq {VQ_DIM} | generator {GEN_LAYERS}x{GEN_DIM}"
)
print("Budget: unload the teacher before training. 0.5B peak < 8 GB, 1.5B peak < 12 GB.")
"""
)

code(
    """
class MultiScaleResidualVQ(nn.Module):
    \"\"\"Residual vector quantizer. One codebook per level. The finest level is one code per token.\"\"\"

    def __init__(self, vocab: int, dim: int, codebook: int, levels: list[int]):
        super().__init__()
        self.levels = levels
        self.dim = dim
        self.embed = nn.Embedding(vocab, dim)
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(codebook, dim) * (0.02 if i < len(levels) - 1 else 0.2)) for i in range(len(levels))]
        )
        self.dec = nn.Linear(dim, vocab, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            last = self.codebooks[-1]
            n = min(vocab, last.shape[0])
            last[:n].copy_(self.embed.weight[:n])
            self.dec.weight.copy_(self.embed.weight)

    def _bins(self, t: int, length: int) -> list[tuple[int, int]]:
        bins = []
        for i in range(length):
            start = (i * t) // length
            end = min(t, max(start + 1, ((i + 1) * t) // length))
            bins.append((start, end))
        return bins

    def _pool(self, residual: torch.Tensor, length: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        bins = self._bins(residual.shape[1], length)
        chunks = [residual[:, start:end].mean(dim=1) for start, end in bins]
        return torch.stack(chunks, dim=1), bins

    def _upsample(self, codes: torch.Tensor, bins: list[tuple[int, int]], t: int) -> torch.Tensor:
        out = codes.new_zeros(codes.shape[0], t, codes.shape[-1])
        for i, (start, end) in enumerate(bins):
            out[:, start:end] = codes[:, i : i + 1]
        return out

    def quantize(self, pooled: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        book = self.codebooks[level]
        # argmin ||p - c||^2 == argmax p·c - 0.5 ||c||^2
        score = pooled @ book.t() - 0.5 * book.pow(2).sum(-1)
        idx = score.argmax(-1)
        return idx, book[idx]

    def encode(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        residual = self.embed(tokens)
        indices = []
        for li, length in enumerate(self.levels):
            pooled, bins = self._pool(residual, length)
            idx, q = self.quantize(pooled, li)
            residual = residual - self._upsample(q, bins, tokens.shape[1])
            indices.append(idx)
        return indices

    def logits_from_codes(self, indices: list[torch.Tensor]) -> torch.Tensor:
        t = self.levels[-1]
        z = self.embed.weight.new_zeros(indices[0].shape[0], t, self.dim)
        for li, idx in enumerate(indices):
            bins = self._bins(t, self.levels[li])
            z = z + self._upsample(self.codebooks[li][idx], bins, t)
        return self.dec(z)

    def reconstruction_loss(self, tokens: torch.Tensor, keep: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.embed(tokens)
        z = torch.zeros_like(residual)
        commit = tokens.new_zeros(())
        n = keep or len(self.levels)
        for li in range(n):
            pooled, bins = self._pool(residual, self.levels[li])
            _idx, q = self.quantize(pooled, li)
            q_st = pooled + (q - pooled).detach()
            up = self._upsample(q_st, bins, tokens.shape[1])
            residual = residual - up
            z = z + up
            commit = commit + (pooled - q.detach()).pow(2).mean()
        logits = self.dec(z)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))
        acc = (logits.argmax(-1) == tokens).float().mean()
        return ce + 0.25 * (commit / n), acc

    @torch.no_grad()
    def ema_update(self, tokens: torch.Tensor, decay: float = 0.96) -> None:
        # Move each codebook toward the residuals assigned to it. Dead codes get a pooled vector.
        residual = self.embed(tokens)
        for li, length in enumerate(self.levels):
            pooled, bins = self._pool(residual, length)
            idx, q = self.quantize(pooled, li)
            residual = residual - self._upsample(q, bins, tokens.shape[1])
            flat_idx = idx.reshape(-1)
            flat_p = pooled.reshape(-1, self.dim)
            book = self.codebooks[li]
            sums = torch.zeros_like(book)
            counts = torch.zeros(book.shape[0], device=book.device, dtype=book.dtype)
            sums.index_add_(0, flat_idx, flat_p)
            counts.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=book.dtype))
            seen = counts > 0
            book[seen] = decay * book[seen] + (1 - decay) * (sums[seen] / counts[seen].unsqueeze(1))
            dead = (~seen).nonzero(as_tuple=False).flatten()
            if dead.numel() and torch.rand((), device=book.device).item() < 0.25:
                pick = dead[torch.randint(0, dead.numel(), (min(4, dead.numel()),), device=book.device)]
                src = flat_p[torch.randint(0, flat_p.shape[0], (pick.shape[0],), device=book.device)]
                book[pick] = src


class NextConcept(nn.Module):
    \"\"\"Predicts level k from coarser codes. Positions inside a level do not attend to each other.\"\"\"

    def __init__(self, vq: MultiScaleResidualVQ, dim: int, n_layer: int, n_head: int):
        super().__init__()
        self.levels = list(vq.levels)
        self.vq = vq
        codebook = vq.codebooks[0].shape[0]
        self.in_proj = nn.Linear(vq.dim, dim)
        self.pos = nn.ParameterList([nn.Parameter(torch.randn(length, dim) * 0.02) for length in self.levels[1:]])
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(dim, n_head, dim * 4, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
                for _ in range(n_layer)
            ]
        )
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, codebook)

    def forward_level(self, codes: list[torch.Tensor], level: int) -> torch.Tensor:
        parts = [self.in_proj(self.vq.codebooks[li][codes[li]]) for li in range(level)]
        mem = torch.cat(parts, dim=1)
        for block in self.blocks:
            mem = block(mem)
        query = self.pos[level - 1].unsqueeze(0).expand(mem.shape[0], -1, -1)
        scale = 1.0 / math.sqrt(query.shape[-1])
        attn = torch.softmax(self.q(query) @ self.k(mem).transpose(1, 2) * scale, dim=-1)
        return self.head(query + attn @ self.v(mem))


def init_from_teacher(vq: MultiScaleResidualVQ, teacher_embed: torch.Tensor) -> None:
    with torch.no_grad():
        src = teacher_embed.float()
        rows = min(vq.embed.weight.shape[0], src.shape[0])
        if src.shape[1] == vq.dim:
            copied = src[:rows]
        elif src.shape[1] > vq.dim:
            copied = src[:rows, : vq.dim]
        else:
            copied = F.pad(src[:rows], (0, vq.dim - src.shape[1]))
        vq.embed.weight[:rows].copy_(copied)
        n = min(rows, vq.codebooks[-1].shape[0])
        vq.codebooks[-1][:n].copy_(vq.embed.weight[:n])
        vq.dec.weight[:rows].copy_(vq.embed.weight[:rows])


@torch.no_grad()
def generate_ncm(gen: NextConcept, vq: MultiScaleResidualVQ, unigram: torch.Tensor, batch: int) -> torch.Tensor:
    level0 = torch.multinomial(unigram, batch, replacement=True).view(batch, 1)
    codes = [level0]
    for li in range(1, len(vq.levels)):
        codes.append(gen.forward_level(codes, li).argmax(-1))
    return vq.logits_from_codes(codes).argmax(-1)


@torch.no_grad()
def hybrid_repair(draft: torch.Tensor, teacher, tau: float) -> tuple[torch.Tensor, float]:
    \"\"\"One causal teacher forward. Low-confidence draft tokens take the teacher's prediction.

    Logits at position t-1 predict token t, so a replacement does not re-enter this forward.
    Forwards: len(levels) from the concept model, plus this one.
    \"\"\"
    logits = teacher(draft).logits
    pred = logits.argmax(-1)
    probs = logits.softmax(-1)
    conf = probs[:, :-1].gather(-1, draft[:, 1:].unsqueeze(-1)).squeeze(-1)
    replace = conf < tau
    hybrid = draft.clone()
    hybrid[:, 1:] = torch.where(replace, pred[:, :-1], draft[:, 1:])
    return hybrid, replace.float().mean().item()


def teacher_nll(teacher, tokens: torch.Tensor) -> float:
    logits = teacher(tokens).logits
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1)).item()


def cuda_ms(fn):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end), torch.cuda.max_memory_allocated() / 1024**3
"""
)

md(
    """
## CodeSearchNet functions, then the teacher's embedding

Training rows are real Python functions from the CodeSearchNet training split, packed to `CONTEXT` tokens. The validation split is held out. The test split is not touched here. Repositories do not cross those splits.

Wikipedia is the wrong corpus for this. A 128-token window of an article cannot be checked, and a lot of it is disputed or stale. A function can be read. This run still only sees a slice, and a few hundred steps will not produce correct programs. What it can show is whether the quantizer and the concept model fit real code tokens.

The ten short prompts this notebook used to carry were only a way to start the teacher. Every training row was then a continuation of "the harbor" or "the red boat". Those prompts are gone. The teacher is loaded in fp16 long enough to copy its input embedding, then deleted. It does not write the training set.
"""
)

code(
    """
import os

try:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        print("using HF_TOKEN from Colab secrets")
except Exception:
    pass

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def code_windows(split: str, limit: int, seed: int) -> torch.Tensor:
    \"\"\"Non-overlapping token windows from CodeSearchNet functions.

    Streaming, so this stops once `limit` windows are filled. It does not
    download the whole 412k-function training split. Functions are not padded:
    a window can start in the middle of a function.
    \"\"\"
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=split, streaming=True)
    ds = ds.shuffle(buffer_size=2048, seed=seed)
    need = limit * CONTEXT
    ids: list[int] = []
    kept = 0
    for row in ds:
        text = (row.get("whole_func_string") or "").replace("\\r\\n", "\\n").strip()
        if len(text) < 40 or "def " not in text:
            continue
        piece = tokenizer(text + "\\n", add_special_tokens=False)["input_ids"]
        if len(piece) < 16 or len(piece) > 1024:
            continue
        ids.extend(piece)
        kept += 1
        if len(ids) >= need + CONTEXT:
            break
    usable = (len(ids) // CONTEXT) * CONTEXT
    if usable < need:
        raise SystemExit(
            f"{split} produced {usable // CONTEXT} windows of {CONTEXT} tokens, need {limit}."
        )
    windows = torch.tensor(ids[:usable], dtype=torch.long).view(-1, CONTEXT)
    order = torch.randperm(windows.shape[0], generator=torch.Generator().manual_seed(seed))
    chosen = windows[order[:limit]].contiguous()
    print(f"{split}: {kept} functions, {tuple(chosen.shape)}")
    return chosen


train = code_windows("train", N_TRAIN, SEED).to(device)
held = code_windows("validation", N_VAL, SEED + 1).to(device)
print(tokenizer.decode(train[0].tolist(), skip_special_tokens=True)[:500])

teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True)
teacher.to(device)
teacher.eval()
teacher_embed = teacher.get_input_embeddings().weight.detach().float().cpu()
print(f"teacher resident {torch.cuda.memory_allocated()/1024**3:.2f} GB, embed {tuple(teacher_embed.shape)}")
del teacher
gc.collect()
torch.cuda.empty_cache()
print(f"teacher unloaded, allocated {torch.cuda.memory_allocated()/1024**3:.2f} GB")
"""
)

md(
    """
## Train the hierarchy

Stage 1 is the multiscale residual tokenizer. Stage 2 freezes it and trains the next-concept generator. Both stay in fp32. They are small next to the teacher. The first third of tokenizer steps always keep every level, so the fine codes learn token identity before coarse levels are asked to reconstruct on their own.
"""
)

code(
    """
vocab = int(tokenizer.vocab_size)
vq = MultiScaleResidualVQ(vocab, VQ_DIM, CODEBOOK, LEVELS).to(device)
init_from_teacher(vq, teacher_embed)
del teacher_embed
print(f"VQ parameters {sum(p.numel() for p in vq.parameters())/1e6:.1f} M, allocated {torch.cuda.memory_allocated()/1024**3:.2f} GB")

opt = torch.optim.AdamW(vq.parameters(), lr=LR)
vq_history = []
vq.train()
t0 = time.time()
for step in range(VQ_STEPS):
    ix = torch.randint(0, train.shape[0], (BATCH,), device=device)
    if step < VQ_STEPS // 3 or torch.rand((), device=device).item() > 0.3:
        keep = len(LEVELS)
    else:
        keep = int(torch.randint(1, len(LEVELS) + 1, ()).item())
    loss, acc = vq.reconstruction_loss(train[ix], keep)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(vq.parameters(), 1.0)
    opt.step()
    vq.ema_update(train[ix])
    if step % 25 == 0 or step == VQ_STEPS - 1:
        vq_history.append((step, loss.item(), acc.item()))
        print(f"[tokenizer] {step:4d}  loss {loss.item():.3f}  acc {acc.item():.3f}  {time.time()-t0:.0f}s", flush=True)

for p in vq.parameters():
    p.requires_grad_(False)
vq.eval()

with torch.no_grad():
    chunks = [vq.encode(train[i : i + 16])[0].reshape(-1) for i in range(0, train.shape[0], 16)]
    counts = torch.bincount(torch.cat(chunks), minlength=CODEBOOK).float()
unigram = (counts + 0.1) / (counts + 0.1).sum()
unigram = unigram.to(device)

gen = NextConcept(vq, GEN_DIM, GEN_LAYERS, GEN_HEADS).to(device)
gopt = torch.optim.AdamW((p for p in gen.parameters() if p.requires_grad), lr=LR)
gen_history = []
gen.train()
t1 = time.time()
for step in range(GEN_STEPS):
    ix = torch.randint(0, train.shape[0], (BATCH,), device=device)
    with torch.no_grad():
        codes = vq.encode(train[ix])
    loss = torch.zeros((), device=device)
    correct = 0
    total = 0
    for li in range(1, len(LEVELS)):
        cond = [c.clone() for c in codes]
        if torch.rand((), device=device).item() < 0.5:
            mask = torch.rand_like(cond[li - 1].float()) < 0.1
            noise = torch.randint(0, CODEBOOK, cond[li - 1].shape, device=device)
            cond[li - 1] = torch.where(mask, noise, cond[li - 1])
        logits = gen.forward_level(cond, li)
        target = codes[li]
        loss = loss + F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        correct += (logits.argmax(-1) == target).sum().item()
        total += target.numel()
    loss = loss / (len(LEVELS) - 1)
    gopt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_((p for p in gen.parameters() if p.requires_grad), 1.0)
    gopt.step()
    if step % 25 == 0 or step == GEN_STEPS - 1:
        acc = correct / max(1, total)
        gen_history.append((step, loss.item(), acc))
        print(f"[concepts]  {step:4d}  loss {loss.item():.3f}  acc {acc:.3f}  {time.time()-t1:.0f}s", flush=True)

gen.eval()
with torch.no_grad():
    recon = (vq.logits_from_codes(vq.encode(held)).argmax(-1) == held).float().mean().item()
print(f"held-out tokenizer reconstruction {recon:.3f}")
print(f"allocated after training {torch.cuda.memory_allocated()/1024**3:.2f} GB")
"""
)

md(
    """
## Benchmark the hybrid

Three writers, same length:

- **Teacher** continues the first `PREFIX` tokens of a validation window, which is the start or middle of a real function. One forward per new token.
- **Next-concept** drafts a full window from the coarsest code. One forward per level. It is not shown that prefix.
- **Hybrid** is that draft plus one teacher forward, which replaces tokens the teacher assigns probability below `TAU`.

Quality is the teacher's negative log-likelihood of each writer (lower is closer to the teacher). Repair rate is the fraction of tokens the teacher actually changed. Read the decoded text. A short run will not match the teacher, and it will not be a working program. The table is the gap, the step count, and the memory.
"""
)

code(
    """
teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True)
teacher.to(device)
teacher.eval()
bench_batch = 4


def teacher_continuations():
    prefixes = held[:bench_batch, :PREFIX]
    attention = torch.ones_like(prefixes)
    out = teacher.generate(
        input_ids=prefixes,
        attention_mask=attention,
        max_new_tokens=CONTEXT,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    cont = out[:, PREFIX:]
    if cont.shape[1] < CONTEXT:
        cont = F.pad(cont, (0, CONTEXT - cont.shape[1]), value=tokenizer.pad_token_id)
    return cont[:, :CONTEXT]


def run_ncm():
    return generate_ncm(gen, vq, unigram, bench_batch)


def run_hybrid(draft: torch.Tensor):
    return hybrid_repair(draft, teacher, TAU)


with torch.no_grad():
    teacher_tokens, teacher_ms, teacher_peak = cuda_ms(teacher_continuations)
    teacher_nll_value = teacher_nll(teacher, teacher_tokens)
    draft, ncm_ms, ncm_peak = cuda_ms(run_ncm)
    ncm_nll_value = teacher_nll(teacher, draft)
    (hybrid, repair_rate), repair_ms, repair_peak = cuda_ms(lambda: run_hybrid(draft))
    hybrid_nll_value = teacher_nll(teacher, hybrid)

rows = [
    ("teacher", teacher_nll_value, CONTEXT, teacher_ms / bench_batch, teacher_peak, "greedy, from a function prefix"),
    ("next-concept", ncm_nll_value, len(LEVELS), ncm_ms / bench_batch, ncm_peak, "draft"),
    ("hybrid", hybrid_nll_value, len(LEVELS) + 1, (ncm_ms + repair_ms) / bench_batch, max(ncm_peak, repair_peak), f"repair rate {repair_rate:.0%}"),
]
print(f"{'writer':16} {'NLL':>8} {'forwards':>10} {'ms/seq':>10} {'peak GB':>8}  note")
for name, nll, forwards, ms, peak, note in rows:
    print(f"{name:16} {nll:8.3f} {forwards:10d} {ms:10.1f} {peak:8.2f}  {note}")

print("\\n--- teacher ---")
print(tokenizer.decode(teacher_tokens[0], skip_special_tokens=True)[:500])
print("\\n--- next-concept draft ---")
print(tokenizer.decode(draft[0], skip_special_tokens=True)[:500])
print("\\n--- hybrid ---")
print(tokenizer.decode(hybrid[0], skip_special_tokens=True)[:500])
"""
)

code(
    """
import matplotlib.pyplot as plt

names = [row[0] for row in rows]
nlls = [row[1] for row in rows]
forwards = [row[2] for row in rows]
colors = ["#1d1a16", "#9c4221", "#c47a4a"]

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
axes[0].bar(names, nlls, color=colors)
axes[0].set_ylabel("teacher NLL (lower is closer)")
axes[0].set_title("Quality under the teacher")
axes[1].bar(names, forwards, color=colors)
axes[1].set_ylabel("forwards per sequence")
axes[1].set_title("Work")
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
plt.show()
"""
)

md(
    """
## Save

Weights land in `/content/ncm-hybrid/hybrid.pt`. Colab wipes `/content` when the runtime dies. The next cell copies that file to Drive if you want to keep it. It asks for Google Drive permission and does nothing until you run it.

Score the checkpoint with `ncm_hybrid_test.ipynb`. That notebook does not train. It reloads this file and tests reconstruction, next-level code accuracy, and the hybrid repair on text this run did not use.
"""
)

code(
    """
OUTPUT.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "levels": LEVELS,
        "codebook": CODEBOOK,
        "vq_dim": VQ_DIM,
        "gen_dim": GEN_DIM,
        "gen_layers": GEN_LAYERS,
        "gen_heads": GEN_HEADS,
        "tau": TAU,
        "vq": vq.state_dict(),
        "generator": gen.state_dict(),
        "unigram": unigram.detach().cpu(),
        "benchmark": [
            {"name": name, "nll": nll, "forwards": forwards, "ms_per_seq": ms, "peak_gb": peak, "note": note}
            for name, nll, forwards, ms, peak, note in rows
        ],
    },
    OUTPUT / "hybrid.pt",
)
print("wrote", OUTPUT / "hybrid.pt")
"""
)

code(
    """
# Optional. Run only if you want the checkpoint after this runtime is gone.
from google.colab import drive

drive.mount("/content/drive")
dest = Path("/content/drive/MyDrive/ncm-hybrid")
dest.mkdir(parents=True, exist_ok=True)
target = dest / "hybrid.pt"
target.write_bytes((OUTPUT / "hybrid.pt").read_bytes())
print("copied to", target)
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
    cell = {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    nb["cells"].append(cell)

path = Path("src/lib/ncm/ncm_hybrid_colab.ipynb")
path.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {path} ({path.stat().st_size} bytes, {len(nb['cells'])} cells)")
