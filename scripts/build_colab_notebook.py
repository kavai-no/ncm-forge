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

Runtime → Change runtime type → **T4 GPU**. Then run the cells in order. Training text is Python functions from CodeSearchNet, in Qwen's own vocabulary. A finished tokenizer checkpoint is frozen. The generator is new: the coarsest level predicts one code at a time from the codes before it, and the finer levels stay parallel. It walks 131,072 windows once and stops when the held-out next-code accuracy stops rising. That number is the score. The training accuracy is not.

This is still not the paper's training run (billions of tokens, a ~100M generator). Judge it by the round trip and by whether the continuation contains code, not by teacher NLL. A string of `=` can score better than a real function.
"""
)

code(
    """
%pip install -q "transformers>=5.3,<6" "datasets>=3" "huggingface-hub>=1.32,<2" accelerate sentencepiece protobuf
"""
)

code(
    """
import gc
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Free Colab T4. Do not point this at a 7B checkpoint.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Qwen's vocabulary. The coarsest code covers 8 tokens. A 64-token prefix
# pins half of every level, so the model continues a function instead of
# guessing 96 tokens from three codes.
CONTEXT = 128
LEVELS = [16, 32, 48, 64, 80, 96, 112, 128]
CODEBOOK = 4096
VQ_DIM = 384
GEN_DIM = 512
GEN_LAYERS = 8
GEN_HEADS = 8
DEC_LAYERS = 4
N_TRAIN = 131072
N_VAL = 512
PREFIX = 64
VQ_STEPS = 6000
GEN_BATCH = 16
GEN_STEPS = N_TRAIN // GEN_BATCH
BATCH = 8
LR = 3e-4
TAU = 0.45
COARSE_LOSS_WEIGHT = 4.0
OBJECTIVE = "causal-next-code"
HELD_EVERY = 1000
HELD_PATIENCE = 3
HELD_MIN_STEP = 4000
HELD_MIN_GAIN = 0.005
SEED = 7
LOG_EVERY = 100
OUTPUT = Path("/content/ncm-hybrid")

# CodeSearchNet Python. Public, no HF gate. Train / validation / test are
# split by repository. Each row is one function and its docstring.
# The test split is left for the test notebook.
DATASET_ID = "code-search-net/code_search_net"
DATASET_CONFIG = "python"

assert GEN_DIM % GEN_HEADS == 0, "GEN_DIM must divide evenly across heads"
assert VQ_DIM % GEN_HEADS == 0, "VQ_DIM must divide evenly across heads"
assert LEVELS[-1] == CONTEXT, "the finest level has to be the token length"
assert N_TRAIN % GEN_BATCH == 0, "one pass has to fill whole batches"
assert GEN_STEPS == N_TRAIN // GEN_BATCH

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
    f"teacher {MODEL_ID} | context {CONTEXT} | prefix {PREFIX} | levels {len(LEVELS)} | "
    f"codebook {CODEBOOK} | vq {VQ_DIM} x {DEC_LAYERS} | generator {GEN_LAYERS}x{GEN_DIM}"
)
print("Budget: unload the teacher before training. 0.5B peak < 8 GB, 1.5B peak < 12 GB.")
"""
)

code(
    """
class MultiScaleResidualVQ(nn.Module):
    \"\"\"Residual vector quantizer. One codebook per level. A small transformer, not a linear map, reads the reconstruction back into tokens.\"\"\"

    def __init__(self, vocab: int, dim: int, codebook: int, levels: list[int], n_dec: int = 2, n_head: int = 6):
        super().__init__()
        self.levels = levels
        self.dim = dim
        self.embed = nn.Embedding(vocab, dim)
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(codebook, dim) * 0.02) for _ in range(len(levels))]
        )
        self.dec_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(dim, n_head, dim * 4, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
                for _ in range(n_dec)
            ]
        )
        self.dec_norm = nn.LayerNorm(dim)
        self.dec_head = nn.Linear(dim, vocab, bias=False)
        self.dec_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

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
        indices, _before = self.contexts(tokens)
        return indices

    def contexts(self, tokens: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        \"\"\"Codes for every level, and the cumulative reconstruction from before that level.\"\"\"
        residual = self.embed(tokens)
        recon = torch.zeros_like(residual)
        indices = []
        before = []
        for li, length in enumerate(self.levels):
            before.append(recon)
            pooled, bins = self._pool(residual, length)
            idx, q = self.quantize(pooled, li)
            up = self._upsample(q, bins, tokens.shape[1])
            residual = residual - up
            recon = recon + up
            indices.append(idx)
        return indices, before

    def add_level(self, recon: torch.Tensor, idx: torch.Tensor, level: int) -> torch.Tensor:
        bins = self._bins(recon.shape[1], self.levels[level])
        return recon + self._upsample(self.codebooks[level][idx], bins, recon.shape[1])

    def _decode(self, recon: torch.Tensor) -> torch.Tensor:
        h = recon
        for block in self.dec_blocks:
            h = block(h)
        return self.dec_head(self.dec_norm(h))

    def logits_from_codes(self, indices: list[torch.Tensor]) -> torch.Tensor:
        t = self.levels[-1]
        recon = self.embed.weight.new_zeros(indices[0].shape[0], t, self.dim)
        for li, idx in enumerate(indices):
            recon = self.add_level(recon, idx, li)
        return self._decode(recon)

    def reconstruction_loss(self, tokens: torch.Tensor, keep: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.embed(tokens)
        recon = torch.zeros_like(residual)
        commit = tokens.new_zeros((), dtype=torch.float)
        book = tokens.new_zeros((), dtype=torch.float)
        n = keep or len(self.levels)
        for li in range(n):
            pooled, bins = self._pool(residual, self.levels[li])
            _idx, q = self.quantize(pooled, li)
            q_st = pooled + (q - pooled).detach()
            up = self._upsample(q_st, bins, tokens.shape[1])
            residual = residual - up
            recon = recon + up
            commit = commit + (pooled - q.detach()).pow(2).mean()
            book = book + (q - pooled.detach()).pow(2).mean()
        logits = self._decode(recon)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))
        acc = (logits.argmax(-1) == tokens).float().mean()
        return ce + 0.25 * (commit / n) + book / n, acc


class NextConcept(nn.Module):
    \"\"\"Finer levels are parallel. The coarsest level is a causal next-code model.

    Position t of the coarsest level predicts that code from codes 0..t-1 only.
    A future code is never an input to its own prediction.
    \"\"\"

    def __init__(self, vq: MultiScaleResidualVQ, dim: int, n_layer: int, n_head: int):
        super().__init__()
        self.levels = list(vq.levels)
        self.vq = vq
        codebook = vq.codebooks[0].shape[0]
        self.in_proj = nn.Linear(vq.dim, dim)
        self.pos = nn.ParameterList([nn.Parameter(torch.randn(length, dim) * 0.02) for length in self.levels])
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(dim, n_head, dim * 4, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
                for _ in range(n_layer)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, codebook)
        coarse_len = self.levels[0]
        self.coarse_in = nn.Linear(vq.dim, dim)
        self.coarse_bos = nn.Parameter(torch.zeros(1, 1, dim))
        self.coarse_pos = nn.Parameter(torch.randn(coarse_len, dim) * 0.02)
        self.coarse_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(dim, n_head, dim * 4, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
                for _ in range(4)
            ]
        )
        self.coarse_norm = nn.LayerNorm(dim)
        self.coarse_head = nn.Linear(dim, codebook)

    def forward_level(self, recon: torch.Tensor, level: int) -> torch.Tensor:
        if level == 0:
            raise RuntimeError("the coarsest level is causal; call forward_coarse")
        pooled, _bins = self.vq._pool(recon, self.levels[level])
        x = self.in_proj(pooled) + self.pos[level]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def forward_coarse(self, codes: torch.Tensor) -> torch.Tensor:
        \"\"\"codes is (batch, level_length). Logits at t depend only on codes before t.\"\"\"
        emb = self.coarse_in(self.vq.codebooks[0][codes])
        bos = self.coarse_bos.expand(codes.shape[0], 1, -1)
        x = torch.cat([bos, emb[:, :-1]], dim=1) + self.coarse_pos
        length = codes.shape[1]
        mask = torch.triu(torch.full((length, length), float("-inf"), device=x.device), diagonal=1)
        for block in self.coarse_blocks:
            x = block(x, src_mask=mask)
        return self.coarse_head(self.coarse_norm(x))


def init_from_teacher(vq: MultiScaleResidualVQ, teacher_embed: torch.Tensor) -> None:
    with torch.no_grad():
        src = teacher_embed.float()
        rows = min(vq.embed.weight.shape[0], src.shape[0])
        if src.shape[1] == vq.dim:
            copied = src[:rows]
        else:
            # First 256 coordinates of an 896-d table are not a projection.
            gen = torch.Generator().manual_seed(0)
            proj = torch.randn(src.shape[1], vq.dim, generator=gen) / math.sqrt(src.shape[1])
            copied = src[:rows] @ proj
        vq.embed.weight[:rows].copy_(copied)
        for book in vq.codebooks:
            pick = torch.randint(0, rows, (book.shape[0],))
            book.copy_(vq.embed.weight[pick])


@torch.no_grad()
def generate_ncm(
    gen: NextConcept,
    vq: MultiScaleResidualVQ,
    unigram: torch.Tensor,
    batch: int | None = None,
    prefixes: torch.Tensor | None = None,
) -> torch.Tensor:
    \"\"\"Draft a window. The prefix keeps its encoded codes.

    The coarsest future codes are predicted one at a time, each from the codes
    before it. Finer levels then run in parallel on that sketch. The returned
    tokens start with the prefix.
    \"\"\"
    if prefixes is not None:
        batch = prefixes.shape[0]
        known_idx, known_mask = _prefix_codes(vq, prefixes)
    else:
        known_idx = known_mask = None
        if batch is None:
            raise ValueError("pass batch or prefixes")
    device = vq.embed.weight.device
    recon = torch.zeros(batch, vq.levels[-1], vq.dim, device=device, dtype=vq.embed.weight.dtype)
    if known_idx is not None:
        coarse = known_idx[0].clone()
        for t in range(coarse.shape[1]):
            if bool(known_mask[0][:, t].all()):
                continue
            coarse[:, t] = gen.forward_coarse(coarse)[:, t].argmax(-1)
    else:
        coarse = torch.zeros(batch, vq.levels[0], dtype=torch.long, device=device)
        for t in range(coarse.shape[1]):
            coarse[:, t] = gen.forward_coarse(coarse)[:, t].argmax(-1)
    codes = [coarse]
    recon = vq.add_level(recon, coarse, 0)
    for li in range(1, len(vq.levels)):
        pred = gen.forward_level(recon, li).argmax(-1)
        if known_mask is not None:
            pred = torch.where(known_mask[li], known_idx[li], pred)
        codes.append(pred)
        recon = vq.add_level(recon, pred, li)
    tokens = vq.logits_from_codes(codes).argmax(-1)
    if prefixes is not None:
        tokens = tokens.clone()
        tokens[:, : prefixes.shape[1]] = prefixes
    return tokens


@torch.no_grad()
def _prefix_codes(vq: MultiScaleResidualVQ, prefixes: torch.Tensor):
    residual = vq.embed(prefixes)
    known_idx = []
    known_mask = []
    p = prefixes.shape[1]
    for li, length in enumerate(vq.levels):
        bins = vq._bins(vq.levels[-1], length)
        idx = torch.zeros(prefixes.shape[0], length, dtype=torch.long, device=prefixes.device)
        mask = torch.zeros(prefixes.shape[0], length, dtype=torch.bool, device=prefixes.device)
        for i, (start, end) in enumerate(bins):
            if end <= p and end > start:
                code, q = vq.quantize(residual[:, start:end].mean(dim=1, keepdim=True), li)
                idx[:, i] = code[:, 0]
                mask[:, i] = True
                residual[:, start:end] = residual[:, start:end] - q
        known_idx.append(idx)
        known_mask.append(mask)
    return known_idx, known_mask


@torch.no_grad()
def hybrid_repair(draft: torch.Tensor, teacher, tau: float, prefix: int = 0) -> tuple[torch.Tensor, float]:
    \"\"\"One causal teacher forward. Low-confidence draft tokens take the teacher's prediction.

    Logits at position t-1 predict token t, so a replacement does not re-enter this forward.
    Tokens inside `prefix` are the given function start and are not rewritten.
    The rate is the fraction rewritten in the continuation.
    \"\"\"
    logits = teacher(draft).logits
    pred = logits.argmax(-1)
    probs = logits.softmax(-1)
    conf = probs[:, :-1].gather(-1, draft[:, 1:].unsqueeze(-1)).squeeze(-1)
    replace = conf < tau
    if prefix > 1:
        replace[:, : prefix - 1] = False
    hybrid = draft.clone()
    hybrid[:, 1:] = torch.where(replace, pred[:, :-1], draft[:, 1:])
    tail = replace if prefix <= 1 else replace[:, prefix - 1 :]
    return hybrid, tail.float().mean().item()


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

from tqdm.auto import tqdm

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
    bar = tqdm(total=limit, desc=f"{split} windows", unit="win", mininterval=1, dynamic_ncols=True)
    for row in ds:
        text = (row.get("whole_func_string") or "").replace("\\r\\n", "\\n").strip()
        if len(text) < 40 or "def " not in text:
            continue
        piece = tokenizer(text + "\\n", add_special_tokens=False)["input_ids"]
        if len(piece) < 16 or len(piece) > 1024:
            continue
        ids.extend(piece)
        kept += 1
        bar.n = min(limit, len(ids) // CONTEXT)
        bar.set_postfix(functions=kept, refresh=False)
        bar.refresh()
        if len(ids) >= need + CONTEXT:
            break
    bar.close()
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


def cached_windows(split: str, limit: int, seed: int, name: str) -> torch.Tensor:
    path = OUTPUT / name
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(cached.shape) == (limit, CONTEXT):
            print(f"loaded {path} {tuple(cached.shape)}")
            return cached.to(device)
        print(f"ignoring {path}, shape {tuple(cached.shape)} is not {(limit, CONTEXT)}")
    windows = code_windows(split, limit, seed).cpu()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(windows, path)
    return windows.to(device)


train = cached_windows("train", N_TRAIN, SEED, "train.pt")
held = cached_windows("validation", N_VAL, SEED + 1, "held.pt")
print(tokenizer.decode(train[0].tolist(), skip_special_tokens=True)[:500])

teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True)
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

Stage 1 trains the tokenizer only when no finished checkpoint is loaded. A matching checkpoint freezes those weights. Stage 2 does not load an older generator. Those predicted every future coarse code at once and peaked at 0.047 on held-out text. This generator predicts the next coarse code from the true codes before it, teacher-forced, then the finer levels run in parallel. The stop metric is the held-out accuracy of that next code, the one that covers the eight tokens after the prefix. Training stops after step 4,000 if that number gains less than 0.005 three checks in a row.
"""
)

code(
    """
vocab = int(tokenizer.vocab_size)
from tqdm.auto import tqdm

vq = MultiScaleResidualVQ(vocab, VQ_DIM, CODEBOOK, LEVELS, n_dec=DEC_LAYERS, n_head=GEN_HEADS).to(device)
init_from_teacher(vq, teacher_embed)
del teacher_embed
print(f"VQ parameters {sum(p.numel() for p in vq.parameters())/1e6:.1f} M, allocated {torch.cuda.memory_allocated()/1024**3:.2f} GB")

OUTPUT.mkdir(parents=True, exist_ok=True)
ckpt_path = OUTPUT / "hybrid.pt"
if not ckpt_path.exists():
    drive_ckpt = Path("/content/drive/MyDrive/ncm-hybrid/hybrid.pt")
    if not drive_ckpt.exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as exc:
            print(f"Drive not available ({exc}). A missing local checkpoint starts from the teacher embedding.")
    if drive_ckpt.exists():
        ckpt_path.write_bytes(drive_ckpt.read_bytes())
        print("copied checkpoint from Drive", drive_ckpt)
resume = None
if ckpt_path.exists():
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    same = (
        list(loaded.get("levels", [])) == list(LEVELS)
        and int(loaded.get("codebook", -1)) == CODEBOOK
        and int(loaded.get("vq_dim", -1)) == VQ_DIM
        and int(loaded.get("gen_dim", -1)) == GEN_DIM
        and int(loaded.get("gen_layers", -1)) == GEN_LAYERS
        and int(loaded.get("dec_layers", -1)) == DEC_LAYERS
        and int(loaded.get("prefix", -1)) == PREFIX
        and "vq" in loaded
    )
    if same:
        resume = loaded
        print(f"resuming from {ckpt_path}  tokenizer {int(loaded.get('vq_step', 0))}  generator {int(loaded.get('gen_step', 0))}")
    else:
        print("checkpoint at", ckpt_path, "is a different architecture, starting over")

stopped_early = False
best_held_coarse = -1.0


def save_checkpoint(vq_step, gen_step, generator=None, prior=None):
    payload = {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "levels": LEVELS,
        "codebook": CODEBOOK,
        "vq_dim": VQ_DIM,
        "gen_dim": GEN_DIM,
        "gen_layers": GEN_LAYERS,
        "gen_heads": GEN_HEADS,
        "dec_layers": DEC_LAYERS,
        "prefix": PREFIX,
        "tau": TAU,
        "n_train": N_TRAIN,
        "objective": OBJECTIVE,
        "stopped_early": stopped_early,
        "best_held_coarse": best_held_coarse,
        "vq_step": vq_step,
        "gen_step": gen_step,
        "vq": vq.state_dict(),
        "vq_history": vq_history,
        "gen_history": gen_history,
    }
    if generator is not None and prior is not None:
        payload["generator"] = generator.state_dict()
        payload["unigram"] = prior.detach().cpu()
    torch.save(payload, ckpt_path)
    tqdm.write(f"saved {ckpt_path}  tokenizer {vq_step}  generator {gen_step}")


decay, plain = [], []
for name, param in vq.named_parameters():
    (plain if "codebooks" in name else decay).append(param)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.01}, {"params": plain, "weight_decay": 0.0}],
    lr=LR,
)
vq_history = []
gen_history = []
start_vq = 0
if resume is not None:
    vq.load_state_dict(resume["vq"])
    start_vq = min(VQ_STEPS, int(resume.get("vq_step", 0)))
    vq_history = list(resume.get("vq_history", []))
vq.train()
t0 = time.time()
if start_vq >= VQ_STEPS:
    print(f"tokenizer frozen at step {start_vq}")
else:
    tokenizer_bar = tqdm(range(start_vq, VQ_STEPS), desc="tokenizer", initial=start_vq, total=VQ_STEPS, mininterval=1, dynamic_ncols=True)
    for step in tokenizer_bar:
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
        tokenizer_bar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc.item():.3f}", refresh=False)
        if step % LOG_EVERY == 0 or step == VQ_STEPS - 1:
            vq_history.append((step, loss.item(), acc.item()))
            tqdm.write(f"[tokenizer] {step:4d}  loss {loss.item():.3f}  acc {acc.item():.3f}  {time.time()-t0:.0f}s")
        if step % 1000 == 999 or step == VQ_STEPS - 1:
            save_checkpoint(step + 1, 0)
    tokenizer_bar.close()

for p in vq.parameters():
    p.requires_grad_(False)
vq.eval()

with torch.no_grad():
    prior_rows = train[:4096]
    chunks = []
    for i in tqdm(range(0, prior_rows.shape[0], 16), desc="coarse prior", mininterval=1, dynamic_ncols=True):
        chunks.append(vq.encode(prior_rows[i : i + 16])[0].reshape(-1))
    counts = torch.bincount(torch.cat(chunks), minlength=CODEBOOK).float()
unigram = (counts + 0.1) / (counts + 0.1).sum()
unigram = unigram.to(device)

gen = NextConcept(vq, GEN_DIM, GEN_LAYERS, GEN_HEADS).to(device)
gopt = torch.optim.AdamW((p for p in gen.parameters() if p.requires_grad), lr=LR)
same_pass = (
    resume is not None
    and resume.get("objective") == OBJECTIVE
    and int(resume.get("n_train", -1)) == N_TRAIN
    and "generator" in resume
)
best_held_coarse = -1.0
stopped_early = False
best_state = None
best_step = 0
if same_pass:
    gen.load_state_dict(resume["generator"])
    start_gen = min(GEN_STEPS, int(resume.get("gen_step", 0)))
    gen_history = list(resume.get("gen_history", []))
    best_held_coarse = float(resume.get("best_held_coarse", -1.0))
    stopped_early = bool(resume.get("stopped_early", False))
    best_step = start_gen
    print(f"resuming the causal generator at step {start_gen}, best held-out next code {best_held_coarse:.3f}")
else:
    start_gen = 0
    print("generator reinitialized. Older generators predicted every future code at once and were not loaded.")
gen.train()
t1 = time.time()
coarse_until = int(GEN_STEPS * 0.4)
coarse_hi = max(i for i, length in enumerate(LEVELS) if length <= PREFIX)
coarse_future = [i for i, (start, _end) in enumerate(vq._bins(CONTEXT, LEVELS[0])) if start >= PREFIX]
if not coarse_future:
    raise RuntimeError("PREFIX covers every coarsest bin, so there is nothing to continue")
order = torch.randperm(train.shape[0], generator=torch.Generator().manual_seed(SEED + 3))
gen_steps_done = start_gen
stall = 0


@torch.no_grad()
def held_next_code(rows: torch.Tensor) -> float:
    \"\"\"Accuracy of the single coarse code that covers the eight tokens after the prefix.\"\"\"
    training = gen.training
    gen.eval()
    hit = count = 0
    t = coarse_future[0]
    for start in range(0, rows.shape[0], 4):
        batch = rows[start : start + 4]
        codes0 = vq.encode(batch)[0]
        ctx = codes0.clone()
        ctx[:, t:] = 0
        pred = gen.forward_coarse(ctx)[:, t].argmax(-1)
        hit += (pred == codes0[:, t]).sum().item()
        count += codes0.shape[0]
    if training:
        gen.train()
    return hit / max(1, count)


if stopped_early or start_gen >= GEN_STEPS:
    print(f"generator left as loaded, step {start_gen}, held-out next code {best_held_coarse:.3f}")
    concept_bar = None
else:
    concept_bar = tqdm(range(start_gen, GEN_STEPS), desc="concepts", initial=start_gen, total=GEN_STEPS, mininterval=1, dynamic_ncols=True)
for step in concept_bar or []:
    ix = order[step * GEN_BATCH : (step + 1) * GEN_BATCH].to(device)
    with torch.no_grad():
        codes, recons = vq.contexts(train[ix])
        known_idx, known_mask = _prefix_codes(vq, train[ix][:, :PREFIX])
    hi = coarse_hi if step < coarse_until else len(LEVELS) - 1
    next_pos = coarse_future[0]
    logits0 = gen.forward_coarse(codes[0])
    future_logits = logits0[:, coarse_future]
    future_target = codes[0][:, coarse_future]
    loss = COARSE_LOSS_WEIGHT * F.cross_entropy(
        future_logits.reshape(-1, future_logits.size(-1)), future_target.reshape(-1)
    )
    weight_sum = float(COARSE_LOSS_WEIGHT)
    next_hit = (logits0[:, next_pos].argmax(-1) == codes[0][:, next_pos]).sum().item()
    next_count = codes[0].shape[0]
    coarse_hit = (future_logits.argmax(-1) == future_target).sum().item()
    coarse_count = future_target.numel()
    correct = coarse_hit
    total = coarse_count
    pred0 = torch.where(known_mask[0], known_idx[0], logits0.argmax(-1).detach())
    own = vq.add_level(torch.zeros_like(recons[0]), pred0, 0)
    ss = min(0.85, 0.25 + 0.75 * (step / max(1, GEN_STEPS - 1)))
    for li in range(1, hi + 1):
        target = codes[li]
        if torch.rand((), device=device).item() < ss:
            cond = own
        else:
            cond = recons[li]
        logits = gen.forward_level(cond, li)
        loss = loss + F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        weight_sum += 1.0
        hit = (logits.argmax(-1) == target).sum().item()
        correct += hit
        total += target.numel()
        pred = torch.where(known_mask[li], known_idx[li], logits.argmax(-1).detach())
        own = vq.add_level(own, pred, li)
    loss = loss / weight_sum
    gopt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_((p for p in gen.parameters() if p.requires_grad), 1.0)
    gopt.step()
    concept_bar.set_postfix(loss=f"{loss.item():.3f}", refresh=False)
    if step % LOG_EVERY == 0 or step == GEN_STEPS - 1:
        acc = correct / max(1, total)
        next_acc = next_hit / max(1, next_count)
        future_acc = coarse_hit / max(1, coarse_count)
        gen_history.append((step, loss.item(), next_acc))
        concept_bar.set_postfix(loss=f"{loss.item():.3f}", next=f"{next_acc:.3f}", refresh=False)
        tqdm.write(
            f"[concepts]  {step:4d}  loss {loss.item():.3f}  acc {acc:.3f}  "
            f"next {next_acc:.3f}  future {future_acc:.3f}  {time.time()-t1:.0f}s"
        )
    if step % HELD_EVERY == HELD_EVERY - 1 or step == GEN_STEPS - 1:
        held_c = held_next_code(held)
        if held_c > best_held_coarse + HELD_MIN_GAIN:
            best_held_coarse = held_c
            best_step = step + 1
            stall = 0
            best_state = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}
        else:
            stall += 1
        gen_steps_done = step + 1
        save_checkpoint(VQ_STEPS, gen_steps_done, gen, unigram)
        tqdm.write(f"held-out next code {held_c:.3f}  best {best_held_coarse:.3f} at step {best_step}")
        if gen_steps_done >= HELD_MIN_STEP and stall >= HELD_PATIENCE:
            stopped_early = True
            tqdm.write("stopping: held-out next code stopped rising")
            break
else:
    if concept_bar is not None:
        stopped_early = True
if concept_bar is not None:
    concept_bar.close()
if best_state is not None:
    gen.load_state_dict(best_state)
    gen_steps_done = best_step
    tqdm.write(f"restored generator from held-out next code {best_held_coarse:.3f} at step {best_step}")
if stopped_early:
    save_checkpoint(VQ_STEPS, gen_steps_done, gen, unigram)

gen.eval()
correct = 0
total = 0
with torch.no_grad():
    for start in tqdm(range(0, held.shape[0], 4), desc="held-out reconstruction", mininterval=1, dynamic_ncols=True):
        batch = held[start : start + 4]
        pred = vq.logits_from_codes(vq.encode(batch)).argmax(-1)
        correct += (pred == batch).sum().item()
        total += batch.numel()
recon = correct / max(1, total)
print(f"held-out tokenizer reconstruction {recon:.3f}")
sample = held[:1]
with torch.no_grad():
    trip = vq.logits_from_codes(vq.encode(sample)).argmax(-1)
print("original  ", tokenizer.decode(sample[0].tolist(), skip_special_tokens=True)[:400])
print("round trip", tokenizer.decode(trip[0].tolist(), skip_special_tokens=True)[:400])
per_level = []
with torch.no_grad():
    probe = held[:64]
    for li in range(1, len(LEVELS)):
        hit = 0
        count = 0
        for start in range(0, probe.shape[0], 4):
            batch = probe[start : start + 4]
            codes, recons = vq.contexts(batch)
            pred = gen.forward_level(recons[li], li).argmax(-1)
            hit += (pred == codes[li]).sum().item()
            count += codes[li].numel()
        per_level.append(hit / max(1, count))
print("held-out next-level", " ".join(f"L{i}:{acc:.2f}" for i, acc in enumerate(per_level, start=1)))
print(f"held-out next code {held_next_code(held):.3f}  (the code for the eight tokens after the prefix)")
print(f"allocated after training {torch.cuda.memory_allocated()/1024**3:.2f} GB")
"""
)

md(
    """
## Benchmark the hybrid

Three writers, same 128 tokens, same function prefix:

- **Teacher** continues the first `PREFIX` tokens of a validation window. One forward per new token.
- **Next-concept** encodes the bins inside the prefix. It predicts the next coarse code from the codes before it, eight times, then the finer levels in parallel, and decodes. The first of those codes is the next eight tokens.
- **Hybrid** is that draft plus one teacher forward. Tokens inside the prefix stay. Elsewhere, a token the teacher assigns probability below `TAU` is replaced. The replacement does not re-enter that forward.

Read the next eight tokens. A low NLL on a string of parentheses is not a function.
"""
)

code(
    """
teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True)
teacher.to(device)
teacher.eval()
bench_batch = 4


def teacher_continuations():
    prefixes = held[:bench_batch, :PREFIX]
    attention = torch.ones_like(prefixes)
    out = teacher.generate(
        input_ids=prefixes,
        attention_mask=attention,
        max_new_tokens=CONTEXT - PREFIX,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    full = out[:, :CONTEXT]
    if full.shape[1] < CONTEXT:
        full = F.pad(full, (0, CONTEXT - full.shape[1]), value=tokenizer.pad_token_id)
    return full


def run_ncm():
    return generate_ncm(gen, vq, unigram, prefixes=held[:bench_batch, :PREFIX])


def run_hybrid(draft: torch.Tensor):
    return hybrid_repair(draft, teacher, TAU, prefix=PREFIX)


with torch.no_grad():
    teacher_tokens, teacher_ms, teacher_peak = cuda_ms(teacher_continuations)
    teacher_nll_value = teacher_nll(teacher, teacher_tokens)
    draft, ncm_ms, ncm_peak = cuda_ms(run_ncm)
    ncm_nll_value = teacher_nll(teacher, draft)
    (hybrid, repair_rate), repair_ms, repair_peak = cuda_ms(lambda: run_hybrid(draft))
    hybrid_nll_value = teacher_nll(teacher, hybrid)

rows = [
    ("teacher", teacher_nll_value, CONTEXT - PREFIX, teacher_ms / bench_batch, teacher_peak, "greedy continuation"),
    ("next-concept", ncm_nll_value, len(coarse_future) + len(LEVELS) - 1, ncm_ms / bench_batch, ncm_peak, "same prefix"),
    ("hybrid", hybrid_nll_value, len(LEVELS) + 1, (ncm_ms + repair_ms) / bench_batch, max(ncm_peak, repair_peak), f"repair rate {repair_rate:.0%} of the continuation"),
]
print(f"{'writer':16} {'NLL':>8} {'forwards':>10} {'ms/seq':>10} {'peak GB':>8}  note")
for name, nll, forwards, ms, peak, note in rows:
    print(f"{name:16} {nll:8.3f} {forwards:10d} {ms:10.1f} {peak:8.2f}  {note}")

print("\\n--- teacher ---")
print(tokenizer.decode(teacher_tokens[0], skip_special_tokens=True)[:500])
print("\\n--- round trip (tokenizer ceiling on this window) ---")
with torch.no_grad():
    trip = vq.logits_from_codes(vq.encode(held[:1])).argmax(-1)
print(tokenizer.decode(trip[0].tolist(), skip_special_tokens=True)[:500])
print("\\n--- next eight tokens ---")
span = CONTEXT // LEVELS[0]
print("teacher", tokenizer.decode(teacher_tokens[0, PREFIX : PREFIX + span].tolist(), skip_special_tokens=True))
print("draft  ", tokenizer.decode(draft[0, PREFIX : PREFIX + span].tolist(), skip_special_tokens=True))
print("hybrid ", tokenizer.decode(hybrid[0, PREFIX : PREFIX + span].tolist(), skip_special_tokens=True))
print("\\n--- next-concept draft ---")
print(tokenizer.decode(draft[0], skip_special_tokens=True)[:500])
print("\\n--- hybrid ---")
print(tokenizer.decode(hybrid[0], skip_special_tokens=True)[:500])


def letter_frac(token_rows, start, end):
    scores = []
    for row in token_rows[:, start:end].tolist():
        hit = 0
        for token_id in row:
            if any(ch.isalpha() for ch in tokenizer.decode([token_id])):
                hit += 1
        scores.append(hit / max(1, len(row)))
    return sum(scores) / max(1, len(scores))


print(
    "next-eight tokens with a letter: "
    f"teacher {letter_frac(teacher_tokens, PREFIX, PREFIX + span):.0%}  "
    f"draft {letter_frac(draft, PREFIX, PREFIX + span):.0%}  "
    f"hybrid {letter_frac(hybrid, PREFIX, PREFIX + span):.0%}"
)
print(
    "full continuation tokens with a letter: "
    f"teacher {letter_frac(teacher_tokens, PREFIX, CONTEXT):.0%}  "
    f"draft {letter_frac(draft, PREFIX, CONTEXT):.0%}  "
    f"hybrid {letter_frac(hybrid, PREFIX, CONTEXT):.0%}"
)
print("A draft of parentheses lands near 0% on the next-eight line. Trust that line, not the NLL.")
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

Weights are already checkpointed every 1,000 steps at `/content/ncm-hybrid/hybrid.pt`. This cell writes the finished file, including the benchmark. Colab wipes `/content` when the runtime dies. The next cell copies that file to Drive if you want to keep it.

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
        "dec_layers": DEC_LAYERS,
        "prefix": PREFIX,
        "tau": TAU,
        "n_train": N_TRAIN,
        "objective": OBJECTIVE,
        "stopped_early": True,
        "best_held_coarse": best_held_coarse,
        "vq_step": VQ_STEPS,
        "gen_step": gen_steps_done,
        "vq": vq.state_dict(),
        "generator": gen.state_dict(),
        "unigram": unigram.detach().cpu(),
        "vq_history": vq_history,
        "gen_history": gen_history,
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
# Optional. The checkpoint is already at OUTPUT / "hybrid.pt". Drive is only a copy.
from google.colab import drive

try:
    drive.mount("/content/drive")
    dest = Path("/content/drive/MyDrive/ncm-hybrid")
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "hybrid.pt"
    target.write_bytes((OUTPUT / "hybrid.pt").read_bytes())
    print("copied to", target)
except Exception as exc:
    print("Drive copy failed:", exc)
    print("The checkpoint is still at", OUTPUT / "hybrid.pt")
    print("Download that file from the Colab file browser before the runtime is deleted.")
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
