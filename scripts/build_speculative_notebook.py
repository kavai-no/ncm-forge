#!/usr/bin/env python3
"""Write the Colab notebooks that draft for Qwen and keep only Qwen's tokens."""

import ast
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "lib" / "ncm"


def notebook(cells: list[tuple[str, str]]) -> dict:
    body = []
    for kind, source in cells:
        text = source.strip() + "\n"
        lines = text.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            if not text.lstrip().startswith("%"):
                ast.parse(text)
        body.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
        },
        "cells": body,
    }


def write(name: str, cells: list[tuple[str, str]]) -> None:
    path = OUT / name
    path.write_text(json.dumps(notebook(cells), indent=1) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(cells)} cells)")


TRAIN: list[tuple[str, str]] = []


def md(source: str) -> None:
    TRAIN.append(("markdown", source))


def code(source: str) -> None:
    TRAIN.append(("code", source))


md(
    """
# A faster Qwen, with Qwen's own text

The drafter proposes the next few tokens. Qwen scores that span in one forward and keeps the longest prefix it would have written itself. The rest of the span is thrown away. The output is greedy Qwen, token for token. It is faster only when several tokens survive each check.

There is no codebook. A codebook that cannot say Qwen's token cannot be a drafter.

Runtime → Change runtime type → **T4 GPU**. The teacher is `Qwen/Qwen2.5-0.5B-Instruct` in fp16. The drafter is a 4-layer model with Qwen's vocabulary and tied embeddings. Training text is CodeSearchNet Python. The drafter is trained to copy Qwen's next token, then unloaded Qwen is loaded again for the check.
"""
)

code(
    """
%pip install -q "transformers>=5.3,<6" "datasets>=3" "huggingface-hub>=1.32,<2" accelerate sentencepiece protobuf
"""
)

code(
    """
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "code-search-net/code_search_net"
DATASET_CONFIG = "python"
CONTEXT = 128
PREFIX = 64
NEW_TOKENS = 64
PROPOSE = 4
N_TRAIN = 4096
N_HELD = 128
N_BENCH = 16
EPOCHS = 2
BATCH = 8
LR = 3e-4
DRAFT_DIM = 256
DRAFT_LAYERS = 4
DRAFT_HEADS = 4
SEED = 7
OUTPUT = Path("/content/qwen-draft")

if not torch.cuda.is_available():
    raise SystemExit("No GPU. Runtime → Change runtime type → T4 GPU, then rerun.")
props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
print(f"GPU {props.name}  {vram_gb:.1f} GB")
if vram_gb < 12:
    raise SystemExit(f"This notebook wants a 16 GB T4. This GPU reports {vram_gb:.1f} GB.")
torch.manual_seed(SEED)
device = torch.device("cuda")
OUTPUT.mkdir(parents=True, exist_ok=True)
"""
)

md(
    """
## Text

Windows are whole Python functions, in Qwen's vocabulary. The teacher is loaded only to label each position with the token it would write next. Those labels are kept. The weights are not.
"""
)

code(
    """
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        print("using HF_TOKEN from Colab secrets")
except Exception:
    pass


def function_windows(split: str, limit: int, seed: int) -> torch.Tensor:
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
        raise SystemExit(f"{split} produced {usable // CONTEXT} windows, need {limit}.")
    windows = torch.tensor(ids[:usable], dtype=torch.long).view(-1, CONTEXT)
    order = torch.randperm(windows.shape[0], generator=torch.Generator().manual_seed(seed))
    chosen = windows[order[:limit]].contiguous()
    print(f"{split}: {kept} functions, {tuple(chosen.shape)}")
    return chosen


train = function_windows("train", N_TRAIN, SEED)
held = function_windows("validation", N_HELD, SEED + 1)
print(tokenizer.decode(held[0].tolist(), skip_special_tokens=True)[:400])
"""
)

code(
    """
teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True)
teacher.to(device)
teacher.eval()


@torch.no_grad()
def qwen_targets(windows: torch.Tensor) -> torch.Tensor:
    \"\"\"Token Qwen would write at each next position, given the true prefix.\"\"\"
    out = torch.empty_like(windows)
    for start in tqdm(range(0, windows.shape[0], 2), desc="Qwen labels", mininterval=1, dynamic_ncols=True):
        batch = windows[start : start + 2].to(device)
        pred = teacher(batch).logits.argmax(-1)
        out[start : start + batch.shape[0]] = pred.cpu()
    return out


train_target = qwen_targets(train)
held_target = qwen_targets(held)
agree = (train_target[:, :-1] == train[:, 1:]).float().mean().item()
print(f"Qwen's next token matches the dataset {agree:.1%} of the time. The drafter copies Qwen, not the dataset.")
del teacher
torch.cuda.empty_cache()
print(f"teacher unloaded, allocated {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
"""
)

md(
    """
## Drafter

Four transformer layers, Qwen's vocabulary, embeddings tied to the output. It is trained with cross-entropy on Qwen's argmax. A codebook is not in this path.
"""
)

code(
    """
class Drafter(nn.Module):
    def __init__(self, vocab: int, dim: int, n_layer: int, n_head: int, context: int):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    dim, n_head, dim * 4, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"
                )
                for _ in range(n_layer)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.context:
            tokens = tokens[:, -self.context :]
            length = tokens.shape[1]
        index = torch.arange(length, device=tokens.device)
        x = self.embed(tokens) + self.pos(index)
        mask = torch.triu(torch.full((length, length), float("-inf"), device=tokens.device), diagonal=1)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        return self.head(self.norm(x))


vocab = int(tokenizer.vocab_size)
drafter = Drafter(vocab, DRAFT_DIM, DRAFT_LAYERS, DRAFT_HEADS, CONTEXT).to(device)
n_param = sum(p.numel() for p in drafter.parameters())
print(f"drafter parameters {n_param / 1e6:.1f} M, allocated {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
opt = torch.optim.AdamW(drafter.parameters(), lr=LR)
loader = DataLoader(TensorDataset(train, train_target), batch_size=BATCH, shuffle=True)
steps = 0
for epoch in range(EPOCHS):
    bar = tqdm(loader, desc=f"epoch {epoch + 1}/{EPOCHS}", mininterval=1, dynamic_ncols=True)
    for tokens, target in bar:
        tokens = tokens.to(device)
        target = target.to(device)
        logits = drafter(tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab), target[:, :-1].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
        opt.step()
        steps += 1
        if steps % 20 == 0:
            hit = (logits.argmax(-1) == target[:, :-1]).float().mean().item()
            bar.set_postfix(loss=f"{loss.item():.3f}", agree=f"{hit:.3f}")
print(f"trained {steps} steps")
"""
)

md(
    """
## The check

Greedy Qwen writes one token at a time. The drafter writes up to `PROPOSE` tokens. One Qwen forward scores them. Tokens that match are kept. The first token that does not match is replaced with Qwen's token, and the rest of the proposal is dropped. If every proposal matches, Qwen's next token comes from that same forward.

The two strings below have to be identical. If they are not, the notebook stops.
"""
)

code(
    """
teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True)
teacher.to(device)
teacher.eval()
drafter.eval()


@torch.no_grad()
def greedy(prefix: torch.Tensor, n_new: int) -> tuple[torch.Tensor, int]:
    out = prefix.view(1, -1)
    for _ in range(n_new):
        nxt = teacher(out).logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
    return out[0], n_new


@torch.no_grad()
def speculate(prefix: torch.Tensor, n_new: int, k: int) -> tuple[torch.Tensor, int, int]:
    \"\"\"Return tokens, Qwen forwards, and proposal tokens that were kept.\"\"\"
    out = prefix.view(1, -1)
    start = out.shape[1]
    forwards = 0
    kept = 0
    while out.shape[1] - start < n_new:
        take = min(k, n_new - (out.shape[1] - start))
        draft = out
        proposed = []
        for _ in range(take):
            nxt = drafter(draft).logits[:, -1].argmax(-1, keepdim=True)
            proposed.append(nxt)
            draft = torch.cat([draft, nxt], dim=1)
        props = torch.cat(proposed, dim=1)
        full = torch.cat([out, props], dim=1)
        logits = teacher(full).logits[0]
        base = out.shape[1] - 1
        forwards += 1
        n_keep = 0
        for i in range(take):
            if int(logits[base + i].argmax()) == int(props[0, i]):
                n_keep += 1
            else:
                break
        if n_keep:
            out = torch.cat([out, props[:, :n_keep]], dim=1)
            kept += n_keep
        if out.shape[1] - start >= n_new:
            break
        if n_keep < take:
            out = torch.cat([out, logits[base + n_keep].argmax().view(1, 1)], dim=1)
        else:
            out = torch.cat([out, logits[base + take].argmax().view(1, 1)], dim=1)
            kept += 1
    return out[0, : start + n_new], forwards, kept


same = 0
qwen_forwards = 0
draft_kept = 0
plain_ms = 0.0
spec_ms = 0.0
rows = []
for i in tqdm(range(N_BENCH), desc="check", mininterval=1, dynamic_ncols=True):
    prefix = held[i, :PREFIX].to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gold, plain_n = greedy(prefix, NEW_TOKENS)
    torch.cuda.synchronize()
    plain_ms += (time.perf_counter() - t0) * 1000
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    got, forwards, kept = speculate(prefix, NEW_TOKENS, PROPOSE)
    torch.cuda.synchronize()
    spec_ms += (time.perf_counter() - t1) * 1000
    match = torch.equal(gold, got)
    same += int(match)
    qwen_forwards += forwards
    draft_kept += kept
    if i == 0:
        rows.append((gold, got, forwards, kept, match))

if same != N_BENCH:
    raise SystemExit(f"{same}/{N_BENCH} continuations matched greedy Qwen. The check is wrong.")

per_forward = (N_BENCH * NEW_TOKENS) / max(1, qwen_forwards)
print(f"matched greedy Qwen on {same}/{N_BENCH}")
print(f"Qwen forwards per continuation: greedy {NEW_TOKENS}, speculative {qwen_forwards / N_BENCH:.1f}")
print(f"tokens per Qwen forward: {per_forward:.2f}")
print(f"ms per continuation: greedy {plain_ms / N_BENCH:.0f}, speculative {spec_ms / N_BENCH:.0f}")
print(f"proposal tokens kept, including a bonus token when the whole span matched: {draft_kept / (N_BENCH * NEW_TOKENS):.0%}")
if per_forward <= 1.05:
    print("No speedup. The drafter is not close enough to Qwen, so the extra pass is overhead.")
else:
    print("Fewer Qwen forwards, and the text is still Qwen's.")

gold, got, forwards, kept, match = rows[0]
print("\\n--- greedy Qwen ---")
print(tokenizer.decode(gold.tolist(), skip_special_tokens=True)[:700])
print("\\n--- speculative ---")
print(tokenizer.decode(got.tolist(), skip_special_tokens=True)[:700])
print("identical" if match else "NOT identical")
"""
)

md(
    """
## Save

The file is the drafter only. Qwen is not in it. The next notebook loads this file and checks new functions.
"""
)

code(
    """
torch.save(
    {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "vocab": vocab,
        "context": CONTEXT,
        "prefix": PREFIX,
        "propose": PROPOSE,
        "draft_dim": DRAFT_DIM,
        "draft_layers": DRAFT_LAYERS,
        "draft_heads": DRAFT_HEADS,
        "drafter": drafter.state_dict(),
        "tokens_per_qwen_forward": per_forward,
    },
    OUTPUT / "drafter.pt",
)
print("wrote", OUTPUT / "drafter.pt")
"""
)

code(
    """
from google.colab import drive

try:
    drive.mount("/content/drive")
    dest = Path("/content/drive/MyDrive/qwen-draft")
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "drafter.pt"
    target.write_bytes((OUTPUT / "drafter.pt").read_bytes())
    print("copied to", target)
except Exception as exc:
    print("Drive copy failed:", exc)
    print("The drafter is still at", OUTPUT / "drafter.pt")
"""
)

TEST: list[tuple[str, str]] = []


def tmd(source: str) -> None:
    TEST.append(("markdown", source))


def tcode(source: str) -> None:
    TEST.append(("code", source))


tmd(
    """
# Check a Qwen drafter

Loads `drafter.pt` and checks it on CodeSearchNet functions the training notebook's bench split did not print as the sample. Nothing here updates weights.

The output has to equal greedy Qwen. A faster run that changes a token is a failure.
"""
)

tcode(
    """
%pip install -q "transformers>=5.3,<6" "datasets>=3" "huggingface-hub>=1.32,<2" accelerate sentencepiece protobuf
"""
)

tcode(
    """
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CANDIDATES = [
    Path("/content/qwen-draft/drafter.pt"),
    Path("/content/drive/MyDrive/qwen-draft/drafter.pt"),
    Path("drafter.pt"),
]
N_TEST = 32
NEW_TOKENS = 64
SEED = 99

if not torch.cuda.is_available():
    raise SystemExit("No GPU. Runtime → Change runtime type → T4 GPU, then rerun.")
device = torch.device("cuda")
path = next((item for item in CANDIDATES if item.exists()), None)
if path is None:
    try:
        from google.colab import files
        print("No drafter.pt found. Upload the file saved by the training notebook.")
        uploaded = files.upload()
        if not uploaded:
            raise SystemExit("No file uploaded.")
        name = next(iter(uploaded))
        path = Path("/content") / name
    except ImportError:
        raise SystemExit("Put drafter.pt at /content/qwen-draft/drafter.pt and rerun this cell.")

ckpt = torch.load(path, map_location="cpu", weights_only=False)
needed = ["model", "vocab", "context", "prefix", "propose", "draft_dim", "draft_layers", "draft_heads", "drafter"]
missing = [key for key in needed if key not in ckpt]
if missing:
    raise SystemExit(f"{path} is missing {missing}.")
print(f"loaded {path}")
print(f"teacher {ckpt['model']} | propose {ckpt['propose']} | trained tokens/forward {ckpt.get('tokens_per_qwen_forward', float('nan')):.2f}")
"""
)

tcode(
    """
class Drafter(nn.Module):
    def __init__(self, vocab: int, dim: int, n_layer: int, n_head: int, context: int):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    dim, n_head, dim * 4, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"
                )
                for _ in range(n_layer)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.context:
            tokens = tokens[:, -self.context :]
            length = tokens.shape[1]
        index = torch.arange(length, device=tokens.device)
        x = self.embed(tokens) + self.pos(index)
        mask = torch.triu(torch.full((length, length), float("-inf"), device=tokens.device), diagonal=1)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        return self.head(self.norm(x))


model_id = ckpt["model"]
tokenizer = AutoTokenizer.from_pretrained(model_id)
context = int(ckpt["context"])
prefix_len = int(ckpt["prefix"])
propose = int(ckpt["propose"])
drafter = Drafter(
    int(ckpt["vocab"]), int(ckpt["draft_dim"]), int(ckpt["draft_layers"]), int(ckpt["draft_heads"]), context
).to(device)
drafter.load_state_dict(ckpt["drafter"])
drafter.eval()
teacher = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, low_cpu_mem_usage=True)
teacher.to(device)
teacher.eval()


def windows(limit: int) -> torch.Tensor:
    ds = load_dataset(ckpt.get("dataset", "code-search-net/code_search_net"), ckpt.get("dataset_config", "python"), split="test", streaming=True)
    ds = ds.shuffle(buffer_size=2048, seed=SEED)
    ids: list[int] = []
    need = limit * context
    for row in ds:
        text = (row.get("whole_func_string") or "").replace("\\r\\n", "\\n").strip()
        if len(text) < 40 or "def " not in text:
            continue
        piece = tokenizer(text + "\\n", add_special_tokens=False)["input_ids"]
        if len(piece) < 16 or len(piece) > 1024:
            continue
        ids.extend(piece)
        if len(ids) >= need + context:
            break
    usable = (len(ids) // context) * context
    if usable < need:
        raise SystemExit(f"test split produced {usable // context} windows, need {limit}.")
    data = torch.tensor(ids[:usable], dtype=torch.long).view(-1, context)
    order = torch.randperm(data.shape[0], generator=torch.Generator().manual_seed(SEED))
    return data[order[:limit]]


held = windows(N_TEST)
print(tokenizer.decode(held[0].tolist(), skip_special_tokens=True)[:400])
"""
)

tcode(
    """
@torch.no_grad()
def greedy(prefix: torch.Tensor, n_new: int) -> torch.Tensor:
    out = prefix.view(1, -1)
    for _ in range(n_new):
        nxt = teacher(out).logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
    return out[0]


@torch.no_grad()
def speculate(prefix: torch.Tensor, n_new: int, k: int) -> tuple[torch.Tensor, int]:
    out = prefix.view(1, -1)
    start = out.shape[1]
    forwards = 0
    while out.shape[1] - start < n_new:
        take = min(k, n_new - (out.shape[1] - start))
        draft = out
        proposed = []
        for _ in range(take):
            nxt = drafter(draft).logits[:, -1].argmax(-1, keepdim=True)
            proposed.append(nxt)
            draft = torch.cat([draft, nxt], dim=1)
        props = torch.cat(proposed, dim=1)
        full = torch.cat([out, props], dim=1)
        logits = teacher(full).logits[0]
        base = out.shape[1] - 1
        forwards += 1
        n_keep = 0
        for i in range(take):
            if int(logits[base + i].argmax()) == int(props[0, i]):
                n_keep += 1
            else:
                break
        if n_keep:
            out = torch.cat([out, props[:, :n_keep]], dim=1)
        if out.shape[1] - start >= n_new:
            break
        if n_keep < take:
            out = torch.cat([out, logits[base + n_keep].argmax().view(1, 1)], dim=1)
        else:
            out = torch.cat([out, logits[base + take].argmax().view(1, 1)], dim=1)
    return out[0, : start + n_new], forwards


same = 0
forwards = 0
plain_ms = 0.0
spec_ms = 0.0
show = None
for i in tqdm(range(held.shape[0]), desc="check", mininterval=1, dynamic_ncols=True):
    prefix = held[i, :prefix_len].to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gold = greedy(prefix, NEW_TOKENS)
    torch.cuda.synchronize()
    plain_ms += (time.perf_counter() - t0) * 1000
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    got, n_forward = speculate(prefix, NEW_TOKENS, propose)
    torch.cuda.synchronize()
    spec_ms += (time.perf_counter() - t1) * 1000
    match = torch.equal(gold, got)
    same += int(match)
    forwards += n_forward
    if show is None:
        show = (gold, got, match)

n = held.shape[0]
per_forward = (n * NEW_TOKENS) / max(1, forwards)
print(f"matched greedy Qwen on {same}/{n}")
print(f"Qwen forwards per continuation: greedy {NEW_TOKENS}, speculative {forwards / n:.1f}")
print(f"tokens per Qwen forward: {per_forward:.2f}")
print(f"ms per continuation: greedy {plain_ms / n:.0f}, speculative {spec_ms / n:.0f}")
if same != n:
    raise SystemExit("Speculative text differed from greedy Qwen.")
if per_forward <= 1.05:
    print("No speedup on this split.")
else:
    print("Fewer Qwen forwards. The text is still Qwen's.")
gold, got, match = show
print("\\n--- greedy Qwen ---")
print(tokenizer.decode(gold.tolist(), skip_special_tokens=True)[:700])
print("\\n--- speculative ---")
print(tokenizer.decode(got.tolist(), skip_special_tokens=True)[:700])
"""
)


def algorithm_holds() -> None:
    gold = list(range(1, 81))

    def teacher_fn(seq: list[int]) -> list[int]:
        preds = []
        for i in range(len(seq)):
            preds.append(gold[i + 1] if seq[: i + 1] == gold[: i + 1] and i + 1 < len(gold) else -1)
        return preds

    def drafter_fn(seq: list[int], agree: float, step: int) -> int:
        nxt = gold[len(seq)]
        if ((len(seq) * 17 + step) % 100) / 100 >= agree:
            return 0 if nxt != 0 else 2
        return nxt

    def speculate(prefix: list[int], n_new: int, k: int, agree: float) -> tuple[list[int], int]:
        out = list(prefix)
        forwards = 0
        step = 0
        while len(out) - len(prefix) < n_new:
            take = min(k, n_new - (len(out) - len(prefix)))
            ctx = list(out)
            props = []
            for _ in range(take):
                tok = drafter_fn(ctx, agree, step)
                step += 1
                props.append(tok)
                ctx.append(tok)
            full = out + props
            preds = teacher_fn(full)
            base = len(out) - 1
            forwards += 1
            n_keep = 0
            for i in range(take):
                if preds[base + i] == props[i]:
                    n_keep += 1
                else:
                    break
            if n_keep:
                out.extend(props[:n_keep])
            if len(out) - len(prefix) >= n_new:
                break
            if n_keep < take:
                out.append(preds[base + n_keep])
            else:
                out.append(preds[base + take])
        return out[: len(prefix) + n_new], forwards

    def greedy(prefix: list[int], n_new: int) -> list[int]:
        out = list(prefix)
        while len(out) - len(prefix) < n_new:
            out.append(teacher_fn(out)[len(out) - 1])
        return out

    for k in (1, 2, 4, 8):
        for agree in (0.0, 0.3, 0.7, 1.0):
            prefix = gold[:8]
            got, forwards = speculate(prefix, 24, k, agree)
            expect = greedy(prefix, 24)
            if got != expect:
                raise SystemExit(f"mismatch k={k} agree={agree}")
            if agree == 1.0 and forwards > math.ceil(24 / (k + 1)):
                raise SystemExit(f"perfect drafter used too many forwards: {forwards} k={k}")
            if agree == 0.0 and forwards != 24:
                raise SystemExit(f"rejected drafter should cost one forward per token, got {forwards}")
    print("speculative check matches greedy")


if __name__ == "__main__":
    algorithm_holds()
    write("qwen_speculative_colab.ipynb", TRAIN)
    write("qwen_speculative_test.ipynb", TEST)
