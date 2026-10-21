import type { ForgeMode } from "./engine";
import type { ExportRecipe, TeacherCard } from "./recipe";

export function pythonTrainer(teacher: TeacherCard, mode: ForgeMode, recipe: ExportRecipe): string {
  const levels = recipe.levels.join(", ");
  return `#!/usr/bin/env python3
"""Train a Next-Concept Model in the token space of ${teacher.id}.

Same two stages as HazyResearch/ncm (arXiv 2610.14162):
  1. Multiscale residual VQ tokenizer over the teacher tokenizer.
  2. Next-concept generator that predicts one level at a time, all positions
     in the level in parallel.

Modes
  distill  Sample text from the teacher, then train NCM on those tokens.
  scratch  Train NCM on a local text file (--data), tokenizer only from the repo.
  hybrid   Distill, and at inference let the teacher rewrite low-confidence
           draft tokens. The teacher's input embedding also initializes the
           VQ token table — the two models share a vocabulary and a starting
           representation.

Lab-scale Forge runs a miniature of this file in the browser. This is the
scale-up: paper-like widths, the real Hugging Face tokenizer, real weights.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = ${JSON.stringify(teacher.id)}
MODE = ${JSON.stringify(mode)}
LEVELS = [${levels}]
CONTEXT = ${recipe.context}
CODEBOOK = ${recipe.codebook}
VQ_DIM = ${recipe.vqEmbed}
GEN_DIM = ${recipe.generatorEmbed}
GEN_LAYERS = ${recipe.generatorLayers}
GEN_HEADS = ${recipe.generatorHeads}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill or train an NCM from a Hugging Face model.")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--mode", choices=["distill", "scratch", "hybrid"], default=MODE)
    p.add_argument("--data", type=Path, default=None, help="UTF-8 text, one document per line. Required for scratch.")
    p.add_argument("--output", type=Path, default=Path("ncm-checkpoint"))
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--vq-steps", type=int, default=2000)
    p.add_argument("--gen-steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.45, help="Hybrid rewrite threshold on draft confidence.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class MultiScaleResidualVQ(nn.Module):
    """Residual vector quantizer with one codebook per level. Finest level has length CONTEXT."""

    def __init__(self, vocab: int, dim: int, codebook: int, levels: list[int]):
        super().__init__()
        self.levels = levels
        self.dim = dim
        self.embed = nn.Embedding(vocab, dim)
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(codebook, dim) * (0.02 if i < len(levels) - 1 else 0.2)) for i in range(len(levels))]
        )
        self.dec = nn.Linear(dim, vocab)
        nn.init.normal_(self.embed.weight, std=0.2)
        with torch.no_grad():
            last = self.codebooks[-1]
            n = min(vocab, last.shape[0])
            last[:n].copy_(self.embed.weight[:n])
            self.dec.weight.copy_(self.embed.weight * 4)

    def _pool(self, residual: torch.Tensor, length: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        b, t, d = residual.shape
        bins: list[tuple[int, int]] = []
        chunks = []
        for i in range(length):
            start = (i * t) // length
            end = max(start + 1, ((i + 1) * t) // length)
            end = min(t, end)
            bins.append((start, end))
            chunks.append(residual[:, start:end].mean(dim=1))
        return torch.stack(chunks, dim=1), bins

    def _upsample(self, codes: torch.Tensor, bins: list[tuple[int, int]], t: int) -> torch.Tensor:
        b, _, d = codes.shape
        out = codes.new_zeros(b, t, d)
        for i, (start, end) in enumerate(bins):
            out[:, start:end] = codes[:, i : i + 1]
        return out

    def quantize(self, pooled: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        book = self.codebooks[level]
        # argmin ||p - c||^2 == argmax p·c - 0.5 ||c||^2
        score = pooled @ book.t() - 0.5 * book.pow(2).sum(-1)
        idx = score.argmax(-1)
        q = book[idx]
        return idx, q

    def encode(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        residual = self.embed(tokens)
        indices = []
        for li, length in enumerate(self.levels):
            pooled, bins = self._pool(residual, length)
            idx, q = self.quantize(pooled, li)
            residual = residual - self._upsample(q, bins, tokens.shape[1])
            indices.append(idx)
        return indices

    def decode_levels(self, indices: list[torch.Tensor]) -> torch.Tensor:
        b = indices[0].shape[0]
        t = self.levels[-1]
        z = self.embed.weight.new_zeros(b, t, self.dim)
        for li, idx in enumerate(indices):
            _, bins = self._pool(z, self.levels[li])
            q = self.codebooks[li][idx]
            z = z + self._upsample(q, bins, t)
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
        return ce + 0.25 * commit / n, (logits.argmax(-1) == tokens).float().mean()


class NextConcept(nn.Module):
    """Predicts level k from coarser code embeddings. Positions in a level do not attend to each other."""

    def __init__(self, vq: MultiScaleResidualVQ, dim: int, n_layer: int, n_head: int):
        super().__init__()
        self.levels = list(vq.levels)
        self.vq = vq
        self.in_proj = nn.Linear(vq.dim, dim)
        self.pos = nn.ParameterList(
            [nn.Parameter(torch.randn(length, dim) * 0.02) for length in self.levels[1:]]
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    dim, n_head, dim * 4, dropout=0.0, batch_first=True, norm_first=True
                )
                for _ in range(n_layer)
            ]
        )
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vq.codebooks[0].shape[0])

    def memory(self, codes: list[torch.Tensor], upto: int) -> torch.Tensor:
        parts = []
        for li in range(upto):
            vec = self.vq.codebooks[li][codes[li]]
            parts.append(self.in_proj(vec))
        return torch.cat(parts, dim=1)

    def forward_level(self, codes: list[torch.Tensor], level: int) -> torch.Tensor:
        mem = self.memory(codes, level)
        for block in self.blocks:
            mem = block(mem)
        query = self.pos[level - 1].unsqueeze(0).expand(mem.shape[0], -1, -1)
        scale = 1.0 / math.sqrt(query.shape[-1])
        attn = torch.softmax(self.q(query) @ self.k(mem).transpose(1, 2) * scale, dim=-1)
        ctx = attn @ self.v(mem)
        return self.head(query + ctx)


def init_from_teacher(vq: MultiScaleResidualVQ, teacher_embed: torch.Tensor) -> None:
    with torch.no_grad():
        src = teacher_embed.float()
        rows = min(vq.embed.weight.shape[0], src.shape[0])
        if src.shape[1] == vq.dim:
            vq.embed.weight[:rows].copy_(src[:rows])
        else:
            proj = src[:rows, : vq.dim] if src.shape[1] > vq.dim else F.pad(src[:rows], (0, vq.dim - src.shape[1]))
            vq.embed.weight[:rows].copy_(proj)
        n = min(rows, vq.codebooks[-1].shape[0])
        vq.codebooks[-1][:n].copy_(vq.embed.weight[:n])


def batch_text(tokenizer, lines: list[str], context: int, device: str) -> torch.Tensor:
    enc = tokenizer(lines, padding="max_length", truncation=True, max_length=context, return_tensors="pt")
    return enc["input_ids"].to(device)


@torch.no_grad()
def sample_teacher(model, tokenizer, n: int, context: int, device: str) -> list[str]:
    prompts = [
        "The harbor", "A quiet", "Once", "The field note", "In the morning",
        "She", "He", "They", "After dawn", "Near the",
    ]
    lines: list[str] = []
    model.eval()
    while len(lines) < n:
        seeds = [prompts[i % len(prompts)] for i in range(min(8, n - len(lines)))]
        ids = tokenizer(seeds, return_tensors="pt", padding=True).to(device)
        out = model.generate(**ids, max_new_tokens=context, do_sample=True, temperature=0.9, top_p=0.95)
        lines.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return lines[:n]


def train_vq(vq: MultiScaleResidualVQ, data: torch.Tensor, steps: int, batch: int, lr: float) -> None:
    opt = torch.optim.AdamW(vq.parameters(), lr=lr)
    vq.train()
    for step in range(steps):
        ix = torch.randint(0, data.shape[0], (batch,), device=data.device)
        keep = len(vq.levels) if step < steps // 3 or torch.rand(()) > 0.3 else int(torch.randint(1, len(vq.levels) + 1, ()))
        loss, acc = vq.reconstruction_loss(data[ix], keep)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vq.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"[tokenizer] step {step} loss {loss.item():.3f} acc {acc.item():.3f}", flush=True)


def train_generator(gen: NextConcept, vq: MultiScaleResidualVQ, data: torch.Tensor, steps: int, batch: int, lr: float) -> None:
    opt = torch.optim.AdamW(gen.parameters(), lr=lr)
    gen.train()
    vq.eval()
    for step in range(steps):
        ix = torch.randint(0, data.shape[0], (batch,), device=data.device)
        with torch.no_grad():
            codes = vq.encode(data[ix])
        loss = data.new_zeros(())
        correct = 0
        total = 0
        for li in range(1, len(vq.levels)):
            cond = [c.clone() for c in codes]
            if torch.rand(()) < 0.5:
                # Corrupt a few conditioning codes so inference errors do not cascade.
                mask = torch.rand_like(cond[li - 1].float()) < 0.1
                random_codes = torch.randint(0, vq.codebooks[0].shape[0], cond[li - 1].shape, device=data.device)
                cond[li - 1] = torch.where(mask, random_codes, cond[li - 1])
            logits = gen.forward_level(cond, li)
            loss = loss + F.cross_entropy(logits.reshape(-1, logits.size(-1)), codes[li].reshape(-1))
            correct += (logits.argmax(-1) == codes[li]).sum().item()
            total += codes[li].numel()
        loss = loss / (len(vq.levels) - 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"[concepts] step {step} loss {loss.item():.3f} acc {correct / max(1, total):.3f}", flush=True)


@torch.no_grad()
def generate_ncm(gen: NextConcept, vq: MultiScaleResidualVQ, batch: int, device: str) -> torch.Tensor:
    codes = [torch.randint(0, vq.codebooks[0].shape[0], (batch, 1), device=device)]
    # Level 0 from the empirical unigram would go here; uniform is the cold start.
    for li in range(1, len(vq.levels)):
        logits = gen.forward_level(codes, li)
        codes.append(logits.argmax(-1))
    return vq.decode_levels(codes).argmax(-1)


@torch.no_grad()
def hybrid_repair(draft: torch.Tensor, teacher, tau: float) -> torch.Tensor:
    """One teacher forward over the NCM draft. Uncertain positions take the teacher's token.

    Causal scores at t-1 predict position t, so a rewrite does not re-enter the
    same forward. That is the whole point of combining them: the concept model
    spent len(levels) parallel steps, the teacher spends one.
    """
    out = teacher(draft)
    logits = out.logits
    pred = logits.argmax(-1)
    # Confidence of the draft token under the teacher, shifted for causal prediction.
    probs = logits.softmax(-1)
    conf = probs[:, :-1].gather(-1, draft[:, 1:].unsqueeze(-1)).squeeze(-1)
    replace = conf < tau
    source = pred[:, :-1]
    hybrid = draft.clone()
    hybrid[:, 1:] = torch.where(replace, source, draft[:, 1:])
    return hybrid


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = args.device
    teacher = None
    if args.mode != "scratch":
        teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32)
        teacher.to(device)
        lines = sample_teacher(teacher, tokenizer, args.samples, CONTEXT, device)
    else:
        if args.data is None:
            raise SystemExit("scratch mode needs --data, a utf-8 file with one document per line")
        lines = [ln.strip() for ln in args.data.read_text().splitlines() if ln.strip()]
    data = batch_text(tokenizer, lines, CONTEXT, device)
    vocab = int(tokenizer.vocab_size)
    vq = MultiScaleResidualVQ(vocab, VQ_DIM, CODEBOOK, LEVELS).to(device)
    if teacher is not None:
        init_from_teacher(vq, teacher.get_input_embeddings().weight.detach().to(device))
        print(f"initialized VQ embeddings from {args.model}", flush=True)
    train_vq(vq, data, args.vq_steps, args.batch, args.lr)
    for p in vq.parameters():
        p.requires_grad_(False)
    gen = NextConcept(vq, GEN_DIM, GEN_LAYERS, GEN_HEADS).to(device)
    train_generator(gen, vq, data, args.gen_steps, args.batch, args.lr)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": vq.state_dict(), "levels": LEVELS, "vocab": vocab}, args.output / "tokenizer.pt")
    torch.save({"model": gen.state_dict()}, args.output / "generator.pt")
    (args.output / "config.yaml").write_text(
        "\\n".join(
            [
                f"teacher: {args.model}",
                f"mode: {args.mode}",
                f"levels: {LEVELS}",
                f"context_length: {CONTEXT}",
                f"codebook: {CODEBOOK}",
                f"vq_embed: {VQ_DIM}",
                f"generator_layers: {GEN_LAYERS}",
                f"generator_embed: {GEN_DIM}",
                f"generator_heads: {GEN_HEADS}",
                "note: Hierarchy matches the TinyStories NCM schedule.",
                "",
            ]
        )
    )
    draft = generate_ncm(gen, vq, 2, device)
    if args.mode == "hybrid" and teacher is not None:
        hybrid = hybrid_repair(draft, teacher, args.tau)
        print("draft ", tokenizer.batch_decode(draft))
        print("hybrid", tokenizer.batch_decode(hybrid))
    else:
        print(tokenizer.batch_decode(draft))
    print(f"wrote {args.output}/tokenizer.pt, generator.pt, config.yaml")


if __name__ == "__main__":
    main()
`;
}
