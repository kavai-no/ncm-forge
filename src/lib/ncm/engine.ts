import { sentence, validTemplate, type Corpus } from "./corpus";

export type ForgeMode = "distill" | "scratch" | "hybrid";

export type StageName = "teacher" | "tokenizer" | "concepts" | "combine";

type Param = {
  w: Float32Array;
  g: Float32Array;
  m: Float32Array;
  v: Float32Array;
};

export type HistoryPoint = {
  stage: StageName;
  step: number;
  loss: number;
  acc: number;
};

export type Metrics = {
  reconAcc: number;
  conceptAcc: number;
  arAcc: number;
  ncmValid: number;
  arValid: number;
  hybridValid: number;
  ncmTokenAcc: number;
  hybridTokenAcc: number;
  avgRewrites: number;
  stepsAr: number;
  stepsNcm: number;
  stepsHybrid: number;
  paramsTokenizer: number;
  paramsGenerator: number;
  paramsToken: number;
};

export type SampleView = {
  kind: "reconstruction" | "generation";
  title: string;
  source: string;
  ladder: { level: number; length: number; tokens: string[]; acc: number }[];
  ncmTokens: string[];
  arTokens: string[];
  hybridTokens: string[];
  repaired: boolean[];
  rewrites: number;
};

export type ForgeSnapshot = {
  stage: StageName | "done";
  stageStep: number;
  stageSteps: number;
  loss: number;
  acc: number;
  history: HistoryPoint[];
  done: boolean;
};

type LevelGen = { pos: Param; outW: Param; outB: Param };

type ArModel = {
  wh: Param;
  wx: Param;
  wc: Param;
  wo: Param;
  bh: Param;
  bo: Param;
  pos: Param;
};

const B = 24;

function makeRng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number) {
  const u = Math.max(rng(), 1e-8);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
}

function param(n: number, std: number, rng: () => number): Param {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = gaussian(rng) * std;
  return { w, g: new Float32Array(n), m: new Float32Array(n), v: new Float32Array(n) };
}

function zeroGrad(p: Param) {
  p.g.fill(0);
}

function clipAndAdam(p: Param, lr: number, t: number) {
  let norm = 0;
  for (let i = 0; i < p.g.length; i++) norm += p.g[i]! * p.g[i]!;
  norm = Math.sqrt(norm);
  if (norm > 4) {
    const s = 4 / norm;
    for (let i = 0; i < p.g.length; i++) p.g[i]! *= s;
  }
  const b1 = 0.9;
  const b2 = 0.99;
  const c1 = 1 - b1 ** t;
  const c2 = 1 - b2 ** t;
  for (let i = 0; i < p.w.length; i++) {
    const g = p.g[i]!;
    p.m[i] = b1 * p.m[i]! + (1 - b1) * g;
    p.v[i] = b2 * p.v[i]! + (1 - b2) * g * g;
    p.w[i] = p.w[i]! - (lr * (p.m[i]! / c1)) / (Math.sqrt(p.v[i]! / c2) + 1e-6);
  }
}

function binsFor(tLen: number, length: number): Array<[number, number]> {
  const bins: Array<[number, number]> = [];
  for (let i = 0; i < length; i++) {
    const start = Math.floor((i * tLen) / length);
    const end = Math.max(start + 1, Math.floor(((i + 1) * tLen) / length));
    bins.push([start, Math.min(tLen, end)]);
  }
  return bins;
}

function softmaxRows(logits: Float32Array, rows: number, cols: number, out: Float32Array) {
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    const base = r * cols;
    for (let c = 0; c < cols; c++) max = Math.max(max, logits[base + c]!);
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const e = Math.exp(logits[base + c]! - max);
      out[base + c] = e;
      sum += e;
    }
    for (let c = 0; c < cols; c++) out[base + c] = out[base + c]! / sum;
  }
}

export class ForgeSession {
  readonly corpus: Corpus;
  readonly mode: ForgeMode;
  readonly levels: number[];
  readonly dim: number;
  readonly codebook: number;
  readonly hidden: number;
  readonly context: number;
  readonly vocabSize: number;

  private rng: () => number;
  private embed: Param;
  private books: Param[];
  private emaSum: Float32Array[];
  private emaCount: Float32Array[];
  private decW: Param;
  private decB: Param;
  private gens: LevelGen[];
  private plain: ArModel;
  private fused: ArModel;
  private unigram: Float32Array;
  private source: number[][];
  private adamStep = 1;
  private stages: StageName[];
  private stageCursor = 0;
  private stageStep = 0;
  private loss = 0;
  private acc = 0;
  history: HistoryPoint[] = [];
  private done = false;
  private cache: { idx: Int16Array; z: Float32Array } | null = null;
  private embedFrozen = false;

  private readonly h: Float32Array;
  private readonly residual: Float32Array;
  private readonly zq: Float32Array;

  constructor(corpus: Corpus, mode: ForgeMode, seed = 5) {
    this.corpus = corpus;
    this.mode = mode;
    this.context = corpus.train[0]!.length;
    this.vocabSize = corpus.vocab.length;
    this.dim = 24;
    this.codebook = Math.max(32, this.vocabSize + 4);
    this.hidden = 28;
    this.levels = levelsFor(this.context);
    this.rng = makeRng(seed);
    const stdE = 0.35;
    this.embed = param(this.vocabSize * this.dim, stdE, this.rng);
    this.books = this.levels.map((_, li) => {
      const std = li === this.levels.length - 1 ? 0.02 : 0.04;
      return param(this.codebook * this.dim, std, this.rng);
    });
    this.emaSum = this.books.map((book) => book.w.slice());
    this.emaCount = this.books.map(() => {
      const c = new Float32Array(this.codebook);
      c.fill(1);
      return c;
    });
    this.decW = param(this.dim * this.vocabSize, 0.02, this.rng);
    this.decB = param(this.vocabSize, 0, this.rng);
    this.tieDecoderToEmbed();
    this.copyEmbedsIntoFinest();
    this.gens = this.levels.slice(1).map((length) => ({
      pos: param(length * this.dim, 0.15, this.rng),
      outW: param(this.dim * this.codebook, 0.05, this.rng),
      outB: param(this.codebook, 0, this.rng),
    }));
    this.plain = this.makeAr(false);
    this.fused = this.makeAr(true);
    this.unigram = new Float32Array(this.codebook);
    this.source = corpus.train.map((row) => row.slice());
    this.stages =
      mode === "distill"
        ? ["teacher", "tokenizer", "concepts", "combine"]
        : mode === "scratch"
          ? ["tokenizer", "concepts", "teacher", "combine"]
          : ["tokenizer", "concepts", "teacher", "combine"];
    this.h = new Float32Array(B * this.context * this.dim);
    this.residual = new Float32Array(B * this.context * this.dim);
    this.zq = new Float32Array(B * this.context * this.dim);
  }

  stageSteps(stage: StageName): number {
    if (stage === "teacher") return this.mode === "distill" ? 160 : 120;
    if (stage === "tokenizer") return 80;
    if (stage === "concepts") return 200;
    return this.mode === "scratch" ? 60 : 120;
  }

  snapshot(): ForgeSnapshot {
    const stage = this.done ? "done" : this.stages[this.stageCursor]!;
    return {
      stage,
      stageStep: this.stageStep,
      stageSteps: stage === "done" ? 1 : this.stageSteps(stage),
      loss: this.loss,
      acc: this.acc,
      history: this.history,
      done: this.done,
    };
  }

  /** Run up to `steps` optimizer updates. Returns true when the recipe is finished. */
  tick(steps: number): boolean {
    for (let n = 0; n < steps && !this.done; n++) {
      const stage = this.stages[this.stageCursor]!;
      if (stage === "teacher") this.stepTeacher();
      else if (stage === "tokenizer") this.stepTokenizer();
      else if (stage === "concepts") this.stepConcepts();
      else this.stepCombine();
      this.stageStep += 1;
      this.adamStep += 1;
      if (this.stageStep % 4 === 0 || this.stageStep === 1) {
        this.history.push({
          stage,
          step: this.history.length,
          loss: this.loss,
          acc: this.acc,
        });
        if (this.history.length > 240) this.history.shift();
      }
      const limit = this.stageSteps(stage);
      if (this.stageStep >= limit) {
        if (this.mode === "distill" && stage === "teacher") this.adoptTeacherSamples();
        if (stage === "tokenizer") {
          this.freezeCodes();
          this.embedFrozen = true;
        }
        this.stageCursor += 1;
        this.stageStep = 0;
        if (this.stageCursor >= this.stages.length) this.done = true;
      }
    }
    return this.done;
  }

  metrics(): Metrics {
    const recon = this.scoreReconstruction(this.corpus.heldout);
    const concept = this.scoreConcepts(this.corpus.heldout);
    const ar = this.scoreAr(this.plain, this.corpus.heldout, null);
    const gens = Array.from({ length: 24 }, () => this.generateOne());
    const ncmValid = mean(gens.map((g) => (validTemplate(this.corpus, g.ncm) ? 1 : 0)));
    const arValid = mean(gens.map((g) => (validTemplate(this.corpus, g.ar) ? 1 : 0)));
    const hybridValid = mean(gens.map((g) => (validTemplate(this.corpus, g.hybrid) ? 1 : 0)));
    const held = this.corpus.heldout;
    let ncmToken = 0;
    let hybridToken = 0;
    let rewrites = 0;
    for (const row of held) {
      const ladder = this.reconstruct(row);
      const full = ladder[ladder.length - 1]!.ids;
      ncmToken += tokenAcc(row, full);
      const gen = this.generateFromSketch(row);
      hybridToken += tokenAcc(row, gen.hybrid);
      rewrites += gen.rewrites;
    }
    const nHeld = Math.max(1, held.length);
    const tokParams =
      this.plain.wh.w.length +
      this.plain.wx.w.length +
      this.plain.wo.w.length +
      this.plain.bh.w.length +
      this.plain.bo.w.length +
      this.plain.pos.w.length;
    const genParams = this.gens.reduce((s, g) => s + g.pos.w.length + g.outW.w.length + g.outB.w.length, 0);
    const tokParamsVq = this.embed.w.length + this.decW.w.length + this.decB.w.length + this.books.reduce((s, b) => s + b.w.length, 0);
    return {
      reconAcc: recon,
      conceptAcc: concept,
      arAcc: ar,
      ncmValid,
      arValid,
      hybridValid,
      ncmTokenAcc: ncmToken / nHeld,
      hybridTokenAcc: hybridToken / nHeld,
      avgRewrites: rewrites / nHeld,
      stepsAr: this.context,
      stepsNcm: this.levels.length,
      stepsHybrid: this.levels.length + rewrites / nHeld,
      paramsTokenizer: tokParamsVq,
      paramsGenerator: genParams,
      paramsToken: tokParams,
    };
  }

  samples(): SampleView[] {
    const views: SampleView[] = [];
    const vocab = this.corpus.vocab;
    const toTokens = (ids: number[]) => ids.map((id) => vocab[id] ?? "?");
    for (let i = 0; i < 2; i++) {
      const row = this.corpus.heldout[i] ?? this.corpus.train[i]!;
      const ladder = this.reconstruct(row);
      const draft = ladder[ladder.length - 1]!.ids;
      const gen = this.generateFromSketch(row);
      views.push({
        kind: "reconstruction",
        title: `Held-out ${i + 1}`,
        source: sentence(vocab, row),
        ladder: ladder.map((level) => ({
          level: level.level,
          length: level.length,
          tokens: toTokens(level.ids),
          acc: tokenAcc(row, level.ids),
        })),
        ncmTokens: toTokens(draft),
        arTokens: toTokens(gen.ar),
        hybridTokens: toTokens(gen.hybrid),
        repaired: gen.repaired,
        rewrites: gen.rewrites,
      });
    }
    for (let i = 0; i < 3; i++) {
      const gen = this.generateOne();
      const ladder = this.reconstructCodes(gen.codes);
      views.push({
        kind: "generation",
        title: `Sample ${i + 1}`,
        source: "Drawn coarse to fine",
        ladder: ladder.map((level) => ({
          level: level.level,
          length: level.length,
          tokens: toTokens(level.ids),
          acc: -1,
        })),
        ncmTokens: toTokens(gen.ncm),
        arTokens: toTokens(gen.ar),
        hybridTokens: toTokens(gen.hybrid),
        repaired: gen.repaired,
        rewrites: gen.rewrites,
      });
    }
    return views;
  }

  reconstructLine(text: string): { error?: string; ladder?: SampleView["ladder"]; source?: string } {
    const parts = text
      .trim()
      .replace(/([.])$/g, " $1")
      .split(/\s+/)
      .filter(Boolean);
    if (parts.length !== this.context) {
      return { error: `Use exactly ${this.context} tokens, the width this run was trained on.` };
    }
    const ids: number[] = [];
    for (const word of parts) {
      const id = this.corpus.vocab.indexOf(word);
      if (id < 0) return { error: `"${word}" is outside this run's vocabulary.` };
      ids.push(id);
    }
    const ladder = this.reconstruct(ids);
    return {
      source: sentence(this.corpus.vocab, ids),
      ladder: ladder.map((level) => ({
        level: level.level,
        length: level.length,
        tokens: level.ids.map((id) => this.corpus.vocab[id] ?? "?"),
        acc: tokenAcc(ids, level.ids),
      })),
    };
  }

  checkpoint() {
    const pack = (p: Param) => Array.from(p.w);
    return {
      format: "ncm-forge-lab-v1",
      mode: this.mode,
      vocab: this.corpus.vocab,
      levels: this.levels,
      dim: this.dim,
      codebook: this.codebook,
      hidden: this.hidden,
      context: this.context,
      tokenizer: {
        embed: pack(this.embed),
        codebooks: this.books.map(pack),
        decoderW: pack(this.decW),
        decoderB: pack(this.decB),
      },
      generator: this.gens.map((g) => ({
        pos: pack(g.pos),
        outW: pack(g.outW),
        outB: pack(g.outB),
      })),
      unigram: Array.from(this.unigram),
      tokenModel: {
        plain: packAr(this.plain),
        fused: packAr(this.fused),
      },
    };
  }

  private makeAr(fused: boolean): ArModel {
    const d = this.dim;
    const h = this.hidden;
    const v = this.vocabSize;
    return {
      wh: param(h * h, 0.04, this.rng),
      wx: param(h * d, 0.2, this.rng),
      wc: param(h * d, fused ? 0.12 : 0, this.rng),
      wo: param(h * v, 0.05, this.rng),
      bh: param(h, 0, this.rng),
      bo: param(v, 0, this.rng),
      pos: param(this.context * v, 0, this.rng),
    };
  }

  private tieDecoderToEmbed() {
    const { dim: d, vocabSize: v } = this;
    for (let vi = 0; vi < v; vi++) {
      for (let di = 0; di < d; di++) {
        this.decW.w[di * v + vi] = this.embed.w[vi * d + di]! * 4;
      }
    }
  }

  private copyEmbedsIntoFinest() {
    const last = this.books[this.books.length - 1]!;
    const n = Math.min(this.vocabSize, this.codebook);
    for (let i = 0; i < n * this.dim; i++) last.w[i] = this.embed.w[i]!;
    this.emaSum[this.emaSum.length - 1] = last.w.slice();
  }

  private batchFrom(data: number[][]): number[][] {
    const out: number[][] = [];
    for (let i = 0; i < B; i++) out.push(data[Math.floor(this.rng() * data.length)]!);
    return out;
  }

  private stepTeacher() {
    const batch = this.batchFrom(this.corpus.train);
    const scored = this.arLoss(this.plain, batch, null, true);
    this.loss = scored.loss;
    this.acc = scored.acc;
    this.applyAr(this.plain, 0.03, false);
  }

  private stepTokenizer() {
    const batch = this.batchFrom(this.source);
    const keep =
      this.stageStep < 40 || this.rng() > 0.28
        ? this.levels.length
        : 1 + Math.floor(this.rng() * this.levels.length);
    const scored = this.vqLoss(batch, keep, true);
    this.loss = scored.loss;
    this.acc = scored.acc;
    clipAndAdam(this.embed, 0.004, this.adamStep);
    clipAndAdam(this.decW, 0.02, this.adamStep);
    clipAndAdam(this.decB, 0.02, this.adamStep);
  }

  private stepConcepts() {
    if (!this.cache) this.freezeCodes();
    const scored = this.conceptLoss(true);
    this.loss = scored.loss;
    this.acc = scored.acc;
    for (const gen of this.gens) {
      clipAndAdam(gen.pos, 0.03, this.adamStep);
      clipAndAdam(gen.outW, 0.03, this.adamStep);
      clipAndAdam(gen.outB, 0.03, this.adamStep);
    }
  }

  private stepCombine() {
    if (!this.cache) this.freezeCodes();
    const cache = this.cache;
    if (!cache) return;
    const batchIdx = this.sampleCacheIndices();
    const tokens: number[][] = [];
    const sketches: Float32Array[] = [];
    for (const index of batchIdx) {
      tokens.push(Array.from(this.source[index]!));
      if (this.rng() < 0.45) {
        const codes = this.codesFromCache(index);
        const predicted = this.predictCodes(codes, true);
        sketches.push(this.sketchFromCodes(predicted));
      } else {
        sketches.push(cache.z.subarray(index * this.context * this.dim, (index + 1) * this.context * this.dim));
      }
    }
    const scored = this.arLoss(this.fused, tokens, sketches, true);
    this.loss = scored.loss;
    this.acc = scored.acc;
    this.applyAr(this.fused, 0.025, true);
  }

  private applyAr(ar: ArModel, lr: number, fused: boolean) {
    clipAndAdam(ar.wh, lr, this.adamStep);
    clipAndAdam(ar.wx, lr, this.adamStep);
    if (fused) clipAndAdam(ar.wc, lr, this.adamStep);
    clipAndAdam(ar.wo, lr, this.adamStep);
    clipAndAdam(ar.bh, lr, this.adamStep);
    clipAndAdam(ar.bo, lr, this.adamStep);
    clipAndAdam(ar.pos, lr, this.adamStep);
    zeroGrad(this.embed);
  }

  private adoptTeacherSamples() {
    const fresh: number[][] = [];
    const want = this.corpus.train.length;
    let guard = 0;
    while (fresh.length < want && guard < want * 8) {
      guard += 1;
      const ids = this.sampleAr(this.plain, null, 0.7);
      if (validTemplate(this.corpus, ids) || this.corpus.slots === null) fresh.push(ids);
    }
    if (fresh.length >= 8) this.source = fresh;
  }

  private freezeCodes() {
    const n = this.source.length;
    const tLen = this.context;
    const idx = new Int16Array(n * this.levels.length * tLen);
    const z = new Float32Array(n * tLen * this.dim);
    this.unigram.fill(0);
    for (let s = 0; s < n; s++) {
      const encoded = this.encodeOne(this.source[s]!);
      for (let li = 0; li < this.levels.length; li++) {
        const codes = encoded.indices[li]!;
        for (let p = 0; p < codes.length; p++) {
          idx[s * this.levels.length * tLen + li * tLen + p] = codes[p]!;
        }
      }
      this.unigram[encoded.indices[0]![0]!] += 1;
      z.set(encoded.z, s * tLen * this.dim);
    }
    const uniSum = this.unigram.reduce((a, b) => a + b, 0) || 1;
    for (let i = 0; i < this.unigram.length; i++) this.unigram[i] = this.unigram[i]! / uniSum;
    this.cache = { idx, z };
  }

  private sampleCacheIndices(): number[] {
    const n = this.source.length;
    const out: number[] = [];
    for (let i = 0; i < B; i++) out.push(Math.floor(this.rng() * n));
    return out;
  }

  private codesFromCache(sample: number): number[][] {
    const tLen = this.context;
    return this.levels.map((length, li) => {
      const codes: number[] = [];
      for (let p = 0; p < length; p++) {
        codes.push(this.cache!.idx[sample * this.levels.length * tLen + li * tLen + p]!);
      }
      return codes;
    });
  }

  private vqLoss(batch: number[][], keep: number, train: boolean): { loss: number; acc: number } {
    const { dim: d, context: tLen, vocabSize: v } = this;
    zeroGrad(this.embed);
    zeroGrad(this.decW);
    zeroGrad(this.decB);
    for (const book of this.books) zeroGrad(book);

    this.embedBatch(batch);
    this.residual.set(this.h);
    const ups: Float32Array[] = [];
    const pooledList: Float32Array[] = [];
    const idxList: Int16Array[] = [];
    let commit = 0;
    const usedKeep = Math.max(1, Math.min(keep, this.levels.length));

    for (let li = 0; li < usedKeep; li++) {
      const length = this.levels[li]!;
      const bins = binsFor(tLen, length);
      const pooled = new Float32Array(B * length * d);
      const idx = new Int16Array(B * length);
      for (let b = 0; b < B; b++) {
        for (let i = 0; i < length; i++) {
          const [start, end] = bins[i]!;
          const span = end - start;
          for (let di = 0; di < d; di++) {
            let sum = 0;
            for (let t = start; t < end; t++) sum += this.residual[(b * tLen + t) * d + di]!;
            pooled[(b * length + i) * d + di] = sum / span;
          }
        }
      }
      this.quantizeInto(li, pooled, B, length, idx);
      const up = new Float32Array(B * tLen * d);
      for (let b = 0; b < B; b++) {
        for (let i = 0; i < length; i++) {
          const code = idx[b * length + i]!;
          const [start, end] = bins[i]!;
          for (let t = start; t < end; t++) {
            for (let di = 0; di < d; di++) {
              const q = this.books[li]!.w[code * d + di]!;
              up[(b * tLen + t) * d + di] = q;
              this.residual[(b * tLen + t) * d + di]! -= q;
            }
          }
        }
      }
      let local = 0;
      const denom = B * length * d;
      for (let i = 0; i < pooled.length; i++) {
        const diff = pooled[i]! - upValue(this.books[li]!.w, idx, i, d, length);
        local += diff * diff;
      }
      commit += local / denom;
      if (train) this.emaUpdate(li, pooled, idx, B, length);
      ups.push(up);
      pooledList.push(pooled);
      idxList.push(idx);
    }

    this.zq.fill(0);
    for (const up of ups) {
      for (let i = 0; i < this.zq.length; i++) this.zq[i]! += up[i]!;
    }

    const logits = new Float32Array(B * tLen * v);
    const probs = new Float32Array(B * tLen * v);
    linear(this.zq, this.decW.w, this.decB.w, logits, B * tLen, d, v);
    softmaxRows(logits, B * tLen, v, probs);
    let loss = 0;
    let correct = 0;
    const dLogits = new Float32Array(B * tLen * v);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < tLen; t++) {
        const target = batch[b]![t]!;
        const base = (b * tLen + t) * v;
        loss += -Math.log(Math.max(probs[base + target]!, 1e-8));
        let best = 0;
        let bestP = -1;
        for (let vi = 0; vi < v; vi++) {
          const p = probs[base + vi]!;
          dLogits[base + vi] = p;
          if (p > bestP) {
            bestP = p;
            best = vi;
          }
        }
        dLogits[base + target]! -= 1;
        if (best === target) correct += 1;
      }
    }
    const denomTok = B * tLen;
    loss = loss / denomTok + 0.25 * (commit / usedKeep);
    if (train) {
      for (let i = 0; i < dLogits.length; i++) dLogits[i]! /= denomTok;
      const dZ = new Float32Array(B * tLen * d);
      linearBackward(this.zq, this.decW, this.decB, dLogits, dZ, B * tLen, d, v);
      const dRes = new Float32Array((usedKeep + 1) * B * tLen * d);
      // dRes[usedKeep] stays 0; z = sum ups; residual chain res_{i+1} = res_i - up_i
      for (let li = usedKeep - 1; li >= 0; li--) {
        const length = this.levels[li]!;
        const bins = binsFor(tLen, length);
        const dUp = new Float32Array(B * tLen * d);
        const next = dRes.subarray((li + 1) * B * tLen * d, (li + 2) * B * tLen * d);
        for (let i = 0; i < dUp.length; i++) dUp[i] = dZ[i]! - next[i]!;
        const dPooled = new Float32Array(B * length * d);
        for (let b = 0; b < B; b++) {
          for (let i = 0; i < length; i++) {
            const [start, end] = bins[i]!;
            for (let t = start; t < end; t++) {
              for (let di = 0; di < d; di++) {
                dPooled[(b * length + i) * d + di]! += dUp[(b * tLen + t) * d + di]!;
              }
            }
          }
        }
        const pooled = pooledList[li]!;
        const idx = idxList[li]!;
        const commitScale = (0.25 * 2) / (usedKeep * B * length * d);
        for (let b = 0; b < B; b++) {
          for (let i = 0; i < length; i++) {
            const code = idx[b * length + i]!;
            for (let di = 0; di < d; di++) {
              const q = this.books[li]!.w[code * d + di]!;
              const p = pooled[(b * length + i) * d + di]!;
              dPooled[(b * length + i) * d + di]! += commitScale * (p - q);
            }
          }
        }
        const dResidual = dRes.subarray(li * B * tLen * d, (li + 1) * B * tLen * d);
        dResidual.set(next);
        for (let b = 0; b < B; b++) {
          for (let i = 0; i < length; i++) {
            const [start, end] = bins[i]!;
            const span = end - start;
            for (let t = start; t < end; t++) {
              for (let di = 0; di < d; di++) {
                dResidual[(b * tLen + t) * d + di]! += dPooled[(b * length + i) * d + di]! / span;
              }
            }
          }
        }
      }
      const dH = dRes.subarray(0, B * tLen * d);
      for (let b = 0; b < B; b++) {
        for (let t = 0; t < tLen; t++) {
          const token = batch[b]![t]!;
          for (let di = 0; di < d; di++) {
            this.embed.g[token * d + di]! += dH[(b * tLen + t) * d + di]!;
          }
        }
      }
    }
    return { loss, acc: correct / denomTok };
  }

  private quantizeInto(level: number, pooled: Float32Array, batch: number, length: number, idx: Int16Array) {
    const d = this.dim;
    const book = this.books[level]!.w;
    const k = this.codebook;
    const c2 = new Float32Array(k);
    for (let c = 0; c < k; c++) {
      let s = 0;
      for (let di = 0; di < d; di++) {
        const v = book[c * d + di]!;
        s += v * v;
      }
      c2[c] = s;
    }
    for (let b = 0; b < batch; b++) {
      for (let i = 0; i < length; i++) {
        let best = 0;
        let bestScore = -Infinity;
        const base = (b * length + i) * d;
        for (let c = 0; c < k; c++) {
          let dot = 0;
          for (let di = 0; di < d; di++) dot += pooled[base + di]! * book[c * d + di]!;
          const score = dot - 0.5 * c2[c]!;
          if (score > bestScore) {
            bestScore = score;
            best = c;
          }
        }
        idx[b * length + i] = best;
      }
    }
  }

  private emaUpdate(level: number, pooled: Float32Array, idx: Int16Array, batch: number, length: number) {
    const d = this.dim;
    const decay = 0.96;
    const book = this.books[level]!.w;
    const sum = this.emaSum[level]!;
    const count = this.emaCount[level]!;
    const seen = new Uint8Array(this.codebook);
    for (let c = 0; c < this.codebook; c++) {
      count[c] = count[c]! * decay;
      for (let di = 0; di < d; di++) sum[c * d + di] = sum[c * d + di]! * decay;
    }
    for (let b = 0; b < batch; b++) {
      for (let i = 0; i < length; i++) {
        const code = idx[b * length + i]!;
        seen[code] = 1;
        count[code] = count[code]! + (1 - decay);
        for (let di = 0; di < d; di++) {
          sum[code * d + di]! += (1 - decay) * pooled[(b * length + i) * d + di]!;
        }
      }
    }
    for (let c = 0; c < this.codebook; c++) {
      const n = Math.max(count[c]!, 1e-4);
      for (let di = 0; di < d; di++) book[c * d + di] = sum[c * d + di]! / n;
    }
    for (let c = 0; c < this.codebook; c++) {
      if (seen[c]) continue;
      if (this.rng() > 0.15) continue;
      const b = Math.floor(this.rng() * batch);
      const i = Math.floor(this.rng() * length);
      for (let di = 0; di < d; di++) {
        const v = pooled[(b * length + i) * d + di]!;
        book[c * d + di] = v;
        sum[c * d + di] = v;
      }
      count[c] = 1;
    }
  }

  private embedBatch(batch: number[][]) {
    const d = this.dim;
    const tLen = this.context;
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < tLen; t++) {
        const token = batch[b]![t]!;
        for (let di = 0; di < d; di++) this.h[(b * tLen + t) * d + di] = this.embed.w[token * d + di]!;
      }
    }
  }

  private conceptLoss(train: boolean): { loss: number; acc: number } {
    for (const gen of this.gens) {
      zeroGrad(gen.pos);
      zeroGrad(gen.outW);
      zeroGrad(gen.outB);
    }
    const picks = this.sampleCacheIndices();
    let loss = 0;
    let correct = 0;
    let total = 0;
    for (const index of picks) {
      const original = this.codesFromCache(index);
      for (let li = 1; li < this.levels.length; li++) {
        const cond = original.map((row) => row.slice());
        for (let c = 0; c < li; c++) {
          const row = cond[c]!;
          if (this.rng() < 0.12) {
            const at = Math.floor(this.rng() * row.length);
            row[at] = Math.floor(this.rng() * this.codebook);
          }
        }
        const pred = this.forwardLevel(li, cond, original[li]!, train);
        loss += pred.loss;
        correct += pred.correct;
        total += original[li]!.length;
      }
    }
    return { loss: loss / Math.max(1, picks.length), acc: correct / Math.max(1, total) };
  }

  private forwardLevel(
    level: number,
    codes: number[][],
    target: number[],
    train: boolean,
  ): { loss: number; correct: number } {
    const d = this.dim;
    const gen = this.gens[level - 1]!;
    const length = this.levels[level]!;
    const memory: number[] = [];
    const memLevel: number[] = [];
    for (let li = 0; li < level; li++) {
      for (const code of codes[li]!) {
        memory.push(code);
        memLevel.push(li);
      }
    }
    const s = memory.length;
    const k = this.codebook;
    const scores = new Float32Array(length * s);
    const scale = 1 / Math.sqrt(d);
    for (let q = 0; q < length; q++) {
      for (let m = 0; m < s; m++) {
        let dot = 0;
        const code = memory[m]!;
        const book = this.books[memLevel[m]!]!.w;
        for (let di = 0; di < d; di++) {
          dot += gen.pos.w[q * d + di]! * book[code * d + di]!;
        }
        scores[q * s + m] = dot * scale;
      }
    }
    const attn = new Float32Array(length * s);
    softmaxRows(scores, length, s, attn);
    const ctx = new Float32Array(length * d);
    for (let q = 0; q < length; q++) {
      for (let m = 0; m < s; m++) {
        const a = attn[q * s + m]!;
        const book = this.books[memLevel[m]!]!.w;
        const code = memory[m]!;
        for (let di = 0; di < d; di++) ctx[q * d + di]! += a * book[code * d + di]!;
      }
    }
    const h = new Float32Array(length * d);
    for (let q = 0; q < length; q++) {
      for (let di = 0; di < d; di++) h[q * d + di] = gen.pos.w[q * d + di]! + ctx[q * d + di]!;
    }
    const logits = new Float32Array(length * k);
    linear(h, gen.outW.w, gen.outB.w, logits, length, d, k);
    const probs = new Float32Array(length * k);
    softmaxRows(logits, length, k, probs);
    let loss = 0;
    let correct = 0;
    const dLogits = new Float32Array(length * k);
    for (let q = 0; q < length; q++) {
      const y = target[q]!;
      loss += -Math.log(Math.max(probs[q * k + y]!, 1e-8));
      let best = 0;
      let bestP = -1;
      for (let c = 0; c < k; c++) {
        const p = probs[q * k + c]!;
        dLogits[q * k + c] = p;
        if (p > bestP) {
          bestP = p;
          best = c;
        }
      }
      dLogits[q * k + y]! -= 1;
      if (best === y) correct += 1;
    }
    loss /= length;
    if (train) {
      for (let i = 0; i < dLogits.length; i++) dLogits[i]! /= length;
      const dH = new Float32Array(length * d);
      linearBackward(h, gen.outW, gen.outB, dLogits, dH, length, d, k);
      const dCtx = new Float32Array(length * d);
      for (let i = 0; i < dH.length; i++) {
        gen.pos.g[i]! += dH[i]!;
        dCtx[i] = dH[i]!;
      }
      const dAttn = new Float32Array(length * s);
      for (let q = 0; q < length; q++) {
        for (let m = 0; m < s; m++) {
          const book = this.books[memLevel[m]!]!.w;
          const code = memory[m]!;
          let dot = 0;
          for (let di = 0; di < d; di++) dot += dCtx[q * d + di]! * book[code * d + di]!;
          dAttn[q * s + m] = dot;
        }
      }
      const dScores = new Float32Array(length * s);
      for (let q = 0; q < length; q++) {
        let dot = 0;
        for (let m = 0; m < s; m++) dot += attn[q * s + m]! * dAttn[q * s + m]!;
        for (let m = 0; m < s; m++) {
          dScores[q * s + m] = attn[q * s + m]! * (dAttn[q * s + m]! - dot);
        }
      }
      for (let q = 0; q < length; q++) {
        for (let m = 0; m < s; m++) {
          const book = this.books[memLevel[m]!]!.w;
          const code = memory[m]!;
          const g = dScores[q * s + m]! * scale;
          for (let di = 0; di < d; di++) gen.pos.g[q * d + di]! += g * book[code * d + di]!;
        }
      }
    }
    return { loss, correct };
  }

  private predictCodes(trueCodes: number[][], greedy: boolean): number[][] {
    const out: number[][] = [sampleUnigram(this.unigram, this.rng)];
    // For teacher-forced repair sketches we still roll out the generator
    // from the true coarse code so the sketch matches inference.
    out[0] = trueCodes[0]!.slice();
    for (let li = 1; li < this.levels.length; li++) {
      out.push(this.sampleLevel(li, out, greedy ? 0 : 0.8));
    }
    return out;
  }

  private sampleLevel(level: number, prev: number[][], temp: number): number[] {
    const d = this.dim;
    const gen = this.gens[level - 1]!;
    const length = this.levels[level]!;
    const memory: number[] = [];
    const memLevel: number[] = [];
    for (let li = 0; li < level; li++) {
      for (const code of prev[li]!) {
        memory.push(code);
        memLevel.push(li);
      }
    }
    const s = memory.length;
    const k = this.codebook;
    const ids: number[] = [];
    for (let q = 0; q < length; q++) {
      const scores = new Float32Array(s);
      const scale = 1 / Math.sqrt(d);
      for (let m = 0; m < s; m++) {
        let dot = 0;
        const book = this.books[memLevel[m]!]!.w;
        const code = memory[m]!;
        for (let di = 0; di < d; di++) dot += gen.pos.w[q * d + di]! * book[code * d + di]!;
        scores[m] = dot * scale;
      }
      const attn = new Float32Array(s);
      softmaxRows(scores, 1, s, attn);
      const h = new Float32Array(d);
      for (let di = 0; di < d; di++) h[di] = gen.pos.w[q * d + di]!;
      for (let m = 0; m < s; m++) {
        const book = this.books[memLevel[m]!]!.w;
        const code = memory[m]!;
        for (let di = 0; di < d; di++) h[di]! += attn[m]! * book[code * d + di]!;
      }
      const logits = new Float32Array(k);
      linear(h, gen.outW.w, gen.outB.w, logits, 1, d, k);
      ids.push(sampleLogits(logits, temp <= 0 ? 0 : temp, this.rng));
    }
    return ids;
  }

  private arLoss(
    ar: ArModel,
    batch: number[][],
    sketches: Float32Array[] | null,
    train: boolean,
  ): { loss: number; acc: number } {
    const d = this.dim;
    const h = this.hidden;
    const v = this.vocabSize;
    const tLen = this.context;
    zeroGrad(ar.wh);
    zeroGrad(ar.wx);
    zeroGrad(ar.wc);
    zeroGrad(ar.wo);
    zeroGrad(ar.bh);
    zeroGrad(ar.bo);
    zeroGrad(ar.pos);
    if (train) zeroGrad(this.embed);
    let loss = 0;
    let correct = 0;
    const states: Float32Array[] = [];
    const inputs: Float32Array[] = [];
    const dLogitsAll: Float32Array[] = [];
    for (let b = 0; b < batch.length; b++) {
      const row = batch[b]!;
      const sketch = sketches ? sketches[b]! : null;
      let prev = this.corpus.bos;
      const hs = new Float32Array((tLen + 1) * h);
      const xs = new Float32Array(tLen * d);
      const dLogits = new Float32Array(tLen * v);
      for (let t = 0; t < tLen; t++) {
        for (let di = 0; di < d; di++) xs[t * d + di] = this.embed.w[prev * d + di]!;
        const hPrev = hs.subarray(t * h, (t + 1) * h);
        const hNow = hs.subarray((t + 1) * h, (t + 2) * h);
        rnnStep(ar, hPrev, xs.subarray(t * d, (t + 1) * d), sketch ? sketch.subarray(t * d, (t + 1) * d) : null, hNow);
        const logits = new Float32Array(v);
        linear(hNow, ar.wo.w, ar.bo.w, logits, 1, h, v);
        for (let vi = 0; vi < v; vi++) logits[vi]! += ar.pos.w[t * v + vi]!;
        const probs = new Float32Array(v);
        softmaxRows(logits, 1, v, probs);
        const y = row[t]!;
        loss += -Math.log(Math.max(probs[y]!, 1e-8));
        let best = 0;
        let bestP = -1;
        for (let vi = 0; vi < v; vi++) {
          dLogits[t * v + vi] = probs[vi]!;
          if (probs[vi]! > bestP) {
            bestP = probs[vi]!;
            best = vi;
          }
        }
        dLogits[t * v + y]! -= 1;
        if (best === y) correct += 1;
        prev = y;
      }
      states.push(hs);
      inputs.push(xs);
      dLogitsAll.push(dLogits);
    }
    const denom = batch.length * tLen;
    if (train) {
      for (let b = 0; b < batch.length; b++) {
        const dLogits = dLogitsAll[b]!;
        for (let i = 0; i < dLogits.length; i++) dLogits[i]! /= denom;
        const hs = states[b]!;
        const xs = inputs[b]!;
        const sketch = sketches ? sketches[b]! : null;
        const dH = new Float32Array(h);
        for (let t = tLen - 1; t >= 0; t--) {
          const hNow = hs.subarray((t + 1) * h, (t + 2) * h);
          const hPrev = hs.subarray(t * h, (t + 1) * h);
          const x = xs.subarray(t * d, (t + 1) * d);
          const sk = sketch ? sketch.subarray(t * d, (t + 1) * d) : null;
          const dLog = dLogits.subarray(t * v, (t + 1) * v);
          for (let hi = 0; hi < h; hi++) {
            let g = 0;
            for (let vi = 0; vi < v; vi++) g += dLog[vi]! * ar.wo.w[hi * v + vi]!;
            dH[hi] = dH[hi]! + g;
          }
          for (let hi = 0; hi < h; hi++) {
            for (let vi = 0; vi < v; vi++) ar.wo.g[hi * v + vi]! += hNow[hi]! * dLog[vi]!;
          }
          for (let vi = 0; vi < v; vi++) ar.bo.g[vi]! += dLog[vi]!;
          for (let vi = 0; vi < v; vi++) ar.pos.g[t * v + vi]! += dLog[vi]!;
          const dPre = new Float32Array(h);
          for (let hi = 0; hi < h; hi++) dPre[hi] = (1 - hNow[hi]! * hNow[hi]!) * dH[hi]!;
          for (let hi = 0; hi < h; hi++) {
            ar.bh.g[hi]! += dPre[hi]!;
            for (let hj = 0; hj < h; hj++) ar.wh.g[hi * h + hj]! += dPre[hi]! * hPrev[hj]!;
            for (let di = 0; di < d; di++) {
              ar.wx.g[hi * d + di]! += dPre[hi]! * x[di]!;
              if (sk) ar.wc.g[hi * d + di]! += dPre[hi]! * sk[di]!;
            }
          }
          const dPrev = new Float32Array(h);
          const dX = new Float32Array(d);
          for (let hj = 0; hj < h; hj++) {
            for (let hi = 0; hi < h; hi++) dPrev[hj]! += ar.wh.w[hi * h + hj]! * dPre[hi]!;
          }
          for (let di = 0; di < d; di++) {
            for (let hi = 0; hi < h; hi++) dX[di]! += ar.wx.w[hi * d + di]! * dPre[hi]!;
          }
          dH.set(dPrev);
          const prevTok = t === 0 ? this.corpus.bos : batch[b]![t - 1]!;
          for (let di = 0; di < d; di++) this.embed.g[prevTok * d + di]! += dX[di]!;
        }
      }
      if (!this.embedFrozen) clipAndAdam(this.embed, 0.008, this.adamStep);
    }
    return { loss: loss / denom, acc: correct / denom };
  }

  private sampleAr(ar: ArModel, sketch: Float32Array | null, temp: number): number[] {
    const d = this.dim;
    const h = this.hidden;
    const tLen = this.context;
    const v = this.vocabSize;
    let prev = this.corpus.bos;
    const state = new Float32Array(h);
    const ids: number[] = [];
    for (let t = 0; t < tLen; t++) {
      const x = this.embed.w.subarray(prev * d, (prev + 1) * d);
      const sk = sketch ? sketch.subarray(t * d, (t + 1) * d) : null;
      const next = new Float32Array(h);
      rnnStep(ar, state, x, sk, next);
      state.set(next);
      const logits = new Float32Array(v);
      linear(next, ar.wo.w, ar.bo.w, logits, 1, h, v);
      for (let vi = 0; vi < v; vi++) logits[vi]! += ar.pos.w[t * v + vi]!;
      const id = sampleLogits(logits, temp, this.rng);
      ids.push(id);
      prev = id;
    }
    return ids;
  }

  private hybridDecode(sketch: Float32Array, draft: number[], tau: number): { ids: number[]; rewrites: number; repaired: boolean[] } {
    const d = this.dim;
    const h = this.hidden;
    const tLen = this.context;
    const v = this.vocabSize;
    const conf = this.tokenConfidence(draft, sketch);
    let prev = this.corpus.bos;
    const state = new Float32Array(h);
    const ids: number[] = [];
    const repaired: boolean[] = [];
    let rewrites = 0;
    for (let t = 0; t < tLen; t++) {
      const x = this.embed.w.subarray(prev * d, (prev + 1) * d);
      const next = new Float32Array(h);
      rnnStep(this.fused, state, x, sketch.subarray(t * d, (t + 1) * d), next);
      state.set(next);
      let id = draft[t]!;
      const uncertain = conf[t]! < tau;
      if (uncertain) {
        const logits = new Float32Array(v);
        linear(next, this.fused.wo.w, this.fused.bo.w, logits, 1, h, v);
        id = sampleLogits(logits, 0, this.rng);
        if (id !== draft[t]) rewrites += 1;
      }
      repaired.push(uncertain && id !== draft[t]);
      ids.push(id);
      prev = id;
    }
    return { ids, rewrites, repaired };
  }

  private tokenConfidence(ids: number[], sketch: Float32Array): Float32Array {
    const { dim: d, context: tLen, vocabSize: v } = this;
    const z = sketch;
    const logits = new Float32Array(tLen * v);
    linear(z, this.decW.w, this.decB.w, logits, tLen, d, v);
    const probs = new Float32Array(tLen * v);
    softmaxRows(logits, tLen, v, probs);
    const conf = new Float32Array(tLen);
    for (let t = 0; t < tLen; t++) conf[t] = probs[t * v + ids[t]!]!;
    return conf;
  }

  private encodeOne(ids: number[]): { indices: number[][]; z: Float32Array } {
    const d = this.dim;
    const tLen = this.context;
    const h = new Float32Array(tLen * d);
    for (let t = 0; t < tLen; t++) {
      const token = ids[t]!;
      for (let di = 0; di < d; di++) h[t * d + di] = this.embed.w[token * d + di]!;
    }
    const residual = h.slice();
    const indices: number[][] = [];
    const z = new Float32Array(tLen * d);
    for (let li = 0; li < this.levels.length; li++) {
      const length = this.levels[li]!;
      const bins = binsFor(tLen, length);
      const pooled = new Float32Array(length * d);
      for (let i = 0; i < length; i++) {
        const [start, end] = bins[i]!;
        const span = end - start;
        for (let di = 0; di < d; di++) {
          let sum = 0;
          for (let t = start; t < end; t++) sum += residual[t * d + di]!;
          pooled[i * d + di] = sum / span;
        }
      }
      const idx = new Int16Array(length);
      this.quantizeInto(li, pooled, 1, length, idx);
      const codes: number[] = [];
      for (let i = 0; i < length; i++) {
        const code = idx[i]!;
        codes.push(code);
        const [start, end] = bins[i]!;
        for (let t = start; t < end; t++) {
          for (let di = 0; di < d; di++) {
            const q = this.books[li]!.w[code * d + di]!;
            residual[t * d + di]! -= q;
            z[t * d + di]! += q;
          }
        }
      }
      indices.push(codes);
    }
    return { indices, z };
  }

  private decodeZ(z: Float32Array): number[] {
    const tLen = this.context;
    const v = this.vocabSize;
    const d = this.dim;
    const logits = new Float32Array(tLen * v);
    linear(z, this.decW.w, this.decB.w, logits, tLen, d, v);
    const ids: number[] = [];
    for (let t = 0; t < tLen; t++) {
      let best = 0;
      let bestV = -Infinity;
      for (let vi = 0; vi < v; vi++) {
        const value = logits[t * v + vi]!;
        if (value > bestV) {
          bestV = value;
          best = vi;
        }
      }
      ids.push(best);
    }
    return ids;
  }

  private sketchFromCodes(codes: number[][]): Float32Array {
    const d = this.dim;
    const tLen = this.context;
    const z = new Float32Array(tLen * d);
    for (let li = 0; li < this.levels.length; li++) {
      const length = this.levels[li]!;
      const bins = binsFor(tLen, length);
      for (let i = 0; i < length; i++) {
        const code = codes[li]![i]!;
        const [start, end] = bins[i]!;
        for (let t = start; t < end; t++) {
          for (let di = 0; di < d; di++) z[t * d + di]! += this.books[li]!.w[code * d + di]!;
        }
      }
    }
    return z;
  }

  private reconstruct(ids: number[]): { level: number; length: number; ids: number[] }[] {
    const encoded = this.encodeOne(ids);
    return this.reconstructCodes(encoded.indices);
  }

  private reconstructCodes(codes: number[][]): { level: number; length: number; ids: number[] }[] {
    const d = this.dim;
    const tLen = this.context;
    const z = new Float32Array(tLen * d);
    const ladder: { level: number; length: number; ids: number[] }[] = [];
    for (let li = 0; li < codes.length; li++) {
      const length = this.levels[li]!;
      const bins = binsFor(tLen, length);
      for (let i = 0; i < length; i++) {
        const code = codes[li]![i]!;
        const [start, end] = bins[i]!;
        for (let t = start; t < end; t++) {
          for (let di = 0; di < d; di++) z[t * d + di]! += this.books[li]!.w[code * d + di]!;
        }
      }
      ladder.push({ level: li + 1, length, ids: this.decodeZ(z) });
    }
    return ladder;
  }

  private scoreReconstruction(rows: number[][]): number {
    let acc = 0;
    for (const row of rows) {
      const ladder = this.reconstruct(row);
      acc += tokenAcc(row, ladder[ladder.length - 1]!.ids);
    }
    return acc / Math.max(1, rows.length);
  }

  private scoreConcepts(rows: number[][]): number {
    let correct = 0;
    let total = 0;
    for (const row of rows) {
      const codes = this.encodeOne(row).indices;
      for (let li = 1; li < this.levels.length; li++) {
        const pred = this.sampleLevel(li, codes, 0);
        const target = codes[li]!;
        for (let i = 0; i < target.length; i++) {
          total += 1;
          if (pred[i] === target[i]) correct += 1;
        }
      }
    }
    return correct / Math.max(1, total);
  }

  private scoreAr(ar: ArModel, rows: number[][], sketches: Float32Array[] | null): number {
    const take = rows.slice(0, 16);
    const sk = sketches ? sketches.slice(0, take.length) : null;
    return this.arLoss(ar, take, sk, false).acc;
  }

  private generateOne(): {
    ncm: number[];
    ar: number[];
    hybrid: number[];
    rewrites: number;
    repaired: boolean[];
    codes: number[][];
  } {
    const codes: number[][] = [sampleUnigram(this.unigram, this.rng)];
    for (let li = 1; li < this.levels.length; li++) codes.push(this.sampleLevel(li, codes, 0.55));
    const sketch = this.sketchFromCodes(codes);
    const ncm = this.decodeZ(sketch);
    const ar = this.sampleAr(this.plain, null, 0.55);
    const hybrid = this.hybridDecode(sketch, ncm, 0.55);
    return { ncm, ar, hybrid: hybrid.ids, rewrites: hybrid.rewrites, repaired: hybrid.repaired, codes };
  }

  private generateFromSketch(row: number[]): { ar: number[]; hybrid: number[]; rewrites: number; repaired: boolean[] } {
    const encoded = this.encodeOne(row);
    const draft = this.decodeZ(encoded.z);
    const hybrid = this.hybridDecode(encoded.z, draft, 0.55);
    const ar = this.sampleAr(this.plain, null, 0.35);
    return { ar, hybrid: hybrid.ids, rewrites: hybrid.rewrites, repaired: hybrid.repaired };
  }
}

function levelsFor(context: number): number[] {
  const levels = [1];
  let value = 2;
  while (value < context) {
    levels.push(value);
    value *= 2;
  }
  if (levels[levels.length - 1] !== context) levels.push(context);
  return levels;
}

function linear(
  x: Float32Array,
  w: Float32Array,
  b: Float32Array,
  out: Float32Array,
  rows: number,
  inDim: number,
  outDim: number,
) {
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < outDim; c++) {
      let sum = b[c]!;
      for (let k = 0; k < inDim; k++) sum += x[r * inDim + k]! * w[k * outDim + c]!;
      out[r * outDim + c] = sum;
    }
  }
}

function linearBackward(
  x: Float32Array,
  w: Param,
  b: Param,
  dOut: Float32Array,
  dX: Float32Array,
  rows: number,
  inDim: number,
  outDim: number,
) {
  for (let r = 0; r < rows; r++) {
    for (let k = 0; k < inDim; k++) {
      let g = 0;
      for (let c = 0; c < outDim; c++) {
        const dy = dOut[r * outDim + c]!;
        g += dy * w.w[k * outDim + c]!;
        w.g[k * outDim + c]! += x[r * inDim + k]! * dy;
      }
      dX[r * inDim + k]! += g;
    }
    for (let c = 0; c < outDim; c++) b.g[c]! += dOut[r * outDim + c]!;
  }
}

function rnnStep(ar: ArModel, hPrev: Float32Array, x: Float32Array, sketch: Float32Array | null, hNow: Float32Array) {
  const h = hPrev.length;
  const d = x.length;
  for (let hi = 0; hi < h; hi++) {
    let pre = ar.bh.w[hi]!;
    for (let hj = 0; hj < h; hj++) pre += ar.wh.w[hi * h + hj]! * hPrev[hj]!;
    for (let di = 0; di < d; di++) pre += ar.wx.w[hi * d + di]! * x[di]!;
    if (sketch) {
      for (let di = 0; di < d; di++) pre += ar.wc.w[hi * d + di]! * sketch[di]!;
    }
    hNow[hi] = Math.tanh(pre);
  }
}

function sampleLogits(logits: Float32Array, temp: number, rng: () => number): number {
  if (temp <= 0) {
    let best = 0;
    let bestV = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i]! > bestV) {
        bestV = logits[i]!;
        best = i;
      }
    }
    return best;
  }
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i]! / temp;
  const probs = new Float32Array(logits.length);
  softmaxRows(scaled, 1, logits.length, probs);
  let draw = rng();
  for (let i = 0; i < probs.length; i++) {
    draw -= probs[i]!;
    if (draw <= 0) return i;
  }
  return probs.length - 1;
}

function sampleUnigram(unigram: Float32Array, rng: () => number): number[] {
  let draw = rng();
  for (let i = 0; i < unigram.length; i++) {
    draw -= unigram[i]!;
    if (draw <= 0) return [i];
  }
  let best = 0;
  for (let i = 1; i < unigram.length; i++) if (unigram[i]! > unigram[best]!) best = i;
  return [best];
}

function upValue(book: Float32Array, idx: Int16Array, pooledIndex: number, d: number, _length: number): number {
  const unit = Math.floor(pooledIndex / d);
  const di = pooledIndex - unit * d;
  const code = idx[unit]!;
  return book[code * d + di]!;
}

function tokenAcc(target: number[], pred: number[]): number {
  let hit = 0;
  const n = Math.min(target.length, pred.length);
  for (let i = 0; i < n; i++) if (target[i] === pred[i]) hit += 1;
  return hit / Math.max(1, target.length);
}

function mean(xs: number[]): number {
  if (!xs.length) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function packAr(ar: ArModel) {
  return {
    wh: Array.from(ar.wh.w),
    wx: Array.from(ar.wx.w),
    wc: Array.from(ar.wc.w),
    wo: Array.from(ar.wo.w),
    bh: Array.from(ar.bh.w),
    bo: Array.from(ar.bo.w),
    pos: Array.from(ar.pos.w),
  };
}

export function stageLabel(stage: StageName | "done"): string {
  if (stage === "teacher") return "Token model";
  if (stage === "tokenizer") return "Multiscale tokenizer";
  if (stage === "concepts") return "Next-concept generator";
  if (stage === "combine") return "Hybrid repair head";
  return "Ready";
}
