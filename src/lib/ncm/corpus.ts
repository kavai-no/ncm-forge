export type Corpus = {
  id: string;
  name: string;
  blurb: string;
  train: number[][];
  heldout: number[][];
  vocab: string[];
  bos: number;
  pad: number;
  /** Token classes per position, when the corpus is a fixed template. */
  slots: string[][] | null;
};

export type CorpusIssue = { error: string };

const DOT = ".";

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildVocab(slots: string[][]): { vocab: string[]; rows: number[][] } {
  const vocab = ["<pad>", "<bos>"];
  const index = new Map<string, number>();
  for (const tok of ["<pad>", "<bos>"]) index.set(tok, vocab.indexOf(tok));
  for (const slot of slots) {
    for (const word of slot) {
      if (!index.has(word)) {
        index.set(word, vocab.length);
        vocab.push(word);
      }
    }
  }
  const rows = slots.map((slot) => slot.map((word) => index.get(word)!));
  return { vocab, rows };
}

function templateCorpus(
  id: string,
  name: string,
  blurb: string,
  slots: string[][],
  nTrain: number,
  nTest: number,
  seed: number,
): Corpus {
  const { vocab, rows } = buildVocab(slots);
  const rng = mulberry32(seed);
  const seen = new Set<string>();
  const draw = () => {
    const ids = rows.map((opts) => opts[Math.floor(rng() * opts.length)]!);
    return ids;
  };
  const train: number[][] = [];
  const heldout: number[][] = [];
  let guard = 0;
  while ((train.length < nTrain || heldout.length < nTest) && guard < 20000) {
    guard += 1;
    const ids = draw();
    const key = ids.join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    if (train.length < nTrain) train.push(ids);
    else heldout.push(ids);
  }
  return {
    id,
    name,
    blurb,
    train,
    heldout,
    vocab,
    bos: 1,
    pad: 0,
    slots,
  };
}

export const HARBOR = templateCorpus(
  "harbor",
  "Harbor log",
  "A closed grammar of dock sentences. Small enough for the lab model to actually learn, same shape as a TinyStories run.",
  [
    ["ana", "bo", "cy", "des"],
    ["finds", "keeps", "moves", "paints"],
    ["a", "the"],
    ["red", "blue", "quiet", "silver"],
    ["boat", "bell", "fish", "lamp"],
    ["at", "near", "before"],
    ["dawn", "dusk", "noon"],
    [DOT],
  ],
  96,
  24,
  11,
);

export const CREEK = templateCorpus(
  "creek",
  "Creek notes",
  "Field notes with the same length and a different vocabulary, so the recipe is not glued to one sentence.",
  [
    ["nia", "jules", "sam", "rio"],
    ["sees", "tracks", "counts", "follows"],
    ["two", "three", "four", "five"],
    ["bright", "faint", "wild", "still"],
    ["herons", "foxes", "otters", "moths"],
    ["along", "above", "near"],
    ["water", "ridge", "reeds"],
    [DOT],
  ],
  96,
  24,
  29,
);

export const CORPORA = [HARBOR, CREEK];

export function sentence(vocab: string[], ids: number[]): string {
  const words: string[] = [];
  for (const id of ids) {
    const tok = vocab[id] ?? "?";
    if (tok === "<pad>" || tok === "<bos>") continue;
    if (tok === "." && words.length) {
      words[words.length - 1] = `${words[words.length - 1]}.`;
    } else {
      words.push(tok);
    }
  }
  return words.join(" ");
}

export function validTemplate(corpus: Corpus, ids: number[]): boolean {
  if (!corpus.slots) return true;
  if (ids.length !== corpus.slots.length) return false;
  for (let i = 0; i < ids.length; i++) {
    const tok = corpus.vocab[ids[i]!] ?? "";
    if (!corpus.slots[i]!.includes(tok)) return false;
  }
  return true;
}

export function corpusFromText(text: string): Corpus | CorpusIssue {
  const lines = text
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 8) {
    return { error: "Paste at least 8 short lines. The lab model needs a small repeated set." };
  }
  const tokenized = lines.map((line) =>
    line
      .replace(/([.])$/g, " $1")
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 12),
  );
  const length = tokenized[0]!.length;
  if (length < 4) return { error: "Each line needs at least 4 tokens." };
  const even = tokenized.filter((row) => row.length === length);
  if (even.length < 8) {
    return { error: "Use lines of the same token length (4–12 words). The hierarchy expects a fixed width." };
  }
  const vocab = ["<pad>", "<bos>"];
  const index = new Map<string, number>([
    ["<pad>", 0],
    ["<bos>", 1],
  ]);
  const ids = even.map((row) =>
    row.map((word) => {
      let id = index.get(word);
      if (id === undefined) {
        id = vocab.length;
        index.set(word, id);
        vocab.push(word);
      }
      return id;
    }),
  );
  if (vocab.length > 80) {
    return { error: "Keep the vocabulary under 80 tokens for the in-browser run. The export is what scales up." };
  }
  const cut = Math.max(4, Math.floor(ids.length * 0.8));
  return {
    id: "custom",
    name: "Pasted text",
    blurb: "Your lines, whitespace-tokenized, held to one width so the hierarchy can train.",
    train: ids.slice(0, cut),
    heldout: ids.slice(cut),
    vocab,
    bos: 1,
    pad: 0,
    slots: null,
  };
}
