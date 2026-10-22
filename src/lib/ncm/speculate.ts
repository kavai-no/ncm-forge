export type Round = {
  proposed: string[];
  kept: number;
  written: string[];
};

export type Speculation = {
  text: string[];
  greedy: string[];
  rounds: Round[];
  qwenForwards: number;
  identical: boolean;
};

function teacherNext(gold: string[], seq: string[], index: number): string {
  if (seq[index] !== gold[index]) {
    throw new Error("the check looked at a token Qwen did not write");
  }
  return gold[index + 1];
}

export function speculate(gold: string[], prefix: number, nNew: number, k: number, agree: number, seed: number): Speculation {
  const greedy = gold.slice(0, prefix + nNew);
  const out = gold.slice(0, prefix);
  const rounds: Round[] = [];
  let state = seed >>> 0;
  const nextRand = () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };

  while (out.length - prefix < nNew) {
    const take = Math.min(k, nNew - (out.length - prefix));
    const ctx = out.slice();
    const proposed: string[] = [];
    for (let i = 0; i < take; i += 1) {
      const right = gold[ctx.length];
      proposed.push(nextRand() < agree ? right : right === "the" ? "a" : "the");
      ctx.push(proposed[proposed.length - 1]);
    }
    const full = out.concat(proposed);
    const base = out.length - 1;
    let nKeep = 0;
    for (let i = 0; i < take; i += 1) {
      if (teacherNext(gold, full, base + i) === proposed[i]) nKeep += 1;
      else break;
    }
    const written: string[] = [];
    if (nKeep > 0) {
      const kept = proposed.slice(0, nKeep);
      out.push(...kept);
      written.push(...kept);
    }
    if (out.length - prefix < nNew) {
      const token = teacherNext(gold, full, base + (nKeep < take ? nKeep : take));
      out.push(token);
      written.push(token);
    }
    rounds.push({ proposed, kept: nKeep, written });
  }

  const text = out.slice(0, prefix + nNew);
  return {
    text,
    greedy,
    rounds,
    qwenForwards: rounds.length,
    identical: text.every((token, index) => token === greedy[index]),
  };
}
