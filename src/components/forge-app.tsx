import { useMemo, useState } from "react";
import { Check, Download } from "lucide-react";
import trainNotebook from "@/lib/ncm/qwen_speculative_colab.ipynb?raw";
import testNotebook from "@/lib/ncm/qwen_speculative_test.ipynb?raw";
import { speculate } from "@/lib/ncm/speculate";

const PREFIX = 8;
const NEW_TOKENS = 16;

const SAMPLES = [
  {
    id: "normalize",
    label: "normalize",
    tokens: [
      "def", "normalize", "(", "text", ")", ":", "if", "text", "is", "None", ":", "return", '""',
      "text", "=", "str", "(", "text", ")", ".", "strip", "(", ")", "return", "text", ".", "lower",
      "(", ")", "def", "blank", "(", "text", ")", ":", "return", "not", "normalize", "(", "text", ")",
    ],
  },
  {
    id: "clamp",
    label: "clamp",
    tokens: [
      "def", "clamp", "(", "value", ",", "low", ",", "high", ")", ":", "if", "value", "<", "low", ":",
      "return", "low", "if", "value", ">", "high", ":", "return", "high", "return", "value", "def",
      "span", "(", "low", ",", "high", ")", ":", "return", "high", "-", "low",
    ],
  },
] as const;

function download(name: string, contents: string) {
  const blob = new Blob([contents], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function show(tokens: string[]) {
  return tokens.join(" ");
}

export function ForgeApp() {
  const [sampleId, setSampleId] = useState<(typeof SAMPLES)[number]["id"]>("normalize");
  const [agree, setAgree] = useState(0.55);
  const [k, setK] = useState(4);
  const sample = SAMPLES.find((item) => item.id === sampleId) ?? SAMPLES[0];
  const result = useMemo(() => speculate(sample.tokens.slice(), PREFIX, NEW_TOKENS, k, agree, 11), [sample, k, agree]);
  const perForward = NEW_TOKENS / result.qwenForwards;

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-8 sm:px-6 sm:py-12">
      <header className="flex flex-col gap-3">
        <p className="text-sm tracking-wide text-muted">Qwen stays the author</p>
        <h1 className="font-display text-4xl leading-tight text-fg sm:text-5xl">A faster Qwen, with Qwen's own text</h1>
        <p className="max-w-2xl text-base leading-relaxed text-muted">
          The drafter proposes the next few tokens. Qwen checks the span in one pass and keeps only the prefix it would
          have written. The text below is always Qwen's. It gets faster only when more than one token survives the check.
        </p>
      </header>

      <section className="flex flex-col gap-4 rounded-md border border-line bg-raised p-4">
        <div className="flex flex-wrap gap-2">
          {SAMPLES.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setSampleId(item.id)}
              className={
                item.id === sampleId
                  ? "min-h-11 rounded-sm bg-accent px-4 text-sm text-raised"
                  : "min-h-11 rounded-sm border border-line bg-bg px-4 text-sm text-fg"
              }
            >
              {item.label}
            </button>
          ))}
        </div>
        <label className="flex flex-col gap-2 text-sm text-fg">
          <span className="flex items-center justify-between">
            Drafter agrees with Qwen
            <span className="text-muted">{Math.round(agree * 100)}%</span>
          </span>
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(agree * 100)}
            onChange={(event) => setAgree(Number(event.target.value) / 100)}
            className="accent-accent"
          />
        </label>
        <label className="flex flex-col gap-2 text-sm text-fg">
          <span className="flex items-center justify-between">
            Tokens proposed per check
            <span className="text-muted">{k}</span>
          </span>
          <input
            type="range"
            min={1}
            max={8}
            value={k}
            onChange={(event) => setK(Number(event.target.value))}
            className="accent-accent"
          />
        </label>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="font-display text-2xl text-fg">Given</h2>
        <p className="rounded-md bg-sunken px-3 py-3 font-mono text-sm leading-relaxed text-fg">{show(sample.tokens.slice(0, PREFIX))}</p>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="font-display text-2xl text-fg">Each check</h2>
        <ol className="flex flex-col gap-3">
          {result.rounds.map((round, index) => (
            <li key={index} className="rounded-md border border-line bg-raised px-3 py-3">
              <p className="mb-2 text-sm text-muted">Qwen forward {index + 1}</p>
              <p className="font-mono text-sm leading-relaxed text-fg">
                {round.proposed.map((token, tokenIndex) => (
                  <span key={tokenIndex} className={tokenIndex < round.kept ? "text-fg" : "text-accent line-through"}>
                    {token}
                    {tokenIndex < round.proposed.length - 1 ? " " : ""}
                  </span>
                ))}
              </p>
              <p className="mt-2 font-mono text-sm leading-relaxed text-muted">kept {show(round.written)}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="flex flex-col gap-3 rounded-md border border-line bg-raised p-4">
        <p className="flex items-center gap-2 text-sm text-fg">
          <Check className="size-4 text-accent" aria-hidden />
          {result.identical ? "Identical to greedy Qwen" : "Not Qwen's text"}
        </p>
        <p className="font-mono text-sm leading-relaxed text-fg">{show(result.text.slice(PREFIX))}</p>
        <p className="text-sm leading-relaxed text-muted">
          {result.qwenForwards} Qwen forwards instead of {NEW_TOKENS}. {perForward.toFixed(2)} tokens per forward.
          {perForward <= 1.05 ? " No speedup: the drafter is not close enough, so the extra pass is overhead." : " Fewer Qwen forwards, same text."}
        </p>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="font-display text-2xl text-fg">Train it on Qwen</h2>
        <p className="max-w-2xl text-sm leading-relaxed text-muted">
          The page above is the rule, on a fixed function. The notebook trains a 4-layer drafter to copy
          Qwen2.5-0.5B on CodeSearchNet, then refuses to continue if the text differs from greedy Qwen. A codebook is
          not in that path.
        </p>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => download("qwen_speculative_colab.ipynb", trainNotebook)}
            className="inline-flex min-h-11 items-center gap-2 rounded-sm bg-accent px-4 text-sm text-raised"
          >
            <Download className="size-4" aria-hidden />
            Train notebook
          </button>
          <button
            type="button"
            onClick={() => download("qwen_speculative_test.ipynb", testNotebook)}
            className="inline-flex min-h-11 items-center gap-2 rounded-sm border border-line bg-bg px-4 text-sm text-fg"
          >
            <Download className="size-4" aria-hidden />
            Test notebook
          </button>
        </div>
      </section>
    </main>
  );
}
