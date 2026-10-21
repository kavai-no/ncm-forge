import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Download, Layers, Search, Shuffle, Square, Workflow } from "lucide-react";
import { CREEK, HARBOR, corpusFromText, type Corpus } from "@/lib/ncm/corpus";
import {
  ForgeSession,
  stageLabel,
  type ForgeMode,
  type ForgeSnapshot,
  type Metrics,
  type SampleView,
} from "@/lib/ncm/engine";
import { readModelConfig, searchModels, type HfHit } from "@/lib/hf.functions";
import { pythonTrainer } from "@/lib/ncm/python";
import notebookTemplate from "@/lib/ncm/ncm_hybrid_colab.ipynb?raw";
import testNotebook from "@/lib/ncm/ncm_hybrid_test.ipynb?raw";
import { PRESETS, exportRecipe, familyOf, formatCount, type TeacherCard } from "@/lib/ncm/recipe";

const MODES: { id: ForgeMode; title: string; body: string }[] = [
  {
    id: "distill",
    title: "Distill",
    body: "A token model learns the corpus, then the concept model is trained on what that teacher would have written.",
  },
  {
    id: "scratch",
    title: "From scratch",
    body: "Train the hierarchy directly. A token model of similar size is kept only as the quality yardstick.",
  },
  {
    id: "hybrid",
    title: "Combine both",
    body: "The concept model drafts every level at once. The token model rewrites only the words the draft does not trust.",
  },
];

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function fitsFreeColab(id: string): boolean {
  return /(\b0\.5b\b|\b0\.6b\b|\b1b\b|\b1\.5b\b|\b1\.7b\b|\b2b\b|360m)/i.test(id);
}

function colabNotebook(modelId: string): string {
  const chosen = fitsFreeColab(modelId) ? modelId : "Qwen/Qwen2.5-0.5B-Instruct";
  return notebookTemplate.replace(
    'MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"',
    `MODEL_ID = ${JSON.stringify(chosen)}`,
  );
}

function download(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function Tokens({ tokens, repaired }: { tokens: string[]; repaired?: boolean[] }) {
  return (
    <span className="flex flex-wrap gap-1">
      {tokens.map((token, index) => (
        <span
          key={`${token}-${index}`}
          className={
            repaired?.[index]
              ? "rounded-sm bg-accent-soft px-1.5 py-0.5 text-accent"
              : "rounded-sm bg-sunken px-1.5 py-0.5 text-fg"
          }
        >
          {token}
        </span>
      ))}
    </span>
  );
}

export function ForgeApp() {
  const [mode, setMode] = useState<ForgeMode>("hybrid");
  const [teacher, setTeacher] = useState<TeacherCard>(PRESETS[0]!);
  const [query, setQuery] = useState("qwen");
  const [hits, setHits] = useState<HfHit[]>([]);
  const [searchNote, setSearchNote] = useState("");
  const [searching, setSearching] = useState(false);
  const [corpusId, setCorpusId] = useState<"harbor" | "creek" | "custom">("harbor");
  const [customText, setCustomText] = useState("");
  const [formError, setFormError] = useState("");
  const [running, setRunning] = useState(false);
  const [snap, setSnap] = useState<ForgeSnapshot | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [samples, setSamples] = useState<SampleView[]>([]);
  const [activeSample, setActiveSample] = useState(0);
  const [probe, setProbe] = useState("des moves the silver fish at dusk .");
  const [probeNote, setProbeNote] = useState("");
  const [probeLadder, setProbeLadder] = useState<SampleView["ladder"] | null>(null);
  const [mounted, setMounted] = useState(false);
  const sessionRef = useRef<ForgeSession | null>(null);
  const stopRef = useRef(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const recipe = exportRecipe(teacher);
  const corpusPreview = corpusId === "creek" ? CREEK : HARBOR;

  async function choose(card: TeacherCard) {
    setTeacher(card);
    try {
      const remote = await readModelConfig({ data: { id: card.id } });
      if (remote.gated) {
        setTeacher({ ...card, gated: true });
        return;
      }
      setTeacher({
        ...card,
        vocab: remote.vocab,
        hidden: remote.hidden,
        layers: remote.layers,
        gated: false,
        params: card.params === "—" ? formatCount(remote.hidden * remote.layers) : card.params,
      });
    } catch {
      setTeacher(card);
    }
  }

  async function onSearch() {
    setSearching(true);
    setSearchNote("");
    try {
      const found = await searchModels({ data: { q: query } });
      setHits(found);
      if (!found.length) setSearchNote("No text-generation models matched.");
    } catch (error) {
      setHits([]);
      setSearchNote(error instanceof Error ? error.message : "Search failed.");
    } finally {
      setSearching(false);
    }
  }

  function resolveCorpus(): Corpus | null {
    if (corpusId === "harbor") return HARBOR;
    if (corpusId === "creek") return CREEK;
    const built = corpusFromText(customText);
    if ("error" in built) {
      setFormError(built.error);
      return null;
    }
    return built;
  }

  async function onTrain() {
    setFormError("");
    const corpus = resolveCorpus();
    if (!corpus) return;
    stopRef.current = false;
    const session = new ForgeSession(corpus, mode, 7);
    sessionRef.current = session;
    setMetrics(null);
    setSamples([]);
    setProbeLadder(null);
    setRunning(true);
    setSnap(session.snapshot());
    let paints = 0;
    while (!stopRef.current) {
      const done = session.tick(16);
      paints += 1;
      if (done || paints % 2 === 0) {
        const next = session.snapshot();
        setSnap({ ...next, history: next.history.slice() });
      }
      if (done) break;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    if (!stopRef.current) {
      setMetrics(session.metrics());
      setSamples(session.samples());
      setActiveSample(0);
      const example = corpus.heldout[0] ?? corpus.train[0]!;
      const words = example
        .map((id) => corpus.vocab[id])
        .filter((word) => word && word !== "<pad>" && word !== "<bos>");
      setProbe(words.join(" "));
    }
    setRunning(false);
  }

  function onProbe() {
    const session = sessionRef.current;
    if (!session) return;
    const result = session.reconstructLine(probe);
    if (result.error) {
      setProbeNote(result.error);
      setProbeLadder(null);
      return;
    }
    setProbeNote("");
    setProbeLadder(result.ladder ?? null);
  }

  function onExportCheckpoint() {
    const session = sessionRef.current;
    if (!session || !metrics) return;
    const payload = {
      ...session.checkpoint(),
      teacher: teacher.id,
      metrics,
    };
    download("ncm-forge-checkpoint.json", JSON.stringify(payload), "application/json");
  }

  function onExportPython() {
    download("ncm_forge_train.py", pythonTrainer(teacher, mode, recipe), "text/x-python");
  }

  function onExportColab() {
    download("ncm_hybrid_colab.ipynb", colabNotebook(teacher.id), "application/json");
  }

  function onExportTest() {
    download("ncm_hybrid_test.ipynb", testNotebook, "application/json");
  }

  const shown = samples[activeSample];
  const history = snap?.history ?? [];

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-8 sm:px-6 sm:py-12">
      <header className="grid gap-6 lg:grid-cols-[1.4fr_0.8fr] lg:items-end">
        <div className="flex flex-col gap-4">
          <p className="text-sm font-medium tracking-wide text-accent uppercase">Next-concept modeling</p>
          <h1 className="font-display text-5xl leading-none text-fg sm:text-6xl">
            Forge a concept model from a token model.
          </h1>
          <p className="max-w-xl text-pretty text-muted">
            Qwen and its cousins write one token at a time. A next-concept model, the 2026 Stanford recipe,
            stores text as a stack of discrete codes and writes a whole level in one step. Pick a Hugging Face
            model, then distill it, train the hierarchy from scratch, or run the two together.
          </p>
        </div>
        <div className="rounded-md border border-line bg-raised p-4 text-sm text-muted">
          <p className="font-medium text-fg">What actually runs here</p>
          <p className="mt-2 text-pretty">
            The browser trains a miniature — same two stages, a few thousand parameters — so you can watch the
            sketch sharpen. The Python file is the scale-up: the selected repo’s tokenizer, its embedding table
            as initialization, teacher samples, and a repair pass.
          </p>
          <p className="mt-3">
            <a className="text-accent underline decoration-line underline-offset-2" href="https://arxiv.org/abs/2610.14162">
              Paper
            </a>
            <span aria-hidden="true"> · </span>
            <a className="text-accent underline decoration-line underline-offset-2" href="https://github.com/HazyResearch/ncm">
              HazyResearch/ncm
            </a>
          </p>
        </div>
      </header>

      <section className="grid gap-3 md:grid-cols-3" aria-label="Training recipe">
        {MODES.map((item) => {
          const on = mode === item.id;
          return (
            <button
              key={item.id}
              type="button"
              aria-pressed={on}
              onClick={() => setMode(item.id)}
              className={
                on
                  ? "min-h-11 rounded-md border border-accent bg-accent-soft p-4 text-left"
                  : "min-h-11 rounded-md border border-line bg-raised p-4 text-left hover:border-accent"
              }
            >
              <span className="flex items-center gap-2 font-medium text-fg">
                {item.id === "hybrid" ? <Workflow className="size-4 text-accent" /> : <Layers className="size-4 text-accent" />}
                {item.title}
              </span>
              <span className="mt-2 block text-sm text-pretty text-muted">{item.body}</span>
            </button>
          );
        })}
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="flex flex-col gap-4">
          <div className="rounded-md border border-line bg-raised p-4">
            <div className="flex items-end justify-between gap-3">
              <h2 className="font-display text-2xl">Teacher</h2>
              <p className="text-sm text-muted">{teacher.family}</p>
            </div>
            <p className="mt-1 font-medium break-all text-fg">{teacher.id}</p>
            <p className="mt-1 text-sm text-muted">
              {teacher.params}
              {teacher.vocab ? ` · vocab ${formatCount(teacher.vocab)}` : ""}
              {teacher.hidden ? ` · width ${teacher.hidden}` : ""}
              {teacher.layers ? ` · ${teacher.layers} layers` : ""}
              {teacher.gated ? " · gated, export still targets it" : ""}
            </p>
            <div className="mt-4 flex flex-col gap-2 sm:flex-row">
              <label className="sr-only" htmlFor="hf-query">
                Search Hugging Face
              </label>
              <input
                id="hf-query"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") void onSearch();
                }}
                placeholder="Search text models"
                className="min-h-11 flex-1 rounded-sm border border-line bg-bg px-3 text-fg outline-none focus:border-accent"
              />
              <button
                type="button"
                onClick={() => void onSearch()}
                disabled={searching}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-sm bg-fg px-4 text-bg disabled:opacity-60"
              >
                <Search className="size-4" />
                {searching ? "Searching" : "Search"}
              </button>
            </div>
            {searchNote ? <p className="mt-2 text-sm text-accent">{searchNote}</p> : null}
            <ul className="mt-3 flex flex-col gap-1">
              {hits.map((hit) => (
                <li key={hit.id}>
                  <button
                    type="button"
                    onClick={() =>
                      void choose({
                        id: hit.id,
                        family: familyOf(hit.id),
                        params: "—",
                        vocab: 151936,
                        hidden: 896,
                        layers: 24,
                        gated: hit.gated,
                      })
                    }
                    className="flex min-h-11 w-full items-center justify-between gap-3 rounded-sm px-2 text-left hover:bg-sunken"
                  >
                    <span className="truncate">{hit.id}</span>
                    <span className="shrink-0 text-sm text-muted tabular-nums">{formatCount(hit.downloads)}</span>
                  </button>
                </li>
              ))}
            </ul>
            <div className="mt-3 flex flex-wrap gap-2">
              {PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => void choose(preset)}
                  className={
                    preset.id === teacher.id
                      ? "min-h-11 rounded-sm bg-fg px-3 text-sm text-bg"
                      : "min-h-11 rounded-sm border border-line bg-bg px-3 text-sm text-fg"
                  }
                >
                  {preset.params} {preset.family}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-md border border-line bg-raised p-4">
            <h2 className="font-display text-2xl">Lab corpus</h2>
            <p className="mt-1 text-sm text-pretty text-muted">
              Closed text, fixed width, so the miniature can actually learn. The export is what you point at {teacher.id}.
            </p>
            <div className="mt-3 grid gap-2 sm:grid-cols-3">
              {(
                [
                  ["harbor", HARBOR.name, HARBOR.blurb],
                  ["creek", CREEK.name, CREEK.blurb],
                  ["custom", "Your lines", "Same token length on every line, 8 lines or more."],
                ] as const
              ).map(([id, name, blurb]) => (
                <button
                  key={id}
                  type="button"
                  aria-pressed={corpusId === id}
                  onClick={() => setCorpusId(id)}
                  className={
                    corpusId === id
                      ? "min-h-11 rounded-sm border border-accent bg-accent-soft p-3 text-left"
                      : "min-h-11 rounded-sm border border-line p-3 text-left"
                  }
                >
                  <span className="block font-medium">{name}</span>
                  <span className="mt-1 block text-sm text-pretty text-muted">{blurb}</span>
                </button>
              ))}
            </div>
            {corpusId === "custom" ? (
              <textarea
                value={customText}
                onChange={(event) => setCustomText(event.target.value)}
                rows={6}
                placeholder={"ana finds the quiet boat at dawn .\nbo keeps a silver lamp near dusk ."}
                className="mt-3 w-full rounded-sm border border-line bg-bg p-3 text-sm text-fg outline-none focus:border-accent"
              />
            ) : (
              <p className="mt-3 text-sm text-muted">
                Example: {corpusPreview.train[0]!.map((id) => corpusPreview.vocab[id]).join(" ")}
              </p>
            )}
          </div>
        </div>

        <aside className="flex flex-col gap-4 rounded-md border border-line bg-fg p-5 text-bg">
          <p className="text-sm tracking-wide text-accent-soft uppercase">Export recipe</p>
          <h2 className="font-display text-3xl leading-none">Scaled like the TinyStories NCM</h2>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
            <div>
              <dt className="text-line">Levels</dt>
              <dd className="tabular-nums">{recipe.levels.length} · last {recipe.levels.at(-1)}</dd>
            </div>
            <div>
              <dt className="text-line">Codebook</dt>
              <dd className="tabular-nums">{formatCount(recipe.codebook)} / level</dd>
            </div>
            <div>
              <dt className="text-line">Context</dt>
              <dd className="tabular-nums">{recipe.context} tokens</dd>
            </div>
            <div>
              <dt className="text-line">Generator</dt>
              <dd className="tabular-nums">
                {recipe.generatorLayers} × {recipe.generatorEmbed}
              </dd>
            </div>
          </dl>
          <ul className="flex flex-col gap-2 text-sm text-pretty text-line">
            <li>Pre-tokenizer is {teacher.id}’s own Hugging Face tokenizer.</li>
            <li>Distill and combine copy the teacher’s input embedding into the quantizer.</li>
            <li>Combine adds one teacher forward that patches low-confidence draft tokens.</li>
          </ul>
          <p className="text-sm text-line">
            Lab width is {corpusId === "custom" ? "your line length" : "8 tokens"}, levels 1–2–4–8. Not the export.
          </p>
          <div className="flex flex-col gap-2">
            <button
              type="button"
              onClick={onExportColab}
              className="inline-flex min-h-11 items-center justify-center gap-2 rounded-sm bg-bg px-4 text-fg"
            >
              <Download className="size-4" />
              Train notebook
            </button>
            <button
              type="button"
              onClick={onExportTest}
              className="inline-flex min-h-11 items-center justify-center gap-2 rounded-sm border border-line px-4 text-bg"
            >
              <Download className="size-4" />
              Test notebook
            </button>
          </div>
          <p className="text-sm text-pretty text-line">
            {fitsFreeColab(teacher.id)
              ? `Train writes hybrid.pt for ${teacher.id} on a free T4. Test loads that file and scores it on new text.`
              : "A 3B or larger teacher does not fit a free 16 GB T4. The train notebook uses Qwen2.5-0.5B-Instruct. 1.5B also fits."}
          </p>
        </aside>
      </section>

      <section className="rounded-md border border-line bg-raised p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="font-display text-2xl">{running ? stageLabel(snap?.stage ?? "tokenizer") : "Train the lab model"}</h2>
            <p className="text-sm text-muted" aria-live="polite">
              {running && snap && snap.stage !== "done"
                ? `Step ${snap.stageStep} of ${snap.stageSteps}`
                : metrics
                  ? "Hierarchy is frozen. Probe a line or export."
                  : "Multiscale residual quantizer, then a next-concept generator."}
            </p>
          </div>
          <div className="flex gap-2">
            {running ? (
              <button
                type="button"
                onClick={() => {
                  stopRef.current = true;
                }}
                className="inline-flex min-h-11 items-center gap-2 rounded-sm border border-line px-4"
              >
                <Square className="size-4" />
                Stop
              </button>
            ) : (
              <button
                type="button"
                onClick={() => void onTrain()}
                className="inline-flex min-h-11 items-center gap-2 rounded-sm bg-accent px-4 text-bg"
              >
                <Shuffle className="size-4" />
                {metrics ? "Train again" : "Train"}
              </button>
            )}
          </div>
        </div>
        {formError ? <p className="mt-3 text-sm text-accent">{formError}</p> : null}
        {snap && snap.stage !== "done" ? (
          <div className="mt-4 h-1.5 overflow-hidden rounded-sm bg-sunken">
            <div
              className="h-full bg-accent"
              style={{ width: `${Math.min(100, (snap.stageStep / snap.stageSteps) * 100)}%` }}
            />
          </div>
        ) : null}
        {mounted && !running && history.length > 1 ? (
          <div className="mt-4 h-28 text-accent">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid stroke="var(--color-line)" vertical={false} />
                <XAxis dataKey="step" hide />
                <YAxis hide domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{
                    background: "var(--color-raised)",
                    border: "1px solid var(--color-line)",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                  formatter={(value) => [typeof value === "number" ? value.toFixed(3) : value, "loss"]}
                />
                <Line type="monotone" dataKey="loss" stroke="currentColor" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : null}
      </section>

      {metrics && shown ? (
        <section className="grid gap-6 lg:grid-cols-[0.85fr_1.15fr]">
          <div className="flex flex-col gap-4">
            <h2 className="font-display text-3xl">Comparable on this corpus</h2>
            <p className="text-sm text-pretty text-muted">
              Exact next-token match on held-out lines sits near 40% for every model — several words are legal in
              each slot. Grammar of free samples is the number that moves.
            </p>
            <table className="w-full text-left text-sm">
              <thead className="text-muted">
                <tr>
                  <th className="py-2 font-medium">Model</th>
                  <th className="py-2 font-medium">Grammar</th>
                  <th className="py-2 font-medium">Forwards</th>
                </tr>
              </thead>
              <tbody className="tabular-nums">
                <tr className="border-t border-line">
                  <td className="py-3">Token model</td>
                  <td>{pct(metrics.arValid)}</td>
                  <td>{metrics.stepsAr}</td>
                </tr>
                <tr className="border-t border-line">
                  <td className="py-3">Next-concept</td>
                  <td>{pct(metrics.ncmValid)}</td>
                  <td>{metrics.stepsNcm}</td>
                </tr>
                <tr className="border-t border-line">
                  <td className="py-3">Combined</td>
                  <td>{pct(metrics.hybridValid)}</td>
                  <td>{metrics.stepsHybrid.toFixed(1)}</td>
                </tr>
              </tbody>
            </table>
            <dl className="grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-sm bg-sunken p-3">
                <dt className="text-muted">Tokenizer reconstruction</dt>
                <dd className="mt-1 text-lg tabular-nums">{pct(metrics.reconAcc)}</dd>
              </div>
              <div className="rounded-sm bg-sunken p-3">
                <dt className="text-muted">Concept-code accuracy</dt>
                <dd className="mt-1 text-lg tabular-nums">{pct(metrics.conceptAcc)}</dd>
              </div>
              <div className="rounded-sm bg-sunken p-3">
                <dt className="text-muted">Avg. repairs</dt>
                <dd className="mt-1 text-lg tabular-nums">{metrics.avgRewrites.toFixed(1)} tokens</dd>
              </div>
              <div className="rounded-sm bg-sunken p-3">
                <dt className="text-muted">Lab parameters</dt>
                <dd className="mt-1 text-lg tabular-nums">
                  {formatCount(metrics.paramsTokenizer + metrics.paramsGenerator + metrics.paramsToken)}
                </dd>
              </div>
            </dl>
            <div className="flex flex-col gap-2 sm:flex-row">
              <button
                type="button"
                onClick={onExportCheckpoint}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-sm bg-fg px-4 text-bg"
              >
                <Download className="size-4" />
                Lab checkpoint
              </button>
              <button
                type="button"
                onClick={onExportPython}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-sm border border-line px-4"
              >
                <Download className="size-4" />
                Python trainer
              </button>
              <button
                type="button"
                onClick={onExportColab}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-sm border border-line px-4"
              >
                <Download className="size-4" />
                Train notebook
              </button>
              <button
                type="button"
                onClick={onExportTest}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-sm border border-line px-4"
              >
                <Download className="size-4" />
                Test notebook
              </button>
            </div>
          </div>

          <div className="rounded-md border border-line bg-raised p-4">
            <div className="flex flex-wrap gap-2">
              {samples.map((sample, index) => (
                <button
                  key={`${sample.kind}-${sample.title}`}
                  type="button"
                  aria-pressed={index === activeSample}
                  onClick={() => setActiveSample(index)}
                  className={
                    index === activeSample
                      ? "min-h-11 rounded-sm bg-fg px-3 text-sm text-bg"
                      : "min-h-11 rounded-sm border border-line px-3 text-sm"
                  }
                >
                  {sample.title}
                </button>
              ))}
            </div>
            <p className="mt-4 text-sm text-muted">
              {shown.kind === "reconstruction" ? `Source · ${shown.source}` : shown.source}
            </p>
            <ol className="mt-3 flex flex-col gap-3">
              {shown.ladder.map((level) => (
                <li key={level.level} className="grid grid-cols-1 gap-1 sm:grid-cols-[5.5rem_1fr] sm:gap-3">
                  <span className="text-xs text-muted tabular-nums">
                    Level {level.level}
                    <span className="block">{level.length === 1 ? "1 code" : `${level.length} codes`}</span>
                    {level.acc >= 0 ? <span className="block text-fg">{pct(level.acc)} match</span> : null}
                  </span>
                  <Tokens tokens={level.tokens} />
                </li>
              ))}
            </ol>
            <div className="mt-5 grid gap-3">
              <div>
                <p className="text-xs tracking-wide text-muted uppercase">Concept draft</p>
                <div className="mt-1 text-sm">
                  <Tokens tokens={shown.ncmTokens} repaired={shown.repaired} />
                </div>
              </div>
              <div>
                <p className="text-xs tracking-wide text-muted uppercase">Token model alone</p>
                <div className="mt-1 text-sm">
                  <Tokens tokens={shown.arTokens} />
                </div>
              </div>
              <div>
                <p className="text-xs tracking-wide text-muted uppercase">
                  Combined · {shown.rewrites} rewrite{shown.rewrites === 1 ? "" : "s"}
                </p>
                <div className="mt-1 text-sm">
                  <Tokens tokens={shown.hybridTokens} repaired={shown.repaired} />
                </div>
              </div>
            </div>
            <p className="mt-3 text-xs text-muted">Copper chips are tokens the token model replaced.</p>

            <form
              className="mt-5 flex flex-col gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                onProbe();
              }}
            >
              <label htmlFor="probe" className="text-sm font-medium">
                Encode a line and decode it level by level
              </label>
              <div className="flex flex-col gap-2 sm:flex-row">
                <input
                  id="probe"
                  value={probe}
                  onChange={(event) => setProbe(event.target.value)}
                  className="min-h-11 flex-1 rounded-sm border border-line bg-bg px-3 text-sm outline-none focus:border-accent"
                />
                <button type="submit" className="min-h-11 rounded-sm border border-line px-4 text-sm">
                  Decode
                </button>
              </div>
              {probeNote ? <p className="text-sm text-accent">{probeNote}</p> : null}
            </form>
            {probeLadder ? (
              <ol className="mt-3 flex flex-col gap-2">
                {probeLadder.map((level) => (
                  <li key={`probe-${level.level}`} className="grid grid-cols-1 gap-1 sm:grid-cols-[5.5rem_1fr]">
                    <span className="text-xs text-muted tabular-nums">
                      Level {level.level}
                      <span className="block text-fg">{pct(level.acc)}</span>
                    </span>
                    <Tokens tokens={level.tokens} />
                  </li>
                ))}
              </ol>
            ) : null}
          </div>
        </section>
      ) : null}
    </main>
  );
}
