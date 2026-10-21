import { createServerFn } from "@tanstack/react-start";

export type HfHit = {
  id: string;
  downloads: number;
  likes: number;
  gated: boolean;
};

export type RemoteConfig = {
  vocab: number;
  hidden: number;
  layers: number;
  gated: boolean;
};

function assertModelId(id: string): string {
  const trimmed = id.trim();
  if (!/^[A-Za-z0-9_.:@+\-/]{3,120}$/.test(trimmed) || trimmed.includes("..")) {
    throw new Error("That model id is not a Hugging Face repo name.");
  }
  return trimmed;
}

export const searchModels = createServerFn({ method: "GET" })
  .validator((input: { q: string }) => {
    const q = (input?.q ?? "").trim();
    if (q.length < 2 || q.length > 80) throw new Error("Search needs 2–80 characters.");
    return { q };
  })
  .handler(async ({ data }): Promise<HfHit[]> => {
    const url = new URL("https://huggingface.co/api/models");
    url.searchParams.set("search", data.q);
    url.searchParams.set("filter", "text-generation");
    url.searchParams.set("sort", "downloads");
    url.searchParams.set("direction", "-1");
    url.searchParams.set("limit", "8");
    const res = await fetch(url, { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error(`Hugging Face search failed (${res.status}).`);
    const json: unknown = await res.json();
    if (!Array.isArray(json)) return [];
    return json.slice(0, 8).map((row) => {
      const item = row as { id?: string; downloads?: number; likes?: number; gated?: boolean | string };
      return {
        id: String(item.id ?? ""),
        downloads: Number(item.downloads ?? 0),
        likes: Number(item.likes ?? 0),
        gated: Boolean(item.gated),
      };
    }).filter((row) => row.id.length > 0);
  });

export const readModelConfig = createServerFn({ method: "GET" })
  .validator((input: { id: string }) => ({ id: assertModelId(input?.id ?? "") }))
  .handler(async ({ data }): Promise<RemoteConfig> => {
    const url = `https://huggingface.co/${data.id}/resolve/main/config.json`;
    const res = await fetch(url, { headers: { Accept: "application/json", "User-Agent": "ncm-forge" } });
    if (res.status === 401 || res.status === 403) {
      return { vocab: 0, hidden: 0, layers: 0, gated: true };
    }
    if (!res.ok) throw new Error(`No public config for ${data.id} (${res.status}).`);
    const json = (await res.json()) as Record<string, unknown>;
    const nested = (json.text_config ?? json) as Record<string, unknown>;
    const vocab = numberFrom(nested.vocab_size, json.vocab_size);
    const hidden = numberFrom(nested.hidden_size, json.hidden_size, nested.n_embd);
    const layers = numberFrom(nested.num_hidden_layers, json.num_hidden_layers, nested.n_layer);
    if (!vocab || !hidden || !layers) throw new Error("Config is missing vocab, width, or depth.");
    return { vocab, hidden, layers, gated: false };
  });

function numberFrom(...values: unknown[]): number {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  }
  return 0;
}
