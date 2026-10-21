export type TeacherCard = {
  id: string;
  family: string;
  params: string;
  vocab: number;
  hidden: number;
  layers: number;
  gated?: boolean;
};

export type ExportRecipe = {
  context: number;
  levels: number[];
  codebook: number;
  vqEmbed: number;
  vqLayers: number;
  generatorLayers: number;
  generatorEmbed: number;
  generatorHeads: number;
};

export const PRESETS: TeacherCard[] = [
  { id: "Qwen/Qwen2.5-0.5B-Instruct", family: "Qwen", params: "0.5B", vocab: 151936, hidden: 896, layers: 24 },
  { id: "Qwen/Qwen2.5-1.5B-Instruct", family: "Qwen", params: "1.5B", vocab: 151936, hidden: 1536, layers: 28 },
  { id: "Qwen/Qwen2.5-3B-Instruct", family: "Qwen", params: "3B", vocab: 151936, hidden: 2048, layers: 36 },
  { id: "Qwen/Qwen2.5-7B-Instruct", family: "Qwen", params: "7B", vocab: 152064, hidden: 3584, layers: 28 },
  { id: "Qwen/Qwen3-0.6B", family: "Qwen", params: "0.6B", vocab: 151936, hidden: 1024, layers: 28 },
  { id: "Qwen/Qwen3-1.7B", family: "Qwen", params: "1.7B", vocab: 151936, hidden: 2048, layers: 28 },
  { id: "Qwen/Qwen3-4B", family: "Qwen", params: "4B", vocab: 151936, hidden: 2560, layers: 36 },
  { id: "HuggingFaceTB/SmolLM2-360M-Instruct", family: "SmolLM", params: "360M", vocab: 49152, hidden: 960, layers: 32 },
  { id: "meta-llama/Llama-3.2-1B-Instruct", family: "Llama", params: "1B", vocab: 128256, hidden: 2048, layers: 16, gated: true },
  { id: "google/gemma-2-2b-it", family: "Gemma", params: "2B", vocab: 256000, hidden: 2304, layers: 26, gated: true },
];

export function familyOf(id: string): string {
  const lower = id.toLowerCase();
  if (lower.includes("qwen")) return "Qwen";
  if (lower.includes("llama")) return "Llama";
  if (lower.includes("gemma")) return "Gemma";
  if (lower.includes("smollm")) return "SmolLM";
  if (lower.includes("mistral")) return "Mistral";
  return "Hugging Face";
}

export function exportRecipe(teacher: TeacherCard): ExportRecipe {
  const nEmbd = Math.min(768, Math.max(256, Math.round(teacher.hidden / 2 / 64) * 64));
  const nLayer = teacher.layers >= 32 ? 12 : teacher.layers >= 20 ? 8 : 6;
  return {
    context: 128,
    levels: [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128],
    codebook: teacher.vocab >= 8000 ? 4096 : 1024,
    vqEmbed: Math.min(384, nEmbd),
    vqLayers: 6,
    generatorLayers: nLayer,
    generatorEmbed: nEmbd,
    generatorHeads: Math.max(4, Math.round(nEmbd / 64)),
  };
}

export function formatCount(value: number): string {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}
