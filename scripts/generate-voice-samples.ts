import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { KokoroTTS } from "kokoro-js";

type KokoroInstance = Awaited<ReturnType<typeof KokoroTTS.from_pretrained>>;
type KokoroVoice = NonNullable<Parameters<KokoroInstance["generate"]>[1]>["voice"];

type VoiceProfile = {
  id: string;
  overallGrade: string;
};

const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const SAMPLE_TEXT = "This is the voice that you will be hearing when you choose me as your audiobook creator.";

const QUALITY_RANK: Record<string, number> = {
  "A+": 8,
  A: 7,
  "A-": 6,
  "B+": 5,
  B: 4,
  "B-": 3,
  "C+": 2,
  C: 1,
  "C-": 0,
  "D+": -1,
  D: -2,
  "D-": -3,
  "F+": -4,
  F: -5,
};

function normalizeGrade(input: string): string {
  return input.replace(/[^A-F+\-]/gi, "").toUpperCase();
}

function gradeAboveC(input: string): boolean {
  const normalized = normalizeGrade(input);
  return (QUALITY_RANK[normalized] ?? -999) > QUALITY_RANK.C;
}

function parseUiVoicesFromMarkdown(markdown: string): VoiceProfile[] {
  const rows = markdown
    .split("\n")
    .filter((line) => line.trim().startsWith("|"))
    .map((line) => line.trim())
    .filter((line) => !line.includes("----"));

  const parsed: VoiceProfile[] = [];

  for (const row of rows) {
    const columns = row
      .split("|")
      .slice(1, -1)
      .map((item) => item.trim());

    if (columns.length < 5) {
      continue;
    }

    const name = columns[0].replace(/\*\*/g, "").replace(/\\_/g, "_").trim();
    const overallGrade = columns[4]?.replace(/\*\*/g, "").trim() ?? "";

    const isEnglishVoice = /^[ab][fm]_/i.test(name);
    if (!isEnglishVoice || !gradeAboveC(overallGrade)) {
      continue;
    }

    parsed.push({
      id: name,
      overallGrade,
    });
  }

  const unique = new Map<string, VoiceProfile>();
  for (const voice of parsed) {
    unique.set(voice.id, voice);
  }

  return Array.from(unique.values()).sort((a, b) => a.id.localeCompare(b.id));
}

async function saveAudio(audio: unknown, filePath: string) {
  const maybeAudio = audio as {
    save?: (target: string) => Promise<void> | void;
    toBlob?: () => Blob;
  };

  if (typeof maybeAudio.save === "function") {
    await maybeAudio.save(filePath);
    return;
  }

  if (typeof maybeAudio.toBlob === "function") {
    const blob = maybeAudio.toBlob();
    const bytes = Buffer.from(await blob.arrayBuffer());
    await writeFile(filePath, bytes);
    return;
  }

  throw new Error("Unsupported audio object returned by kokoro-js.");
}

async function main() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const projectRoot = path.resolve(__dirname, "..");
  const voicesMarkdownPath = path.join(projectRoot, "public", "voices.md");
  const outputDir = path.join(projectRoot, "public", "samples");

  const markdown = await readFile(voicesMarkdownPath, "utf8");
  const voices = parseUiVoicesFromMarkdown(markdown);

  if (voices.length === 0) {
    throw new Error("No UI voices were found from public/voices.md.");
  }

  await mkdir(outputDir, { recursive: true });

  console.log(`Found ${voices.length} voices. Loading Kokoro model...`);
  const tts = await KokoroTTS.from_pretrained(MODEL_ID, {
    dtype: "fp32",
    device: "cpu",
  });

  for (let i = 0; i < voices.length; i += 1) {
    const voice = voices[i];
    const outputPath = path.join(outputDir, `${voice.id}.wav`);
    console.log(`[${i + 1}/${voices.length}] Generating ${voice.id} (${voice.overallGrade})...`);

    const audio = await tts.generate(SAMPLE_TEXT, { voice: voice.id as KokoroVoice });
    await saveAudio(audio, outputPath);
  }

  console.log(`Done. Samples saved to: ${outputDir}`);
}

main().catch((error) => {
  console.error("Voice sample generation failed:", (error as Error).message);
  process.exitCode = 1;
});
