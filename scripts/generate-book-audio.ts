import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { KokoroTTS, TextSplitterStream } from "kokoro-js";
import { splitBookByUrl, splitRawTextByWords, type Chapter, type ManualSplitResult } from "../src/lib/book-parser";
import { expandTtsAbbreviations, removeTtsLineArtifacts } from "../src/lib/tts-abbreviation-rules";

type KokoroInstance = Awaited<ReturnType<typeof KokoroTTS.from_pretrained>>;
type KokoroVoice = NonNullable<Parameters<KokoroInstance["generate"]>[1]>["voice"];
type KokoroQuantization = "fp32" | "fp16" | "q8" | "q4";

type WavQuality = "low" | "medium" | "high";

type CliOptions = {
  sourceInput: string;
  splitName: string;
  textLength: number;
  bufferLength: number;
  wavQuality: WavQuality;
  voice: string;
  quantization: KokoroQuantization;
  useStream: boolean;
  outDir: string;
  maxSplits?: number;
};

type ParsedWav = {
  audioFormat: number;
  numChannels: number;
  sampleRate: number;
  bitsPerSample: number;
  samplesFloat: Float32Array;
};

type SynthesizedChunk = {
  chunkIndex: number;
  wordCount: number;
  text: string;
};

type SynthesizedSplitResult = {
  wavBuffer: Buffer;
  chunks: SynthesizedChunk[];
};

const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const DEFAULT_VOICE = "af_heart";
const DEFAULT_TEXT_LENGTH = 100;
const DEFAULT_BUFFER_LENGTH = 50;
const DEFAULT_SPLIT_NAME = "Chapter";
const DEFAULT_WAV_QUALITY: WavQuality = "high";
const DEFAULT_QUANTIZATION: KokoroQuantization = "fp32";
const TTS_DEVICE = "cpu" as const;

function printUsage() {
  console.log(`
Usage:
  npm run generate:book-audio -- <bookUrlOrPath> [splitName] [textLength] [bufferLength] [wavQuality] [--voice=<voiceId>] [--quantization=<q>] [--out=<dir>] [--max-splits=<n>]

Examples:
  npm run generate:book-audio -- "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"
  npm run generate:book-audio -- "https://www.gutenberg.org/cache/epub/2701/pg2701.txt" Chapter 120 40 high
  npm run generate:book-audio -- "https://www.gutenberg.org/cache/epub/2701/pg2701.txt" CHAPTER 90 30 medium --voice=am_michael --max-splits=2
  npm run generate:book-audio -- "public/books/Count_of_Monte_Christo.txt" Chapter 90 30 medium --quantization=q8
  npm run generate:book-audio -- "public/books/Count_of_Monte_Christo.txt" Chapter 90 30 medium --use_stream

Arguments:
  bookUrlOrPath HTTP(S) URL or local path to a plain text book (required)
  splitName     Word that starts each split/chapter (default: Chapter)
  textLength    Target words per generated text chunk (default: 100)
  bufferLength  Allowed +/- word buffer around target chunk size (default: 50)
  wavQuality    low | medium | high (default: high)

Flags:
  --voice       Kokoro voice id (default: af_heart)
  --quantization Kokoro quantization/dtype: fp32 | fp16 | q8 | q4 (default: fp32)
  --use_stream  Use chunkless streaming synthesis per split (default: false)
  --out         Output directory (default: ./generated-audio/<timestamp>)
  --max-splits  Limit number of splits generated (useful for testing)
`);
}

function parseBooleanFlag(input: string | undefined, fallback = false): boolean {
  if (!input) {
    return fallback;
  }

  const normalized = input.trim().toLowerCase();
  if (normalized === "true" || normalized === "1" || normalized === "yes" || normalized === "on") {
    return true;
  }
  if (normalized === "false" || normalized === "0" || normalized === "no" || normalized === "off") {
    return false;
  }

  // Presence-only flag (e.g. --use_stream) is parsed as "true" by parser.
  if (normalized === "") {
    return true;
  }

  throw new Error(`Invalid boolean flag value: ${input}`);
}

function sanitizeFileNamePart(value: string): string {
  return value
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-z0-9\-_. ]/gi, "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/ /g, "_")
    .slice(0, 90);
}

function parsePositiveInt(input: string | undefined, fallback: number, label: string): number {
  if (!input) {
    return fallback;
  }

  const parsed = Number(input);
  if (!Number.isFinite(parsed) || parsed < 1 || !Number.isInteger(parsed)) {
    throw new Error(`${label} must be a positive integer. Received: ${input}`);
  }

  return parsed;
}

function parseWavQuality(input: string | undefined): WavQuality {
  const normalized = (input ?? DEFAULT_WAV_QUALITY).toLowerCase();
  if (normalized === "low" || normalized === "medium" || normalized === "high") {
    return normalized;
  }
  throw new Error(`wavQuality must be one of: low, medium, high. Received: ${input}`);
}

function parseQuantization(input: string | undefined): KokoroQuantization {
  const normalized = (input ?? DEFAULT_QUANTIZATION).toLowerCase();
  if (normalized === "fp32" || normalized === "fp16" || normalized === "q8" || normalized === "q4") {
    return normalized;
  }

  throw new Error(`quantization must be one of: fp32, fp16, q8, q4. Received: ${input}`);
}

function parseArgs(argv: string[]): CliOptions {
  if (argv.includes("--help") || argv.includes("-h")) {
    printUsage();
    process.exit(0);
  }

  const positional: string[] = [];
  const flags = new Map<string, string>();

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i] ?? "";
    if (arg.startsWith("--")) {
      const normalized = arg.slice(2);
      if (normalized.includes("=")) {
        const [key, ...rest] = normalized.split("=");
        flags.set(key, rest.join("=") || "true");
      } else {
        const next = argv[i + 1];
        if (next && !next.startsWith("--")) {
          flags.set(normalized, next);
          i += 1;
        } else {
          flags.set(normalized, "true");
        }
      }
    } else {
      positional.push(arg);
    }
  }

  const sourceInput = positional[0]?.trim();
  if (!sourceInput) {
    printUsage();
    throw new Error("bookUrlOrPath is required.");
  }

  const splitName = positional[1]?.trim() || DEFAULT_SPLIT_NAME;
  const textLength = parsePositiveInt(positional[2], DEFAULT_TEXT_LENGTH, "textLength");
  const bufferLength = parsePositiveInt(positional[3], DEFAULT_BUFFER_LENGTH, "bufferLength");
  const wavQuality = parseWavQuality(positional[4]);
  const voice = (flags.get("voice") || DEFAULT_VOICE).trim();
  const quantization = parseQuantization(flags.get("quantization")?.trim());
  const useStream = parseBooleanFlag(flags.get("use_stream"), false);

  const maxSplitsRaw = flags.get("max-splits");
  const maxSplits = maxSplitsRaw ? parsePositiveInt(maxSplitsRaw, 1, "max-splits") : undefined;

  const customOutDir = flags.get("out")?.trim();
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outDir = customOutDir || path.join(process.cwd(), "generated-audio", timestamp);

  return {
    sourceInput,
    splitName,
    textLength,
    bufferLength,
    wavQuality,
    voice,
    quantization,
    useStream,
    outDir,
    maxSplits,
  };
}

function isHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value.trim());
}

function getCandidateLocalPaths(sourceInput: string): string[] {
  const input = sourceInput.trim();
  const candidates = new Set<string>();

  const projectRoot = process.cwd();
  const withoutLeadingSlashes = input.replace(/^\/+/, "");

  candidates.add(path.resolve(projectRoot, input));
  candidates.add(path.resolve(projectRoot, withoutLeadingSlashes));

  if (path.isAbsolute(input)) {
    candidates.add(path.normalize(input));
  }

  return Array.from(candidates);
}

async function splitBookFromInput(sourceInput: string, splitName: string): Promise<ManualSplitResult> {
  if (isHttpUrl(sourceInput)) {
    return splitBookByUrl({
      sourceUrl: sourceInput,
      splitWords: [splitName],
      caseSensitive: false,
    });
  }

  const candidates = getCandidateLocalPaths(sourceInput);

  for (const candidatePath of candidates) {
    try {
      const rawText = await readFile(candidatePath, "utf8");
      if (!rawText.trim()) {
        throw new Error(`Local file is empty: ${candidatePath}`);
      }

      return splitRawTextByWords({
        rawText,
        splitWords: [splitName],
        caseSensitive: false,
      });
    } catch {
      // Try next candidate.
    }
  }

  throw new Error(
    `Could not read local file '${sourceInput}'. Try an absolute path or a path relative to the project root.`,
  );
}

function splitIntoSentences(text: string): string[] {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return [];
  }

  const dotPlaceholder = "∯";
  const abbreviationPattern = /\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|No|vs|etc|e\.g|i\.e)\./gi;

  const protectedText = normalized
    .replace(abbreviationPattern, (_match, abbr: string) => `${abbr}${dotPlaceholder}`)
    .replace(/\b(?:[A-Z]\.){2,}/g, (match) => match.replace(/\./g, dotPlaceholder))
    .replace(/\b([A-Z])\.(?=\s+[A-Z])/g, `$1${dotPlaceholder}`)
    .replace(/\b\d+\.\d+\b/g, (match) => match.replace(/\./g, dotPlaceholder));

  const rawSentences = protectedText.match(/[^.!?]+[.!?]+(?:["')\]]+)?|[^.!?]+$/g) ?? [];

  return rawSentences
    .map((sentence) => sentence.replace(new RegExp(dotPlaceholder, "g"), ".").trim())
    .filter(Boolean);
}

function preprocessTextForKokoro(text: string): string {
  const withoutArtifacts = removeTtsLineArtifacts(text);
  const sanitized = withoutArtifacts
    .replace(/_/g, "")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/\s+/g, " ")
    .trim();

  return expandTtsAbbreviations(sanitized);
}

function chunkTextBySentenceWordBudget(text: string, targetWords: number, buffer: number): string[] {
  const sentences = splitIntoSentences(text);

  if (sentences.length === 0) {
    return [];
  }

  const minWords = Math.max(1, targetWords - buffer);
  const maxWords = targetWords + buffer;
  const chunks: string[] = [];
  let current: string[] = [];
  let currentWordCount = 0;

  const flushCurrent = () => {
    if (current.length > 0) {
      chunks.push(current.join(" "));
      current = [];
      currentWordCount = 0;
    }
  };

  for (const sentence of sentences) {
    const sentenceWordCount = sentence.split(/\s+/).filter(Boolean).length;

    if (sentenceWordCount > maxWords) {
      flushCurrent();

      const words = sentence.split(/\s+/).filter(Boolean);
      for (let i = 0; i < words.length; i += maxWords) {
        chunks.push(words.slice(i, i + maxWords).join(" "));
      }
      continue;
    }

    if (currentWordCount + sentenceWordCount > maxWords && currentWordCount >= minWords) {
      flushCurrent();
    }

    current.push(sentence);
    currentWordCount += sentenceWordCount;
  }

  flushCurrent();
  return chunks;
}

function buildChapterChunks(chapter: Chapter, textLength: number, bufferLength: number): string[] {
  const titleChunk = preprocessTextForKokoro(chapter.title);
  const bodyChunkSource = preprocessTextForKokoro(chapter.content);
  const bodyChunks = chunkTextBySentenceWordBudget(bodyChunkSource, textLength, bufferLength)
    .map((chunk) => chunk.trim())
    .filter(Boolean);

  return [titleChunk, ...bodyChunks].filter(Boolean);
}

function buildWholeSplitText(chapter: Chapter): string {
  const titleChunk = preprocessTextForKokoro(chapter.title);
  const contentChunk = preprocessTextForKokoro(chapter.content);
  return [titleChunk, contentChunk].filter(Boolean).join(" ").trim();
}

async function audioToBuffer(audio: unknown): Promise<Buffer> {
  const maybeAudio = audio as {
    save?: (target: string) => Promise<void> | void;
    toBlob?: () => Blob;
  };

  if (typeof maybeAudio.toBlob === "function") {
    const blob = maybeAudio.toBlob();
    return Buffer.from(await blob.arrayBuffer());
  }

  throw new Error("Unsupported audio object returned by kokoro-js.");
}

function parseWavChunk(buffer: Buffer): ParsedWav {
  if (buffer.length < 44) {
    throw new Error("Invalid WAV: too short.");
  }

  if (buffer.toString("ascii", 0, 4) !== "RIFF" || buffer.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error("Invalid WAV header.");
  }

  let audioFormat = 0;
  let numChannels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataSize = 0;

  let offset = 12;
  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString("ascii", offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    const chunkDataStart = offset + 8;
    const chunkDataEnd = chunkDataStart + chunkSize;

    if (chunkDataEnd > buffer.length) {
      throw new Error("Invalid WAV: corrupted chunk size.");
    }

    if (chunkId === "fmt ") {
      audioFormat = buffer.readUInt16LE(chunkDataStart);
      numChannels = buffer.readUInt16LE(chunkDataStart + 2);
      sampleRate = buffer.readUInt32LE(chunkDataStart + 4);
      bitsPerSample = buffer.readUInt16LE(chunkDataStart + 14);
    } else if (chunkId === "data") {
      dataOffset = chunkDataStart;
      dataSize = chunkSize;
      break;
    }

    offset = chunkDataEnd + (chunkSize % 2);
  }

  if (dataOffset < 0 || dataSize <= 0) {
    throw new Error("Invalid WAV: missing data chunk.");
  }

  const bytesPerSample = bitsPerSample / 8;
  if (!Number.isInteger(bytesPerSample) || bytesPerSample <= 0) {
    throw new Error(`Unsupported WAV bit depth: ${bitsPerSample}.`);
  }

  if ((audioFormat !== 1 && audioFormat !== 3) || (audioFormat === 3 && bitsPerSample !== 32)) {
    throw new Error(
      `Unsupported WAV format: audioFormat=${audioFormat}, bitsPerSample=${bitsPerSample}. Expected PCM (1) or IEEE float 32-bit (3).`,
    );
  }

  const sampleCount = Math.floor(dataSize / bytesPerSample);
  const samples = new Float32Array(sampleCount);

  for (let i = 0; i < sampleCount; i += 1) {
    const sampleOffset = dataOffset + i * bytesPerSample;

    if (audioFormat === 3) {
      samples[i] = Math.max(-1, Math.min(1, buffer.readFloatLE(sampleOffset)));
      continue;
    }

    if (bitsPerSample === 16) {
      const value = buffer.readInt16LE(sampleOffset);
      samples[i] = Math.max(-1, Math.min(1, value / 32768));
      continue;
    }

    if (bitsPerSample === 24) {
      const b0 = buffer[sampleOffset] ?? 0;
      const b1 = buffer[sampleOffset + 1] ?? 0;
      const b2 = buffer[sampleOffset + 2] ?? 0;
      const int24 = (b0 | (b1 << 8) | (b2 << 16)) << 8 >> 8;
      samples[i] = Math.max(-1, Math.min(1, int24 / 8388608));
      continue;
    }

    if (bitsPerSample === 32) {
      const value = buffer.readInt32LE(sampleOffset);
      samples[i] = Math.max(-1, Math.min(1, value / 2147483648));
      continue;
    }

    throw new Error(`Unsupported PCM bit depth: ${bitsPerSample}.`);
  }

  return {
    audioFormat,
    numChannels,
    sampleRate,
    bitsPerSample,
    samplesFloat: samples,
  };
}

function getBitDepthFromQuality(quality: WavQuality): 16 | 24 | 32 {
  if (quality === "low") {
    return 16;
  }
  if (quality === "medium") {
    return 24;
  }
  return 32;
}

function encodeWavFromFloat(args: {
  samplesFloat: Float32Array;
  numChannels: number;
  sampleRate: number;
  quality: WavQuality;
}): Buffer {
  const { samplesFloat, numChannels, sampleRate, quality } = args;

  const bitDepth = getBitDepthFromQuality(quality);
  const bytesPerSample = bitDepth === 16 ? 2 : bitDepth === 24 ? 3 : 4;
  const audioFormat = bitDepth === 32 ? 3 : 1; // 3 = IEEE float
  const dataSize = samplesFloat.length * bytesPerSample;
  const byteRate = sampleRate * numChannels * bytesPerSample;
  const blockAlign = numChannels * bytesPerSample;

  const output = Buffer.allocUnsafe(44 + dataSize);

  output.write("RIFF", 0, 4, "ascii");
  output.writeUInt32LE(36 + dataSize, 4);
  output.write("WAVE", 8, 4, "ascii");

  output.write("fmt ", 12, 4, "ascii");
  output.writeUInt32LE(16, 16);
  output.writeUInt16LE(audioFormat, 20);
  output.writeUInt16LE(numChannels, 22);
  output.writeUInt32LE(sampleRate, 24);
  output.writeUInt32LE(byteRate, 28);
  output.writeUInt16LE(blockAlign, 32);
  output.writeUInt16LE(bitDepth, 34);

  output.write("data", 36, 4, "ascii");
  output.writeUInt32LE(dataSize, 40);

  let offset = 44;

  if (bitDepth === 16) {
    for (let i = 0; i < samplesFloat.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, samplesFloat[i] ?? 0));
      const value = sample < 0 ? sample * 32768 : sample * 32767;
      output.writeInt16LE(Math.round(value), offset);
      offset += 2;
    }
    return output;
  }

  if (bitDepth === 24) {
    for (let i = 0; i < samplesFloat.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, samplesFloat[i] ?? 0));
      const value = Math.max(-8388608, Math.min(8388607, Math.round(sample * 8388607)));
      output[offset] = value & 0xff;
      output[offset + 1] = (value >> 8) & 0xff;
      output[offset + 2] = (value >> 16) & 0xff;
      offset += 3;
    }
    return output;
  }

  for (let i = 0; i < samplesFloat.length; i += 1) {
    const normalized = Math.max(-1, Math.min(1, samplesFloat[i] ?? 0));
    output.writeFloatLE(normalized, offset);
    offset += 4;
  }

  return output;
}

function combineSampleArrays(parts: Float32Array[]): Float32Array {
  const totalLength = parts.reduce((sum, current) => sum + current.length, 0);
  const merged = new Float32Array(totalLength);

  let offset = 0;
  for (const part of parts) {
    merged.set(part, offset);
    offset += part.length;
  }

  return merged;
}

async function synthesizeSplitWav(args: {
  tts: KokoroInstance;
  chapter: Chapter;
  voice: string;
  textLength: number;
  bufferLength: number;
  wavQuality: WavQuality;
  onChunkSynthesized?: (info: {
    chunkIndex: number;
    totalChunks: number | null;
    wordCount: number;
  }) => void;
}): Promise<SynthesizedSplitResult> {
  const { tts, chapter, voice, textLength, bufferLength, wavQuality, onChunkSynthesized } = args;

  const chunks = buildChapterChunks(chapter, textLength, bufferLength);
  if (chunks.length === 0) {
    throw new Error(`Split '${chapter.title}' does not contain any text to synthesize.`);
  }

  const parsedChunksByIndex: Array<ParsedWav | undefined> = new Array(chunks.length);
  const synthesizedChunksByIndex: Array<SynthesizedChunk | undefined> = new Array(chunks.length);

  let nextIndex = 0;
  const workerCount = Math.max(1, Math.min(2, chunks.length));

  const runWorker = async () => {
    while (true) {
      const chunkIndex = nextIndex;
      nextIndex += 1;

      if (chunkIndex >= chunks.length) {
        return;
      }

      const chunk = chunks[chunkIndex] ?? "";
      const audio = await tts.generate(chunk, { voice: voice as KokoroVoice });
      const wavBuffer = await audioToBuffer(audio);
      const parsed = parseWavChunk(wavBuffer);
      const words = chunk.split(/\s+/).filter(Boolean).length;

      parsedChunksByIndex[chunkIndex] = parsed;
      synthesizedChunksByIndex[chunkIndex] = {
        chunkIndex: chunkIndex + 1,
        wordCount: words,
        text: chunk,
      };

      onChunkSynthesized?.({
        chunkIndex: chunkIndex + 1,
        totalChunks: chunks.length,
        wordCount: words,
      });
    }
  };

  await Promise.all(Array.from({ length: workerCount }, () => runWorker()));

  const parsedChunks = parsedChunksByIndex.filter((item): item is ParsedWav => Boolean(item));
  const synthesizedChunks = synthesizedChunksByIndex.filter((item): item is SynthesizedChunk => Boolean(item));

  if (parsedChunks.length !== chunks.length || synthesizedChunks.length !== chunks.length) {
    throw new Error("Some chunks failed to generate.");
  }

  const reference = parsedChunks[0];
  for (let i = 1; i < parsedChunks.length; i += 1) {
    const parsed = parsedChunks[i];
    if (parsed.numChannels !== reference.numChannels || parsed.sampleRate !== reference.sampleRate) {
      throw new Error("Chunk WAV format mismatch; cannot stitch this split.");
    }
  }

  const first = parsedChunks[0];
  const mergedSamples = combineSampleArrays(parsedChunks.map((chunk) => chunk.samplesFloat));
  const wavBuffer = encodeWavFromFloat({
    samplesFloat: mergedSamples,
    numChannels: first.numChannels,
    sampleRate: first.sampleRate,
    quality: wavQuality,
  });

  return {
    wavBuffer,
    chunks: synthesizedChunks,
  };
}

async function synthesizeSplitWavViaStream(args: {
  tts: KokoroInstance;
  chapter: Chapter;
  voice: string;
  wavQuality: WavQuality;
  onChunkSynthesized?: (info: {
    chunkIndex: number;
    totalChunks: number | null;
    wordCount: number;
  }) => void;
}): Promise<SynthesizedSplitResult> {
  const { tts, chapter, voice, wavQuality, onChunkSynthesized } = args;
  const fullText = buildWholeSplitText(chapter);

  if (!fullText) {
    throw new Error(`Split '${chapter.title}' does not contain any text to synthesize.`);
  }

  const splitter = new TextSplitterStream();
  const stream = tts.stream(splitter, {
    voice: voice as KokoroVoice,
  });

  const feedPromise = (async () => {
    splitter.push(fullText);
    splitter.flush();
    splitter.close();
  })();

  const parsedChunks: ParsedWav[] = [];
  let streamSegmentIndex = 0;

  for await (const part of stream as AsyncIterable<{ audio?: unknown }>) {
    const audio = part?.audio;
    if (!audio) {
      continue;
    }

    const wavBuffer = await audioToBuffer(audio);
    const parsed = parseWavChunk(wavBuffer);
    parsedChunks.push(parsed);

    streamSegmentIndex += 1;
    onChunkSynthesized?.({
      chunkIndex: streamSegmentIndex,
      totalChunks: null,
      wordCount: 0,
    });
  }

  await feedPromise;

  if (parsedChunks.length === 0) {
    throw new Error("Stream synthesis produced no audio chunks.");
  }

  const reference = parsedChunks[0];
  for (let i = 1; i < parsedChunks.length; i += 1) {
    const parsed = parsedChunks[i];
    if (parsed.numChannels !== reference.numChannels || parsed.sampleRate !== reference.sampleRate) {
      throw new Error("Stream WAV format mismatch; cannot stitch this split.");
    }
  }

  const mergedSamples = combineSampleArrays(parsedChunks.map((chunk) => chunk.samplesFloat));
  const wavBuffer = encodeWavFromFloat({
    samplesFloat: mergedSamples,
    numChannels: reference.numChannels,
    sampleRate: reference.sampleRate,
    quality: wavQuality,
  });

  return {
    wavBuffer,
    chunks: [
      {
        chunkIndex: 1,
        wordCount: fullText.split(/\s+/).filter(Boolean).length,
        text: fullText,
      },
    ],
  };
}

async function writeSummaryFile(args: {
  outputDir: string;
  options: CliOptions;
  splitResult: ManualSplitResult;
  generatedCount: number;
  ttsRuntime: {
    modelId: string;
    quantization: string;
    device: string;
  };
  performance: {
    totalWords: number;
    elapsedSeconds: number;
    wordsPerSecond: number;
  };
}) {
  const { outputDir, options, splitResult, generatedCount, ttsRuntime, performance } = args;

  const summary = {
    generatedAt: new Date().toISOString(),
    source: options.sourceInput,
    splitName: options.splitName,
    splitCount: splitResult.chapters.length,
    generatedSplits: generatedCount,
    textLength: options.textLength,
    bufferLength: options.bufferLength,
    useStream: options.useStream,
    wavQuality: options.wavQuality,
    voice: options.voice,
    ttsRuntime,
    performance,
    metadata: splitResult.metadata,
  };

  const summaryPath = path.join(outputDir, "summary.json");
  await writeFile(summaryPath, JSON.stringify(summary, null, 2), "utf8");
}

async function main() {
  const options = parseArgs(process.argv.slice(2));

  console.log("Fetching and splitting book...");
  const splitResult = await splitBookFromInput(options.sourceInput, options.splitName);

  if (splitResult.chapters.length === 0) {
    throw new Error("No splits were found using the provided splitName.");
  }

  const splits = options.maxSplits ? splitResult.chapters.slice(0, options.maxSplits) : splitResult.chapters;

  await mkdir(options.outDir, { recursive: true });

  console.log(`Loading Kokoro model (${MODEL_ID})...`);
  console.log(`TTS runtime -> device: ${TTS_DEVICE}, quantization: ${options.quantization}`);
  console.log(`Synthesis mode: ${options.useStream ? "stream (chunkless per split)" : "generate (chunked)"}`);
  const tts = await KokoroTTS.from_pretrained(MODEL_ID, {
    dtype: options.quantization,
    device: TTS_DEVICE,
  });

  console.log(`Generating ${splits.length} split WAV files...`);

  const generationStartAt = process.hrtime.bigint();
  let totalGeneratedWords = 0;

  for (let index = 0; index < splits.length; index += 1) {
    const chapter = splits[index];
    const safeTitle = sanitizeFileNamePart(chapter.title) || `split_${chapter.index}`;
    const baseName = `${String(chapter.index).padStart(3, "0")}_${safeTitle}`;
    const fileName = `${baseName}.wav`;
    const outputPath = path.join(options.outDir, fileName);
    const chunkDebugPath = path.join(options.outDir, `${baseName}.chunks.json`);

    console.log(`[${index + 1}/${splits.length}] ${chapter.title}`);

    const synthesized = await (options.useStream
      ? synthesizeSplitWavViaStream({
          tts,
          chapter,
          voice: options.voice,
          wavQuality: options.wavQuality,
          onChunkSynthesized: ({ chunkIndex }) => {
            const elapsedSecondsSoFar = Number(process.hrtime.bigint() - generationStartAt) / 1_000_000_000;
            const wordsPerSecondSoFar = elapsedSecondsSoFar > 0 ? totalGeneratedWords / elapsedSecondsSoFar : 0;
            console.log(`      stream segment ${chunkIndex} · ${wordsPerSecondSoFar.toFixed(2)} words/s`);
          },
        })
      : synthesizeSplitWav({
          tts,
          chapter,
          voice: options.voice,
          textLength: options.textLength,
          bufferLength: options.bufferLength,
          wavQuality: options.wavQuality,
          onChunkSynthesized: ({ chunkIndex, totalChunks, wordCount }) => {
            totalGeneratedWords += wordCount;
            const elapsedSecondsSoFar = Number(process.hrtime.bigint() - generationStartAt) / 1_000_000_000;
            const wordsPerSecondSoFar = elapsedSecondsSoFar > 0 ? totalGeneratedWords / elapsedSecondsSoFar : 0;
            if (totalChunks) {
              console.log(`      chunk ${chunkIndex}/${totalChunks} (${wordCount} words) · ${wordsPerSecondSoFar.toFixed(2)} words/s`);
            } else {
              console.log(`      chunk ${chunkIndex} (${wordCount} words) · ${wordsPerSecondSoFar.toFixed(2)} words/s`);
            }
          },
        }));

    // In stream mode, count words once per split from full input text.
    if (options.useStream) {
      totalGeneratedWords += synthesized.chunks.reduce((sum, chunk) => sum + chunk.wordCount, 0);
    }

    await writeFile(outputPath, synthesized.wavBuffer);
    await writeFile(
      chunkDebugPath,
      JSON.stringify(
        {
          splitIndex: chapter.index,
          splitTitle: chapter.title,
          wavFileName: fileName,
          source: options.sourceInput,
          splitName: options.splitName,
          textLength: options.textLength,
          bufferLength: options.bufferLength,
          useStream: options.useStream,
          wavQuality: options.wavQuality,
          voice: options.voice,
          ttsRuntime: {
            modelId: MODEL_ID,
            quantization: options.quantization,
            device: TTS_DEVICE,
          },
          chunkCount: synthesized.chunks.length,
          chunks: synthesized.chunks,
        },
        null,
        2,
      ),
      "utf8",
    );
  }

  const elapsedSeconds = Number(process.hrtime.bigint() - generationStartAt) / 1_000_000_000;
  const wordsPerSecond = elapsedSeconds > 0 ? totalGeneratedWords / elapsedSeconds : 0;

  await writeSummaryFile({
    outputDir: options.outDir,
    options,
    splitResult,
    generatedCount: splits.length,
    ttsRuntime: {
      modelId: MODEL_ID,
      quantization: options.quantization,
      device: TTS_DEVICE,
    },
    performance: {
      totalWords: totalGeneratedWords,
      elapsedSeconds: Number(elapsedSeconds.toFixed(3)),
      wordsPerSecond: Number(wordsPerSecond.toFixed(3)),
    },
  });

  console.log("Done.");
  console.log(`Output directory: ${options.outDir}`);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Book audio generation failed: ${message}`);
  process.exitCode = 1;
});
