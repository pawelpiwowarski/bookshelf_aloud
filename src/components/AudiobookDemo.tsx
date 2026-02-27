"use client";

import JSZip from "jszip";
import { KokoroTTS } from "kokoro-js";
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { expandTtsAbbreviations, removeTtsLineArtifacts } from "@/lib/tts-abbreviation-rules";

type Chapter = {
  index: number;
  title: string;
  content: string;
};

type ManualSplitResult = {
  bookId: string;
  metadata: {
    title: string;
    author: string;
  };
  splitWords: string[];
  caseSensitive: boolean;
  strategyLabel: string;
  chapters: Chapter[];
};

const DEFAULT_SPLIT_WORDS = ["Chapter"];
const SPLIT_PRESETS: string[][] = [["Chapter"], ["CHAPTER"], ["Chapter", "CHAPTER"], ["Book"], ["Volume"]];

const KOKORO_MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const KOKORO_DEVICES = ["webgpu", "wasm", "cpu"] as const;

type KokoroDtype = "q8" | "q4" | "fp16" | "fp32";
type KokoroDevice = (typeof KOKORO_DEVICES)[number];
type KokoroInstance = Awaited<ReturnType<typeof KokoroTTS.from_pretrained>>;
type KokoroVoice = NonNullable<Parameters<KokoroInstance["generate"]>[1]>["voice"];

const FIXED_TTS_DTYPE: KokoroDtype = "fp32";

type VoiceProfile = {
  id: string;
  traits: string;
  targetQuality: string;
  overallGrade: string;
};

type GeneratedChunkAudio = {
  chunkIndex: number;
  wordCount: number;
  text: string;
  blob: Blob;
};

type OutputAudioFormat = "wav" | "mp3";

const FALLBACK_VOICE_PROFILES: VoiceProfile[] = [
  { id: "af_heart", traits: "🚺❤️", targetQuality: "A", overallGrade: "A" },
  { id: "af_aoede", traits: "🚺", targetQuality: "B", overallGrade: "C+" },
  { id: "af_bella", traits: "🚺🔥", targetQuality: "A", overallGrade: "A-" },
  { id: "af_kore", traits: "🚺", targetQuality: "B", overallGrade: "C+" },
  { id: "af_nicole", traits: "🚺🎧", targetQuality: "B", overallGrade: "B-" },
  { id: "af_sarah", traits: "🚺", targetQuality: "B", overallGrade: "C+" },
  { id: "am_fenrir", traits: "🚹", targetQuality: "B", overallGrade: "C+" },
  { id: "am_michael", traits: "🚹", targetQuality: "B", overallGrade: "C+" },
  { id: "am_puck", traits: "🚹", targetQuality: "B", overallGrade: "C+" },
  { id: "bf_emma", traits: "🚺", targetQuality: "B", overallGrade: "B-" },
];

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

function getVoiceIdentity(voiceId: string) {
  const parsed = voiceId.match(/^([ab])([fm])_(.+)$/i);
  if (!parsed) {
    return {
      accentFlag: "🏳️",
      accentLabel: "Unknown",
      genderLabel: "unknown",
      displayName: voiceId,
    };
  }

  const accentFlag = parsed[1].toLowerCase() === "a" ? "🇺🇸" : "🇬🇧";
  const accentLabel = parsed[1].toLowerCase() === "a" ? "American" : "British";
  const genderLabel = parsed[2].toLowerCase() === "m" ? "male" : "female";
  const displayName = parsed[3]
    .split("_")
    .map((part) => (part ? `${part[0].toUpperCase()}${part.slice(1)}` : part))
    .join(" ");

  return { accentFlag, accentLabel, genderLabel, displayName };
}

function formatVoiceLabel(voice: VoiceProfile): string {
  const identity = getVoiceIdentity(voice.id);
  return `${identity.accentFlag} ${identity.displayName} (${identity.genderLabel}) ${voice.traits || ""}`.trim();
}

function sanitizeFileNamePart(value: string): string {
  return value
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-z0-9\-_. ]/gi, "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/ /g, "_")
    .slice(0, 80);
}

function parseQualifiedVoicesFromMarkdown(markdown: string): VoiceProfile[] {
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
    const traits = columns[1] ?? "";
    const targetQuality = columns[2]?.replace(/\*\*/g, "").trim() ?? "";
    const overallGrade = columns[4]?.replace(/\*\*/g, "").trim() ?? "";

    const isEnglishVoice = /^[ab][fm]_/i.test(name);
    if (!isEnglishVoice || !gradeAboveC(overallGrade)) {
      continue;
    }

    parsed.push({
      id: name,
      traits,
      targetQuality,
      overallGrade,
    });
  }

  const unique = new Map<string, VoiceProfile>();
  for (const voice of parsed) {
    unique.set(voice.id, voice);
  }

  return Array.from(unique.values()).sort((a, b) => a.id.localeCompare(b.id));
}

function audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const numFrames = buffer.length;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = numFrames * blockAlign;
  const wavSize = 44 + dataSize;

  const arrayBuffer = new ArrayBuffer(wavSize);
  const view = new DataView(arrayBuffer);

  let offset = 0;
  const writeString = (value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
    offset += value.length;
  };

  writeString("RIFF");
  view.setUint32(offset, 36 + dataSize, true);
  offset += 4;
  writeString("WAVE");
  writeString("fmt ");
  view.setUint32(offset, 16, true);
  offset += 4;
  view.setUint16(offset, 1, true);
  offset += 2;
  view.setUint16(offset, numChannels, true);
  offset += 2;
  view.setUint32(offset, sampleRate, true);
  offset += 4;
  view.setUint32(offset, sampleRate * blockAlign, true);
  offset += 4;
  view.setUint16(offset, blockAlign, true);
  offset += 2;
  view.setUint16(offset, bytesPerSample * 8, true);
  offset += 2;
  writeString("data");
  view.setUint32(offset, dataSize, true);
  offset += 4;

  const channels = Array.from({ length: numChannels }, (_, channel) => buffer.getChannelData(channel));

  for (let i = 0; i < numFrames; i += 1) {
    for (let channel = 0; channel < numChannels; channel += 1) {
      const sample = Math.max(-1, Math.min(1, channels[channel][i] ?? 0));
      const pcm = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(offset, Math.round(pcm), true);
      offset += 2;
    }
  }

  return arrayBuffer;
}

function formatDuration(seconds: number): string {
  const safeSeconds = Math.max(0, Math.round(seconds));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const secs = safeSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
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

function normalizeInlineWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function isLikelyWrappedHeadingLine(line: string): boolean {
  const compact = line.trim();
  if (!compact) {
    return false;
  }

  const lettersOnly = compact.replace(/[^A-Za-z]/g, "");
  if (!lettersOnly) {
    return false;
  }

  const uppercaseLetters = lettersOnly.replace(/[^A-Z]/g, "").length;
  const uppercaseRatio = uppercaseLetters / lettersOnly.length;
  const wordCount = compact.split(/\s+/).filter(Boolean).length;

  return uppercaseRatio >= 0.75 && wordCount >= 2;
}

function promoteLeadingSplitTextToTitle(chapter: Chapter): Chapter {
  const originalTitle = chapter.title.trim();
  const originalContent = chapter.content;

  const lines = originalContent.split(/\r?\n/);
  const firstNonEmptyLineIndex = lines.findIndex((line) => line.trim().length > 0);
  const firstNonEmptyLine = firstNonEmptyLineIndex >= 0 ? (lines[firstNonEmptyLineIndex] ?? "").trim() : "";

  let consumedUntilLine = firstNonEmptyLineIndex;
  const headingLines: string[] = [];

  if (firstNonEmptyLineIndex >= 0) {
    headingLines.push(firstNonEmptyLine);

    let nextIndex = firstNonEmptyLineIndex + 1;
    while (nextIndex < lines.length) {
      const nextLine = (lines[nextIndex] ?? "").trim();
      if (!nextLine) {
        break;
      }

      if (!isLikelyWrappedHeadingLine(nextLine)) {
        break;
      }

      headingLines.push(nextLine);
      consumedUntilLine = nextIndex;
      nextIndex += 1;
    }
  }

  let extractedTitlePart = normalizeInlineWhitespace(headingLines.join(" "));

  if (!extractedTitlePart) {
    const firstSentence = splitIntoSentences(originalContent)[0]?.trim() ?? "";
    extractedTitlePart = firstSentence;
  }

  if (!extractedTitlePart) {
    return chapter;
  }

  const titleAlreadyContainsPart = originalTitle.toLowerCase().includes(extractedTitlePart.toLowerCase());
  const nextTitle = titleAlreadyContainsPart
    ? originalTitle
    : normalizeInlineWhitespace(`${originalTitle} ${extractedTitlePart}`);

  let nextContent = originalContent;
  if (consumedUntilLine >= firstNonEmptyLineIndex && firstNonEmptyLineIndex >= 0) {
    nextContent = lines.slice(consumedUntilLine + 1).join("\n").trimStart();
  } else {
    const escapedLeading = extractedTitlePart.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const removeLeadingPattern = new RegExp(`^\\s*${escapedLeading}\\s*`, "i");
    nextContent = originalContent.replace(removeLeadingPattern, "").trimStart();
  }

  return {
    ...chapter,
    title: nextTitle,
    content: nextContent,
  };
}

function removeLeadingHeadingBlock(text: string): string {
  return text.replace(
    /^\s*(?:chapter|book|volume)[^\n]*\n(?:\s*[A-Z0-9][A-Z0-9 ,;:'"’“”()\-—–]*\n){0,4}\s*/i,
    "",
  );
}

function getWordTokens(value: string): string[] {
  return (value.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter(Boolean);
}

function removeLeadingWordTokens(value: string, tokenCount: number): string {
  if (tokenCount <= 0) {
    return value;
  }

  const matcher = /[A-Za-z0-9]+/g;
  let match: RegExpExecArray | null = null;
  let seen = 0;

  while ((match = matcher.exec(value)) !== null) {
    seen += 1;
    if (seen >= tokenCount) {
      const cutIndex = (match.index ?? 0) + match[0].length;
      return value.slice(cutIndex).replace(/^\s*[-–—:;,.!?]+\s*/, "").trimStart();
    }
  }

  return "";
}

function stripRepeatedIntroPrefix(bodyText: string, introText: string): string {
  const bodyTokens = getWordTokens(bodyText);
  const introTokens = getWordTokens(introText);

  if (bodyTokens.length === 0 || introTokens.length === 0) {
    return bodyText;
  }

  let matched = 0;
  while (matched < bodyTokens.length && matched < introTokens.length && bodyTokens[matched] === introTokens[matched]) {
    matched += 1;
  }

  const requiredMatch = Math.min(introTokens.length, Math.max(4, Math.floor(introTokens.length * 0.7)));
  if (matched < requiredMatch) {
    return bodyText;
  }

  return removeLeadingWordTokens(bodyText, matched).trim();
}

function stripOverlapWithPreviousChunk(currentText: string, previousText: string): string {
  const currentTokens = getWordTokens(currentText);
  const previousTokens = getWordTokens(previousText);

  if (currentTokens.length === 0 || previousTokens.length === 0) {
    return currentText;
  }

  const maxOverlap = Math.min(currentTokens.length, previousTokens.length);
  let bestOverlap = 0;

  for (let overlap = maxOverlap; overlap >= 3; overlap -= 1) {
    let same = true;
    for (let i = 0; i < overlap; i += 1) {
      if (currentTokens[i] !== previousTokens[previousTokens.length - overlap + i]) {
        same = false;
        break;
      }
    }

    if (same) {
      bestOverlap = overlap;
      break;
    }
  }

  if (bestOverlap === 0) {
    return currentText;
  }

  return removeLeadingWordTokens(currentText, bestOverlap).trim();
}

function enforceNoAdjacentChunkOverlap(chunks: string[]): string[] {
  if (chunks.length <= 1) {
    return chunks;
  }

  const result: string[] = [chunks[0] ?? ""];

  for (let i = 1; i < chunks.length; i += 1) {
    const previous = result[result.length - 1] ?? "";
    const current = chunks[i] ?? "";
    const withoutOverlap = stripOverlapWithPreviousChunk(current, previous);
    if (withoutOverlap.trim()) {
      result.push(withoutOverlap.trim());
    }
  }

  return result;
}

export default function AudiobookDemo() {
  const ttsRef = useRef<KokoroInstance | null>(null);
  const ttsConfigRef = useRef<{ dtype: KokoroDtype; device: KokoroDevice } | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const resumePlaybackRef = useRef<{ shouldResume: boolean; currentTime: number }>({
    shouldResume: false,
    currentTime: 0,
  });

  const [books, setBooks] = useState<string[]>([]);
  const [booksLoading, setBooksLoading] = useState(true);
  const [booksError, setBooksError] = useState<string | null>(null);

  const [selectedBook, setSelectedBook] = useState<string>("");
  const [customWordsInput, setCustomWordsInput] = useState(DEFAULT_SPLIT_WORDS.join(", "));
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [treatFirstSentenceAsTitle, setTreatFirstSentenceAsTitle] = useState(false);

  const [isSplitting, setIsSplitting] = useState(false);
  const [splitError, setSplitError] = useState<string | null>(null);
  const [result, setResult] = useState<ManualSplitResult | null>(null);
  const [showDefaultSplitWarning, setShowDefaultSplitWarning] = useState(false);
  const [editingChapterIndex, setEditingChapterIndex] = useState<number | null>(null);
  const [editingChapterTitle, setEditingChapterTitle] = useState("");
  const [editingChapterContent, setEditingChapterContent] = useState("");
  const [editSplitError, setEditSplitError] = useState<string | null>(null);
  const [expandedChapterIndexes, setExpandedChapterIndexes] = useState<Set<number>>(new Set());
  const [voiceProfiles, setVoiceProfiles] = useState<VoiceProfile[]>([]);
  const [voicesLoading, setVoicesLoading] = useState(true);
  const [voicesError, setVoicesError] = useState<string | null>(null);
  const [ttsVoice, setTtsVoice] = useState<KokoroVoice>("af_heart");
  const [voiceSampleLoadError, setVoiceSampleLoadError] = useState<string | null>(null);
  const [previewAudioUrl, setPreviewAudioUrl] = useState<string | null>(null);
  const [zipDownloadUrl, setZipDownloadUrl] = useState<string | null>(null);
  const [zipFileName, setZipFileName] = useState<string | null>(null);
  const [generatedChunkAudios, setGeneratedChunkAudios] = useState<GeneratedChunkAudio[]>([]);
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false);
  const [generationProgressPct, setGenerationProgressPct] = useState(0);
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState<string | null>(null);
  const [ttsStatus, setTtsStatus] = useState("Select voice and generate preview.");
  const [outputAudioFormat, setOutputAudioFormat] = useState<OutputAudioFormat>("wav");
  const [currentStep, setCurrentStep] = useState(1);

  // Book source: "library" | "file" | "url"
  const [bookSource, setBookSource] = useState<"library" | "file" | "url">("library");
  const [droppedFileName, setDroppedFileName] = useState<string | null>(null);
  const [droppedFileText, setDroppedFileText] = useState<string | null>(null);
  const [urlInput, setUrlInput] = useState("");
  const [urlLoading, setUrlLoading] = useState(false);
  const [urlError, setUrlError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Split selection for download
  const [selectedSplitIndexes, setSelectedSplitIndexes] = useState<Set<number>>(new Set());

  useEffect(() => {
    const loadBooks = async () => {
      setBooksLoading(true);
      setBooksError(null);

      try {
        const response = await fetch("/api/books");
        const data = (await response.json()) as { books?: string[]; error?: string };
        if (!response.ok) {
          throw new Error(data.error || "Failed to load books.");
        }

        const nextBooks = data.books ?? [];
        setBooks(nextBooks);
        if (nextBooks.length > 0) {
          setSelectedBook((prev) => prev || nextBooks[0]);
        }
      } catch (error) {
        setBooksError((error as Error).message);
      } finally {
        setBooksLoading(false);
      }
    };

    void loadBooks();
  }, []);

  useEffect(() => {
    const loadVoiceProfiles = async () => {
      setVoicesLoading(true);
      setVoicesError(null);
      try {
        const response = await fetch("/voices.md");
        if (!response.ok) {
          throw new Error("Failed to fetch voices metadata.");
        }
        const markdown = await response.text();
        const parsed = parseQualifiedVoicesFromMarkdown(markdown);
        const profiles = parsed.length > 0 ? parsed : FALLBACK_VOICE_PROFILES;
        setVoiceProfiles(profiles);

        if (profiles.some((voice) => voice.id === "af_heart")) {
          setTtsVoice("af_heart");
        } else if (profiles.length > 0) {
          setTtsVoice(profiles[0].id as KokoroVoice);
        }
      } catch (error) {
        setVoicesError((error as Error).message);
        setVoiceProfiles(FALLBACK_VOICE_PROFILES);
      } finally {
        setVoicesLoading(false);
      }
    };

    void loadVoiceProfiles();
  }, []);

  const canSplit = useMemo(() => {
    if (isSplitting) return false;
    if (bookSource === "library") return Boolean(selectedBook);
    if (bookSource === "file") return Boolean(droppedFileText);
    if (bookSource === "url") return Boolean(urlInput.trim());
    return false;
  }, [bookSource, selectedBook, droppedFileText, urlInput, isSplitting]);

  const isDefaultSplitWords = useCallback((words: string[]) => {
    if (words.length !== DEFAULT_SPLIT_WORDS.length) {
      return false;
    }

    return words.every((word, index) => word.trim().toLowerCase() === (DEFAULT_SPLIT_WORDS[index] ?? "").trim().toLowerCase());
  }, []);

  const runSplit = useCallback(async (words: string[]) => {
    if (!selectedBook || words.length === 0) {
      return;
    }

    setIsSplitting(true);
    setSplitError(null);

    try {
      const response = await fetch(`/api/books/${encodeURIComponent(selectedBook)}/split`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          splitWords: words,
          caseSensitive,
        }),
      });

      const data = (await response.json()) as ManualSplitResult & { error?: string };

      if (!response.ok) {
        throw new Error(data.error || "Failed to split book.");
      }

      const nextResult = treatFirstSentenceAsTitle
        ? {
            ...data,
            chapters: data.chapters.map((chapter) => promoteLeadingSplitTextToTitle(chapter)),
          }
        : data;

      setResult(nextResult);
      setShowDefaultSplitWarning(isDefaultSplitWords(words) && data.chapters.length <= 1);
      setExpandedChapterIndexes(new Set());
    } catch (error) {
      setSplitError((error as Error).message);
      setResult(null);
      setShowDefaultSplitWarning(false);
    } finally {
      setIsSplitting(false);
    }
  }, [selectedBook, caseSensitive, isDefaultSplitWords, treatFirstSentenceAsTitle]);

  useEffect(() => {
    if (!selectedBook || booksLoading) {
      return;
    }

    void runSplit(DEFAULT_SPLIT_WORDS);
  }, [selectedBook, booksLoading, runSplit]);

  // Sync selectedSplitIndexes when result changes
  useEffect(() => {
    if (result && result.chapters.length > 0) {
      setSelectedSplitIndexes(new Set(result.chapters.map((ch) => ch.index)));
    } else {
      setSelectedSplitIndexes(new Set());
    }
  }, [result]);

  const runSplitFromUrl = useCallback(async (url: string, words: string[]) => {
    if (!url || words.length === 0) return;
    setIsSplitting(true);
    setSplitError(null);
    setUrlError(null);
    try {
      const response = await fetch("/api/books/url/split", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sourceUrl: url, splitWords: words, caseSensitive }),
      });
      const data = (await response.json()) as ManualSplitResult & { error?: string };
      if (!response.ok) throw new Error(data.error || "Failed to split from URL.");
      const nextResult = treatFirstSentenceAsTitle
        ? { ...data, chapters: data.chapters.map((ch) => promoteLeadingSplitTextToTitle(ch)) }
        : data;
      setResult(nextResult);
      setShowDefaultSplitWarning(false);
      setExpandedChapterIndexes(new Set());
    } catch (error) {
      setSplitError((error as Error).message);
      setResult(null);
    } finally {
      setIsSplitting(false);
    }
  }, [caseSensitive, treatFirstSentenceAsTitle]);

  const runSplitFromText = useCallback(async (rawText: string, words: string[]) => {
    if (!rawText || words.length === 0) return;
    setIsSplitting(true);
    setSplitError(null);
    try {
      const response = await fetch("/api/books/text/split", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rawText, splitWords: words, caseSensitive }),
      });
      const data = (await response.json()) as ManualSplitResult & { error?: string };
      if (!response.ok) throw new Error(data.error || "Failed to split text.");
      const nextResult = treatFirstSentenceAsTitle
        ? { ...data, chapters: data.chapters.map((ch) => promoteLeadingSplitTextToTitle(ch)) }
        : data;
      setResult(nextResult);
      setShowDefaultSplitWarning(false);
      setExpandedChapterIndexes(new Set());
    } catch (error) {
      setSplitError((error as Error).message);
      setResult(null);
    } finally {
      setIsSplitting(false);
    }
  }, [caseSensitive, treatFirstSentenceAsTitle]);

  const handleFileDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".txt")) {
      setSplitError("Only .txt files are supported.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const text = reader.result as string;
      setDroppedFileName(file.name);
      setDroppedFileText(text);
      setSelectedBook("");
      setUrlInput("");
      setSplitError(null);
    };
    reader.onerror = () => {
      setSplitError("Failed to read file.");
    };
    reader.readAsText(file);
  }, []);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".txt")) {
      setSplitError("Only .txt files are supported.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const text = reader.result as string;
      setDroppedFileName(file.name);
      setDroppedFileText(text);
      setSelectedBook("");
      setUrlInput("");
      setSplitError(null);
    };
    reader.onerror = () => {
      setSplitError("Failed to read file.");
    };
    reader.readAsText(file);
  }, []);

  const handleUrlSubmit = useCallback(async () => {
    const url = urlInput.trim();
    if (!url) { setUrlError("Enter a URL."); return; }
    setUrlLoading(true);
    setUrlError(null);
    setDroppedFileName(null);
    setDroppedFileText(null);
    setSelectedBook("");
    // Just validate and proceed — actual split happens in step 2
    try {
      new URL(url);
    } catch {
      setUrlError("Invalid URL format.");
      setUrlLoading(false);
      return;
    }
    setUrlLoading(false);
  }, [urlInput]);

  const applyCustomSplit = async () => {
    const words = customWordsInput
      .split(",")
      .map((word) => word.trim())
      .filter(Boolean);

    if (words.length === 0) {
      setSplitError("Add at least one split word.");
      return;
    }

    if (bookSource === "url" && urlInput.trim()) {
      await runSplitFromUrl(urlInput.trim(), words);
    } else if (bookSource === "file" && droppedFileText) {
      await runSplitFromText(droppedFileText, words);
    } else {
      await runSplit(words);
    }
  };

  const clearGeneratedAudio = useCallback(() => {
    setGeneratedChunkAudios(() => {
      return [];
    });

    setPreviewAudioUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return null;
    });

    setZipDownloadUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return null;
    });
    setZipFileName(null);
  }, []);

  useEffect(() => {
    return () => {
      clearGeneratedAudio();
    };
  }, [clearGeneratedAudio]);

  useEffect(() => {
    const element = previewAudioRef.current;
    if (!element || !previewAudioUrl) {
      return;
    }

    const { shouldResume, currentTime } = resumePlaybackRef.current;
    if (!shouldResume) {
      return;
    }

    const resumePlayback = () => {
      const safeTime = Math.max(0, Math.min(currentTime, Number.isFinite(element.duration) ? element.duration : currentTime));
      element.currentTime = safeTime;
      void element.play().catch(() => {
        // Ignore autoplay restrictions.
      });
      resumePlaybackRef.current = { shouldResume: false, currentTime: 0 };
    };

    if (element.readyState >= 1) {
      resumePlayback();
      return;
    }

    element.addEventListener("loadedmetadata", resumePlayback, { once: true });
    return () => {
      element.removeEventListener("loadedmetadata", resumePlayback);
    };
  }, [previewAudioUrl]);

  useEffect(() => {
    const originalWarn = console.warn.bind(console);
    const originalError = console.error.bind(console);

    const shouldIgnoreOrtWarning = (args: unknown[]) => {
      const text = args
        .map((arg) => {
          if (typeof arg === "string") return arg;
          if (arg instanceof Error) return arg.message;
          return "";
        })
        .join(" ");

      return (
        text.includes("VerifyEachNodeIsAssignedToAnEp") ||
        text.includes("Some nodes were not assigned to the preferred execution providers")
      );
    };

    console.warn = (...args: unknown[]) => {
      if (shouldIgnoreOrtWarning(args)) {
        return;
      }
      originalWarn(...args);
    };

    console.error = (...args: unknown[]) => {
      if (shouldIgnoreOrtWarning(args)) {
        return;
      }
      originalError(...args);
    };

    return () => {
      console.warn = originalWarn;
      console.error = originalError;
    };
  }, []);

  const chunkTextBySentenceWordBudget = useCallback((text: string, targetWords = 50, buffer = 30) => {
    const sentences = splitIntoSentences(text);

    if (sentences.length === 0) {
      return [] as string[];
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
  }, []);

  const sanitizeTextForKokoro = useCallback((text: string) => {
    return text
      .replace(/_/g, "")
      .replace(/[“”]/g, '"')
      .replace(/[‘’]/g, "'")
      .replace(/\s+/g, " ")
      .trim();
  }, []);

  const preprocessTextForKokoro = useCallback((text: string) => {
    const withoutArtifacts = removeTtsLineArtifacts(text);
    const sanitized = sanitizeTextForKokoro(withoutArtifacts);
    return expandTtsAbbreviations(sanitized);
  }, [sanitizeTextForKokoro]);

  const selectedVoiceSampleUrl = useMemo(() => {
    return `/samples/${encodeURIComponent(String(ttsVoice))}.wav`;
  }, [ttsVoice]);

  useEffect(() => {
    setVoiceSampleLoadError(null);
  }, [selectedVoiceSampleUrl]);

  const buildTtsChunksForChapter = useCallback((chapter: Chapter, introOverride = "", targetWords = 50, buffer = 30) => {
    const chapterIntro = normalizeInlineWhitespace(introOverride);
    const spokenIntro = chapterIntro ? preprocessTextForKokoro(chapterIntro) : "";
    const bodySource = chapterIntro ? removeLeadingHeadingBlock(chapter.content) : chapter.content;
    const preprocessed = preprocessTextForKokoro(bodySource);
    const normalizedIntroForMatch = spokenIntro;
    const dedupedBody = normalizedIntroForMatch ? stripRepeatedIntroPrefix(preprocessed, normalizedIntroForMatch) : preprocessed;
    const sentenceParts = splitIntoSentences(dedupedBody).map((sentence) => sentence.trim()).filter(Boolean);

    if (sentenceParts.length === 0) {
      return spokenIntro ? [spokenIntro] : [];
    }

    const bodyText = sentenceParts.join(" ").trim();
    const bodyChunks = bodyText ? chunkTextBySentenceWordBudget(bodyText, targetWords, buffer) : [];

    if (normalizedIntroForMatch && bodyChunks.length > 0) {
      const firstBodyChunk = stripRepeatedIntroPrefix(bodyChunks[0], normalizedIntroForMatch).trim();
      if (firstBodyChunk) {
        bodyChunks[0] = firstBodyChunk;
      } else {
        bodyChunks.shift();
      }
    }

    const merged = [spokenIntro, ...bodyChunks].map((chunk) => chunk.trim()).filter(Boolean);
    return enforceNoAdjacentChunkOverlap(merged);
  }, [chunkTextBySentenceWordBudget, preprocessTextForKokoro]);

  const stitchAudioBlobsToAudioBuffer = useCallback(async (blobs: Blob[]) => {
    if (blobs.length === 0) {
      throw new Error("No audio chunks to stitch.");
    }

    const audioContext = new AudioContext();

    try {
      const decodedBuffers: AudioBuffer[] = [];

      for (const blob of blobs) {
        const arrayBuffer = await blob.arrayBuffer();
        const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
        decodedBuffers.push(decoded);
      }

      const sampleRate = decodedBuffers[0].sampleRate;
      const numberOfChannels = decodedBuffers[0].numberOfChannels;
      const totalLength = decodedBuffers.reduce((sum, buffer) => sum + buffer.length, 0);

      const stitched = audioContext.createBuffer(numberOfChannels, totalLength, sampleRate);
      let writeOffset = 0;

      for (const buffer of decodedBuffers) {
        for (let channel = 0; channel < numberOfChannels; channel += 1) {
          stitched.copyToChannel(buffer.getChannelData(channel), channel, writeOffset);
        }
        writeOffset += buffer.length;
      }

      return stitched;
    } finally {
      await audioContext.close();
    }
  }, []);

  const convertWavToMp3ViaApi = useCallback(async (wavBuffer: ArrayBuffer): Promise<Blob> => {
    const response = await fetch("/api/audio/mp3", {
      method: "POST",
      headers: {
        "Content-Type": "audio/wav",
      },
      body: wavBuffer,
    });

    if (!response.ok) {
      let message = "Failed to convert WAV to MP3.";
      try {
        const data = (await response.json()) as { error?: string };
        if (data?.error) {
          message = data.error;
        }
      } catch {
        // Ignore JSON parsing errors.
      }
      throw new Error(message);
    }

    return response.blob();
  }, []);

  const ensureTtsLoaded = useCallback(async () => {
    const hasWebGpu = Boolean((navigator as Navigator & { gpu?: unknown }).gpu);
    const hasWasm = typeof WebAssembly !== "undefined";
    const preferredDevice =
      KOKORO_DEVICES.find((device) => {
        if (device === "webgpu") return hasWebGpu;
        if (device === "wasm") return hasWasm;
        return true;
      }) ?? "cpu";

    const hasSameConfig =
      ttsRef.current &&
      ttsConfigRef.current &&
      ttsConfigRef.current.dtype === FIXED_TTS_DTYPE &&
      ttsConfigRef.current.device === preferredDevice;

    if (hasSameConfig) {
      return ttsRef.current as KokoroInstance;
    }

    setTtsStatus(`Loading Kokoro (${FIXED_TTS_DTYPE}, ${preferredDevice})...`);

    try {
      const ortModule = (await import("onnxruntime-web")) as {
        env?: { logLevel?: string };
      };
      if (ortModule.env) {
        ortModule.env.logLevel = "error";
      }
    } catch {
      // Optional optimization only.
    }

    const tts = await KokoroTTS.from_pretrained(KOKORO_MODEL_ID, {
      dtype: FIXED_TTS_DTYPE,
      device: preferredDevice,
    });

    ttsRef.current = tts;
    ttsConfigRef.current = { dtype: FIXED_TTS_DTYPE, device: preferredDevice };
    return tts;
  }, []);

  const synthesizeChunkAudios = useCallback(async (
    chunks: string[],
    onChunkProgress?: (chunkNumber: number, totalChunks: number) => void,
  ): Promise<GeneratedChunkAudio[]> => {
    const tts = await ensureTtsLoaded();
    const nextChunkAudios: GeneratedChunkAudio[] = [];

    for (let index = 0; index < chunks.length; index += 1) {
      const visibleChunkText = chunks[index] ?? "";
      const wordCount = visibleChunkText.split(/\s+/).filter(Boolean).length;
      if (wordCount === 0) {
        continue;
      }

      const chunkNumber = index + 1;
      const audio = await tts.generate(visibleChunkText, {
        voice: ttsVoice as KokoroVoice,
      });

      onChunkProgress?.(chunkNumber, chunks.length);

      nextChunkAudios.push({
        chunkIndex: chunkNumber,
        wordCount,
        text: visibleChunkText,
        blob: audio.toBlob(),
      });
    }

    return nextChunkAudios;
  }, [ensureTtsLoaded, ttsVoice]);

  const playAudiobookPreviewForAllSplits = useCallback(async () => {
    if (!result || result.chapters.length === 0) {
      setTtsStatus("No splits found. Run a split first.");
      return;
    }

    const chaptersToGenerate = result.chapters.filter((ch) => selectedSplitIndexes.has(ch.index));
    if (chaptersToGenerate.length === 0) {
      setTtsStatus("No splits selected for generation.");
      return;
    }

    setIsGeneratingPreview(true);
    setGenerationProgressPct(0);
    setEstimatedTimeRemaining(null);
    setTtsStatus(`Generating ${chaptersToGenerate.length} selected split ${outputAudioFormat.toUpperCase()} files...`);

    try {
      clearGeneratedAudio();
      const zip = new JSZip();
      const totalSplits = chaptersToGenerate.length;
      let addedFiles = 0;
      const generationStartedAtMs = Date.now();

      const bookTitleChunk = normalizeInlineWhitespace(
        `${result.metadata.title.trim()}${result.metadata.author.trim() ? ` by ${result.metadata.author.trim()}` : ""}`,
      );

      // Pre-compute all TTS chunks per split so we know the true total
      const allSplitChunks: { chapter: typeof chaptersToGenerate[number]; chunks: string[] }[] = [];
      let totalChunksAllSplits = 0;

      for (let splitIndex = 0; splitIndex < totalSplits; splitIndex += 1) {
        const chapter = chaptersToGenerate[splitIndex];
        const isFirstChapter = chapter.index === result.chapters[0]?.index;
        const chapterTitleChunk = preprocessTextForKokoro(normalizeInlineWhitespace(chapter.title));
        const chapterChunks = buildTtsChunksForChapter(chapter, chapter.title, 50, 30)
          .map((chunk) => chunk.trim())
          .filter(Boolean);

        const chunksForSplit = isFirstChapter
          ? enforceNoAdjacentChunkOverlap([preprocessTextForKokoro(bookTitleChunk), chapterTitleChunk, ...chapterChunks].filter(Boolean))
          : enforceNoAdjacentChunkOverlap([chapterTitleChunk, ...chapterChunks].filter(Boolean));

        if (chunksForSplit.length > 0) {
          allSplitChunks.push({ chapter, chunks: chunksForSplit });
          totalChunksAllSplits += chunksForSplit.length;
        }
      }

      let chunksProcessedSoFar = 0;

      for (let i = 0; i < allSplitChunks.length; i += 1) {
        const { chapter, chunks } = allSplitChunks[i];

        setTtsStatus(`Generating split ${i + 1}/${allSplitChunks.length}...`);

        const splitChunkAudios = await synthesizeChunkAudios(chunks, (chunkNumber) => {
          const processed = chunksProcessedSoFar + chunkNumber;
          const pct = Math.min(95, Math.round((processed / totalChunksAllSplits) * 95));
          const elapsedSeconds = Math.max(1, (Date.now() - generationStartedAtMs) / 1000);
          const chunksPerSecond = processed / elapsedSeconds;
          const remainingChunks = Math.max(0, totalChunksAllSplits - processed);
          const etaSeconds = chunksPerSecond > 0 ? remainingChunks / chunksPerSecond : 0;
          const etaLabel = formatDuration(etaSeconds);

          setEstimatedTimeRemaining(etaLabel);
          setGenerationProgressPct(pct);
          setTtsStatus(`Generating chunk ${processed}/${totalChunksAllSplits}... (${pct}%) · ETA ~${etaLabel}`);
        });

        chunksProcessedSoFar += chunks.length;

        if (splitChunkAudios.length === 0) {
          continue;
        }

        const stitchedBuffer = await stitchAudioBlobsToAudioBuffer(splitChunkAudios.map((item) => item.blob));
        const safeTitle = sanitizeFileNamePart(chapter.title) || `split_${chapter.index}`;
        if (outputAudioFormat === "mp3") {
          const wavBuffer = audioBufferToWav(stitchedBuffer);
          const splitMp3Blob = await convertWavToMp3ViaApi(wavBuffer);
          zip.file(`${safeTitle}.mp3`, splitMp3Blob);
        } else {
          const wavBuffer = audioBufferToWav(stitchedBuffer);
          const splitWavBlob = new Blob([wavBuffer], { type: "audio/wav" });
          zip.file(`${safeTitle}.wav`, splitWavBlob);
        }
        addedFiles += 1;
      }

      if (addedFiles === 0) {
        throw new Error("No non-empty splits to export.");
      }

      setTtsStatus("Packaging ZIP archive...");
      const zipBlob = await zip.generateAsync(
        { type: "blob" },
        (metadata) => {
          const pct = Math.min(100, Math.max(95, Math.round(95 + metadata.percent * 0.05)));
          setGenerationProgressPct(pct);
        },
      );

      const safeBookTitle = sanitizeFileNamePart(result.metadata.title || "audiobook") || "audiobook";
      const nextZipFileName = `${safeBookTitle}.zip`;
      const nextZipUrl = URL.createObjectURL(zipBlob);

      setZipDownloadUrl((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous);
        }
        return nextZipUrl;
      });
      setZipFileName(nextZipFileName);
      setGenerationProgressPct(100);
      setEstimatedTimeRemaining(null);
      setTtsStatus(`Done. ZIP is ready with ${addedFiles} split ${outputAudioFormat.toUpperCase()} files.`);

      const anchor = document.createElement("a");
      anchor.href = nextZipUrl;
      anchor.download = nextZipFileName;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    } catch (error) {
      setEstimatedTimeRemaining(null);
      setTtsStatus(`ZIP export error: ${(error as Error).message}`);
    } finally {
      setIsGeneratingPreview(false);
    }
  }, [buildTtsChunksForChapter, clearGeneratedAudio, convertWavToMp3ViaApi, outputAudioFormat, preprocessTextForKokoro, result, selectedSplitIndexes, stitchAudioBlobsToAudioBuffer, synthesizeChunkAudios]);

  const getChapterPreviewText = (text: string) => {
    const normalized = text.replace(/\s+/g, " ").trim();
    const sentenceParts = splitIntoSentences(normalized);

    if (sentenceParts.length <= 5) {
      return normalized;
    }

    const start = sentenceParts.slice(0, 2).join(" ");
    const end = sentenceParts.slice(-2).join(" ");
    return `${start}\n\n…\n\n${end}`;
  };

  const toggleExpandedChapter = (chapterIndex: number) => {
    setExpandedChapterIndexes((previous) => {
      const next = new Set(previous);
      if (next.has(chapterIndex)) {
        next.delete(chapterIndex);
      } else {
        next.add(chapterIndex);
      }
      return next;
    });
  };

  const openEditSplitDialog = useCallback((chapter: Chapter) => {
    setEditingChapterIndex(chapter.index);
    setEditingChapterTitle(chapter.title);
    setEditingChapterContent(chapter.content);
    setEditSplitError(null);
  }, []);

  const closeEditSplitDialog = useCallback(() => {
    setEditingChapterIndex(null);
    setEditingChapterTitle("");
    setEditingChapterContent("");
    setEditSplitError(null);
  }, []);

  const saveEditedSplit = useCallback(() => {
    if (editingChapterIndex === null) {
      return;
    }

    const nextTitle = editingChapterTitle.trim();
    const nextContent = editingChapterContent.trim();

    if (!nextTitle) {
      setEditSplitError("Split name cannot be empty.");
      return;
    }

    if (!nextContent) {
      setEditSplitError("Split text cannot be empty.");
      return;
    }

    setResult((previous) => {
      if (!previous) {
        return previous;
      }

      return {
        ...previous,
        chapters: previous.chapters.map((chapter) => {
          if (chapter.index !== editingChapterIndex) {
            return chapter;
          }

          return {
            ...chapter,
            title: nextTitle,
            content: nextContent,
          };
        }),
      };
    });

    closeEditSplitDialog();
  }, [closeEditSplitDialog, editingChapterContent, editingChapterIndex, editingChapterTitle]);

  return (
    <main className="min-h-screen bg-[var(--bg-primary)] font-[family-name:var(--font-lora)]">
      <div className="mx-auto w-full max-w-4xl px-4 py-8 md:px-8 md:py-12">

        {/* ═══ HEADER ═══ */}
        <header className="mb-10 text-center animate-fadeIn">
          <div className="mb-3 text-5xl select-none">📖</div>
          <h1 className="font-[family-name:var(--font-playfair)] text-4xl font-bold tracking-tight text-[var(--text-primary)] md:text-5xl">
            Bookshelf Aloud
          </h1>
          <p className="mt-2 text-lg text-[var(--text-secondary)] italic">
            Transform your favorite texts into audiobooks — entirely in your browser
          </p>
        </header>

        {/* ═══ STEP INDICATOR ═══ */}
        <nav className="mb-10 flex items-center justify-center gap-0">
          {[
            { num: 1, label: "Select Book" },
            { num: 2, label: "Configure Splits" },
            { num: 3, label: "Generate Audio" },
          ].map((step, i) => (
            <Fragment key={step.num}>
              {i > 0 && (
                <div
                  className={`h-px w-8 sm:w-14 transition-colors duration-500 ${
                    currentStep >= step.num ? "bg-[var(--accent-gold)]" : "bg-[var(--border)]"
                  }`}
                />
              )}
              <button
                type="button"
                onClick={() => {
                  if (step.num < currentStep) setCurrentStep(step.num);
                  if (step.num === 2 && currentStep < 2 && result) setCurrentStep(2);
                  if (step.num === 3 && currentStep < 3 && result) setCurrentStep(3);
                }}
                disabled={step.num > currentStep && !result}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs sm:text-sm font-medium transition-all duration-300 ${
                  currentStep === step.num
                    ? "bg-[var(--accent-gold)] text-white shadow-md shadow-[var(--accent-gold)]/20"
                    : currentStep > step.num
                      ? "bg-[var(--bg-muted)] text-[var(--accent-gold)] hover:bg-[var(--border)] cursor-pointer"
                      : "bg-[var(--bg-muted)] text-[var(--text-muted)] cursor-default"
                }`}
              >
                <span className="font-semibold">{step.num}</span>
                <span className="hidden sm:inline">{step.label}</span>
              </button>
            </Fragment>
          ))}
        </nav>

        {/* ═══════════════════════════════════════════════ */}
        {/* ═══ STEP 1: SELECT BOOK ═══ */}
        {/* ═══════════════════════════════════════════════ */}
        {currentStep === 1 && (
          <div className="animate-fadeSlideIn">
            {/* Info card */}
            <div className="mb-8 rounded-xl border border-[var(--border)] bg-[var(--bg-muted)]/60 p-6">
              <div className="flex items-start gap-4">
                <span className="mt-0.5 shrink-0 text-xl text-[var(--accent-gold)]">✦</span>
                <div className="space-y-2 text-sm leading-relaxed text-[var(--text-secondary)]">
                  <p className="font-semibold text-[var(--text-primary)]">Welcome to Bookshelf Aloud</p>
                  <p>
                    This app uses <strong>Kokoro-JS</strong>, a state-of-the-art text-to-speech engine, to generate
                    high-quality audiobooks <em>entirely in your browser</em> — no server processing, no uploads,
                    complete privacy.
                  </p>
                  <p>
                    It works best with <strong>novels and long prose</strong>. The currently supported format
                    is <code className="rounded bg-[var(--border)]/50 px-1.5 py-0.5 text-xs font-mono">.txt</code>.
                    Pick a book from the library, drop your own file, or paste a URL.
                  </p>
                </div>
              </div>
            </div>

            {/* Ornamental divider */}
            <div className="mb-8 flex items-center gap-4">
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-[var(--border)] to-transparent" />
              <span className="text-sm text-[var(--border)] select-none">❧</span>
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-[var(--border)] to-transparent" />
            </div>

            {/* Source tabs */}
            <div className="mx-auto max-w-lg">
              <div className="mb-5 flex rounded-xl border border-[var(--border)] bg-[var(--bg-muted)] p-1">
                {([
                  { key: "library" as const, label: "📚 Library" },
                  { key: "file" as const, label: "📄 Drop File" },
                  { key: "url" as const, label: "🔗 Paste URL" },
                ]).map((tab) => (
                  <button
                    key={tab.key}
                    type="button"
                    onClick={() => {
                      setBookSource(tab.key);
                      setSplitError(null);
                      setUrlError(null);
                    }}
                    className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium transition-all ${
                      bookSource === tab.key
                        ? "bg-[var(--bg-card)] text-[var(--accent-gold)] shadow-sm"
                        : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* ── Library tab ── */}
              {bookSource === "library" && (
                <div className="animate-fadeIn">
                  <label
                    className="mb-3 block text-center text-sm font-medium text-[var(--text-secondary)]"
                    htmlFor="book-select"
                  >
                    Choose a book from the library
                  </label>
                  {booksLoading ? (
                    <div className="flex items-center justify-center gap-2 py-8 text-sm text-[var(--text-muted)]">
                      <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-[var(--accent-gold)] border-t-transparent" />
                      Loading library…
                    </div>
                  ) : booksError ? (
                    <p className="py-4 text-center text-sm text-[var(--accent-danger)]">{booksError}</p>
                  ) : (
                    <select
                      id="book-select"
                      className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-card)] px-4 py-3 text-base text-[var(--text-primary)] shadow-sm outline-none transition-all focus:border-[var(--accent-gold)] focus:ring-2 focus:ring-[var(--accent-gold)]/20"
                      value={selectedBook}
                      onChange={(event) => {
                        setSelectedBook(event.target.value);
                        setDroppedFileName(null);
                        setDroppedFileText(null);
                        setUrlInput("");
                      }}
                      disabled={books.length === 0 || isSplitting}
                    >
                      {books.map((book) => (
                        <option key={book} value={book}>{book}</option>
                      ))}
                    </select>
                  )}
                </div>
              )}

              {/* ── File drop tab ── */}
              {bookSource === "file" && (
                <div className="animate-fadeIn">
                  <div
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleFileDrop}
                    className={`flex min-h-[160px] flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition-all ${
                      isDragging
                        ? "border-[var(--accent-gold)] bg-[var(--accent-gold)]/5"
                        : droppedFileName
                          ? "border-[var(--accent-success)] bg-[var(--accent-success)]/5"
                          : "border-[var(--border)] bg-[var(--bg-card)] hover:border-[var(--accent-gold)]/50"
                    }`}
                  >
                    {droppedFileName ? (
                      <>
                        <span className="mb-2 text-3xl">✅</span>
                        <p className="text-sm font-semibold text-[var(--text-primary)]">{droppedFileName}</p>
                        <p className="mt-1 text-xs text-[var(--text-muted)]">
                          {droppedFileText ? `${(droppedFileText.length / 1024).toFixed(0)} KB loaded` : ""}
                        </p>
                        <button
                          type="button"
                          onClick={() => { setDroppedFileName(null); setDroppedFileText(null); }}
                          className="mt-2 text-xs text-[var(--accent-danger)] underline hover:text-[var(--accent-danger)]/80"
                        >
                          Remove
                        </button>
                      </>
                    ) : (
                      <>
                        <span className="mb-2 text-3xl select-none">📄</span>
                        <p className="text-sm font-medium text-[var(--text-secondary)]">
                          Drag &amp; drop a <code className="rounded bg-[var(--border)]/50 px-1.5 py-0.5 text-xs font-mono">.txt</code> file here
                        </p>
                        <p className="mt-1.5 text-xs text-[var(--text-muted)]">or</p>
                        <label
                          htmlFor="file-upload"
                          className="mt-1.5 cursor-pointer rounded-lg border border-[var(--border)] px-4 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-all hover:border-[var(--accent-gold)] hover:text-[var(--accent-gold)]"
                        >
                          Browse files
                        </label>
                        <input
                          id="file-upload"
                          type="file"
                          accept=".txt"
                          onChange={handleFileSelect}
                          className="hidden"
                        />
                      </>
                    )}
                  </div>
                  <p className="mt-3 text-center text-xs text-[var(--text-muted)]">
                    💡 The best place to find free books is{" "}
                    <a
                      href="https://www.gutenberg.org"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-medium text-[var(--accent-gold)] underline hover:text-[var(--accent-gold-hover)]"
                    >
                      Project Gutenberg
                    </a>{" "}
                    — download the &ldquo;Plain Text UTF-8&rdquo; version.
                  </p>
                </div>
              )}

              {/* ── URL tab ── */}
              {bookSource === "url" && (
                <div className="animate-fadeIn">
                  <label
                    className="mb-2 block text-center text-sm font-medium text-[var(--text-secondary)]"
                    htmlFor="url-input"
                  >
                    Paste a link to a plain-text file
                  </label>
                  <div className="flex gap-2">
                    <input
                      id="url-input"
                      type="url"
                      value={urlInput}
                      onChange={(e) => { setUrlInput(e.target.value); setUrlError(null); }}
                      placeholder="https://www.gutenberg.org/cache/epub/…/pg….txt"
                      className="flex-1 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] px-4 py-3 text-sm text-[var(--text-primary)] shadow-sm outline-none transition-all focus:border-[var(--accent-gold)] focus:ring-2 focus:ring-[var(--accent-gold)]/20"
                      disabled={urlLoading}
                    />
                    <button
                      type="button"
                      onClick={() => void handleUrlSubmit()}
                      disabled={urlLoading || !urlInput.trim()}
                      className="shrink-0 rounded-xl bg-[var(--text-primary)] px-4 py-3 text-sm font-semibold text-[var(--bg-primary)] transition-all hover:bg-[var(--text-secondary)] disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {urlLoading ? "…" : "Set"}
                    </button>
                  </div>
                  {urlError && <p className="mt-2 text-center text-xs text-[var(--accent-danger)]">{urlError}</p>}
                  {!urlError && urlInput.trim() && (
                    <p className="mt-2 text-center text-xs text-[var(--accent-success)]">✓ URL ready</p>
                  )}
                  <p className="mt-3 text-center text-xs text-[var(--text-muted)]">
                    💡 The best place to find free books is{" "}
                    <a
                      href="https://www.gutenberg.org"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-medium text-[var(--accent-gold)] underline hover:text-[var(--accent-gold-hover)]"
                    >
                      Project Gutenberg
                    </a>{" "}
                    — copy the link to the &ldquo;Plain Text UTF-8&rdquo; version.
                  </p>
                </div>
              )}

              {splitError && <p className="mt-3 text-center text-sm text-[var(--accent-danger)]">{splitError}</p>}

              {/* Continue button */}
              <button
                type="button"
                onClick={() => {
                  if (bookSource === "library" && selectedBook) {
                    setCurrentStep(2);
                  } else if (bookSource === "file" && droppedFileText) {
                    setCurrentStep(2);
                    void runSplitFromText(droppedFileText, DEFAULT_SPLIT_WORDS);
                  } else if (bookSource === "url" && urlInput.trim()) {
                    void handleUrlSubmit().then(() => {
                      setCurrentStep(2);
                      void runSplitFromUrl(urlInput.trim(), DEFAULT_SPLIT_WORDS);
                    });
                  }
                }}
                disabled={
                  isSplitting ||
                  (bookSource === "library" && !selectedBook) ||
                  (bookSource === "file" && !droppedFileText) ||
                  (bookSource === "url" && !urlInput.trim())
                }
                className="mt-6 w-full rounded-xl bg-[var(--accent-gold)] px-6 py-3 text-base font-semibold text-white shadow-md shadow-[var(--accent-gold)]/20 transition-all hover:bg-[var(--accent-gold-hover)] hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isSplitting ? "Preparing…" : "Continue →"}
              </button>
            </div>
          </div>
        )}

        {/* ═══════════════════════════════════════════════ */}
        {/* ═══ STEP 2: CONFIGURE SPLITS ═══ */}
        {/* ═══════════════════════════════════════════════ */}
        {currentStep === 2 && (
          <div className="animate-fadeSlideIn">
            {/* Step 2 navigation (top) */}
            <div className="mb-6 flex items-center justify-between">
              <button
                type="button"
                onClick={() => setCurrentStep(1)}
                className="rounded-lg border border-[var(--border)] px-5 py-2.5 text-sm font-medium text-[var(--text-secondary)] transition-all hover:border-[var(--text-secondary)]"
              >
                ← Back
              </button>
              <button
                type="button"
                onClick={() => setCurrentStep(3)}
                disabled={!result || result.chapters.length === 0}
                className="rounded-xl bg-[var(--accent-gold)] px-6 py-2.5 text-sm font-semibold text-white shadow-md shadow-[var(--accent-gold)]/20 transition-all hover:bg-[var(--accent-gold-hover)] hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-50"
              >
                Continue to Voice Selection →
              </button>
            </div>

            {/* Info card */}
            <div className="mb-6 rounded-xl border border-[var(--border)] bg-[var(--bg-muted)]/60 p-5">
              <div className="flex items-start gap-4">
                <span className="mt-0.5 shrink-0 text-xl text-[var(--accent-gold)]">✦</span>
                <div className="space-y-2 text-sm leading-relaxed text-[var(--text-secondary)]">
                  <p className="font-semibold text-[var(--text-primary)]">How splitting works</p>
                  <p>
                    <strong>Split words</strong> divide your text into meaningful sections. Most commonly these are
                    words like <em>Chapter</em>, <em>Book</em>, <em>Volume</em>, or <em>Part</em> — any word that marks
                    the beginning of a new section in your text.
                  </p>
                  <p>
                    You can <strong>edit the title and content</strong> of each split to remove unwanted text
                    (headers, footnotes, etc.). The more time you spend perfecting your splits, the better
                    your final audiobook will sound.
                  </p>
                </div>
              </div>
            </div>

            {/* Default-split warning */}
            {showDefaultSplitWarning && result && (
              <div className="mb-6 animate-fadeIn rounded-xl border border-[var(--accent-gold)]/40 bg-[var(--accent-gold)]/5 p-5">
                <p className="text-sm font-semibold text-[var(--accent-gold)]">⚠ Only one split was found with the default setting.</p>
                <p className="mt-1 text-xs leading-relaxed text-[var(--text-secondary)]">
                  This usually means the book uses a different chapter format. Try one of the presets below.
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {SPLIT_PRESETS.map((preset) => (
                    <button
                      key={`warning-${preset.join("|")}`}
                      type="button"
                      onClick={() => {
                        const value = preset.join(", ");
                        setCustomWordsInput(value);
                        setShowDefaultSplitWarning(false);
                        if (bookSource === "url" && urlInput.trim()) {
                          void runSplitFromUrl(urlInput.trim(), preset);
                        } else if (bookSource === "file" && droppedFileText) {
                          void runSplitFromText(droppedFileText, preset);
                        } else {
                          void runSplit(preset);
                        }
                      }}
                      disabled={!canSplit}
                      className="rounded-full border border-[var(--accent-gold)]/40 bg-white/60 px-3 py-1 text-xs font-medium text-[var(--accent-gold)] transition hover:bg-[var(--accent-gold)]/10 disabled:opacity-40"
                    >
                      {preset.join(" + ")}
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => setShowDefaultSplitWarning(false)}
                  className="mt-2 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] underline"
                >
                  Dismiss
                </button>
              </div>
            )}

            {/* Split controls card */}
            <div className="mb-6 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-6 shadow-sm">
              <h3 className="font-[family-name:var(--font-playfair)] text-lg font-semibold text-[var(--text-primary)]">
                Split Settings
              </h3>

              <div className="mt-4">
                <p className="mb-2 text-xs font-medium uppercase tracking-widest text-[var(--text-muted)]">Quick presets</p>
                <div className="flex flex-wrap gap-2">
                  {SPLIT_PRESETS.map((preset) => (
                    <button
                      key={preset.join("|")}
                      type="button"
                      onClick={() => {
                        const value = preset.join(", ");
                        setCustomWordsInput(value);
                        if (bookSource === "url" && urlInput.trim()) {
                          void runSplitFromUrl(urlInput.trim(), preset);
                        } else if (bookSource === "file" && droppedFileText) {
                          void runSplitFromText(droppedFileText, preset);
                        } else {
                          void runSplit(preset);
                        }
                      }}
                      disabled={!canSplit}
                      className="rounded-full border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-all hover:border-[var(--accent-gold)] hover:text-[var(--accent-gold)] disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {preset.join(" + ")}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mt-5">
                <label className="mb-1.5 block text-sm text-[var(--text-secondary)]" htmlFor="split-words">
                  Custom split words <span className="text-[var(--text-muted)]">(comma-separated)</span>
                </label>
                <input
                  id="split-words"
                  value={customWordsInput}
                  onChange={(event) => setCustomWordsInput(event.target.value)}
                  className="w-full rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2.5 text-sm text-[var(--text-primary)] outline-none transition-all focus:border-[var(--accent-gold)] focus:ring-2 focus:ring-[var(--accent-gold)]/20"
                  placeholder="Chapter, CHAPTER, Book…"
                  disabled={!canSplit}
                />
              </div>

              <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2">
                <label className="flex items-center gap-2 text-sm text-[var(--text-secondary)] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={caseSensitive}
                    onChange={(event) => setCaseSensitive(event.target.checked)}
                    className="h-4 w-4 rounded border-[var(--border)] accent-[var(--accent-gold)]"
                    disabled={!canSplit}
                  />
                  Case-sensitive
                </label>
                <label className="flex items-center gap-2 text-sm text-[var(--text-secondary)] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={treatFirstSentenceAsTitle}
                    onChange={(event) => setTreatFirstSentenceAsTitle(event.target.checked)}
                    className="h-4 w-4 rounded border-[var(--border)] accent-[var(--accent-gold)]"
                    disabled={!canSplit}
                  />
                  First line as title
                </label>
              </div>

              <button
                type="button"
                onClick={applyCustomSplit}
                disabled={!canSplit}
                className="mt-5 w-full rounded-lg bg-[var(--text-primary)] px-4 py-2.5 text-sm font-semibold text-[var(--bg-primary)] transition-all hover:bg-[var(--text-secondary)] disabled:cursor-not-allowed disabled:opacity-40"
              >
                {isSplitting ? "Splitting…" : "Apply Split"}
              </button>

              {splitError && <p className="mt-3 text-sm text-[var(--accent-danger)]">{splitError}</p>}
            </div>

            {/* Loading state */}
            {isSplitting && (
              <div className="flex items-center justify-center gap-2 py-8 text-sm text-[var(--text-muted)]">
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-[var(--accent-gold)] border-t-transparent" />
                Analyzing text structure…
              </div>
            )}

            {/* Results */}
            {result && !isSplitting && (
              <>
                {/* Divider */}
                <div className="mb-6 flex items-center gap-4">
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-[var(--border)] to-transparent" />
                  <span className="text-sm text-[var(--border)] select-none">✦</span>
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-[var(--border)] to-transparent" />
                </div>

                {/* Book info summary */}
                <div className="mb-6 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-5 shadow-sm">
                  <h2 className="font-[family-name:var(--font-playfair)] text-xl font-bold text-[var(--text-primary)]">
                    {result.metadata.title}
                  </h2>
                  <p className="mt-0.5 text-sm italic text-[var(--text-secondary)]">by {result.metadata.author}</p>
                  <div className="mt-3 flex flex-wrap gap-3">
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--bg-muted)] px-3 py-1 text-xs font-medium text-[var(--text-secondary)]">
                      ✂ {result.chapters.length} {result.chapters.length === 1 ? "split" : "splits"}
                    </span>
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--bg-muted)] px-3 py-1 text-xs font-medium text-[var(--text-secondary)]">
                      🔑 {result.splitWords.join(", ")}
                    </span>
                  </div>
                </div>

                {/* Chapter cards */}
                <div className="space-y-3">
                  {result.chapters.map((chapter) => (
                    <article
                      key={chapter.index}
                      className="group rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-5 shadow-sm transition-all hover:shadow-md hover:border-[var(--accent-gold)]/30"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <span className="text-[10px] font-medium uppercase tracking-widest text-[var(--text-muted)]">
                            Split {chapter.index}
                          </span>
                          <h4 className="mt-0.5 text-base font-semibold text-[var(--text-primary)]">
                            {chapter.title}
                          </h4>
                        </div>
                        <button
                          type="button"
                          onClick={() => openEditSplitDialog(chapter)}
                          className="shrink-0 rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-all hover:border-[var(--accent-gold)] hover:text-[var(--accent-gold)]"
                        >
                          ✎ Edit
                        </button>
                      </div>
                      <p className="mt-3 text-sm leading-relaxed text-[var(--text-secondary)] whitespace-pre-wrap">
                        {expandedChapterIndexes.has(chapter.index)
                          ? chapter.content
                          : getChapterPreviewText(chapter.content)}
                      </p>
                      <button
                        type="button"
                        onClick={() => toggleExpandedChapter(chapter.index)}
                        className="mt-2 text-xs font-medium text-[var(--accent-gold)] hover:text-[var(--accent-gold-hover)] transition-colors"
                      >
                        {expandedChapterIndexes.has(chapter.index) ? "↑ Show less" : "↓ Read more"}
                      </button>
                    </article>
                  ))}
                </div>

                {/* Step 2 navigation */}
                <div className="mt-8 flex items-center justify-between">
                  <button
                    type="button"
                    onClick={() => setCurrentStep(1)}
                    className="rounded-lg border border-[var(--border)] px-5 py-2.5 text-sm font-medium text-[var(--text-secondary)] transition-all hover:border-[var(--text-secondary)]"
                  >
                    ← Back
                  </button>
                  <button
                    type="button"
                    onClick={() => setCurrentStep(3)}
                    disabled={!result || result.chapters.length === 0}
                    className="rounded-xl bg-[var(--accent-gold)] px-6 py-2.5 text-sm font-semibold text-white shadow-md shadow-[var(--accent-gold)]/20 transition-all hover:bg-[var(--accent-gold-hover)] hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Continue to Voice Selection →
                  </button>
                </div>
              </>
            )}

            {!result && !isSplitting && (
              <p className="py-8 text-center text-sm italic text-[var(--text-muted)]">
                Apply a split to see chapter previews.
              </p>
            )}
          </div>
        )}

        {/* ═══════════════════════════════════════════════ */}
        {/* ═══ STEP 3: VOICE & GENERATE ═══ */}
        {/* ═══════════════════════════════════════════════ */}
        {currentStep === 3 && result && (
          <div className="animate-fadeSlideIn">
            {/* Info card */}
            <div className="mb-6 rounded-xl border border-[var(--border)] bg-[var(--bg-muted)]/60 p-5">
              <div className="flex items-start gap-4">
                <span className="mt-0.5 shrink-0 text-xl text-[var(--accent-gold)]">✦</span>
                <div className="space-y-2 text-sm leading-relaxed text-[var(--text-secondary)]">
                  <p className="font-semibold text-[var(--text-primary)]">Voice selection &amp; generation</p>
                  <p>
                    Voice quality ranges from <strong className="text-[var(--accent-success)]">A</strong> (excellent)
                    to <strong className="text-[var(--accent-danger)]">C</strong> (may be unstable). We recommend
                    starting with an A-rated voice for the best results.
                  </p>
                  <p>
                    When you click <strong>Generate &amp; Download ZIP</strong>, a ZIP file containing a separate
                    audio file for each selected split will be downloaded. You can choose <strong>WAV</strong> or
                    <strong> MP3</strong> below. The ZIP is named after the book title and each file is named after
                    the split title. Generation runs entirely in your browser.
                  </p>
                  <p className="rounded-lg border border-[var(--accent-gold)]/30 bg-[var(--accent-gold)]/10 px-3 py-2 text-xs text-[var(--text-primary)]">
                    ⚠️ Full-book generation of a 500 page book can take up to <strong>20 hours</strong>, since it runs entirely in your browser. It is recommended to start with couple of initial splits. Then create more as you need. 
                    when you are not using your computer and let it run.
                  </p>
                </div>
              </div>
            </div>

            {/* Book summary */}
            <div className="mb-6 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-5 shadow-sm">
              <h2 className="font-[family-name:var(--font-playfair)] text-xl font-bold text-[var(--text-primary)]">
                {result.metadata.title}
              </h2>
              <p className="mt-0.5 text-sm italic text-[var(--text-secondary)]">by {result.metadata.author}</p>
              <p className="mt-1 text-xs text-[var(--text-muted)]">{result.chapters.length} splits ready for generation</p>
            </div>

            {/* Voice selection card */}
            <div className="mb-6 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-6 shadow-sm">
              <h3 className="font-[family-name:var(--font-playfair)] text-lg font-semibold text-[var(--text-primary)]">
                Choose a Voice
              </h3>

              <div className="mt-4">
                <select
                  className="w-full rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2.5 text-sm text-[var(--text-primary)] outline-none transition-all focus:border-[var(--accent-gold)] focus:ring-2 focus:ring-[var(--accent-gold)]/20"
                  value={ttsVoice}
                  onChange={(event) => setTtsVoice(event.target.value as KokoroVoice)}
                  disabled={isGeneratingPreview || voicesLoading || voiceProfiles.length === 0}
                >
                  {voiceProfiles.map((voice) => (
                    <option key={voice.id} value={voice.id}>
                      {formatVoiceLabel(voice)}
                    </option>
                  ))}
                </select>

                {voicesError && <p className="mt-2 text-xs text-[var(--accent-danger)]">{voicesError}</p>}
              </div>

              {/* Voice detail badges */}
              {voiceProfiles.find((voice) => voice.id === ttsVoice) && (() => {
                const selectedVoice = voiceProfiles.find((voice) => voice.id === ttsVoice);
                if (!selectedVoice) return null;
                const identity = getVoiceIdentity(selectedVoice.id);
                const isAGrade = selectedVoice.overallGrade.startsWith("A");
                const isBGrade = selectedVoice.overallGrade.startsWith("B");
                return (
                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="inline-flex items-center gap-1 rounded-full bg-[var(--bg-muted)] px-2.5 py-1 text-xs text-[var(--text-secondary)]">
                      {identity.accentFlag} {identity.accentLabel}
                    </span>
                    <span className="inline-flex items-center gap-1 rounded-full bg-[var(--bg-muted)] px-2.5 py-1 text-xs text-[var(--text-secondary)]">
                      {identity.genderLabel}
                    </span>
                    {selectedVoice.traits && (
                      <span className="inline-flex items-center gap-1 rounded-full bg-[var(--bg-muted)] px-2.5 py-1 text-xs text-[var(--text-secondary)]">
                        {selectedVoice.traits}
                      </span>
                    )}
                    <span
                      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-semibold ${
                        isAGrade
                          ? "bg-[var(--accent-success)]/10 text-[var(--accent-success)] border-[var(--accent-success)]/20"
                          : isBGrade
                            ? "bg-[var(--accent-gold)]/10 text-[var(--accent-gold)] border-[var(--accent-gold)]/20"
                            : "bg-[var(--accent-danger)]/10 text-[var(--accent-danger)] border-[var(--accent-danger)]/20"
                      }`}
                    >
                      Grade: {selectedVoice.overallGrade}
                    </span>
                  </div>
                );
              })()}

              {/* Voice sample */}
              <div className="mt-4">
                <p className="mb-1.5 text-xs font-medium text-[var(--text-muted)]">Voice sample</p>
                <audio
                  key={selectedVoiceSampleUrl}
                  controls
                  preload="none"
                  className="w-full"
                  src={selectedVoiceSampleUrl}
                  onError={() => setVoiceSampleLoadError("Sample not available. Run voice sample generator.")}
                />
                {voiceSampleLoadError && (
                  <p className="mt-1 text-xs text-[var(--accent-gold)]">{voiceSampleLoadError}</p>
                )}
              </div>
            </div>

            {/* Generation controls card */}
            <div className="mb-6 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-6 shadow-sm">
              <h3 className="font-[family-name:var(--font-playfair)] text-lg font-semibold text-[var(--text-primary)]">
                Generate Audio
              </h3>

              {/* Split selection list */}
              <div className="mt-4">
                <div className="mb-3 flex items-center justify-between">
                  <p className="text-sm font-medium text-[var(--text-secondary)]">
                    Select splits to include{" "}
                    <span className="text-xs text-[var(--text-muted)]">
                      ({selectedSplitIndexes.size} of {result.chapters.length} selected)
                    </span>
                  </p>
                  <button
                    type="button"
                    onClick={() => {
                      if (selectedSplitIndexes.size === result.chapters.length) {
                        setSelectedSplitIndexes(new Set());
                      } else {
                        setSelectedSplitIndexes(new Set(result.chapters.map((ch) => ch.index)));
                      }
                    }}
                    disabled={isGeneratingPreview}
                    className="text-xs font-medium text-[var(--accent-gold)] hover:text-[var(--accent-gold-hover)] transition-colors disabled:opacity-40"
                  >
                    {selectedSplitIndexes.size === result.chapters.length ? "Deselect all" : "Select all"}
                  </button>
                </div>
                <div className="max-h-64 overflow-y-auto rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] divide-y divide-[var(--border)]">
                  {result.chapters.map((chapter) => (
                    <label
                      key={chapter.index}
                      className={`flex items-center gap-3 px-4 py-2.5 cursor-pointer transition-colors hover:bg-[var(--bg-muted)]/60 ${
                        selectedSplitIndexes.has(chapter.index) ? "" : "opacity-50"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={selectedSplitIndexes.has(chapter.index)}
                        onChange={() => {
                          setSelectedSplitIndexes((prev) => {
                            const next = new Set(prev);
                            if (next.has(chapter.index)) {
                              next.delete(chapter.index);
                            } else {
                              next.add(chapter.index);
                            }
                            return next;
                          });
                        }}
                        disabled={isGeneratingPreview}
                        className="h-4 w-4 shrink-0 rounded border-[var(--border)] accent-[var(--accent-gold)]"
                      />
                      <span className="min-w-0 truncate text-sm text-[var(--text-primary)]">
                        <span className="text-xs text-[var(--text-muted)] mr-1.5">{chapter.index}.</span>
                        {chapter.title}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="mt-4 rounded-lg border border-[var(--border)] bg-[var(--bg-muted)]/40 p-3">
                <p className="text-sm font-medium text-[var(--text-secondary)]">Output format</p>
                <div className="mt-2 flex flex-wrap gap-3">
                  {(["wav", "mp3"] as const).map((formatOption) => (
                    <label
                      key={formatOption}
                      className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 text-sm text-[var(--text-primary)]"
                    >
                      <input
                        type="radio"
                        name="output-audio-format"
                        value={formatOption}
                        checked={outputAudioFormat === formatOption}
                        onChange={() => setOutputAudioFormat(formatOption)}
                        disabled={isGeneratingPreview}
                        className="accent-[var(--accent-gold)]"
                      />
                      {formatOption.toUpperCase()}
                    </label>
                  ))}
                </div>
                <p className="mt-2 text-xs text-[var(--text-muted)]">
                  MP3 uses much less disk space. WAV keeps uncompressed quality but can be very large.
                </p>
              </div>

              {/* Generate button */}
              <button
                type="button"
                onClick={() => void playAudiobookPreviewForAllSplits()}
                disabled={isGeneratingPreview || selectedSplitIndexes.size === 0}
                className="mt-5 w-full rounded-lg bg-[var(--accent-gold)] px-4 py-3 text-sm font-semibold text-white shadow-md shadow-[var(--accent-gold)]/20 transition-all hover:bg-[var(--accent-gold-hover)] hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-40"
              >
                {isGeneratingPreview
                  ? "Generating…"
                  : `⬇ Generate & Download ZIP (${selectedSplitIndexes.size} ${selectedSplitIndexes.size === 1 ? "split" : "splits"})`}
              </button>

              {/* Progress */}
              {(isGeneratingPreview || generationProgressPct > 0) && (
                <div className="mt-5">
                  <div className="mb-2 flex items-center gap-2">
                    {isGeneratingPreview && (
                      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-[var(--accent-gold)] border-t-transparent" />
                    )}
                    <p className="text-xs text-[var(--text-muted)]">{ttsStatus}</p>
                  </div>
                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-[var(--bg-muted)]">
                    <div
                      className="progress-bar h-full rounded-full transition-all duration-500"
                      style={{ width: `${generationProgressPct}%` }}
                    />
                  </div>
                  <p className="mt-1.5 text-right text-xs text-[var(--text-muted)]">{generationProgressPct}%</p>
                  {isGeneratingPreview && estimatedTimeRemaining && (
                    <p className="mt-1 text-right text-xs text-[var(--text-muted)]">Estimated remaining time: ~{estimatedTimeRemaining}</p>
                  )}
                </div>
              )}

              {/* Clear button */}
              {(previewAudioUrl || generatedChunkAudios.length > 0) && (
                <button
                  type="button"
                  onClick={clearGeneratedAudio}
                  className="mt-4 text-xs font-medium text-[var(--text-muted)] underline hover:text-[var(--text-secondary)] transition-colors"
                >
                  Clear generated audio
                </button>
              )}
            </div>

            {/* Audio output card */}
            {(previewAudioUrl || zipDownloadUrl) && (
              <div className="mb-6 rounded-xl border border-[var(--accent-gold)]/30 bg-[var(--accent-gold)]/5 p-6 shadow-sm animate-fadeIn">
                <h3 className="font-[family-name:var(--font-playfair)] text-lg font-semibold text-[var(--text-primary)]">
                  🎧 Your Audio
                </h3>
                {previewAudioUrl && (
                  <div className="mt-4">
                    <audio ref={previewAudioRef} controls className="w-full" src={previewAudioUrl} />
                  </div>
                )}
                {zipDownloadUrl && zipFileName && (
                  <a
                    href={zipDownloadUrl}
                    download={zipFileName}
                    className="mt-4 inline-flex items-center gap-2 rounded-lg border-2 border-[var(--accent-gold)] px-5 py-2.5 text-sm font-semibold text-[var(--accent-gold)] transition-all hover:bg-[var(--accent-gold)] hover:text-white"
                  >
                    ⬇ Download: {zipFileName}
                  </a>
                )}
              </div>
            )}

            {/* Generated chunks detail */}
            {generatedChunkAudios.length > 0 && (
              <details className="mb-6 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] shadow-sm">
                <summary className="cursor-pointer px-5 py-4 text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors">
                  View generated chunks ({generatedChunkAudios.length})
                </summary>
                <div className="border-t border-[var(--border)] p-5 space-y-2">
                  {generatedChunkAudios.map((chunk) => (
                    <details key={chunk.chunkIndex} className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]">
                      <summary className="cursor-pointer px-3 py-2 text-xs text-[var(--text-secondary)]">
                        Chunk {chunk.chunkIndex} · {chunk.wordCount} words
                      </summary>
                      <p className="px-3 pb-3 text-xs leading-relaxed text-[var(--text-muted)] whitespace-pre-wrap">
                        {chunk.text}
                      </p>
                    </details>
                  ))}
                </div>
              </details>
            )}

            {/* Step 3 navigation */}
            <div className="flex items-center justify-between">
              <button
                type="button"
                onClick={() => setCurrentStep(2)}
                className="rounded-lg border border-[var(--border)] px-5 py-2.5 text-sm font-medium text-[var(--text-secondary)] transition-all hover:border-[var(--text-secondary)]"
              >
                ← Back to Splits
              </button>
            </div>
          </div>
        )}

        {/* ═══ EDIT MODAL ═══ */}
        {editingChapterIndex !== null && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-[var(--text-primary)]/40 p-4 backdrop-blur-sm animate-fadeIn">
            <div className="w-full max-w-2xl rounded-2xl border border-[var(--border)] bg-[var(--bg-card)] p-6 shadow-2xl animate-fadeSlideIn">
              <h3 className="font-[family-name:var(--font-playfair)] text-xl font-bold text-[var(--text-primary)]">
                Edit Split
              </h3>
              <p className="mt-1 text-xs text-[var(--text-muted)]">
                Adjust the title and content to improve your audiobook quality.
              </p>

              <label className="mt-5 block text-sm font-medium text-[var(--text-secondary)]" htmlFor="edit-split-title">
                Split title
              </label>
              <input
                id="edit-split-title"
                value={editingChapterTitle}
                onChange={(event) => setEditingChapterTitle(event.target.value)}
                className="mt-1.5 w-full rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2.5 text-sm text-[var(--text-primary)] outline-none transition-all focus:border-[var(--accent-gold)] focus:ring-2 focus:ring-[var(--accent-gold)]/20"
              />

              <label className="mt-4 block text-sm font-medium text-[var(--text-secondary)]" htmlFor="edit-split-content">
                Split content
              </label>
              <textarea
                id="edit-split-content"
                value={editingChapterContent}
                onChange={(event) => setEditingChapterContent(event.target.value)}
                className="mt-1.5 h-72 w-full resize-y rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2.5 text-sm text-[var(--text-primary)] outline-none transition-all focus:border-[var(--accent-gold)] focus:ring-2 focus:ring-[var(--accent-gold)]/20"
              />

              {editSplitError && <p className="mt-3 text-sm text-[var(--accent-danger)]">{editSplitError}</p>}

              <div className="mt-5 flex justify-end gap-3">
                <button
                  type="button"
                  onClick={closeEditSplitDialog}
                  className="rounded-lg border border-[var(--border)] px-5 py-2.5 text-sm font-medium text-[var(--text-secondary)] transition-all hover:border-[var(--text-secondary)]"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={saveEditedSplit}
                  className="rounded-lg bg-[var(--accent-gold)] px-5 py-2.5 text-sm font-semibold text-white transition-all hover:bg-[var(--accent-gold-hover)]"
                >
                  Save Changes
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

