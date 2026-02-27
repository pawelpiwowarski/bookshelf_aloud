import path from "node:path";
import { promises as fs } from "node:fs";

export type Chapter = {
  index: number;
  title: string;
  content: string;
};

export type BookMetadata = {
  title: string;
  author: string;
};

export type ChapterStrategy = {
  id: string;
  name: string;
  type: "regex" | "grep-like";
  pattern: string;
  reason: string;
};

export type StrategyCandidate = {
  id: string;
  name: string;
  type: "regex" | "grep-like";
  pattern: string;
  description: string;
  regex?: RegExp;
};

export type BookAnalysis = {
  bookId: string;
  model: string;
  metadata: BookMetadata;
  previewTokenCount: number;
  strategy: ChapterStrategy;
  chapters: Chapter[];
};

export type ManualSplitResult = {
  bookId: string;
  metadata: BookMetadata;
  splitWords: string[];
  caseSensitive: boolean;
  strategyLabel: string;
  chapters: Chapter[];
};

const BOOKS_DIR = path.join(process.cwd(), "public", "books");
export const PREVIEW_TOKENS = 10_000;

const TOKEN_SPLIT_REGEX = /\s+/;

export const STRATEGY_CANDIDATES: StrategyCandidate[] = [
  {
    id: "chapter-numbered",
    name: "Numbered chapter headings",
    type: "regex",
    regex: /^\s*Chapter\s+\d+[\.:]?\s+.+$/gim,
    pattern: "^\\s*Chapter\\s+\\d+[\\.:]?\\s+.+$",
    description: "Matches headings like 'Chapter 12. The Return'.",
  },
  {
    id: "chapter-roman",
    name: "Roman numeral chapter headings",
    type: "regex",
    regex: /^\s*Chapter\s+[IVXLCDM]+[\.:]?\s+.+$/gim,
    pattern: "^\\s*Chapter\\s+[IVXLCDM]+[\\.:]?\\s+.+$",
    description: "Matches headings like 'Chapter XIV: ...'.",
  },
  {
    id: "chapter-uppercase",
    name: "Uppercase CHAPTER headings",
    type: "regex",
    regex: /^\s*CHAPTER\s+[\w\-]+[\.:]?\s*.*$/gm,
    pattern: "^\\s*CHAPTER\\s+[\\w\\-]+[\\.:]?\\s*.*$",
    description: "Matches headings where the word CHAPTER is uppercase.",
  },
  {
    id: "paragraph-fallback",
    name: "Paragraph chunk fallback",
    type: "grep-like",
    pattern: "Split by large blank sections and volume markers",
    description: "Use broad section chunking when chapter headings are inconsistent.",
  },
];

export async function listBooks(): Promise<string[]> {
  const entries = await fs.readdir(BOOKS_DIR, { withFileTypes: true });

  return entries
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".txt"))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));
}

function sanitizeBookId(bookId: string): string {
  if (!bookId || bookId.includes("/") || bookId.includes("\\") || bookId.includes("..")) {
    throw new Error("Invalid book id.");
  }

  return bookId;
}

async function readBook(bookId: string): Promise<string> {
  const safeBookId = sanitizeBookId(bookId);
  const fullPath = path.join(BOOKS_DIR, safeBookId);
  return fs.readFile(fullPath, "utf8");
}

function tokenize(text: string): string[] {
  return text.split(TOKEN_SPLIT_REGEX).filter(Boolean);
}

function getPreviewByTokens(text: string, tokenCount: number): { preview: string; tokens: string[] } {
  const tokens = tokenize(text);
  const selected = tokens.slice(0, tokenCount);
  return { preview: selected.join(" "), tokens: selected };
}

function parseMetadata(text: string): BookMetadata {
  const titleMatch = text.match(/^Title:\s*(.+)$/im);
  const authorMatch = text.match(/^Author:\s*(.+(?:\n\s+.+)*)/im);

  const title = titleMatch?.[1]?.trim() || "Unknown title";
  const author = authorMatch
    ? authorMatch[1]
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .join(", ")
    : "Unknown author";

  return { title, author };
}

export type BookPreview = {
  bookId: string;
  metadata: BookMetadata;
  previewTokenCount: number;
  previewText: string;
  candidates: Array<Omit<StrategyCandidate, "regex">>;
};

function stripGutenbergWrapper(text: string): string {
  const startMatch = text.match(/\*\*\*\s*START OF[^\n]*\*\*\*/i);
  const endMatch = text.match(/\*\*\*\s*END OF[^\n]*\*\*\*/i);

  const startIndex = startMatch?.index ?? 0;
  const endIndex = endMatch?.index ?? text.length;

  return text.slice(startIndex, endIndex);
}

function chooseChapterStrategy(previewText: string, model: string): ChapterStrategy {
  const regexCandidates = STRATEGY_CANDIDATES.filter((candidate) => candidate.type === "regex" && candidate.regex);

  const scored = regexCandidates.map((candidate) => {
    const matches = previewText.match(candidate.regex as RegExp);
    return { candidate, score: matches?.length ?? 0 };
  }).sort((a, b) => b.score - a.score);

  const best = scored[0];

  if (!best || best.score === 0) {
    const fallback = STRATEGY_CANDIDATES.find((candidate) => candidate.id === "paragraph-fallback");
    if (!fallback) {
      throw new Error("Fallback chapter strategy is missing.");
    }

    return {
      id: fallback.id,
      name: fallback.name,
      type: fallback.type,
      pattern: fallback.pattern,
      reason: `${model} could not detect a stable chapter heading pattern in the first ${PREVIEW_TOKENS} tokens.`,
    };
  }

  return {
    id: best.candidate.id,
    name: best.candidate.name,
    type: best.candidate.type,
    pattern: best.candidate.pattern,
    reason: `${model} selected this pattern because it appeared ${best.score} times in the first ${PREVIEW_TOKENS} tokens.`,
  };
}

function getCandidateById(candidateId: string): StrategyCandidate | null {
  return STRATEGY_CANDIDATES.find((candidate) => candidate.id === candidateId) ?? null;
}

function extractChaptersByRegex(text: string, headingRegex: RegExp): Chapter[] {
  const source = text;
  const globalRegex = new RegExp(headingRegex.source, headingRegex.flags.includes("g") ? headingRegex.flags : `${headingRegex.flags}g`);

  const headingMatches = Array.from(source.matchAll(globalRegex)).filter((match) => typeof match.index === "number");

  if (headingMatches.length === 0) {
    return [];
  }

  let startHeadingIndex = 0;
  for (let i = 0; i < headingMatches.length - 1; i += 1) {
    const current = headingMatches[i];
    const next = headingMatches[i + 1];
    if ((next.index ?? 0) - (current.index ?? 0) > 1200) {
      startHeadingIndex = i;
      break;
    }
  }

  const activeHeadings = headingMatches.slice(startHeadingIndex);

  const chapters: Chapter[] = [];

  for (let i = 0; i < activeHeadings.length; i += 1) {
    const heading = activeHeadings[i];
    const chapterStart = heading.index ?? 0;
    const contentStart = chapterStart + heading[0].length;
    const chapterEnd = activeHeadings[i + 1]?.index ?? source.length;

    const title = heading[0].trim();
    const content = source.slice(contentStart, chapterEnd).trim();

    if (content.length < 80) {
      continue;
    }

    chapters.push({
      index: chapters.length + 1,
      title,
      content,
    });
  }

  return chapters;
}

function extractFallbackChapters(text: string): Chapter[] {
  const sections = text
    .split(/\n\s*\n\s*\n+/)
    .map((block) => block.trim())
    .filter((block) => block.length > 250);

  return sections.slice(0, 60).map((content, index) => ({
    index: index + 1,
    title: `Section ${index + 1}`,
    content,
  }));
}

export async function analyzeBook(bookId: string, model: string): Promise<BookAnalysis> {
  const raw = await readBook(bookId);
  const { preview, tokens } = getPreviewByTokens(raw, PREVIEW_TOKENS);
  const strategy = chooseChapterStrategy(preview, model);
  const metadata = parseMetadata(raw);
  const cleanedText = stripGutenbergWrapper(raw);

  const selectedCandidate = getCandidateById(strategy.id);
  const headingRegex = selectedCandidate?.regex ?? null;
  const chapters = headingRegex
    ? extractChaptersByRegex(cleanedText, headingRegex)
    : extractFallbackChapters(cleanedText);

  return {
    bookId,
    model,
    metadata,
    previewTokenCount: tokens.length,
    strategy,
    chapters,
  };
}

export async function getBookPreview(bookId: string): Promise<BookPreview> {
  const raw = await readBook(bookId);
  const { preview, tokens } = getPreviewByTokens(raw, PREVIEW_TOKENS);

  return {
    bookId,
    metadata: parseMetadata(raw),
    previewTokenCount: tokens.length,
    previewText: preview,
    candidates: STRATEGY_CANDIDATES.map((candidate) => ({
      id: candidate.id,
      name: candidate.name,
      type: candidate.type,
      pattern: candidate.pattern,
      description: candidate.description,
    })),
  };
}

export async function analyzeBookWithSelectedStrategy(args: {
  bookId: string;
  model: string;
  strategyId?: string;
  llmReason?: string;
}): Promise<BookAnalysis> {
  const raw = await readBook(args.bookId);
  const { preview, tokens } = getPreviewByTokens(raw, PREVIEW_TOKENS);
  const metadata = parseMetadata(raw);
  const cleanedText = stripGutenbergWrapper(raw);

  const requestedCandidate = args.strategyId ? getCandidateById(args.strategyId) : null;
  const heuristicStrategy = chooseChapterStrategy(preview, args.model);
  const selectedCandidate = requestedCandidate ?? getCandidateById(heuristicStrategy.id);

  if (!selectedCandidate) {
    throw new Error("No chapter strategy could be selected.");
  }

  const strategy: ChapterStrategy = {
    id: selectedCandidate.id,
    name: selectedCandidate.name,
    type: selectedCandidate.type,
    pattern: selectedCandidate.pattern,
    reason:
      args.llmReason?.trim() ||
      (requestedCandidate
        ? `${args.model} selected '${selectedCandidate.name}' for chapter parsing.`
        : heuristicStrategy.reason),
  };

  const chapters = selectedCandidate.regex
    ? extractChaptersByRegex(cleanedText, selectedCandidate.regex)
    : extractFallbackChapters(cleanedText);

  return {
    bookId: args.bookId,
    model: args.model,
    metadata,
    previewTokenCount: tokens.length,
    strategy,
    chapters,
  };
}

function escapeRegexLiteral(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function splitBookTextByWords(args: {
  sourceId: string;
  rawText: string;
  splitWords: string[];
  caseSensitive?: boolean;
}): ManualSplitResult {
  const cleanedText = stripGutenbergWrapper(args.rawText);
  const metadata = parseMetadata(args.rawText);
  const splitWords = args.splitWords.map((word) => word.trim()).filter(Boolean);

  if (splitWords.length === 0) {
    throw new Error("At least one split word is required.");
  }

  const alternation = splitWords.map((word) => escapeRegexLiteral(word)).join("|");
  const flags = args.caseSensitive ? "gm" : "gim";
  const headingRegex = new RegExp(`^\\s*(?:${alternation})\\b.*$`, flags);
  const chapters = extractChaptersByRegex(cleanedText, headingRegex);

  return {
    bookId: args.sourceId,
    metadata,
    splitWords,
    caseSensitive: Boolean(args.caseSensitive),
    strategyLabel: `Heading starts with: ${splitWords.join(", ")}`,
    chapters: chapters.length > 0 ? chapters : extractFallbackChapters(cleanedText),
  };
}

export async function splitBookByWords(args: {
  bookId: string;
  splitWords: string[];
  caseSensitive?: boolean;
}): Promise<ManualSplitResult> {
  const raw = await readBook(args.bookId);
  return splitBookTextByWords({
    sourceId: args.bookId,
    rawText: raw,
    splitWords: args.splitWords,
    caseSensitive: args.caseSensitive,
  });
}

function validateTextUrl(input: string): URL {
  let parsed: URL;
  try {
    parsed = new URL(input);
  } catch {
    throw new Error("Invalid URL.");
  }

  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("Only http/https URLs are supported.");
  }

  return parsed;
}

export async function splitBookByUrl(args: {
  sourceUrl: string;
  splitWords: string[];
  caseSensitive?: boolean;
}): Promise<ManualSplitResult> {
  const url = validateTextUrl(args.sourceUrl.trim());
  const response = await fetch(url.toString());

  if (!response.ok) {
    throw new Error(`Failed to fetch text from URL (status ${response.status}).`);
  }

  const raw = await response.text();
  if (!raw.trim()) {
    throw new Error("Fetched text file is empty.");
  }

  return splitBookTextByWords({
    sourceId: url.toString(),
    rawText: raw,
    splitWords: args.splitWords,
    caseSensitive: args.caseSensitive,
  });
}

export function splitRawTextByWords(args: {
  rawText: string;
  splitWords: string[];
  caseSensitive?: boolean;
}): ManualSplitResult {
  if (!args.rawText.trim()) {
    throw new Error("Text is empty.");
  }

  return splitBookTextByWords({
    sourceId: "uploaded-file",
    rawText: args.rawText,
    splitWords: args.splitWords,
    caseSensitive: args.caseSensitive,
  });
}
