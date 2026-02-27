import { NextRequest, NextResponse } from "next/server";
import { splitRawTextByWords } from "@/lib/book-parser";

type SplitFromTextRequestBody = {
  rawText?: string;
  splitWords?: string[];
  caseSensitive?: boolean;
};

export async function POST(request: NextRequest) {
  const body = (await request.json().catch(() => ({}))) as SplitFromTextRequestBody;

  const rawText = typeof body.rawText === "string" ? body.rawText : "";
  const splitWords = Array.isArray(body.splitWords)
    ? body.splitWords.map((word) => String(word))
    : [];

  if (!rawText.trim()) {
    return NextResponse.json({ error: "rawText is required." }, { status: 400 });
  }

  if (splitWords.length === 0) {
    return NextResponse.json({ error: "splitWords is required." }, { status: 400 });
  }

  try {
    const result = splitRawTextByWords({
      rawText,
      splitWords,
      caseSensitive: Boolean(body.caseSensitive),
    });

    return NextResponse.json(result);
  } catch (error) {
    const message = (error as Error).message;
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
