import { NextRequest, NextResponse } from "next/server";
import { splitBookByUrl } from "@/lib/book-parser";

type SplitFromUrlRequestBody = {
  sourceUrl?: string;
  splitWords?: string[];
  caseSensitive?: boolean;
};

export async function POST(request: NextRequest) {
  const body = (await request.json().catch(() => ({}))) as SplitFromUrlRequestBody;

  const sourceUrl = typeof body.sourceUrl === "string" ? body.sourceUrl.trim() : "";
  const splitWords = Array.isArray(body.splitWords)
    ? body.splitWords.map((word) => String(word))
    : [];

  if (!sourceUrl) {
    return NextResponse.json({ error: "sourceUrl is required." }, { status: 400 });
  }

  if (splitWords.length === 0) {
    return NextResponse.json({ error: "splitWords is required." }, { status: 400 });
  }

  try {
    const result = await splitBookByUrl({
      sourceUrl,
      splitWords,
      caseSensitive: Boolean(body.caseSensitive),
    });

    return NextResponse.json(result);
  } catch (error) {
    const message = (error as Error).message;
    const isUserInputError =
      message.toLowerCase().includes("invalid") ||
      message.toLowerCase().includes("required") ||
      message.toLowerCase().includes("supported") ||
      message.toLowerCase().includes("empty") ||
      message.toLowerCase().includes("failed to fetch");

    return NextResponse.json({ error: message }, { status: isUserInputError ? 400 : 500 });
  }
}
