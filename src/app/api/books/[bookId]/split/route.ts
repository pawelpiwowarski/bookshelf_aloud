import { NextRequest, NextResponse } from "next/server";
import { splitBookByWords } from "@/lib/book-parser";

type RouteContext = {
  params: Promise<{ bookId: string }>;
};

type SplitRequestBody = {
  splitWords?: string[];
  caseSensitive?: boolean;
};

export async function POST(request: NextRequest, context: RouteContext) {
  const { bookId } = await context.params;
  const body = (await request.json().catch(() => ({}))) as SplitRequestBody;

  const splitWords = Array.isArray(body.splitWords)
    ? body.splitWords.map((word) => String(word))
    : [];

  if (splitWords.length === 0) {
    return NextResponse.json({ error: "splitWords is required." }, { status: 400 });
  }

  try {
    const result = await splitBookByWords({
      bookId,
      splitWords,
      caseSensitive: Boolean(body.caseSensitive),
    });

    return NextResponse.json(result);
  } catch (error) {
    const message = (error as Error).message;
    const status = message.toLowerCase().includes("invalid") ? 400 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
