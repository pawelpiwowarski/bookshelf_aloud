import { NextRequest, NextResponse } from "next/server";
import { analyzeBook, analyzeBookWithSelectedStrategy } from "@/lib/book-parser";

type RouteContext = {
  params: Promise<{ bookId: string }>;
};

export async function GET(request: NextRequest, context: RouteContext) {
  const { bookId } = await context.params;
  const model = request.nextUrl.searchParams.get("model")?.trim();

  if (!model) {
    return NextResponse.json({ error: "Model is required." }, { status: 400 });
  }

  try {
    const result = await analyzeBook(bookId, model);
    return NextResponse.json(result);
  } catch (error) {
    const message = (error as Error).message;
    const status = message.toLowerCase().includes("invalid") ? 400 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}

type AnalyzeRequestBody = {
  model?: string;
  strategyId?: string;
  llmReason?: string;
};

export async function POST(request: NextRequest, context: RouteContext) {
  const { bookId } = await context.params;
  const body = (await request.json().catch(() => ({}))) as AnalyzeRequestBody;
  const model = body.model?.trim();

  if (!model) {
    return NextResponse.json({ error: "Model is required." }, { status: 400 });
  }

  try {
    const result = await analyzeBookWithSelectedStrategy({
      bookId,
      model,
      strategyId: body.strategyId,
      llmReason: body.llmReason,
    });
    return NextResponse.json(result);
  } catch (error) {
    const message = (error as Error).message;
    const status = message.toLowerCase().includes("invalid") ? 400 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
