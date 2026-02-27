import { NextResponse } from "next/server";
import { getBookPreview } from "@/lib/book-parser";

type RouteContext = {
  params: Promise<{ bookId: string }>;
};

export async function GET(_request: Request, context: RouteContext) {
  const { bookId } = await context.params;

  try {
    const result = await getBookPreview(bookId);
    return NextResponse.json(result);
  } catch (error) {
    const message = (error as Error).message;
    const status = message.toLowerCase().includes("invalid") ? 400 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
