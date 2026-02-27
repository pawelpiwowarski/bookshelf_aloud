import { NextResponse } from "next/server";
import { listBooks } from "@/lib/book-parser";

export async function GET() {
  try {
    const books = await listBooks();
    return NextResponse.json({ books });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to list books: ${(error as Error).message}` },
      { status: 500 },
    );
  }
}
