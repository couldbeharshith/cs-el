import { NextResponse } from "next/server";
import OpenAI from "openai";

export async function POST(request: Request) {
  const form = await request.formData().catch(() => null);
  const audio = form?.get("audio");
  if (!(audio instanceof File) || !audio.size) {
    return NextResponse.json({ error: "An audio file is required." }, { status: 400 });
  }
  if (audio.size > 25_000_000) {
    return NextResponse.json({ error: "Audio must be under 25 MB." }, { status: 413 });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "Voice transcription is unavailable." }, { status: 503 });
  }

  try {
    const openai = new OpenAI({ apiKey, baseURL: process.env.OPENAI_BASE_URL, timeout: 25_000, maxRetries: 0 });
    const text = await openai.audio.transcriptions.create({
      file: audio,
      model: "whisper-1",
      response_format: "text",
    }, { signal: request.signal });
    return NextResponse.json({ text });
  } catch (error) {
    console.error("Voice transcription failed", error);
    return NextResponse.json({ error: "Couldn’t transcribe this recording." }, { status: 502 });
  }
}
