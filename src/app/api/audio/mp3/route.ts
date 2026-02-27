import { NextRequest, NextResponse } from "next/server";
import { Mp3Encoder } from "lamejs";

export const runtime = "nodejs";

type ParsedWav = {
  numChannels: number;
  sampleRate: number;
  bitsPerSample: number;
  dataOffset: number;
  dataSize: number;
};

function parseWavHeader(buffer: Buffer): ParsedWav {
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

  if (audioFormat !== 1 || bitsPerSample !== 16) {
    throw new Error("Unsupported WAV format. Expected PCM 16-bit WAV.");
  }

  return {
    numChannels,
    sampleRate,
    bitsPerSample,
    dataOffset,
    dataSize,
  };
}

function wavPcm16ToMp3Buffer(wavBuffer: Buffer, kbps: number): Buffer {
  const parsed = parseWavHeader(wavBuffer);
  const channels = Math.max(1, Math.min(2, parsed.numChannels));
  const encoder = new Mp3Encoder(channels, parsed.sampleRate, kbps);

  const frameSize = 1152;
  const bytesPerFrame = parsed.bitsPerSample / 8;
  const bytesPerSampleAcrossChannels = bytesPerFrame * parsed.numChannels;
  const totalSamplesPerChannel = Math.floor(parsed.dataSize / bytesPerSampleAcrossChannels);

  const chunks: Buffer[] = [];

  for (let sampleIndex = 0; sampleIndex < totalSamplesPerChannel; sampleIndex += frameSize) {
    const frameLength = Math.min(frameSize, totalSamplesPerChannel - sampleIndex);
    const left = new Int16Array(frameLength);
    const right = channels === 2 ? new Int16Array(frameLength) : null;

    for (let i = 0; i < frameLength; i += 1) {
      const absoluteSample = sampleIndex + i;
      const base = parsed.dataOffset + absoluteSample * bytesPerSampleAcrossChannels;

      left[i] = wavBuffer.readInt16LE(base);
      if (right) {
        right[i] = wavBuffer.readInt16LE(base + bytesPerFrame);
      }
    }

    const encoded = channels === 2 && right ? encoder.encodeBuffer(left, right) : encoder.encodeBuffer(left);
    if (encoded.length > 0) {
      chunks.push(Buffer.from(encoded.buffer, encoded.byteOffset, encoded.byteLength));
    }
  }

  const flush = encoder.flush();
  if (flush.length > 0) {
    chunks.push(Buffer.from(flush.buffer, flush.byteOffset, flush.byteLength));
  }

  return Buffer.concat(chunks);
}

export async function POST(request: NextRequest) {
  try {
    const bitrate = Number(request.nextUrl.searchParams.get("bitrate") || "128");
    const safeBitrate = Number.isFinite(bitrate) && bitrate >= 64 && bitrate <= 320 ? Math.round(bitrate) : 128;

    const wavArrayBuffer = await request.arrayBuffer();
    const wavBuffer = Buffer.from(wavArrayBuffer);

    if (wavBuffer.length === 0) {
      return NextResponse.json({ error: "Request body is empty." }, { status: 400 });
    }

    const mp3Buffer = wavPcm16ToMp3Buffer(wavBuffer, safeBitrate);

    return new NextResponse(new Uint8Array(mp3Buffer), {
      status: 200,
      headers: {
        "Content-Type": "audio/mpeg",
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to convert WAV to MP3.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
