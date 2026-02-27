"use client";

import AudiobookDemo from "@/components/AudiobookDemo";

export default function HomePage() {
  return <AudiobookDemo />;
}

/*

import { CreateMLCEngine, type MLCEngineInterface } from "@mlc-ai/web-llm";
import { KokoroTTS } from "kokoro-js";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type ChatRole = "user" | "assistant" | "system";

type ChatMessage = {
  role: ChatRole;
  content: string;
};

type UsageSummary = {
  totalTokens?: number;
  decodeTokensPerSecond?: number;
};

type KokoroInstance = Awaited<ReturnType<typeof KokoroTTS.from_pretrained>>;

type ModelOption = {
  id: string;
  sizeMB: number;
};

const MODEL_OPTIONS: ModelOption[] = [
  { id: "Llama-3.2-1B-Instruct-q4f32_1-MLC", sizeMB: 650 },
  { id: "Llama-3.2-3B-Instruct-q4f32_1-MLC", sizeMB: 1650 },
  { id: "gemma-2-2b-it-q4f32_1-MLC-1k", sizeMB: 1350 },
  { id: "Qwen2.5-1.5B-Instruct-q4f32_1-MLC", sizeMB: 900 },
];

const KOKORO_MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const KOKORO_DTYPES = ["fp32", "fp16", "q8", "q4", "q4f16"] as const;
const KOKORO_VOICES = [
  "af_heart",
  "af_bella",
  "am_fenrir",
  "am_michael",
  "am_puck",
  "bf_emma",
  "bf_isabella",
  "bm_george",
  "bm_fable",
] as const;

type KokoroVoice = (typeof KOKORO_VOICES)[number];
type KokoroDtype = (typeof KOKORO_DTYPES)[number];
type KokoroDevice = "webgpu" | "wasm" | "cpu";

export default function Home() {
  const engineRef = useRef<MLCEngineInterface | null>(null);
  const ttsRef = useRef<KokoroInstance | null>(null);
  const ttsDtypeRef = useRef<KokoroDtype | null>(null);
  const ttsQueueRef = useRef<string[]>([]);
  const ttsBusyRef = useRef(false);
  const activeAudioRef = useRef<HTMLAudioElement | null>(null);
  const activeAudioUrlRef = useRef<string | null>(null);
  const [model, setModel] = useState(MODEL_OPTIONS[0].id);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(512);
  const [ttsVoice, setTtsVoice] = useState<KokoroVoice>("af_heart");
  const [ttsDtype, setTtsDtype] = useState<KokoroDtype>("q8");
  const [ttsDevice, setTtsDevice] = useState<KokoroDevice | null>(null);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [ttsLoading, setTtsLoading] = useState(false);
  const [ttsReady, setTtsReady] = useState(false);
  const [availableVoices] = useState<KokoroVoice[]>([...KOKORO_VOICES]);
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Hi! Load a model, then ask me anything. Responses run locally in your browser via WebLLM.",
    },
  ]);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [status, setStatus] = useState("Pick a model and click “Load model”.");
  const [usage, setUsage] = useState<UsageSummary | null>(null);

  const canSend = modelLoaded && !isGenerating && prompt.trim().length > 0;
  const selectedModel =
    MODEL_OPTIONS.find((option) => option.id === model) ?? MODEL_OPTIONS[0];

  const cleanupAudioPlayback = useCallback(() => {
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current.src = "";
      activeAudioRef.current = null;
    }

    if (activeAudioUrlRef.current) {
      URL.revokeObjectURL(activeAudioUrlRef.current);
      activeAudioUrlRef.current = null;
    }
  }, []);

  const ensureTtsLoaded = useCallback(async () => {
    if (ttsRef.current && ttsDtypeRef.current === ttsDtype) {
      return ttsRef.current;
    }

    if (ttsRef.current && ttsDtypeRef.current !== ttsDtype) {
      ttsRef.current = null;
      ttsDtypeRef.current = null;
      setTtsReady(false);
    }

    setTtsLoading(true);
    try {
      try {
        const ortModule = (await import("onnxruntime-web")) as {
          env?: { logLevel?: string };
        };

        if (ortModule.env) {
          ortModule.env.logLevel = "error";
        }
      } catch {
        // Optional: if ORT module cannot be configured, continue with default logging.
      }

      const deviceCandidates: KokoroDevice[] = [];
      const hasWebGpu = Boolean((navigator as Navigator & { gpu?: unknown }).gpu);
      const hasWasm = typeof WebAssembly !== "undefined";

      if (hasWebGpu) {
        deviceCandidates.push("webgpu");
      }
      if (hasWasm) {
        deviceCandidates.push("wasm");
      }
      deviceCandidates.push("cpu");

      let tts: KokoroInstance | null = null;
      let selectedDevice: KokoroDevice | null = null;
      let lastError: Error | null = null;

      for (const device of deviceCandidates) {
        try {
          tts = await KokoroTTS.from_pretrained(KOKORO_MODEL_ID, {
            dtype: ttsDtype,
            device,
          });
          selectedDevice = device;
          break;
        } catch (error) {
          lastError = error as Error;
        }
      }

      if (!tts || !selectedDevice) {
        throw lastError ?? new Error("No compatible Kokoro device is available.");
      }

      ttsRef.current = tts;
      ttsDtypeRef.current = ttsDtype;
      setTtsDevice(selectedDevice);
      setTtsReady(true);
      return tts;
    } finally {
      setTtsLoading(false);
    }
  }, [ttsDtype]);

  const processTtsQueue = useCallback(async () => {
    if (ttsBusyRef.current || !ttsEnabled) {
      return;
    }

    ttsBusyRef.current = true;

    try {
      const tts = await ensureTtsLoaded();

      while (ttsQueueRef.current.length > 0 && ttsEnabled) {
        const line = ttsQueueRef.current.shift();
        if (!line) {
          continue;
        }

        const audio = await tts.generate(line, { voice: ttsVoice });
        const blob = audio.toBlob();
        const url = URL.createObjectURL(blob);
        activeAudioUrlRef.current = url;

        await new Promise<void>((resolve) => {
          const playback = new Audio(url);
          activeAudioRef.current = playback;

          const done = () => {
            playback.onended = null;
            playback.onerror = null;

            if (activeAudioUrlRef.current) {
              URL.revokeObjectURL(activeAudioUrlRef.current);
              activeAudioUrlRef.current = null;
            }

            activeAudioRef.current = null;
            resolve();
          };

          playback.onended = done;
          playback.onerror = done;
          void playback.play().catch(done);
        });
      }
    } catch {
      setStatus("Error in Kokoro TTS playback.");
    } finally {
      ttsBusyRef.current = false;
    }
  }, [ensureTtsLoaded, ttsEnabled, ttsVoice]);

  const enqueueTtsLines = useCallback((lines: string[]) => {
    if (!ttsEnabled) {
      return;
    }

    const filtered = lines.map((line) => line.trim()).filter((line) => line.length > 0);
    if (filtered.length === 0) {
      return;
    }

    ttsQueueRef.current.push(...filtered);
    void processTtsQueue();
  }, [processTtsQueue, ttsEnabled]);

  useEffect(() => {
    const originalWarn = console.warn.bind(console);
    const originalError = console.error.bind(console);

    const shouldIgnoreOrtEpAssignmentMessage = (args: unknown[]) => {
      const ansiRegex = /\u001b\[[0-9;]*m/g;
      const joined = args
        .map((arg) => {
          if (typeof arg === "string") return arg;
          if (arg instanceof Error) return arg.message;
          return "";
        })
        .join(" ")
        .replace(ansiRegex, "");

      return (
        joined.includes("VerifyEachNodeIsAssignedToAnEp") ||
        joined.includes("Some nodes were not assigned to the preferred execution providers") ||
        joined.includes("Rerunning with verbose output on a non-minimal build will show node assignments")
      );
    };

    console.warn = (...args: unknown[]) => {
      if (shouldIgnoreOrtEpAssignmentMessage(args)) {
        return;
      }

      originalWarn(...args);
    };

    console.error = (...args: unknown[]) => {
      if (shouldIgnoreOrtEpAssignmentMessage(args)) {
        return;
      }

      originalError(...args);
    };

    return () => {
      console.warn = originalWarn;
      console.error = originalError;
    };
  }, []);

  useEffect(() => {
    if (!ttsEnabled) {
      ttsQueueRef.current = [];
      cleanupAudioPlayback();
      return;
    }

    void processTtsQueue();
  }, [cleanupAudioPlayback, processTtsQueue, ttsEnabled]);

  useEffect(() => {
    if (ttsReady && ttsEnabled && ttsQueueRef.current.length > 0) {
      void processTtsQueue();
    }
  }, [processTtsQueue, ttsReady, ttsEnabled, ttsVoice]);

  useEffect(() => {
    ttsQueueRef.current = [];
    cleanupAudioPlayback();
    ttsRef.current = null;
    ttsDtypeRef.current = null;
    setTtsDevice(null);
    setTtsReady(false);
  }, [cleanupAudioPlayback, ttsDtype]);

  useEffect(() => {
    return () => {
      cleanupAudioPlayback();
    };
  }, [cleanupAudioPlayback]);

  const statusTone = useMemo(() => {
    if (status.toLowerCase().includes("error")) return "text-rose-200";
    if (modelLoaded) return "text-emerald-200";
    return "text-sky-200";
  }, [modelLoaded, status]);

  const loadModel = async () => {
    setIsLoadingModel(true);
    setModelLoaded(false);
    setUsage(null);

    try {
      setStatus(
        `Preparing model download (~${selectedModel.sizeMB} MB). Files will be cached in your browser after first load.`,
      );

      const webGpu = (navigator as Navigator & { gpu?: unknown }).gpu;

      if (!webGpu) {
        throw new Error(
          "WebGPU is not available in this browser. Use a recent Chrome/Edge build with WebGPU enabled.",
        );
      }

      const engine = engineRef.current
        ? engineRef.current
        : await CreateMLCEngine(model, {
            initProgressCallback: (report) => {
              const pct = Math.round((report.progress ?? 0) * 100);
              setStatus(
                `${report.text ?? "Loading model"}${Number.isFinite(pct) ? ` (${pct}%)` : ""}`,
              );
            },
          });

      if (engineRef.current) {
        await engine.reload(model);
      }

      engineRef.current = engine;
      setModelLoaded(true);
      setStatus(`Model ready: ${model}`);
    } catch (error) {
      setStatus(`Error loading model: ${(error as Error).message}`);
      setModelLoaded(false);
    } finally {
      setIsLoadingModel(false);
    }
  };

  const clearChat = async () => {
    setMessages([
      {
        role: "assistant",
        content: "Chat cleared. Send a new message whenever you’re ready.",
      },
    ]);
    setUsage(null);
    ttsQueueRef.current = [];
    cleanupAudioPlayback();

    if (engineRef.current) {
      await engineRef.current.resetChat();
    }
  };

  const sendMessage = async (event: FormEvent) => {
    event.preventDefault();
    if (!canSend || !engineRef.current) return;

    const input = prompt.trim();
    setPrompt("");
    setIsGenerating(true);
    setStatus("Generating...");

    const userMessage: ChatMessage = { role: "user", content: input };
    const history = [...messages, userMessage];
    const assistantIndex = history.length;
    let assistantReply = "";
    let spokenLineCount = 0;

    setMessages([...history, { role: "assistant", content: "" }]);

    try {
      const chunks = await engineRef.current.chat.completions.create({
        messages: history,
        temperature,
        top_p: topP,
        max_tokens: maxTokens,
        stream: true,
        stream_options: { include_usage: true },
      });

      for await (const chunk of chunks) {
        const token = chunk.choices[0]?.delta?.content ?? "";
        assistantReply += token;

        const lines = assistantReply.split(/\r?\n/);
        const completeLineCount = assistantReply.endsWith("\n")
          ? lines.length
          : Math.max(lines.length - 1, 0);

        if (completeLineCount > spokenLineCount) {
          enqueueTtsLines(lines.slice(spokenLineCount, completeLineCount));
          spokenLineCount = completeLineCount;
        }

        setMessages((current) => {
          const next = [...current];
          if (next[assistantIndex]) {
            next[assistantIndex] = {
              role: "assistant",
              content: assistantReply,
            };
          }
          return next;
        });

        if (chunk.usage) {
          setUsage({
            totalTokens: chunk.usage.total_tokens,
            decodeTokensPerSecond: chunk.usage.extra?.decode_tokens_per_s,
          });
        }
      }

      const finalLines = assistantReply.split(/\r?\n/);
      if (spokenLineCount < finalLines.length) {
        enqueueTtsLines(finalLines.slice(spokenLineCount));
      }

      setStatus("Done.");
    } catch (error) {
      setStatus(`Error generating response: ${(error as Error).message}`);
      setMessages((current) => {
        const next = [...current];
        if (next[assistantIndex]) {
          next[assistantIndex] = {
            role: "assistant",
            content:
              "I could not generate a response. Try reducing max tokens or loading a smaller model.",
          };
        }
        return next;
      });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_#67e8f940,_transparent_45%),radial-gradient(circle_at_bottom_right,_#a855f740,_transparent_40%)]" />

      <main className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
        <section className="rounded-3xl border border-white/15 bg-white/10 p-6 shadow-2xl shadow-black/40 backdrop-blur-xl">
          <h1 className="text-2xl font-semibold tracking-tight text-white md:text-3xl">
            Browser LLM Chat
          </h1>
          <p className="mt-2 text-sm text-slate-200 md:text-base">
            Next.js + TypeScript + WebLLM. Your model runs directly on your device.
          </p>
          <p className="mt-1 text-xs text-slate-300 md:text-sm">
            Kokoro TTS reads each assistant output line and downloads its voice model into browser cache on first use.
          </p>

          <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-6">
            <label className="xl:col-span-2">
              <span className="mb-2 block text-sm font-medium text-slate-100">Model</span>
              <select
                value={model}
                onChange={(e) => {
                  setModel(e.target.value);
                  setModelLoaded(false);
                  setStatus("Model changed. Click “Load model”.");
                }}
                className="w-full rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-300"
              >
                {MODEL_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id} className="bg-slate-900 text-white">
                    {option.id} (~{option.sizeMB} MB)
                  </option>
                ))}
              </select>
              <span className="mt-2 block text-xs text-slate-300">
                First load downloads about {selectedModel.sizeMB} MB and stores it in browser cache.
              </span>
            </label>

            <label>
              <span className="mb-2 block text-sm font-medium text-slate-100">
                Temperature: {temperature.toFixed(2)}
              </span>
              <input
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full accent-cyan-300"
              />
            </label>

            <label>
              <span className="mb-2 block text-sm font-medium text-slate-100">Top-p: {topP.toFixed(2)}</span>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={topP}
                onChange={(e) => setTopP(Number(e.target.value))}
                className="w-full accent-violet-300"
              />
            </label>

            <label>
              <span className="mb-2 block text-sm font-medium text-slate-100">Max tokens</span>
              <input
                type="number"
                min={32}
                max={2048}
                step={32}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                className="w-full rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-sm text-white outline-none transition focus:border-violet-300"
              />
            </label>

            <label>
              <span className="mb-2 block text-sm font-medium text-slate-100">Kokoro voice</span>
              <select
                value={ttsVoice}
                onChange={(e) => setTtsVoice(e.target.value as KokoroVoice)}
                className="w-full rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-300"
              >
                {availableVoices.map((voice) => (
                  <option key={voice} value={voice} className="bg-slate-900 text-white">
                    {voice}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span className="mb-2 block text-sm font-medium text-slate-100">Kokoro quantization</span>
              <select
                value={ttsDtype}
                onChange={(e) => setTtsDtype(e.target.value as KokoroDtype)}
                className="w-full rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-300"
              >
                {KOKORO_DTYPES.map((dtype) => (
                  <option key={dtype} value={dtype} className="bg-slate-900 text-white">
                    {dtype}
                  </option>
                ))}
              </select>
              <span className="mt-2 block text-xs text-slate-300">Options: fp32, fp16, q8, q4, q4f16</span>
              <span className="mt-1 block text-xs text-cyan-200">
                Active Kokoro device: {ttsLoading ? "detecting..." : ttsDevice ?? "not loaded yet"} (fallback: webgpu → wasm → cpu)
              </span>
            </label>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={loadModel}
              disabled={isLoadingModel || isGenerating}
              className="rounded-xl bg-cyan-400/90 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isLoadingModel ? "Loading..." : "Load model"}
            </button>

            <button
              type="button"
              onClick={() => setTtsEnabled((enabled) => !enabled)}
              className="rounded-xl border border-white/25 bg-white/10 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/20"
            >
              {ttsEnabled ? "Voice: on" : "Voice: off"}
            </button>

            <button
              type="button"
              onClick={clearChat}
              disabled={isGenerating}
              className="rounded-xl border border-white/25 bg-white/10 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Clear chat
            </button>

            <p className={`text-sm ${statusTone}`}>{status}</p>
            <p className="text-xs text-slate-300">
              Kokoro: {ttsLoading ? "loading..." : ttsReady ? `ready (${ttsVoice}, ${ttsDtype}, ${ttsDevice ?? "-"})` : `idle (${ttsDtype})`}
            </p>
          </div>

          {usage && (
            <p className="mt-2 text-xs text-slate-300">
              Tokens: {usage.totalTokens ?? "-"} · Decode speed: {usage.decodeTokensPerSecond?.toFixed(2) ?? "-"} tok/s
            </p>
          )}
        </section>

        <section className="flex min-h-[420px] flex-1 flex-col rounded-3xl border border-white/15 bg-white/10 p-4 shadow-2xl shadow-black/30 backdrop-blur-xl md:p-6">
          <div className="chat-scroll flex-1 space-y-4 overflow-y-auto pr-1">
            {messages.map((message, index) => (
              <div
                key={`${index}-${message.role}`}
                className={`max-w-[90%] rounded-2xl border px-4 py-3 text-sm leading-6 md:text-base ${
                  message.role === "user"
                    ? "ml-auto border-cyan-300/40 bg-cyan-200/20 text-cyan-50"
                    : "border-white/20 bg-white/10 text-slate-100"
                }`}
              >
                <p className="mb-1 text-xs uppercase tracking-wide text-slate-300">{message.role}</p>
                <p className="whitespace-pre-wrap">{message.content || (isGenerating ? "..." : "")}</p>
              </div>
            ))}
          </div>

          <form onSubmit={sendMessage} className="mt-4 flex flex-col gap-3 md:flex-row">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              placeholder="Type your message..."
              className="min-h-[86px] flex-1 resize-y rounded-2xl border border-white/20 bg-black/20 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-400 focus:border-cyan-300 md:text-base"
            />

            <button
              type="submit"
              disabled={!canSend}
              className="rounded-2xl bg-violet-400/90 px-6 py-3 text-sm font-semibold text-slate-950 transition hover:bg-violet-300 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isGenerating ? "Thinking..." : "Send"}
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}

*/
