"use client";

import { useEffect, useRef, useState } from "react";

const audioTypes = { "audio/webm": "webm", "audio/mp4": "mp4", "audio/ogg": "ogg" };

export function useVoiceRecorder({ onTranscription, silenceTimeout = 2_000 }: {
  onTranscription: (text: string) => void;
  silenceTimeout?: number;
}) {
  const [state, setState] = useState<"idle" | "requesting" | "recording" | "processing">("idle");
  const [error, setError] = useState<string | null>(null);
  const transcription = useRef(onTranscription);
  const stopRef = useRef<(() => void) | null>(null);
  const cancelRef = useRef<AbortController | null>(null);

  useEffect(() => { transcription.current = onTranscription; }, [onTranscription]);
  useEffect(() => () => { cancelRef.current?.abort(); }, []);

  async function startRecording() {
    if (cancelRef.current) return;
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("Voice recording is unavailable in this browser.");
      return;
    }

    const controller = new AbortController();
    cancelRef.current = controller;
    setError(null);
    setState("requesting");
    let stream: MediaStream | undefined;
    let recorder: MediaRecorder | undefined;
    let context: AudioContext | undefined;
    let frame = 0;
    let limit: ReturnType<typeof setTimeout> | undefined;
    const chunks: Blob[] = [];
    const cleanup = () => {
      cancelAnimationFrame(frame);
      clearTimeout(limit);
      stream?.getTracks().forEach((track) => track.stop());
      if (context && context.state !== "closed") void context.close().catch(() => {});
      stopRef.current = null;
      controller.signal.removeEventListener("abort", cleanup);
    };
    const finish = () => {
      cleanup();
      if (cancelRef.current === controller) cancelRef.current = null;
      if (!controller.signal.aborted) setState("idle");
    };

    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (controller.signal.aborted) { finish(); return; }
      controller.signal.addEventListener("abort", cleanup, { once: true });
      const mimeType = Object.keys(audioTypes).find((type) => MediaRecorder.isTypeSupported(type));
      recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      const currentRecorder = recorder;
      stopRef.current = () => { if (currentRecorder.state === "recording") currentRecorder.stop(); };
      recorder.ondataavailable = ({ data }) => { if (data.size) chunks.push(data); };
      recorder.onerror = () => { setError("The recording was interrupted."); finish(); };
      recorder.onstop = async () => {
        cleanup();
        if (controller.signal.aborted) { finish(); return; }
        setState("processing");
        try {
          const type = currentRecorder.mimeType.split(";")[0] as keyof typeof audioTypes;
          const audio = new Blob(chunks, { type });
          if (!audio.size) throw new Error("No audio was recorded.");
          const form = new FormData();
          form.append("audio", audio, `recording.${audioTypes[type] ?? "webm"}`);
          const response = await fetch("/api/transcribe", {
            method: "POST", body: form, signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30_000)]),
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.error || "Transcription failed.");
          if (typeof data.text === "string" && data.text.trim()) transcription.current(data.text.trim());
          else setError("No speech was detected.");
        } catch (error) {
          if (!controller.signal.aborted) setError(error instanceof Error ? error.message : "Transcription failed.");
        } finally {
          finish();
        }
      };
      recorder.start();
      setState("recording");
      limit = setTimeout(() => stopRef.current?.(), 120_000);

      if (typeof AudioContext === "undefined") return;
      try {
        context = new AudioContext();
        const analyser = context.createAnalyser();
        analyser.fftSize = 256;
        context.createMediaStreamSource(stream).connect(analyser);
        const bins = new Uint8Array(analyser.frequencyBinCount);
        let lastSound = 0;
        const check = (now: number) => {
          analyser.getByteFrequencyData(bins);
          if (bins.some((value) => value > 32)) lastSound = now;
          if (lastSound && now - lastSound > silenceTimeout) stopRef.current?.();
          else frame = requestAnimationFrame(check);
        };
        frame = requestAnimationFrame(check);
      } catch {
        // Manual stop remains available when audio analysis is unsupported.
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setError(error instanceof DOMException && error.name === "NotAllowedError"
          ? "Microphone access was denied." : "Couldn’t start recording.");
      }
      finish();
    }
  }

  return {
    isRecording: state === "recording",
    isProcessing: state === "processing" || state === "requesting",
    startRecording,
    stopRecording: () => stopRef.current?.(),
    error,
  };
}
