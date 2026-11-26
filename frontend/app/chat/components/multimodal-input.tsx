"use client";

import type { FormEvent } from "react";
import { ArrowUp, LoaderCircle, Mic, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useVoiceRecorder } from "../hooks/use-voice-recorder";
import { models } from "../lib/ai/models";
import { ModelSelector } from "./model-selector";

interface MultimodalInputProps {
  value: string;
  setInput: (value: string) => void;
  handleSubmit: (event: FormEvent<HTMLFormElement>) => void;
  busy: boolean;
  responding: boolean;
  onStop: () => void;
  modelState: {
    selectedModel: string;
    setSelectedModel: (model: string) => void;
  };
}

export function MultimodalInput({
  value, setInput, handleSubmit, busy, responding, onStop, modelState,
}: MultimodalInputProps) {
  const { isRecording, isProcessing, startRecording, stopRecording, error } = useVoiceRecorder({
    onTranscription: (text) => setInput(value ? `${value} ${text}` : text),
  });

  return (
    <form onSubmit={handleSubmit} className="overflow-hidden rounded-2xl border bg-card shadow-sm focus-within:border-primary/60">
      <textarea
        value={value}
        onChange={(event) => setInput(event.target.value)}
        placeholder="Ask a question or paste a document link…"
        aria-label="Message"
        rows={2}
        className="block min-h-16 w-full resize-none bg-transparent px-4 pt-3 text-base outline-none placeholder:text-muted-foreground"
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            if (!busy && !isRecording && !isProcessing) event.currentTarget.form?.requestSubmit();
          }
        }}
      />
      {error && <p role="alert" className="px-4 text-xs text-destructive">{error}</p>}
      <div className="flex items-center justify-between gap-2 px-2.5 pb-2.5">
        <ModelSelector models={models} {...modelState} />
        <div className="flex items-center gap-1.5">
          <Button type="button" size="icon" variant={isRecording ? "destructive" : "ghost"}
            className="rounded-full" disabled={isProcessing || (busy && !isRecording)}
            aria-label={isRecording ? "Stop recording" : isProcessing ? "Transcribing audio" : "Voice input"}
            onClick={isRecording ? stopRecording : startRecording}>
            {isProcessing ? <LoaderCircle className="animate-spin" /> : isRecording ? <Square /> : <Mic />}
          </Button>
          {responding ? (
            <Button type="button" size="icon" variant="outline" className="rounded-full" onClick={onStop} aria-label="Stop response">
              <Square />
            </Button>
          ) : (
            <Button type="submit" size="icon" className="rounded-full" disabled={busy || !value.trim() || isRecording || isProcessing} aria-label="Send message">
              {busy ? <LoaderCircle className="animate-spin" /> : <ArrowUp />}
            </Button>
          )}
        </div>
      </div>
    </form>
  );
}
