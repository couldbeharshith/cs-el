"use client";

import { useState, useRef, useCallback, useEffect } from "react";

interface UseVoiceRecorderOptions {
  onTranscription: (text: string) => void;
  silenceTimeout?: number; // ms to wait after silence before stopping
}

interface UseVoiceRecorderReturn {
  isRecording: boolean;
  isProcessing: boolean;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  error: string | null;
}

export function useVoiceRecorder({
  onTranscription,
  silenceTimeout = 2000,
}: UseVoiceRecorderOptions): UseVoiceRecorderReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Cleanup function
  const cleanup = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  // Send audio to Whisper API
  const transcribeAudio = useCallback(
    async (audioBlob: Blob) => {
      setIsProcessing(true);
      setError(null);

      try {
        const formData = new FormData();
        // Create a file from the blob with proper extension
        const audioFile = new File([audioBlob], "recording.webm", {
          type: audioBlob.type,
        });
        formData.append("audio", audioFile);

        const response = await fetch("/api/transcribe", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Transcription failed");
        }

        const data = await response.json();
        if (data.text && data.text.trim()) {
          onTranscription(data.text.trim());
        }
      } catch (err: any) {
        console.error("Transcription error:", err);
        setError(err.message || "Failed to transcribe audio");
      } finally {
        setIsProcessing(false);
      }
    },
    [onTranscription]
  );

  // Monitor audio levels for silence detection
  const monitorSilence = useCallback(() => {
    if (!analyserRef.current || !isRecording) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const checkAudioLevel = () => {
      if (!isRecording || !analyserRef.current) return;

      analyser.getByteFrequencyData(dataArray);

      // Calculate average volume
      const average = dataArray.reduce((a, b) => a + b, 0) / bufferLength;
      const threshold = 10; // Adjust this threshold as needed

      if (average < threshold) {
        // Silence detected - start or continue silence timer
        if (!silenceTimerRef.current) {
          silenceTimerRef.current = setTimeout(() => {
            if (mediaRecorderRef.current?.state === "recording") {
              mediaRecorderRef.current.stop();
            }
          }, silenceTimeout);
        }
      } else {
        // Sound detected - reset silence timer
        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = null;
        }
      }

      animationFrameRef.current = requestAnimationFrame(checkAudioLevel);
    };

    checkAudioLevel();
  }, [isRecording, silenceTimeout]);

  const startRecording = useCallback(async () => {
    setError(null);
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Set up audio analysis for silence detection
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      analyserRef.current.fftSize = 256;

      // Determine supported MIME type
      const mimeType = MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : MediaRecorder.isTypeSupported("audio/mp4")
        ? "audio/mp4"
        : "audio/ogg";

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        setIsRecording(false);
        cleanup();

        if (audioChunksRef.current.length > 0) {
          const audioBlob = new Blob(audioChunksRef.current, {
            type: mimeType,
          });
          await transcribeAudio(audioBlob);
        }
      };

      mediaRecorder.start(100); // Collect data every 100ms
      setIsRecording(true);

      // Start monitoring for silence
      setTimeout(monitorSilence, 500); // Give a small delay before starting silence detection
    } catch (err: any) {
      console.error("Error starting recording:", err);
      setError(
        err.name === "NotAllowedError"
          ? "Microphone access denied. Please allow microphone access."
          : "Failed to start recording"
      );
      cleanup();
    }
  }, [cleanup, monitorSilence, transcribeAudio]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  return {
    isRecording,
    isProcessing,
    startRecording,
    stopRecording,
    error,
  };
}

