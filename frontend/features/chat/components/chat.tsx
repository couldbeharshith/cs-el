"use client";

import { useChat } from "@ai-sdk/react";
import type { Message } from "ai";
import { useRef, useState } from "react";
import { CircleAlert, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ChatHeader } from "./chat-header";
import { Messages } from "./messages";
import { MultimodalInput } from "./multimodal-input";
import { SuggestedActions } from "./suggested-actions";
import { saveMessages } from "../actions";
import { models } from "../lib/ai/models";
import { useConversationsContext } from "../hooks/conversations-context";

interface ChatProps {
  id: string;
  title: string;
  initialInput: string;
  initialMessages: Message[];
}

export function Chat({ id, title, initialInput, initialMessages }: ChatProps) {
  const [selectedModel, setSelectedModel] = useState<string>(models[0].value);
  const [savingMessage, setSavingMessage] = useState(false);
  const [savingReply, setSavingReply] = useState(false);
  const sending = useRef(false);
  const [sendError, setSendError] = useState("");
  const [unsaved, setUnsaved] = useState<Message>();
  const { conversations, refreshConversations } = useConversationsContext();
  const currentTitle = conversations.find((conversation) => conversation.id === id)?.title ?? title;

  const { messages, input, setInput, append, reload, status, error, stop } = useChat({
    id,
    initialMessages,
    initialInput,
    api: "/api/chat",
    experimental_prepareRequestBody: ({ messages }) => ({ messages, selectedModel }),
    onFinish: async (message) => {
      setSavingReply(true);
      try {
        await saveMessages([message], id);
        void refreshConversations();
      } catch {
        setUnsaved(message);
      } finally {
        setSavingReply(false);
      }
    },
  });

  const responding = status === "submitted" || status === "streaming";
  const busy = responding || savingMessage || savingReply;

  async function send(content: string) {
    const text = content.trim();
    if (!text || busy || sending.current || unsaved) return;
    sending.current = true;
    setSavingMessage(true);
    setSendError("");
    const message: Message = { id: crypto.randomUUID(), role: "user", content: text };
    try {
      try {
        await saveMessages([message], id);
      } catch {
        setSendError("Your message could not be saved. Try sending it again.");
        return;
      } finally {
        setSavingMessage(false);
      }
      setInput("");
      void refreshConversations();
      await append(message);
    } finally {
      sending.current = false;
    }
  }

  async function retrySave() {
    if (!unsaved || savingReply) return;
    setSavingReply(true);
    try {
      await saveMessages([unsaved], id);
      setUnsaved(undefined);
      void refreshConversations();
    } catch {
      setUnsaved(unsaved);
    } finally {
      setSavingReply(false);
    }
  }

  return (
    <div className="flex h-full min-w-0 flex-col">
      <ChatHeader title={currentTitle} />
      {messages.length ? (
        <Messages messages={messages} isLoading={status === "submitted"} />
      ) : (
        <main className="flex flex-1 flex-col overflow-y-auto">
          <div className="mx-auto my-auto w-full max-w-4xl px-4 py-6 sm:px-8 sm:py-7">
            <div className="mb-5 sm:mb-6">
              <div className="mb-3 grid size-11 place-items-center rounded-xl border border-primary/25 bg-primary/10 text-primary">
                <ShieldCheck className="size-6" aria-hidden />
              </div>
              <h1 className="text-3xl font-semibold tracking-tight sm:text-5xl">What are you researching?</h1>
              <p className="mt-3 max-w-xl text-sm leading-relaxed text-muted-foreground sm:text-base">
                Explore the markets, compare companies, or work through the details of an insurance policy.
              </p>
            </div>
            <SuggestedActions onSelect={send} disabled={busy} />
          </div>
        </main>
      )}

      <div className="shrink-0 bg-background/90">
        <div className="mx-auto max-w-4xl px-4 pb-[max(1rem,env(safe-area-inset-bottom))] pt-3 sm:px-8">
          {(sendError || error || unsaved) && (
            <div role="alert" className="mb-3 flex items-center gap-2 rounded-xl border border-destructive/25 bg-destructive/5 px-3 py-2 text-sm">
              <CircleAlert className="size-4 shrink-0 text-destructive" aria-hidden />
              <span className="flex-1">{sendError || (unsaved ? "This response hasn’t been saved to history." : formatChatError(error))}</span>
              {unsaved ? <Button type="button" size="sm" variant="outline" disabled={savingReply} onClick={() => void retrySave()}>Save again</Button> :
                error && !sendError && <Button type="button" size="sm" variant="outline" disabled={busy} onClick={() => void reload()}>Retry</Button>}
            </div>
          )}
          <MultimodalInput
            value={input}
            setInput={setInput}
            handleSubmit={(event) => { event.preventDefault(); void send(input); }}
            busy={busy || Boolean(unsaved)}
            readOnly={savingMessage}
            responding={responding}
            onStop={stop}
            modelState={{ selectedModel, setSelectedModel }}
          />
          <p className="mt-2 text-center text-xs text-muted-foreground">Verify important figures and policy details with the original source.</p>
        </div>
      </div>
    </div>
  );
}

function formatChatError(error: Error | undefined) {
  const message = error?.message ?? "";
  try {
    const payload = JSON.parse(message);
    if (typeof payload.error === "string") return payload.error;
  } catch {
    if (message.startsWith("The model or")) return message;
  }
  return "The response was interrupted. You can retry.";
}
