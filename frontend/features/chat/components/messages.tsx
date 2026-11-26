"use client";

import dynamic from "next/dynamic";
import type { Message } from "ai";
import { ArrowDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PreviewMessage, ThinkingMessage } from "./message";
import { useScrollToBottom } from "./use-scroll-to-bottom";
const ToolResultDisplay = dynamic(() => import("./tool-result-display").then((module) => module.ToolResultDisplay));

export function Messages({ isLoading, messages }: { isLoading: boolean; messages: Message[] }) {
  const { container, content, onScroll, showJump, scrollToBottom } = useScrollToBottom();

  return (
    <main className="relative flex min-h-0 flex-1 flex-col">
      <div ref={container} onScroll={onScroll} className="flex-1 overflow-y-auto" aria-label="Conversation">
        <div ref={content} className="flex flex-col gap-7 py-6 sm:py-8">
          {messages.map((message) => (
            <div key={message.id} className="min-w-0">
              {message.toolInvocations?.map((call) => <ToolResultDisplay key={call.toolCallId} toolCall={call} />)}
              {message.content && <PreviewMessage message={message} />}
            </div>
          ))}
          {isLoading && <ThinkingMessage />}
        </div>
      </div>
      {showJump && <Button type="button" size="sm" variant="outline"
        className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-card shadow"
        onClick={scrollToBottom}><ArrowDown /> Latest message</Button>}
    </main>
  );
}
