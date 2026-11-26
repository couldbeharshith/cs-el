"use client";

import type { Message } from "ai";
import { LoaderCircle, Sparkles } from "lucide-react";
import { memo } from "react";
import { MarkdownContent } from "@/components/ui/markdown-content";

export const PreviewMessage = memo(function PreviewMessage({ message }: { message: Message }) {
  const user = message.role === "user";

  return (
    <article className="mx-auto flex w-full max-w-4xl min-w-0 items-start gap-3 px-4 sm:px-8"
      aria-label={user ? "Your message" : "Assistant response"}>
      {!user && <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary">
        <Sparkles className="size-4" aria-hidden />
      </span>}
      {user ? (
        <div className="ml-auto max-w-[90%] min-w-0 overflow-x-auto rounded-2xl rounded-br-sm bg-primary/10 px-4 py-2.5 text-sm leading-relaxed">
          <MarkdownContent content={message.content} id={message.id} />
        </div>
      ) : (
        <div className="min-w-0 flex-1 overflow-x-auto pt-0.5">
          <MarkdownContent content={message.content} id={message.id} />
        </div>
      )}
    </article>
  );
});

export function ThinkingMessage() {
  return (
    <div role="status" className="mx-auto flex w-full max-w-4xl items-center gap-3 px-4 text-sm text-muted-foreground sm:px-8">
      <span className="grid size-8 place-items-center rounded-lg bg-primary/10 text-primary">
        <LoaderCircle className="size-4 animate-spin" aria-hidden />
      </span>
      Thinking…
    </div>
  );
}
