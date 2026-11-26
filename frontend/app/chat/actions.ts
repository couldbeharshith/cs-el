"use server";

import { createHash } from "node:crypto";
import type { Message } from "ai";
import { createSupabaseServer } from "@/lib/supabase/server";
import { getUser } from "./hooks/get-user";

export async function saveMessages(messages: Message[], conversationId: string) {
  const supabase = await createSupabaseServer();
  const user = await getUser();
  const { data: conversation, error: lookupError } = await supabase.from("conversations")
    .select("title").eq("id", conversationId).eq("user_id", user.id).maybeSingle();

  if (lookupError || !conversation) throw new Error("Conversation unavailable");

  const rows = messages.map((message) => ({
    id: messageId(conversationId, message.id),
    conversation_id: conversationId,
    content: message.content,
    role: message.role,
    tool_invocations: message.toolInvocations,
  }));

  const { error } = await supabase.from("messages").upsert(rows, {
    onConflict: "id",
    ignoreDuplicates: true,
  });
  if (error) {
    console.error("Could not save messages", error);
    throw new Error("Could not save messages");
  }

  const firstUserMessage = messages.find((message) => message.role === "user");
  const updatedAt = new Date().toISOString();
  const update = supabase.from("conversations").update({ updated_at: updatedAt })
    .eq("id", conversationId).eq("user_id", user.id).lt("updated_at", updatedAt);
  const operations = [update];
  if (conversation.title === "New Chat" && firstUserMessage?.content) {
    const text = firstUserMessage.content;
    const title = text.length > 50 ? text.slice(0, 47) + "..." : text;
    operations.push(supabase.from("conversations").update({ title })
      .eq("id", conversationId).eq("user_id", user.id).eq("title", "New Chat"));
  }
  for (const result of await Promise.all(operations)) {
    if (result.error) console.error("Could not update conversation metadata", result.error);
  }
}

function messageId(conversationId: string, sourceId: string) {
  const hash = createHash("sha256").update(`${conversationId}:${sourceId}`).digest("hex");
  return `${hash.slice(0, 8)}-${hash.slice(8, 12)}-8${hash.slice(13, 16)}-8${hash.slice(17, 20)}-${hash.slice(20, 32)}`;
}
