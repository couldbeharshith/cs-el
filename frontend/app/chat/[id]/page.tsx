import { createSupabaseServer } from "@/lib/supabase/server";
import { Chat } from "../components/chat";
import { notFound } from "next/navigation";
import { getUser } from "../hooks/get-user";
import type { Message } from "ai";

export default async function ChatPage({ params, searchParams }: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ q?: string }>;
}) {
  const [{ id }, { q }, user] = await Promise.all([params, searchParams, getUser()]);
  if (!/^[\da-f]{8}(?:-[\da-f]{4}){3}-[\da-f]{12}$/i.test(id)) notFound();

  const supabase = await createSupabaseServer();
  const { data: conversation, error } = await supabase
    .from("conversations")
    .select("id, title")
    .eq("id", id)
    .eq("user_id", user.id)
    .maybeSingle();
  if (error) throw new Error("Could not load the conversation", { cause: error });
  if (!conversation) notFound();

  const { data: messages, error: messageError } = await supabase
    .from("messages")
    .select("id, content, role, tool_invocations, created_at")
    .eq("conversation_id", id)
    .order("created_at", { ascending: true });
  if (messageError) throw new Error("Could not load messages", { cause: messageError });

  const initialMessages: Message[] = messages.map((message) => ({
    id: message.id,
    role: message.role,
    content: message.content,
    toolInvocations: message.tool_invocations ?? undefined,
    createdAt: new Date(message.created_at),
  }));

  return <Chat key={id} id={id} title={conversation.title} initialMessages={initialMessages}
    initialInput={typeof q === "string" ? q.slice(0, 4000) : ""} />;
}
