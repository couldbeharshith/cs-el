import { redirect } from "next/navigation";
import { createConversationServer } from "@/lib/supabase/queries/server/conversations";
import { getUser } from "@/features/chat/hooks/get-user";

export const dynamic = "force-dynamic";

export default async function NewChatPage({ searchParams }: {
  searchParams: Promise<{ q?: string }>;
}) {
  const { id } = await getUser();
  const { q } = await searchParams;
  const conversation = await createConversationServer(id);
  const query = typeof q === "string" && q.trim()
    ? `?${new URLSearchParams({ q: q.trim().slice(0, 4000) })}`
    : "";
  redirect(`/chat/${conversation.id}${query}`);
}
