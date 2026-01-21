import { createSupabaseServer } from "@/lib/supabase/server";
import { redirect } from "next/navigation";
import { generateUUID } from "./lib/utils/generate-uuid";

export default async function NewChatPage() {
  // Auth removed - demo mode only
  const demoUserId = "demo-user";
  const supabase = await createSupabaseServer();

  // Create a new conversation
  const { data: conversation } = await supabase
    .from("conversations")
    .insert({
      id: generateUUID(),
      title: "New Chat",
      user_id: demoUserId,
    })
    .select()
    .single();
    
  if (!conversation) {
    throw new Error("Failed to create conversation");
  }

  // Redirect to the new conversation
  return redirect(`/chat/${conversation.id}`);
}
