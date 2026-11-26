"use client";

import { createContext, useContext, useEffect, type ReactNode } from "react";
import { useConversations } from "./use-conversations";

const ConversationsContext = createContext<ReturnType<typeof useConversations> | null>(null);

export function ConversationsProvider({ userId, children }: { userId: string; children: ReactNode }) {
  const conversations = useConversations(userId);
  const { refreshConversations } = conversations;

  useEffect(() => { void refreshConversations(); }, [refreshConversations]);

  return <ConversationsContext.Provider value={conversations}>{children}</ConversationsContext.Provider>;
}

export function useConversationsContext() {
  const context = useContext(ConversationsContext);
  if (!context) throw new Error("Chat history requires a conversations provider");
  return context;
}
