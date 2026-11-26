import { useState, useCallback, useRef } from "react";
import {
  createConversationBrowser,
  fetchConversationsBrowser,
  renameConversationBrowser,
  deleteConversationBrowser,
} from "@/lib/supabase/queries/client/conversations";
import type { Conversation } from "../types";
import { toast } from "sonner";

export function useConversations(userId: string) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [historyState, setHistoryState] = useState<"loading" | "ready" | "error">("loading");
  const request = useRef(0);

  const refreshConversations = useCallback(async () => {
    const version = ++request.current;
    try {
      const rows = await fetchConversationsBrowser(userId);
      if (request.current !== version) return;
      setConversations(rows);
      setHistoryState("ready");
    } catch (error) {
      if (request.current !== version) return;
      console.error("Could not load chat history", error);
      setHistoryState("error");
    }
  }, [userId]);

  const createConversation = useCallback(async () => {
    try {
      const row = await createConversationBrowser(userId);
      setConversations((previous) => [row, ...previous.filter(({ id }) => id !== row.id)]);
      void refreshConversations();
      return row;
    } catch {
      toast.error("Could not create a chat.");
      return null;
    }
  }, [userId, refreshConversations]);

  const renameConversation = useCallback(async (id: string, title: string) => {
    try {
      await renameConversationBrowser(id, title);
      setConversations((previous) => previous.map((row) => row.id === id ? { ...row, title } : row));
      void refreshConversations();
      return true;
    } catch {
      toast.error("Could not rename the chat.");
      return false;
    }
  }, [refreshConversations]);

  const deleteConversation = useCallback(async (id: string) => {
    try {
      await deleteConversationBrowser(id);
      setConversations((previous) => previous.filter((row) => row.id !== id));
      void refreshConversations();
      return true;
    } catch {
      toast.error("Could not delete the chat.");
      return false;
    }
  }, [refreshConversations]);

  return { conversations, historyState, refreshConversations, createConversation, renameConversation, deleteConversation };
}
