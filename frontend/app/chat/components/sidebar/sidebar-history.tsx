"use client";

import { isToday, isYesterday, subMonths, subWeeks } from "date-fns";
import { MessageSquare, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { SidebarGroup, SidebarGroupLabel, SidebarMenu, SidebarMenuAction, SidebarMenuButton, SidebarMenuItem, useSidebar } from "@/components/ui/sidebar";
import { useConversationsContext } from "../../hooks/conversations-context";
import type { Conversation } from "../../types";

export function SidebarHistory() {
  const { id } = useParams();
  const router = useRouter();
  const { setOpenMobile } = useSidebar();
  const { conversations, historyState, refreshConversations, renameConversation, deleteConversation } = useConversationsContext();
  const [dialog, setDialog] = useState<{ mode: "rename" | "delete"; chat: Conversation }>();
  const [title, setTitle] = useState("");
  const [pending, setPending] = useState(false);

  const now = new Date();
  const groups = { Today: [], Yesterday: [], "Last week": [], "Last month": [], Older: [] } as Record<string, Conversation[]>;
  for (const chat of conversations) {
    const date = new Date(chat.updated_at);
    const group = isToday(date) ? "Today" : isYesterday(date) ? "Yesterday" :
      date > subWeeks(now, 1) ? "Last week" : date > subMonths(now, 1) ? "Last month" : "Older";
    groups[group].push(chat);
  }

  async function confirm() {
    if (!dialog || pending) return;
    setPending(true);
    try {
      if (dialog.mode === "delete") {
        if (!(await deleteConversation(dialog.chat.id))) return;
        if (id === dialog.chat.id) {
          const next = conversations.find((chat) => chat.id !== id);
          router.replace(next ? `/chat/${next.id}` : "/chat");
        }
      } else if (!(await renameConversation(dialog.chat.id, title.trim()))) return;
      setDialog(undefined);
    } finally {
      setPending(false);
    }
  }

  return (
    <SidebarGroup className="px-3">
      <SidebarGroupLabel>Chat history</SidebarGroupLabel>
      {historyState === "loading" && <p className="px-2 text-xs text-muted-foreground">Loading history…</p>}
      {historyState === "error" && (
        <div className="space-y-2 px-2 py-1 text-xs text-muted-foreground">
          <p>History is unavailable.</p>
          <Button type="button" size="sm" variant="outline" onClick={() => void refreshConversations()}>Retry</Button>
        </div>
      )}
      {historyState === "ready" && !conversations.length && <p className="px-2 text-xs text-muted-foreground">Recent chats will appear here.</p>}
      {Object.entries(groups).map(([label, chats]) => chats.length > 0 && (
        <div key={label} className="mb-3">
          <p className="px-2 pb-1 pt-2 text-xs font-medium text-muted-foreground">{label}</p>
          <SidebarMenu>
            {chats.map((chat) => (
              <SidebarMenuItem key={chat.id}>
                <SidebarMenuButton asChild isActive={chat.id === id} tooltip={chat.title}>
                  <Link href={`/chat/${chat.id}`} prefetch={false} onClick={() => setOpenMobile(false)} aria-current={chat.id === id ? "page" : undefined}>
                    <MessageSquare /><span>{chat.title}</span>
                  </Link>
                </SidebarMenuButton>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <SidebarMenuAction aria-label={`Options for ${chat.title}`}><MoreHorizontal /></SidebarMenuAction>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent side="bottom" align="end">
                    <DropdownMenuItem onSelect={() => { setTitle(chat.title); setDialog({ mode: "rename", chat }); }}>
                      <Pencil /> Rename
                    </DropdownMenuItem>
                    <DropdownMenuItem variant="destructive" onSelect={() => setDialog({ mode: "delete", chat })}>
                      <Trash2 /> Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </div>
      ))}
      <Dialog open={Boolean(dialog)} onOpenChange={(open) => { if (!open && !pending) setDialog(undefined); }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{dialog?.mode === "delete" ? "Delete chat?" : "Rename chat"}</DialogTitle>
            <DialogDescription>{dialog?.mode === "delete" ? `“${dialog.chat.title}” and its messages will be deleted.` : "Choose a title for this conversation."}</DialogDescription>
          </DialogHeader>
          <form className="grid gap-4" onSubmit={(event) => { event.preventDefault(); void confirm(); }}>
            {dialog?.mode === "rename" && <Input aria-label="Chat title" value={title} maxLength={120} onChange={(event) => setTitle(event.target.value)} required />}
            <div className="flex justify-end gap-2">
              <Button type="button" variant="outline" disabled={pending} onClick={() => setDialog(undefined)}>Cancel</Button>
              <Button type="submit" variant={dialog?.mode === "delete" ? "destructive" : "default"} disabled={pending || (dialog?.mode === "rename" && !title.trim())}>
                {dialog?.mode === "delete" ? "Delete chat" : "Save title"}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </SidebarGroup>
  );
}
