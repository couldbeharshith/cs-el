"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Info, LayoutDashboard, LoaderCircle, Plus, ShieldCheck } from "lucide-react";
import { useConversationsContext } from "../../hooks/conversations-context";
import { SidebarHistory } from "./sidebar-history";
import { Button } from "@/components/ui/button";
import { Sidebar, SidebarContent, SidebarFooter, SidebarHeader, useSidebar } from "@/components/ui/sidebar";

export function AppSidebar() {
  const router = useRouter();
  const { setOpenMobile } = useSidebar();
  const { createConversation } = useConversationsContext();
  const [creating, setCreating] = useState(false);

  async function startChat() {
    setCreating(true);
    try {
      const conversation = await createConversation();
      if (!conversation) return;
      setOpenMobile(false);
      router.push(`/chat/${conversation.id}`);
    } finally {
      setCreating(false);
    }
  }

  return (
    <Sidebar>
      <SidebarHeader className="gap-5 px-4 pb-4 pt-5">
        <Link href="/dashboard" className="flex items-center gap-3 rounded-lg" onClick={() => setOpenMobile(false)}>
          <span className="grid size-9 place-items-center rounded-lg bg-primary text-primary-foreground"><ShieldCheck className="size-5" /></span>
          <span className="font-semibold tracking-tight">{process.env.NEXT_PUBLIC_APP_NAME || "FinSync AI"}</span>
        </Link>
        <Button type="button" variant="outline" className="justify-start bg-background" disabled={creating} onClick={startChat}>
          {creating ? <LoaderCircle className="animate-spin" /> : <Plus />} New chat
        </Button>
      </SidebarHeader>
      <SidebarContent><SidebarHistory /></SidebarContent>
      <SidebarFooter className="gap-1 border-t p-3">
        <Button variant="ghost" className="justify-start" asChild>
          <Link href="/dashboard" onClick={() => setOpenMobile(false)}><LayoutDashboard /> Market dashboard</Link>
        </Button>
        <Button variant="ghost" className="justify-start" asChild>
          <Link href="/about" onClick={() => setOpenMobile(false)}><Info /> About FinSync AI</Link>
        </Button>
      </SidebarFooter>
    </Sidebar>
  );
}
