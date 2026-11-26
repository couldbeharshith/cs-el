"use client";

import Link from "next/link";
import { LayoutDashboard, PanelLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";
import { ModeToggle } from "@/components/global/theme-switcher";

export function ChatHeader({ title }: { title: string }) {
  const { toggleSidebar } = useSidebar();

  return (
    <header className="flex h-16 shrink-0 items-center gap-2 border-b px-3 sm:gap-3 sm:px-6">
      <Button type="button" variant="ghost" size="icon" onClick={toggleSidebar} aria-label="Toggle chat history">
        <PanelLeft />
      </Button>
      <span className="min-w-0 flex-1 truncate text-sm font-semibold" title={title}>{title}</span>
      <Button variant="ghost" asChild>
        <Link href="/dashboard"><LayoutDashboard /><span className="hidden sm:inline">Dashboard</span><span className="sr-only sm:hidden">Dashboard</span></Link>
      </Button>
      <ModeToggle />
    </header>
  );
}
