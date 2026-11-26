"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldCheck } from "lucide-react";
import type { HeaderConfig } from "@/lib/config/header";
import { cn } from "@/lib/utils";
import { NavigationMobile } from "./header-mobile";
import { ModeToggle } from "./theme-switcher";

export function Header({ config }: { config: HeaderConfig }) {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b bg-background/95 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-[1450px] items-center justify-between gap-3 px-4 sm:px-6">
        <Link href="/" prefetch={false} className="flex items-center gap-2 rounded-lg font-semibold tracking-tight">
          {config.brand.icon === "shield" && (
            <span className="grid size-8 place-items-center rounded-lg bg-primary text-primary-foreground"><ShieldCheck className="size-5" /></span>
          )}
          {config.brand.title}
        </Link>
        <nav aria-label="Main navigation" className="hidden items-center gap-1 md:flex">
          {config.navigationLinks.map(({ href, label }) => {
            const active = pathname === href || (href !== "/" && pathname.startsWith(`${href}/`));
            return <Link key={href} href={href} prefetch={href !== "/chat" && href !== "/"}
              aria-current={active ? "page" : undefined}
              className={cn("rounded-lg px-3 py-1.5 text-sm transition-colors hover:bg-accent", active ? "bg-primary/10 font-medium text-primary" : "text-muted-foreground")}>
              {label}
            </Link>;
          })}
        </nav>
        <div className="flex items-center gap-2">
          <NavigationMobile navigationLinks={config.navigationLinks} />
          <ModeToggle />
        </div>
      </div>
    </header>
  );
}
