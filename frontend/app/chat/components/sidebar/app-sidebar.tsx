"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import { PlusIcon } from "../icons";
import { useConversationsContext } from "../../hooks/conversations-context";
import { SidebarHistory } from "./sidebar-history";
import Image from "next/image"
import { Button } from "@/components/ui/button";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface AppSidebarProps {
  user: any;
}

export function AppSidebar({ user }: AppSidebarProps) {
  const router = useRouter();
  const { setOpenMobile } = useSidebar();
  const { createConversation } = useConversationsContext();
  const appName = process.env.NEXT_PUBLIC_APP_NAME!;
  const appIcon = process.env.NEXT_PUBLIC_APP_ICON!;
  
  // Demo mode - use demo user ID
  const demoUserId = "demo-user";
  
  return (
    <Sidebar className="group-data-[side=left]:border-r-0">
      <SidebarHeader>
        <SidebarMenu>
          <div className="flex flex-row justify-between items-center px-4 py-2">
            <Link
              href="/"
              onClick={() => {
                setOpenMobile(false);
              }}
              className="flex flex-row gap-3 items-center"
            >
              <Image
                src={appIcon}
                alt={appName}
                width={40}
                height={40}
                className="h-10 w-10 rounded-md"
                />
              <span className="text-lg font-semibold px-2 hover:bg-muted rounded-md cursor-pointer">
                {appName}
              </span>
            </Link>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  type="button"
                  className="p-2 h-fit"
                  onClick={async () => {
                    setOpenMobile(false);
                    const newConv = await createConversation(demoUserId, "New Chat");
                    if (newConv) {
                      router.push(`/chat/${newConv.id}`);
                    }
                  }}
                >
                  <PlusIcon />
                </Button>
              </TooltipTrigger>
              <TooltipContent align="end">New Chat</TooltipContent>
            </Tooltip>
          </div>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarHistory user={user} />
      </SidebarContent>

      <SidebarFooter>
        {/* Auth removed - demo mode only */}
      </SidebarFooter>
    </Sidebar>
  );
}
