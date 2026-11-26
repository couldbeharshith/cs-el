import { SidebarProvider } from "@/components/ui/sidebar";
import { ConversationsProvider } from "./hooks/conversations-context";
import { AppSidebar } from "./components/sidebar/app-sidebar";

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  return (
    <ConversationsProvider userId="demo-user">
      <SidebarProvider className="h-dvh overflow-hidden">
        <AppSidebar />
        <div className="h-full min-w-0 flex-1">{children}</div>
      </SidebarProvider>
    </ConversationsProvider>
  );
}
