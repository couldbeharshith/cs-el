import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { ConversationsProvider } from "./hooks/conversations-context";
import { AppSidebar } from "./components/sidebar/app-sidebar";

export default async function ChatLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Auth removed - demo mode only
  const demoUserId = "demo-user";
  
  return (
    <div className="flex h-screen">
      <ConversationsProvider userId={demoUserId}>
        <SidebarProvider defaultOpen={true}>
          <AppSidebar user={null} />
          <div className="flex-1 ">{children}</div>
        </SidebarProvider>
      </ConversationsProvider>
    </div>
  );
}
