import { AppSidebar } from "@/components/global/app-sidebar";
import { Header } from "@/components/global/header";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { headerConfig } from "@/lib/config/header";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <Header config={headerConfig} />
      <SidebarProvider className="min-h-0">
        <AppSidebar />
        <SidebarInset>
          <main className="mx-auto w-full max-w-[1450px] px-4 pb-10 pt-4 sm:px-6">{children}</main>
        </SidebarInset>
      </SidebarProvider>
    </>
  );
}
