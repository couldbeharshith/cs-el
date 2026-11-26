import { Header } from "@/components/global/header";
import Footer from "@/components/global/footer";
import { headerConfig } from "@/lib/config/header";

export default function MarketingLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <Header config={headerConfig} />
      {children}
      <Footer />
    </>
  );
}
