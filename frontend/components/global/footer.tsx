import Link from "next/link";
import { footerLinks } from "@/lib/config/footer";

export default function Footer() {
  return (
    <footer className="border-t">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-4 px-5 py-6 text-sm sm:px-8">
        <span className="font-semibold">FinSync AI</span>
        <nav aria-label="Footer navigation" className="flex flex-wrap items-center gap-4 text-muted-foreground">
          {footerLinks.map(({ href, label }) => (
            <Link key={href} href={href} prefetch={href !== "/chat"} className="hover:text-foreground">{label}</Link>
          ))}
        </nav>
      </div>
    </footer>
  );
}
