import Link from "next/link";
import { ArrowUpRight, ChartNoAxesCombined, MessageSquare } from "lucide-react";
import { MarketOverviewCharts } from "@/components/dashboard/market-overview-charts";
import { MarketNarrative } from "@/components/dashboard/market-narrative";
import { MarketSidebar } from "@/components/dashboard/market-sidebar";
import { LatestUpdates } from "@/components/dashboard/latest-updates";
import { Button } from "@/components/ui/button";

const researchLinks = [
  { label: "Crypto", query: "Summarize the latest developments in the cryptocurrency market." },
  { label: "Earnings", query: "What upcoming earnings should I watch in India?" },
  { label: "Screener", query: "Help me design a screen for Indian companies with strong revenue growth." },
  { label: "Tata Motors Analysis", query: "Tata Motors Analysis" },
  { label: "Nifty 50 Outlook", query: "Nifty 50 Outlook" },
];

export default function DashboardPage() {
  return (
    <div className="mx-auto max-w-[1400px] space-y-8">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b pb-5">
        <div>
          <div className="mb-2 flex items-center gap-2 text-sm font-medium text-primary"><ChartNoAxesCombined className="size-4" /> India markets</div>
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Market overview</h1>
          <p className="mt-2 text-sm text-muted-foreground">Sample snapshot · 29 Nov 2025 (IST) · Market closed</p>
        </div>
        <Button asChild><Link href="/chat" prefetch={false}><MessageSquare /> Open chat <ArrowUpRight /></Link></Button>
      </header>
      <section aria-label="Market indices"><MarketOverviewCharts /></section>
      <div className="grid min-w-0 gap-7 xl:grid-cols-[minmax(0,2fr)_minmax(18rem,1fr)]">
        <div className="min-w-0 space-y-7">
          <section className="rounded-2xl border bg-card p-4 sm:p-6">
            <h2 className="text-lg font-semibold">Research the market</h2>
            <p className="mt-1 text-sm text-muted-foreground">Choose a question to start a new chat, or jump to the sample watchlist.</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {researchLinks.map(({ label, query }) => (
                <Button key={label} size="sm" variant="outline" asChild>
                  <Link href={`/chat?${new URLSearchParams({ q: query })}`} prefetch={false}>{label} <ArrowUpRight /></Link>
                </Button>
              ))}
              <Button size="sm" variant="ghost" asChild><a href="#watchlist">Watchlist</a></Button>
            </div>
          </section>
          <MarketNarrative />
          <LatestUpdates />
        </div>
        <MarketSidebar />
      </div>
    </div>
  );
}
