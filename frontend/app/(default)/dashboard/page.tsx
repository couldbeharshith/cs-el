import { MarketOverviewCharts } from "@/components/dashboard/market-overview-charts"
import { MarketNarrative } from "@/components/dashboard/market-narrative"
import { MarketSidebar } from "@/components/dashboard/market-sidebar"
import { LatestUpdates } from "@/components/dashboard/latest-updates"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { Search, Sparkles } from "lucide-react"

export default async function DashboardPage() {
  return (
    <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/20">
      <div className="container mx-auto p-6 max-w-[1600px] space-y-8">
        
        {/* Header Section */}
        <header className="flex items-center justify-between pb-2 border-b border-border/40">
          <div className="flex items-center gap-4">
             {/* Could be a dropdown for market selection */}
             <Button variant="outline" className="h-8 rounded-full border-emerald-500/50 text-emerald-500 hover:bg-emerald-500/10 hover:text-emerald-400 bg-emerald-500/5 text-xs font-medium">
                <span className="mr-2 text-lg">🇮🇳</span> India Markets
             </Button>
             <div className="flex gap-2">
                 {["Crypto", "Earnings", "Screener", "Watchlist"].map((tab) => (
                     <Button key={tab} variant="ghost" className="h-8 rounded-full text-muted-foreground hover:text-foreground hover:bg-accent/50 text-xs">
                         {tab}
                     </Button>
                 ))}
             </div>
          </div>
          <div className="text-xs text-right">
              <div className="text-muted-foreground">29 Nov 2025, IST · Market Closed</div>
              <div className="font-medium text-emerald-500 tracking-widest text-[10px] uppercase mt-0.5">
                  |||||||||| Upbeat Sentiment
              </div>
          </div>
        </header>

        {/* Top Charts */}
        <section>
          <MarketOverviewCharts />
        </section>
        
        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Content Column */}
          <div className="lg:col-span-8 space-y-10">
             <MarketNarrative />
             <LatestUpdates />
             
             {/* Ask Anything Search Bar */}
             <div className="pt-4">
                 <h3 className="text-lg font-medium mb-4 text-muted-foreground">Ask anything about Indian markets</h3>
                 <Link href="/chat" className="block group">
                    <div className="relative flex items-center w-full h-14 rounded-full bg-muted/30 border border-border/50 hover:border-primary/50 hover:bg-muted/50 transition-all px-4 shadow-lg hover:shadow-primary/5">
                        <Search className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors ml-2" />
                        <div className="ml-4 text-muted-foreground/70 text-sm group-hover:text-foreground/80">
                            Try "Why did HDFC Bank fall today?" or "Top EV stocks in India"
                        </div>
                        <div className="absolute right-2 top-2 bottom-2 aspect-square rounded-full bg-primary/10 flex items-center justify-center group-hover:bg-primary group-hover:text-primary-foreground transition-all">
                            <Sparkles className="h-4 w-4" />
                        </div>
                    </div>
                 </Link>
                 <div className="flex gap-3 mt-4 ml-2">
                     <Button variant="outline" size="sm" className="h-8 rounded-lg text-xs bg-transparent border-border/40 hover:bg-accent/50" asChild>
                        <Link href="/chat?q=Tata+Motors+Analysis">
                           <Sparkles className="mr-2 h-3 w-3" />
                           Tata Motors Analysis
                        </Link>
                     </Button>
                     <Button variant="outline" size="sm" className="h-8 rounded-lg text-xs bg-transparent border-border/40 hover:bg-accent/50" asChild>
                        <Link href="/chat?q=Nifty+50+Outlook">
                            <Sparkles className="mr-2 h-3 w-3" />
                            Nifty 50 Outlook
                        </Link>
                     </Button>
                 </div>
             </div>
          </div>

          {/* Right Sidebar Column */}
          <div className="lg:col-span-4">
             <MarketSidebar />
             
             {/* Disclaimer */}
             <div className="mt-8 p-4 rounded-xl bg-muted/20 border border-border/30 text-[10px] text-muted-foreground leading-relaxed">
                Financial information provided by Financial Modeling Prep. Options data provided by Unusual Whales. Earnings transcripts, audio, and documents provided by Quartr. Reported revenue and EPS data from Earnings powered by Fiscal.ai. Estimates provided by S&P. All data is provided for informational purposes only, and is not intended for trading purposes or financial, investment, tax, legal, accounting or other advice.
             </div>
          </div>
        </div>
      </div>
    </div>
  )
}
