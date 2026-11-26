"use client"

import Link from "next/link"
import { ArrowUpRight } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

const watchlist = [
  { name: "HDFC Bank Limited", symbol: "HDFCBANK", exchange: "NSE", price: "1,007.6", change: "-0.19%", trend: "down" },
  { name: "Reliance Industries", symbol: "RELIANCE", exchange: "BSE", price: "1,566.85", change: "+0.21%", trend: "up" },
  { name: "Infosys Limited", symbol: "INFY", exchange: "NSE", price: "1,560.1", change: "-0.40%", trend: "down" },
  { name: "ICICI Lombard General Insurance", symbol: "ICICIGI", exchange: "NSE", price: "1,970.5", change: "-0.51%", trend: "down" },
]

const gainers = [
  { name: "63 Moons Technologies", symbol: "63MOONS", exchange: "BSE", price: "907.05", change: "+15.98%", trend: "up" },
  { name: "Rico Auto Industries", symbol: "RICOAUTO", exchange: "BSE", price: "121.10", change: "+13.78%", trend: "up" },
  { name: "Welspun Living Limited", symbol: "WELSPUNLIV", exchange: "BSE", price: "148.70", change: "+12.35%", trend: "up" },
  { name: "Sigachi Industries", symbol: "SIGACHI", exchange: "BSE", price: "38.25", change: "+10.90%", trend: "up" },
]

const losers = [
  { name: "Paytm (One97)", symbol: "PAYTM", exchange: "NSE", price: "345.20", change: "-4.50%", trend: "down" },
  { name: "Zomato Ltd", symbol: "ZOMATO", exchange: "BSE", price: "156.45", change: "-3.20%", trend: "down" },
  { name: "Adani Enterprises", symbol: "ADANIENT", exchange: "NSE", price: "3,120.00", change: "-2.80%", trend: "down" },
  { name: "Tata Steel", symbol: "TATASTEEL", exchange: "BSE", price: "142.80", change: "-1.95%", trend: "down" },
]

const active = [
  { name: "Vodafone Idea", symbol: "IDEA", exchange: "NSE", price: "13.45", change: "+2.10%", trend: "up" },
  { name: "Yes Bank", symbol: "YESBANK", exchange: "NSE", price: "24.60", change: "+0.80%", trend: "up" },
  { name: "Tata Power", symbol: "TATAPOWER", exchange: "BSE", price: "398.10", change: "+1.20%", trend: "up" },
  { name: "Suzlon Energy", symbol: "SUZLON", exchange: "NSE", price: "41.25", change: "-0.50%", trend: "down" },
]

const moverGroups = { Gainers: gainers, Losers: losers, Active: active };

export function MarketSidebar() {
  return (
    <aside className="min-w-0 space-y-5">
      <section id="watchlist" className="scroll-mt-4 rounded-2xl border bg-card p-4">
        <h2 className="mb-2 font-semibold">Watchlist</h2>
        <StockList stocks={watchlist} />
      </section>
      <section className="rounded-2xl border bg-card p-4">
        <h2 className="mb-3 font-semibold">Market movers</h2>
        <Tabs defaultValue="Gainers">
          <TabsList aria-label="Market movers">
            {Object.keys(moverGroups).map((group) => <TabsTrigger key={group} value={group}>{group}</TabsTrigger>)}
          </TabsList>
          {Object.entries(moverGroups).map(([group, stocks]) => (
            <TabsContent key={group} value={group}><StockList stocks={stocks} /></TabsContent>
          ))}
        </Tabs>
      </section>
    </aside>
  );
}

function StockList({ stocks }: { stocks: typeof watchlist }) {
  return (
    <ul className="divide-y">
      {stocks.map((stock) => (
        <li key={stock.symbol}>
          <Link
            href={`/chat?${new URLSearchParams({ q: `Analyze ${stock.name} (${stock.symbol}).` })}`}
            prefetch={false}
            className="group grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded-md py-3 hover:bg-accent focus-visible:outline-2"
          >
            <span className="grid size-8 place-items-center rounded-lg bg-primary/10 text-xs font-semibold text-primary">{stock.symbol.slice(0, 2)}</span>
            <span className="min-w-0">
              <span className="block truncate text-sm font-medium">{stock.name}</span>
              <span className="block truncate text-xs text-muted-foreground">{stock.symbol} · {stock.exchange}</span>
            </span>
            <span className="text-right tabular-nums">
              <span className="flex items-center justify-end gap-1 text-sm font-medium">₹{stock.price}<ArrowUpRight className="size-3 text-muted-foreground" /></span>
              <span className={stock.trend === "up" ? "text-xs text-emerald-700 dark:text-emerald-400" : "text-xs text-rose-700 dark:text-rose-400"}>{stock.change}</span>
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}
