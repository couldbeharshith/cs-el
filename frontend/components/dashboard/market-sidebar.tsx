"use client"

import { Search, Plus, Settings2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"

const watchlist = [
  { name: "HDFC Bank Limited", symbol: "HDFCBANK", exchange: "NSE", price: "1,007.6", change: "-0.19%", trend: "down" },
  { name: "Reliance Industries", symbol: "RELIANCE", exchange: "BSE", price: "1,566.85", change: "+0.21%", trend: "up" },
  { name: "Infosys Limited", symbol: "INFY", exchange: "NSE", price: "1,560.1", change: "-0.40%", trend: "down" },
  { name: "ICICI Lombard Gener...", symbol: "ICICIGI", exchange: "NSE", price: "1,970.5", change: "-0.51%", trend: "down" },
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

export function MarketSidebar() {
  return (
    <div className="space-y-8">
      {/* Watchlist Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">Create Watchlist</h3>
          <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground">
            <Settings2 className="h-4 w-4" />
          </Button>
        </div>
        
        <div className="bg-card/30 border border-border/40 rounded-xl backdrop-blur-sm overflow-hidden">
            <div className="space-y-1">
                {watchlist.map((stock) => (
                    <div key={stock.symbol} className="flex items-center justify-between p-3 hover:bg-accent/50 transition-colors group cursor-pointer border-b border-border/20 last:border-0">
                        <div className="flex items-center gap-3">
                             <div className={`h-8 w-8 rounded-md flex items-center justify-center text-xs font-bold text-white ${
                                 stock.name.includes("HDFC") ? "bg-blue-700" : 
                                 stock.name.includes("Reliance") ? "bg-blue-500" : 
                                 stock.name.includes("Infosys") ? "bg-blue-400" : "bg-orange-600"
                             }`}>
                                 {stock.symbol.slice(0,1)}
                             </div>
                             <div>
                                 <div className="font-medium text-sm line-clamp-1">{stock.name}</div>
                                 <div className="text-xs text-muted-foreground">{stock.symbol} · {stock.exchange}</div>
                             </div>
                        </div>
                         <div className="text-right">
                             <div className="font-medium text-sm">₹{stock.price}</div>
                             <div className={`text-xs ${stock.trend === "up" ? "text-emerald-400" : "text-rose-400"}`}>
                                 {stock.change}
                             </div>
                         </div>
                         <Button size="icon" variant="ghost" className="h-6 w-6 opacity-0 group-hover:opacity-100 absolute right-2">
                            <Plus className="h-3 w-3" />
                         </Button>
                    </div>
                ))}
            </div>
        </div>
      </div>

      {/* Movers Section */}
      <div className="space-y-4">
        <Tabs defaultValue="gainers" className="w-full">
            <TabsList className="bg-transparent p-0 h-auto space-x-4 justify-start">
                <TabsTrigger 
                    value="gainers" 
                    className="data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 pb-1 text-muted-foreground hover:text-foreground transition-colors"
                >
                    Gainers
                </TabsTrigger>
                <TabsTrigger 
                    value="losers" 
                    className="data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 pb-1 text-muted-foreground hover:text-foreground transition-colors"
                >
                    Losers
                </TabsTrigger>
                <TabsTrigger 
                    value="active" 
                    className="data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 pb-1 text-muted-foreground hover:text-foreground transition-colors"
                >
                    Active
                </TabsTrigger>
            </TabsList>
            
            {/* Shared rendering logic for stock items to keep it clean */}
            {[
                { value: "gainers", data: gainers },
                { value: "losers", data: losers },
                { value: "active", data: active }
            ].map(({ value, data }) => (
                <TabsContent key={value} value={value} className="mt-4">
                    <div className="bg-card/30 border border-border/40 rounded-xl backdrop-blur-sm overflow-hidden">
                        {data.map((stock, i) => (
                            <div key={stock.symbol} className="flex items-center justify-between p-3 hover:bg-accent/50 transition-colors border-b border-border/20 last:border-0 cursor-pointer">
                                <div className="flex items-center gap-3">
                                    <div className={`h-8 w-8 rounded-md flex items-center justify-center text-xs font-bold text-white ${
                                        i % 4 === 0 ? "bg-blue-600" : i % 4 === 1 ? "bg-gray-600" : i % 4 === 2 ? "bg-blue-800" : "bg-orange-500"
                                    }`}>
                                        {stock.symbol.substring(0,2)}
                                    </div>
                                    <div>
                                        <div className="font-medium text-sm line-clamp-1">{stock.name}</div>
                                        <div className="text-xs text-muted-foreground">{stock.symbol} · {stock.exchange}</div>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className="font-medium text-sm">₹{stock.price}</div>
                                    <div className={`text-xs ${stock.trend === "up" ? "text-emerald-400" : "text-rose-400"}`}>
                                        {stock.change}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </TabsContent>
            ))}
        </Tabs>
      </div>
    </div>
  )
}
