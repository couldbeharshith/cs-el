"use client"

import { TrendingUp, TrendingDown, Activity } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"

const mostActiveStocks = [
  { name: "HDFC Bank", price: "1,456.00", change: "+1.2%", volume: "12.5M" },
  { name: "Tata Motors", price: "987.45", change: "+2.5%", volume: "10.2M" },
  { name: "Reliance", price: "2,980.10", change: "-0.5%", volume: "8.9M" },
  { name: "SBI", price: "765.30", change: "+0.8%", volume: "7.5M" },
  { name: "Infosys", price: "1,543.20", change: "-1.1%", volume: "6.2M" },
]

const sectorPerformance = [
  { name: "Energy", value: 75, trend: "up", change: "+2.4%" },
  { name: "Technology", value: 45, trend: "down", change: "-1.2%" },
  { name: "Banking", value: 60, trend: "up", change: "+0.8%" },
  { name: "Auto", value: 85, trend: "up", change: "+3.1%" },
  { name: "Pharma", value: 30, trend: "down", change: "-0.5%" },
]

export function MarketSummary() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
      <Card className="col-span-4">
        <CardHeader>
          <CardTitle>Most Active Stocks</CardTitle>
          <CardDescription>
            Top traded companies by volume in the Indian market today.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {mostActiveStocks.map((stock, index) => (
              <div
                key={stock.name}
                className="flex items-center justify-between border-b pb-4 last:border-0 last:pb-0"
              >
                <div className="flex items-center gap-4">
                  <div className="flex h-9 w-9 items-center justify-center rounded-full bg-muted/50 font-medium">
                    {index + 1}
                  </div>
                  <div>
                    <p className="text-sm font-medium leading-none">{stock.name}</p>
                    <p className="text-xs text-muted-foreground mt-1">Vol: {stock.volume}</p>
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <span className="font-medium">{stock.price}</span>
                  <Badge
                    variant={stock.change.startsWith("+") ? "default" : "destructive"}
                    className={`text-xs font-normal ${
                      stock.change.startsWith("+") 
                        ? "bg-emerald-500/15 text-emerald-500 hover:bg-emerald-500/25 shadow-none" 
                        : "bg-rose-500/15 text-rose-500 hover:bg-rose-500/25 shadow-none"
                    }`}
                  >
                    {stock.change}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
      <Card className="col-span-3">
        <CardHeader>
          <CardTitle>Sector Performance</CardTitle>
          <CardDescription>
            Daily sector-wise performance tracking.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {sectorPerformance.map((sector) => (
              <div key={sector.name} className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">{sector.name}</span>
                  <span
                    className={
                      sector.trend === "up" ? "text-emerald-500" : "text-rose-500"
                    }
                  >
                    {sector.change}
                  </span>
                </div>
                <Progress
                  value={sector.value}
                  className={`h-2 ${
                    sector.trend === "up" ? "[&>div]:bg-emerald-500" : "[&>div]:bg-rose-500"
                  }`}
                />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

