"use client"

import * as React from "react"
import { Area, AreaChart, ResponsiveContainer, YAxis } from "recharts"
import { ArrowUpIcon, ArrowDownIcon } from "lucide-react"

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

const generateChartData = (startValue: number, points: number, trend: "up" | "down") => {
  let currentValue = startValue
  const data = []
  for (let i = 0; i < points; i++) {
    // Increase volatility: using 0.4% daily volatility approximation for intraday
    const volatility = startValue * 0.001 
    const change = (Math.random() - 0.5) * volatility + (trend === "up" ? startValue * 0.0001 : -startValue * 0.0001)
    currentValue += change
    data.push({
      time: i,
      value: currentValue,
    })
  }
  return data
}

const marketData = [
  {
    name: "NIFTY 50",
    symbol: "^NSEI · INDEX",
    value: "26,202.95",
    change: "-12.6",
    changePercent: "0.05%",
    trend: "down" as const,
    data: generateChartData(26215, 50, "down"),
    color: "hsl(0, 84%, 60%)", // Red
  },
  {
    name: "S&P BSE Sensex",
    symbol: "^BSESN · INDEX",
    value: "85,706.67",
    change: "-13.71",
    changePercent: "0.02%",
    trend: "down" as const,
    data: generateChartData(85720, 50, "down"),
    color: "hsl(0, 84%, 60%)",
  },
  {
    name: "Nifty Bank Index",
    symbol: "^NSEBANK · INDEX",
    value: "59,752.7",
    change: "+15.4",
    changePercent: "0.03%",
    trend: "up" as const,
    data: generateChartData(59730, 50, "up"),
    color: "hsl(173, 58%, 39%)", // Teal/Green
  },
  {
    name: "Bitcoin",
    symbol: "BTCUSD · CRYPTO",
    value: "$90,959.30",
    change: "-346.26",
    changePercent: "0.38%",
    trend: "down" as const,
    data: generateChartData(91300, 50, "down"),
    color: "hsl(0, 84%, 60%)",
  },
]

const chartConfig = {
  value: {
    label: "Price",
  },
} satisfies ChartConfig

export function MarketOverviewCharts() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {marketData.map((market) => (
        <Card key={market.name} className="flex flex-col justify-between overflow-hidden bg-card/50 border-border/50 backdrop-blur-sm hover:bg-card/80 transition-colors">
          <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2 px-4 pt-4">
            <div className="space-y-1">
              <CardTitle className="text-base font-bold text-foreground">
                {market.name}
              </CardTitle>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                {market.symbol}
              </p>
            </div>
            <div
              className={`flex flex-col items-end text-xs font-medium ${
                market.trend === "up" ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              <div className="flex items-center bg-background/20 px-1.5 py-0.5 rounded text-xs">
                 {market.trend === "up" ? "↗" : "↘"} {market.changePercent}
              </div>
              <div className="mt-1">
                 {market.change}
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-0">
             <div className="px-4 pb-2">
                <span className="text-xl font-bold tracking-tight">{market.value}</span>
             </div>
            <div className="h-[60px] w-full">
              <ChartContainer config={chartConfig} className="h-full w-full">
                <AreaChart data={market.data} margin={{ top: 5, right: 0, bottom: 0, left: 0 }}>
                  <defs>
                    <linearGradient id={`fill-${market.name}`} x1="0" y1="0" x2="0" y2="1">
                      <stop
                        offset="0%"
                        stopColor={market.color}
                        stopOpacity={0.2}
                      />
                      <stop
                        offset="100%"
                        stopColor={market.color}
                        stopOpacity={0}
                      />
                    </linearGradient>
                  </defs>
                  <ChartTooltip
                    cursor={false}
                    content={<ChartTooltipContent hideLabel hideIndicator />}
                  />
                  <YAxis domain={['dataMin', 'dataMax']} hide />
                  <Area
                    dataKey="value"
                    type="monotone"
                    stroke={market.color}
                    fill={`url(#fill-${market.name})`}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ChartContainer>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
