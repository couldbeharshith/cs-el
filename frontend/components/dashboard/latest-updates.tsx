"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

const newsUpdates = [
  {
    id: 1,
    title: "St. Louis Fed US Q3 GDP Nowcast Estimate 0.468% Gain vs Previous 0.5% Gain",
    source: "MT Newswires",
    time: "1 hour ago",
    category: "Macro",
    sentiment: "neutral",
  },
  {
    id: 2,
    title: "If You Invested $100 In Advanced Drainage Systems Stock 5 Years Ago, You Would Have This Much Today",
    source: "Benzinga",
    time: "1 hour ago",
    category: "Analysis",
    sentiment: "positive",
  },
  {
    id: 3,
    title: "A 62-Year-Old Dave Ramsey Caller Has No Retirement Savings But Wants To Buy A First Home.",
    source: "Benzinga",
    time: "1 hour ago",
    category: "Personal Finance",
    sentiment: "warning",
  },
  {
    id: 4,
    title: "Here's How Much You Would Have Made Owning Lockheed Martin Stock In The Last 20 Years",
    source: "Benzinga",
    time: "1 hour ago",
    category: "Analysis",
    sentiment: "positive",
  },
  {
    id: 5,
    title: "$100 Invested In AbbVie 10 Years Ago Would Be Worth This Much Today",
    source: "Benzinga",
    time: "1 hour ago",
    category: "Analysis",
    sentiment: "positive",
  },
  {
    id: 6,
    title: "$100 Invested In ITT 5 Years Ago Would Be Worth This Much Today",
    source: "Benzinga",
    time: "1 hour ago",
    category: "Analysis",
    sentiment: "positive",
  },
]

export function LatestUpdates() {
  return (
    <div className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight flex items-center">
            Latest updates 
            <span className="ml-2 text-muted-foreground text-sm font-normal">›</span>
        </h2>
        <div className="grid gap-4 md:grid-cols-2">
        {newsUpdates.map((item) => (
            <div 
                key={item.id} 
                className="group flex flex-col justify-between gap-3 rounded-lg border border-transparent hover:border-border/50 bg-card/20 hover:bg-card/40 p-4 transition-all cursor-pointer"
            >
                <div className="space-y-2">
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span className="font-medium text-foreground/80 flex items-center gap-1">
                            <span className="h-1.5 w-1.5 rounded-full bg-blue-500"></span>
                            {item.source}
                        </span>
                        <span>·</span>
                        <span>{item.time}</span>
                    </div>
                    <h3 className="font-medium leading-snug group-hover:text-primary transition-colors line-clamp-3">
                        {item.title}
                    </h3>
                </div>
                {/* Optional footer for tags if needed */}
            </div>
        ))}
        </div>
    </div>
  )
}
