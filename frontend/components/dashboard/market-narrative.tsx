"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function MarketNarrative() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold tracking-tight">Market Summary</h2>
        <span className="text-xs text-muted-foreground">Updated 47 minutes ago</span>
      </div>

      <Card className="bg-card/30 border-border/40 backdrop-blur-sm shadow-sm">
        <CardContent className="p-6 space-y-6">
          <div className="space-y-2">
            <h3 className="font-semibold text-foreground text-base">
              Headline indices end flat amid cautious global cues
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              The Sensex slipped 13.7 points (-0.02%) to 85,706.7 and Nifty 50 edged down 0.05% to 26,203, with trade largely directionless as US markets were shut for Thanksgiving and Asia traded mixed on China property worries and Japan rate-hike speculation.
            </p>
          </div>

          <div className="space-y-2">
            <h3 className="font-semibold text-foreground text-base">
              Banks provide relative support; HDFC Bank, ICICI Bank recent laggards
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Nifty Bank outperformed slightly, up 0.03% at 59,753, as financials remain a preferred 2026 theme, but sentiment is capped by recent profit-taking in heavyweights like HDFC Bank and ICICI Bank after record highs earlier in the week.
            </p>
          </div>

          <div className="space-y-2">
            <h3 className="font-semibold text-foreground text-base">
              Mid- and small-caps soften as market breadth turns negative
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              The BSE Mid-Cap and Small-Cap indices slipped 0.04% and 0.13% respectively, with NSE data showing marginally more decliners than gainers, signalling selective profit-booking after a strong November in broader markets.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

