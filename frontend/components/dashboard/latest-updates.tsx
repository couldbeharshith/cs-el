const newsUpdates = [
  {
    id: 1,
    title: "St. Louis Fed US Q3 GDP Nowcast Estimate 0.468% Gain vs Previous 0.5% Gain",
    source: "MT Newswires",
    category: "Macro",
  },
  {
    id: 2,
    title: "If You Invested $100 In Advanced Drainage Systems Stock 5 Years Ago, You Would Have This Much Today",
    source: "Benzinga",
    category: "Analysis",
  },
  {
    id: 3,
    title: "A 62-Year-Old Dave Ramsey Caller Has No Retirement Savings But Wants To Buy A First Home.",
    source: "Benzinga",
    category: "Personal Finance",
  },
  {
    id: 4,
    title: "Here's How Much You Would Have Made Owning Lockheed Martin Stock In The Last 20 Years",
    source: "Benzinga",
    category: "Analysis",
  },
  {
    id: 5,
    title: "$100 Invested In AbbVie 10 Years Ago Would Be Worth This Much Today",
    source: "Benzinga",
    category: "Analysis",
  },
  {
    id: 6,
    title: "$100 Invested In ITT 5 Years Ago Would Be Worth This Much Today",
    source: "Benzinga",
    category: "Analysis",
  },
]

export function LatestUpdates() {
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold tracking-tight">News from the snapshot</h2>
      <div className="grid gap-3 sm:grid-cols-2">
        {newsUpdates.map(({ id, title, source, category }) => (
          <article key={id} className="space-y-2 rounded-xl border bg-card p-4">
            <p className="text-xs font-medium text-muted-foreground">{source} · {category}</p>
            <h3 className="text-sm font-medium leading-relaxed">{title}</h3>
          </article>
        ))}
      </div>
    </section>
  );
}
