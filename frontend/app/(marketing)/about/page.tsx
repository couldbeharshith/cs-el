import Link from "next/link";
import { ArrowUpRight, ChartNoAxesCombined, FileSearch, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function AboutPage() {
  return (
    <main className="mx-auto w-full max-w-5xl px-5 py-14 sm:px-8 sm:py-24">
      <ShieldCheck className="mb-5 size-8 text-primary" />
      <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">About FinSync AI</h1>
      <p className="mt-4 max-w-xl text-lg leading-relaxed text-muted-foreground">
        A workspace for financial research, company screening, and insurance document analysis.
      </p>
      <div className="mt-6 flex flex-wrap gap-2">
        <Button asChild><Link href="/chat" prefetch={false}>Open chat <ArrowUpRight /></Link></Button>
        <Button variant="outline" asChild><Link href="/dashboard">Market dashboard</Link></Button>
      </div>
      <div className="mt-10 grid gap-4 sm:grid-cols-2">
        <section className="rounded-xl border bg-card p-5">
          <ChartNoAxesCombined className="mb-3 size-5 text-primary" />
          <h2 className="font-semibold">Market and company research</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">Review the sample market dashboard and use chat to screen companies, examine fundamentals, and request charts.</p>
        </section>
        <section className="rounded-xl border bg-card p-5">
          <FileSearch className="mb-3 size-5 text-primary" />
          <h2 className="font-semibold">Insurance document analysis</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">Supply a document link, ask about coverage and exclusions, or use one of the policy prompts to get started.</p>
        </section>
      </div>
    </main>
  );
}
