"use client";

import Link from "next/link";
import { CircleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function ChatError({ reset }: { reset: () => void }) {
  return (
    <main className="grid h-full min-h-64 place-items-center p-6">
      <div className="max-w-md space-y-4 text-center">
        <CircleAlert className="mx-auto size-8 text-primary" aria-hidden />
        <h1 className="text-2xl font-semibold">Chat is unavailable</h1>
        <p className="text-sm text-muted-foreground">
          We couldn’t open your workspace. Check the Supabase connection and try again.
        </p>
        <div className="flex justify-center gap-2">
          <Button onClick={reset}>Try again</Button>
          <Button variant="outline" asChild><Link href="/dashboard">Market dashboard</Link></Button>
        </div>
      </div>
    </main>
  );
}
