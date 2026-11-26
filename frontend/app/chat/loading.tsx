import { LoaderCircle } from "lucide-react";

export default function Loading() {
  return (
    <div className="grid h-full min-h-64 place-items-center text-sm text-muted-foreground" role="status">
      <span className="flex items-center gap-2"><LoaderCircle className="size-4 animate-spin" /> Opening chat…</span>
    </div>
  );
}
