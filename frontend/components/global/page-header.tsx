import Link from "next/link";
import { Button } from "@/components/ui/button";
import { PlusIcon } from "lucide-react";
import { Separator } from "@/components/ui/separator";

interface PageHeaderProps {
  title: string;
  description: string;
  newItemHref?: string;
  newItemLabel?: string;
}

export function PageHeader({ title, description, newItemHref, newItemLabel = "New" }: PageHeaderProps) {
  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold">{title}</h1>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
        {newItemHref && (
          <Button asChild>
            <Link href={newItemHref} className="flex items-center gap-2">
              <PlusIcon className="h-4 w-4" />
              {newItemLabel}
            </Link>
          </Button>
        )}
      </div>
      <Separator className="my-4" />
    </div>
  );
}
