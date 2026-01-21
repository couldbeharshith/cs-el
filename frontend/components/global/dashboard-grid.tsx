import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LucideIcon } from "lucide-react";
import { ArrowRight } from "lucide-react";

interface DashboardLink {
  name: string;
  href: string;
  description?: string;
  icon?: LucideIcon;
}

interface DashboardGridProps {
  title?: string;
  description?: string;
  links: DashboardLink[];
}

export function DashboardGrid({ title, description, links }: DashboardGridProps) {
  return (
    <div className="space-y-6">
      {(title || description) && (
        <div>
          {title && <h1 className="text-3xl font-semibold">{title}</h1>}
          {description && <p className="text-sm text-muted-foreground">{description}</p>}
        </div>
      )}
      
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {links.map((link) => (
          <Link href={link.href} key={link.name} className="group">
            <Card className="h-full transition-all hover:shadow-lg hover:border-primary/50">
              <CardHeader className="space-y-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    {link.icon && (
                      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                        <link.icon className="h-6 w-6 text-primary" />
                      </div>
                    )}
                    <div>
                      <CardTitle className="text-xl group-hover:text-primary transition-colors">
                        {link.name}
                      </CardTitle>
                      {link.description && (
                        <p className="text-sm text-muted-foreground mt-1">{link.description}</p>
                      )}
                    </div>
                  </div>
                  <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
                </div>
              </CardHeader>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
