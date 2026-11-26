"use client";

import type { ToolInvocation } from "ai";
import { AlertCircle, BarChart3, Database, FileText, Globe, LoaderCircle, Search, Wrench } from "lucide-react";
import { memo } from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { ApiRequestDisplay } from "./tool-displays/api-request-display";
import { ChartDisplay } from "./tool-displays/chart-display";
import { GenericToolDisplay } from "./tool-displays/generic-tool-display";
import { RagResultDisplay } from "./tool-displays/rag-result-display";
import { SupabaseQuery } from "./tool-displays/supabase-query";
import { TavilySearchResult } from "./tool-displays/tavily-search-result";
import { V0ToolDisplay } from "./tool-displays/v0-tool-display";

const toolDetails = {
  querySupabase: { label: "Database query", icon: Database },
  generateChart: { label: "Chart generation", icon: BarChart3 },
  exaSearch: { label: "Web search", icon: Search },
  ragRetrieval: { label: "Document retrieval", icon: FileText },
  rag_process_document_rag: { label: "Document analysis", icon: FileText },
  makeApiRequest: { label: "API request", icon: Globe },
  screenerQueryAgent: { label: "Company screening", icon: Search },
  sendHealthReport: { label: "Sending health report", icon: Globe },
  stagehandRun: { label: "Stagehand automation", icon: Wrench },
};

export const ToolResultDisplay = memo(function ToolResultDisplay({ toolCall }: { toolCall: ToolInvocation }) {
  const { toolName } = toolCall;
  const details = Object.hasOwn(toolDetails, toolName)
    ? toolDetails[toolName as keyof typeof toolDetails]
    : { label: toolName.startsWith("v0_") ? "v0 website tool" : toolName, icon: Wrench };
  const Icon = details.icon;

  return (
    <div className="mx-auto mb-3 w-full max-w-4xl px-4 sm:px-8">
      <div className="rounded-xl border bg-card">
        {toolCall.state === "result" ? (
          <CompletedTool toolCall={toolCall} label={details.label} icon={Icon} />
        ) : (
          <div role="status" className="flex items-center gap-2 px-4 py-3 text-sm text-muted-foreground">
            <LoaderCircle className="size-4 animate-spin text-primary" aria-hidden /> {details.label} in progress
          </div>
        )}
      </div>
    </div>
  );
});

function CompletedTool({ toolCall, label, icon: Icon }: {
  toolCall: Extract<ToolInvocation, { state: "result" }>;
  label: string;
  icon: typeof Wrench;
}) {
  const result = parseResult(toolCall.result);
  const error = result && typeof result === "object" && "error" in result ? result.error : null;
  if (error) {
    return (
      <div className="space-y-1 px-4 py-3 text-sm">
        <p className="flex items-center gap-2 font-medium text-destructive"><AlertCircle className="size-4" /> {label} failed</p>
        <p className="break-words text-muted-foreground">{typeof error === "string" ? error : "The service returned an error."}</p>
      </div>
    );
  }

  return (
    <Accordion type="single" collapsible>
      <AccordionItem value={toolCall.toolCallId}>
        <AccordionTrigger className="rounded-xl px-4 hover:no-underline">
          <span className="flex items-center gap-2"><Icon className="size-4 text-primary" />{label} results</span>
        </AccordionTrigger>
        <AccordionContent className="min-w-0 space-y-3 px-4">
          <details>
            <summary className="cursor-pointer text-xs font-medium text-muted-foreground">Request details</summary>
            <pre className="mt-2 overflow-x-auto rounded-lg bg-muted p-3 text-xs">{JSON.stringify(toolCall.args, null, 2)}</pre>
          </details>
          <Result toolName={toolCall.toolName} result={result} args={toolCall.args} />
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

function Result({ toolName, result, args }: { toolName: string; result: any; args: unknown }) {
  if (result && typeof result === "object") {
    switch (toolName) {
      case "querySupabase": return <SupabaseQuery data={result} />;
      case "generateChart": if (Array.isArray(result.data)) return <ChartDisplay {...result} />; break;
      case "exaSearch": if (Array.isArray(result.results)) return <TavilySearchResult data={result} />; break;
      case "ragRetrieval": if (Array.isArray(result.chunks)) return <RagResultDisplay chunks={result.chunks} />; break;
      case "makeApiRequest": return <ApiRequestDisplay data={result} />;
    }
    if (toolName.startsWith("v0_")) return <V0ToolDisplay data={result} />;
  }
  return <GenericToolDisplay args={args} result={result} />;
}

function parseResult(result: unknown): unknown {
  if (typeof result !== "string") return result;
  try {
    return JSON.parse(result);
  } catch {
    return result;
  }
}
