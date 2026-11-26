"use client";

import { ArrowUpRight, BookOpen, Braces, ClipboardList, FileQuestion, Mail, NotebookPen } from "lucide-react";

const prompts = [
    {
      title: "Summarize Insurance Policy (RAG)",
      label: "Extract key coverages, limits, and exclusions from the insurance PDF",
      action:
        `Use RAG/document tools to read the insurance PDF at https://micbdyubdfqefphlaouz.supabase.co/storage/v1/object/public/documents/OrientalInsure.pdf. Produce a concise summary (3-8 bullets) containing: named insured, policy number, policy period (effective and expiry dates), main coverages and limits, key exclusions, deductible(s), and premium if available. If a field is not present, respond with "Not found" for that field.`,
    },
    {
      title: "Build Q&A from Policy",
      label: "Create 6–8 user-facing Q&A pairs based on the PDF",
      action:
        `Read the insurance PDF at https://micbdyubdfqefphlaouz.supabase.co/storage/v1/object/public/documents/OrientalInsure.pdf and generate 6–8 concise question-and-answer pairs a customer might ask about their policy (one-line questions and one-line answers). For each answer, include a short source reference (page or section) if available.`,
    },
    {
      title: "Extract Structured Entities",
      label: "Return JSON with policy fields (policy_number, insured, dates, limits)",
      action:
        `Using the PDF at https://micbdyubdfqefphlaouz.supabase.co/storage/v1/object/public/documents/OrientalInsure.pdf, extract and return a JSON object with the following keys: policy_number, named_insured, effective_date, expiry_date, total_premium, coverages (array of {name, limit}), deductibles (array), and key_exclusions (array). If a value is missing, set it to null.`,
    },
    {
      title: "Identify Coverage Gaps",
      label: "List up to 5 potential gaps or ambiguous terms and suggested questions",
      action:
        `Scan the PDF at https://micbdyubdfqefphlaouz.supabase.co/storage/v1/object/public/documents/OrientalInsure.pdf and list up to 5 potential coverage gaps, ambiguous policy terms, or items that typically trigger claims disputes. For each item, include a suggested clarifying question to ask the insurer.`,
    },
    {
      title: "Draft Client Email",
      label: "Write a short email summarizing coverage and next steps",
      action:
        `Based on the insurance PDF (https://micbdyubdfqefphlaouz.supabase.co/storage/v1/object/public/documents/OrientalInsure.pdf), draft a plain-language email to a client. Include a subject line, a 2–3 sentence summary of main coverages, one sentence calling out any urgent issues, and 2 recommended next steps (e.g., confirm insured property list, obtain clarifications).`,
    },
    {
      title: "Notepad Summary",
      label: "Produce a detailed RAG summary, then type it into Notepad",
      action:
        `First, using RAG/document tools, read the insurance PDF at https://micbdyubdfqefphlaouz.supabase.co/storage/v1/object/public/documents/OrientalInsure.pdf and produce a well-structured, detailed summary containing these sections:\n- Named insured\n- Policy number\n- Policy period (effective and expiry dates)\n- Main coverages and limits (list each coverage and its limit)\n- Key exclusions\n- Deductibles\n- Total premium (if present)\n- Any endorsements or special conditions\n- Up to 3 ambiguous terms or potential gaps to clarify (with suggested clarifying questions)\n\nSecond, open Notepad (Windows) and type the structured summary you created. Format the Notepad file with clear headings (one section per heading).`,
    },
  ];

const icons = [BookOpen, FileQuestion, Braces, ClipboardList, Mail, NotebookPen];

export function SuggestedActions({ onSelect, disabled }: {
  onSelect: (prompt: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3" aria-label="Suggested research prompts">
      {prompts.map(({ title, label, action }, index) => {
        const Icon = icons[index];
        return (
          <button key={title} type="button" disabled={disabled} onClick={() => onSelect(action)}
            className="group flex min-w-0 items-start gap-3 rounded-xl border bg-card p-3 text-left transition-colors hover:border-primary/50 hover:bg-accent focus-visible:outline-2 focus-visible:outline-ring disabled:opacity-50 sm:flex-col sm:p-4">
            <span className="flex shrink-0 items-center justify-between text-primary sm:w-full">
              <Icon className="size-4" aria-hidden />
              <ArrowUpRight className="hidden size-4 text-muted-foreground transition-colors group-hover:text-primary sm:block" aria-hidden />
            </span>
            <span className="min-w-0 space-y-1">
              <span className="block text-sm font-semibold">{title}</span>
              <span className="block text-xs leading-relaxed text-muted-foreground">{label}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}
