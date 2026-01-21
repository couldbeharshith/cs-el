"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Message } from "ai";
import { memo } from "react";
import { saveMessages } from "../actions";
interface SuggestedActionsProps {
  chatId: string;
  append: (message: Message) => Promise<string | null | undefined>;
  handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
}
function PureSuggestedActions({
  chatId,
  append,
  handleSubmit,
}: SuggestedActionsProps) {
  const suggestedActions = [
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
  return (
    <div className="grid sm:grid-cols-2 gap-2 w-full pb-2">
      {suggestedActions.map((suggestedAction, index) => (
        <motion.div
          initial={{
            opacity: 0,
            y: 20,
          }}
          animate={{
            opacity: 1,
            y: 0,
          }}
          exit={{
            opacity: 0,
            y: 20,
          }}
          transition={{
            delay: 0.05 * index,
          }}
          key={`suggested-action-${suggestedAction.title}-${index}`}
          className={index > 1 ? "hidden sm:block" : "block"}
        >
          <Button
            variant="ghost"
            onClick={async () => {
              const userMessage: Message = {
                id: crypto.randomUUID(),
                role: 'user',
                content: suggestedAction.action,
              };
              await saveMessages([userMessage], chatId);
              await append(userMessage);
            }}
            className="text-left border rounded-xl px-4 py-3.5 text-sm gap-1 sm:flex-col w-full h-auto justify-start items-start sm:items-stretch"
          >
            <span className="font-medium truncate">{suggestedAction.title}</span>
            <span className="text-muted-foreground truncate">
              {suggestedAction.label}
            </span>
          </Button>
        </motion.div>
      ))}
    </div>
  );
}
export const SuggestedActions = memo(PureSuggestedActions, () => true);
