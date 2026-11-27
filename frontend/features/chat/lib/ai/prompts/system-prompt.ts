import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { getTableSchema } from "../../utils/get-table-schema";

const loadPrompt = {
  openai: "llm1-system-prompt.md",
  gemini: "llm2-system-prompt.md",
  groq: "llm3-system-prompt.md",
};

const prompts = new Map<string, Promise<string>>();

export async function getSystemPrompt(user: { id: string; email: string }, selectedModel: string) {
  const profile = Object.hasOwn(loadPrompt, selectedModel) ? selectedModel as keyof typeof loadPrompt : "openai";
  let prompt = prompts.get(profile);
  if (!prompt) {
    const file = join(process.cwd(), "features/chat/lib/ai/prompts", loadPrompt[profile]);
    prompt = readFile(file, "utf8").catch((error) => {
      prompts.delete(profile);
      throw error;
    });
    prompts.set(profile, prompt);
  }

  const [base, schema] = await Promise.all([prompt, getTableSchema()]);
  const date = new Intl.DateTimeFormat("en", { year: "numeric", month: "long" }).format(new Date());
  return `${base}

## Current Context

The current user's ID is: ${user.id}.
The current user's email is: ${user.email}.
The current date is ${date}. Use this for date-related questions when no date is specified.

## Database Schema

${schema}
`;
}
