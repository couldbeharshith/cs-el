import { customProvider, wrapLanguageModel } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createGroq } from "@ai-sdk/groq";
import { createXai } from "@ai-sdk/xai";
import { createGeminiFlashModel } from "./gemini-flash";

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL || undefined,
});

const google = createGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
  baseURL: process.env.GOOGLE_GENERATIVE_AI_BASE_URL || undefined,
});

const groq = createGroq({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: process.env.GROQ_BASE_URL || undefined,
});

const xai = createXai({ apiKey: process.env.XAI_API_KEY });

const lmstudio = createOpenAI({
  name: "lmstudio",
  apiKey: process.env.LMSTUDIO_API_KEY || "lm-studio",
  baseURL: process.env.LMSTUDIO_BASE_URL || "http://localhost:1234/v1",
});

const provider = customProvider({
  languageModels: {
    openai: wrapLanguageModel({
      model: openai.responses("gpt-6-luna"),
      middleware: {
        // AI SDK 4 defaults temperature to zero and predates Luna's reasoning mode.
        transformParams: async ({ params }) => ({ ...params, temperature: undefined }),
      },
    }),
    groq: groq("openai/gpt-oss-120b"),
    "gpt-4o-mini": openai("gpt-4o-mini"),
    "gpt-4.1-mini": openai("gpt-4.1-mini"),
    "gpt-4.1": openai("gpt-4.1"),
    "gemini-3-flash-preview": google("gemini-3-flash-preview"),
    "gemini-2.5-flash": google("gemini-2.5-flash"),
    "gemini-2.0-flash": google("gemini-2.0-flash"),
    "gemini-2.5-pro": google("gemini-2.5-pro"),
    "grok-3-mini": xai("grok-3-mini"),
    "llama-3.3-70b-versatile": groq("llama-3.3-70b-versatile"),
    "lmstudio-local": lmstudio("local-model"),
  },
  fallbackProvider: openai,
});

const providers: Record<string, { label: string; key?: string }> = {
  openai: { label: "OpenAI", key: "OPENAI_API_KEY" },
  google: { label: "Gemini", key: "GOOGLE_GENERATIVE_AI_API_KEY" },
  groq: { label: "Groq", key: "GROQ_API_KEY" },
  xai: { label: "xAI", key: "XAI_API_KEY" },
  lmstudio: { label: "LM Studio" },
};

export function getChatModel(id: string) {
  const model = id === "gemini" ? createGeminiFlashModel() : provider.languageModel(id);
  return { model, ...providers[model.provider.split(".")[0]] };
}
