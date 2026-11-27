export const models = [
  { value: "openai", label: "OpenAI" },
  { value: "gemini", label: "Gemini" },
  { value: "groq", label: "Groq" },
];

const legacyModels: Record<string, string> = {
  "local-llm1": "openai",
  "local-llm2": "gemini",
  "local-llm3": "groq",
};

export function resolveModelId(model: string) {
  return Object.hasOwn(legacyModels, model) ? legacyModels[model] : model;
}
