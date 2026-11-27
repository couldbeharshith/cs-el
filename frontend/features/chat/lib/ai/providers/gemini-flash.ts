import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { wrapLanguageModel } from "ai";
import { createParser } from "eventsource-parser";

type GeminiContent = {
  parts: { functionCall?: unknown; thoughtSignature?: string }[];
};

type FunctionDeclaration = {
  name: string;
  description: string;
  parametersJsonSchema: unknown;
};

export function createGeminiFlashModel() {
  // AI SDK 4 drops Gemini's tool-call signatures. Keep them for the steps of
  // this stream only. Each step sends the same history plus any new tool calls.
  const signatures: (string | undefined)[] = [];
  let functionDeclarations: FunctionDeclaration[] = [];
  const google = createGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
    baseURL: process.env.GOOGLE_GENERATIVE_AI_BASE_URL || undefined,
    fetch: async (url, options) => {
      const body = JSON.parse(options?.body as string) as { contents: GeminiContent[]; tools?: unknown };
      if (functionDeclarations.length) body.tools = [{ functionDeclarations }];
      let callIndex = 0;
      for (const content of body.contents) {
        for (const part of content.parts) {
          if (!part.functionCall) continue;
          const signature = signatures[callIndex++];
          if (signature) part.thoughtSignature = signature;
        }
      }

      const response = await fetch(url, { ...options, body: JSON.stringify(body) });
      if (!response.ok || !response.body) return response;

      const decoder = new TextDecoder();
      const parser = createParser({
        onEvent({ data }) {
          let chunk: { candidates?: { content?: GeminiContent }[] };
          try {
            chunk = JSON.parse(data);
          } catch {
            return; // The SDK handles malformed provider events.
          }
          for (const part of chunk?.candidates?.[0]?.content?.parts ?? []) {
            if (part.functionCall) signatures[callIndex++] = part.thoughtSignature;
          }
        },
      });

      const stream = response.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, controller) {
          parser.feed(decoder.decode(chunk, { stream: true }));
          controller.enqueue(chunk);
        },
      }));
      return new Response(stream, response);
    },
  });

  return wrapLanguageModel({
    model: google("gemini-3.8-flash"),
    middleware: {
      transformParams: async ({ params }) => {
        const tools = params.mode.type === "regular" ? (params.mode.tools ?? []) : [];
        // Native JSON Schema keeps record fields such as the Screener request body.
        functionDeclarations = tools.filter((tool) => tool.type === "function").map((tool) => ({
          name: tool.name,
          description: tool.description ?? "",
          parametersJsonSchema: tool.parameters,
        }));
        return { ...params, temperature: undefined };
      },
    },
  });
}
