import { convertToCoreMessages, streamText, smoothStream, type Message } from "ai";
import { z } from "zod";
import { getSystemPrompt } from "@/app/chat/lib/ai/prompts/system-prompt";
import { querySupabaseTool } from "@/app/chat/lib/ai/tools/query-supabase";
import { generateChart } from "@/app/chat/lib/ai/tools/generate-chart";
import { exaSearchTool } from "@/app/chat/lib/ai/tools/exa-search";
import { makeApiRequestTool } from "@/app/chat/lib/ai/tools/make-api-request";
import { screenerQueryAgent } from "@/app/chat/lib/ai/tools/screener-query-agent";
import { myProvider } from "@/app/chat/lib/ai/providers/providers";
import { getAllMCPTools } from "@/app/chat/lib/simple-mcp-manager";
import { getUser } from "@/app/chat/hooks/get-user";

export const maxDuration = 60;

const requestSchema = z.object({
  selectedModel: z.string().trim().min(1).max(100),
  messages: z.array(z.object({
    role: z.enum(["user", "assistant", "system"]),
    content: z.string(),
  }).passthrough()).min(1),
});

export async function POST(request: Request) {
  const input = requestSchema.safeParse(await request.json().catch(() => null));
  if (!input.success || !input.data.messages.some((message) => message.role === "user")) {
    return Response.json({ error: "A model and at least one user message are required." }, { status: 400 });
  }

  const { messages, selectedModel } = input.data;
  if ((selectedModel.startsWith("local-llm") || selectedModel.startsWith("gpt-")) && !process.env.OPENAI_API_KEY) {
    return Response.json({ error: "Chat needs an OpenAI API key for this profile." }, { status: 503 });
  }

  try {
    const [system, mcpTools] = await Promise.all([
      getUser().then((user) => getSystemPrompt(user, selectedModel)),
      getAllMCPTools(),
    ]);

    const result = streamText({
      model: myProvider.languageModel(selectedModel),
      system,
      messages: convertToCoreMessages(messages as Omit<Message, "id">[]),
      tools: {
        querySupabase: querySupabaseTool,
        generateChart,
        exaSearch: exaSearchTool,
        makeApiRequest: makeApiRequestTool,
        screenerQueryAgent,
        ...mcpTools,
      },
      maxSteps: 10,
      abortSignal: request.signal,
      onError: ({ error }) => console.error("Chat stream failed", error),
      experimental_transform: smoothStream({ chunking: "word" }),
      toolCallStreaming: true,
    });

    return result.toDataStreamResponse({
      getErrorMessage: () => "The model or a connected tool could not finish responding.",
    });
  } catch (error) {
    console.error("Chat request failed", error);
    return Response.json({ error: "Couldn’t start a response. Please try again." }, { status: 502 });
  }
}
