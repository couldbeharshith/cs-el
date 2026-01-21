import { streamText, CoreMessage , smoothStream} from 'ai';
import { getSystemPrompt } from '@/app/chat/lib/ai/prompts/system-prompt';
import { querySupabaseTool } from '@/app/chat/lib/ai/tools/query-supabase';
import { generateChart } from '@/app/chat/lib/ai/tools/generate-chart';
import { exaSearchTool } from '@/app/chat/lib/ai/tools/exa-search';
import { makeApiRequestTool } from '@/app/chat/lib/ai/tools/make-api-request';
import { screenerQueryAgent } from '@/app/chat/lib/ai/tools/screener-query-agent';
import { myProvider } from '@/app/chat/lib/ai/providers/providers';
import { getAllMCPTools } from '@/app/chat/lib/simple-mcp-manager';

export const maxDuration = 30;

export async function POST(req: Request) {
  try {
    // Auth removed - demo mode only
    const demoUser = { id: 'demo-user', email: 'demo@example.com' };

    const { messages, data, selectedModel }: { messages: CoreMessage[], data: any, selectedModel: string } = await req.json();
    const systemPrompt = await getSystemPrompt(demoUser);
    console.log(selectedModel);

    // Get MCP tools
    const mcpTools = await getAllMCPTools();

    const result = streamText({
      model: myProvider.languageModel(selectedModel as any),
      system: systemPrompt,
      messages,
      tools: {
        querySupabase: querySupabaseTool,
        generateChart: generateChart,
        exaSearch: exaSearchTool,
        makeApiRequest: makeApiRequestTool,
        screenerQueryAgent: screenerQueryAgent,
        ...mcpTools, 
      },
      maxSteps: 10, 
      onError: (error) => {
        console.error('Error:', error);
      },
      experimental_transform: smoothStream({
        chunking: 'word',
      }),
      toolCallStreaming: true,
    });

    // Stream the response back to the client so `useChat` can consume it
    return result.toDataStreamResponse();
  } catch (error: any) {
    console.error('[API] Error:', error);
    return new Response(JSON.stringify({ error: error.message || 'An unexpected error occurred.' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
