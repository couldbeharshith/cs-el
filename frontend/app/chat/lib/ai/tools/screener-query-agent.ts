import { z } from 'zod';
import { tool } from 'ai';

const screenerQueryAgentParameters = z.object({
  query: z.string().describe('The natural language query to send to the screener query agent.'),
});

export const screenerQueryAgent = tool({
  description: 'Query the screener agent with natural language to get intelligent analysis about companies. Use this tool when the user asks questions like "find companies with high revenue growth", "which companies have strong fundamentals", "analyze market trends", or any complex analytical query about stocks/companies that requires reasoning beyond simple data retrieval. This is the primary tool for stock screening and company analysis queries.',
  parameters: screenerQueryAgentParameters,
  execute: async ({ query }) => {
    try {
      const response = await fetch('http://127.0.0.1:8080/query_agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`Failed to query agent: ${response.statusText}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error querying screener agent:', error);
      return { error: 'Failed to query screener agent.' };
    }
  },
});
