import { tool } from 'ai';
import { z } from 'zod';
import Exa from 'exa-js';

const exa = new Exa(process.env.EXA_API_KEY);

export const exaSearchTool = tool({
  description: 'Search the web using Exa AI for up-to-date information, news, and research.',
  parameters: z.object({
    query: z.string().describe('The search query to use.'),
  }),
  execute: async ({ query }) => {
    try {
      const searchResult = await exa.searchAndContents(query, {
        numResults: 5,
        text: true,
        highlights: true,
      });
      
      return JSON.stringify(searchResult);
    } catch (error) {
      console.error('Error searching with Exa:', error);
      return JSON.stringify({ error: 'Failed to perform search. Please try again.' });
    }
  },
});
