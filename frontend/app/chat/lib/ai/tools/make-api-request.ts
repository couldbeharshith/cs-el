import { tool } from 'ai';
import { z } from 'zod';

export const makeApiRequestTool = tool({
  description: 'Make an HTTP API request to a specified URL. Supports GET, POST, PUT, DELETE, etc. The screener API is available at http://127.0.0.1:8080. The available endpoints are 1. POST /accelerated_concall :-this gets the concall of that company , POST /accelerated_peers : This gets peers of that company , POST /accelerated_profit_loss : This gets the profit and loss of the company , POST /accelerated_quarterly_results :- This gets the quarterly results of that company , POST /accelerated_announcements :- This gets the announcements of that company , body for all these endpoints have company_name : str' ,
  parameters: z.object({
    url: z.string().describe('The URL to make the request to.'),
    method: z.enum(['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']).describe('The HTTP method to use.'),
    headers: z.record(z.string()).optional().describe('Optional headers to include in the request.'),
    body: z.any().optional().describe('Optional body for the request (for POST, PUT, etc.). Can be a JSON object or string.'),
  }),
  execute: async ({ url, method, headers, body }) => {
    try {
      const options: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...headers,
        },
      };

      if (body && method !== 'GET' && method !== 'HEAD') {
        options.body = typeof body === 'string' ? body : JSON.stringify(body);
      }

      const response = await fetch(url, options);
      
      let responseData;
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        responseData = await response.json();
      } else {
        responseData = await response.text();
      }

      return JSON.stringify({
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries()),
        data: responseData,
        request: {
            url,
            method,
            headers: options.headers,
            body
        }
      });
    } catch (error) {
      console.error('Error making API request:', error);
      return JSON.stringify({ 
          error: error instanceof Error ? error.message : 'Failed to make API request.',
          request: {
            url,
            method,
            headers,
            body
        }
      });
    }
  },
});
