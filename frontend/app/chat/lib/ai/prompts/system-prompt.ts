import { getTableSchema } from '../../utils/get-table-schema';
import fs from 'fs';
import path from 'path';

// Model-specific prompt file mapping
// All three use gpt-4.1 but with different system prompts
const MODEL_PROMPT_FILES = {
  'local-llm1': 'llm1-system-prompt.md',
  'local-llm2': 'llm2-system-prompt.md',
  'local-llm3': 'llm3-system-prompt.md',
};

// Cache for loaded prompts
const promptCache: Record<string, string> = {};

function loadPromptFromFile(filename: string): string {
  if (promptCache[filename]) {
    return promptCache[filename];
  }

  const promptPath = path.join(process.cwd(), 'app/chat/lib/ai/prompts', filename);
  const promptContent = fs.readFileSync(promptPath, 'utf-8');
  promptCache[filename] = promptContent;
  return promptContent;
}

export async function getSystemPrompt(user: { id: string; email: string }, selectedModel?: string) {
    const tableSchema = await getTableSchema();
    const now = new Date();
    const year = now.getFullYear();
    const month = now.toLocaleString('default', { month: 'long' });

    // Get model-specific prompt from file or use default
    let basePrompt = '';
    if (selectedModel && MODEL_PROMPT_FILES[selectedModel as keyof typeof MODEL_PROMPT_FILES]) {
      const promptFile = MODEL_PROMPT_FILES[selectedModel as keyof typeof MODEL_PROMPT_FILES];
      basePrompt = loadPromptFromFile(promptFile);
    } else {
      // Default prompt for unknown models
      basePrompt = loadPromptFromFile('llm1-system-prompt.md');
    }

    // Build the complete prompt with dynamic information
    return `${basePrompt}

## Current Context

The current user's ID is: ${user.id}.
The current user's email is: ${user.email}.
The current date is ${month} ${year}. Use this for any date-related questions if the user doesn't specify a date.

## Database Schema

${tableSchema}
`;
}
