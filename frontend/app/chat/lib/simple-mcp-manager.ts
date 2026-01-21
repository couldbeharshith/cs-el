import { experimental_createMCPClient } from 'ai';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

// Simple MCP server configurations
const MCP_SERVERS = {
  // HTTP servers
  computer: {
    type: 'http' as const,
    url: 'http://127.0.0.1:8002/mcp',
    enabled: true,
  },
  rag: {
    type: 'http' as const,
    url: 'http://127.0.0.1:8000/mcp',
    enabled: true,
  },
  playwright: {
    type: 'http' as const,
    url: 'http://localhost:8001/mcp',
    enabled: false,
  },

  // Stdio server (Not needed because we have HTTP Playwright above, but here for reference)
  playwright_stdio: {
    type: 'stdio' as const,
    command: 'cmd',
    args: [
      '/c',
      'npx',
      '-y',
      '@smithery/cli@latest',
      'run',
      '@microsoft/playwright-mcp',
      '--key',
      'c7eefd65-fbfe-405c-b071-f54ad4165201'
    ],
    enabled: false,  // Disabled - causing errors
  },
  
} as const;

type MCPServerName = keyof typeof MCP_SERVERS;

// Simple cache for clients and tools
const clientCache = new Map<string, any>();
const toolsCache = new Map<string, Record<string, any>>();

export async function getMCPClient(serverName: MCPServerName) {
  const cacheKey = serverName;
  
  if (clientCache.has(cacheKey)) {
    return clientCache.get(cacheKey);
  }

  try {
    const serverConfig = MCP_SERVERS[serverName];
    
    if (!serverConfig.enabled) {
      return null;
    }

    let transport;
    // Cast to any to avoid TS narrowing based on the 'enabled' property's current values
    const config = serverConfig as any;

    if (config.type === 'http') {
      transport = new StreamableHTTPClientTransport(new URL(config.url));
    } else if (config.type === 'stdio') {
      transport = new StdioClientTransport({
        command: config.command,
        args: [...config.args],
      });
    } else {
      throw new Error(`Unknown transport type for ${serverName}`);
    }

    const client = await experimental_createMCPClient({ transport });
    
    clientCache.set(cacheKey, client);
    return client;
  } catch (error) {
    console.error(`Failed to create MCP client for ${serverName}:`, error);
    return null;
  }
}

export async function getMCPTools(serverName: MCPServerName) {
  const cacheKey = serverName;
  
  if (toolsCache.has(cacheKey)) {
    return toolsCache.get(cacheKey);
  }

  try {
    const client = await getMCPClient(serverName);
    if (!client) return {};
    
    const tools = await client.tools();
    toolsCache.set(cacheKey, tools);
    return tools;
  } catch (error) {
    console.error(`Failed to get MCP tools for ${serverName}:`, error);
    return {};
  }
}

export async function getAllMCPTools() {
  const allTools: Record<string, any> = {};
  
  for (const serverName of Object.keys(MCP_SERVERS) as MCPServerName[]) {
    // Skip disabled servers
    if (!MCP_SERVERS[serverName].enabled) {
      continue;
    }

    try {
      const tools = await getMCPTools(serverName);
      
      // Prefix tool names with server name to avoid conflicts
      for (const [toolName, tool] of Object.entries(tools)) {
        const prefixedName = `${serverName}_${toolName}`;
        allTools[prefixedName] = tool;
      }
    } catch (error) {
      console.warn(`Failed to load tools from ${serverName}:`, error);
    }
  }
  
  return allTools;
}

export function getAvailableServers(): MCPServerName[] {
  return (Object.keys(MCP_SERVERS) as MCPServerName[]).filter(
    (name) => MCP_SERVERS[name].enabled
  );
}