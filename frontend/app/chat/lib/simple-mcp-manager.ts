import { experimental_createMCPClient, type ToolSet } from "ai";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const MCP_SERVERS = {
  rag: process.env.RAG_MCP_URL ?? process.env.NEXT_PUBLIC_MCP_URL ?? "http://127.0.0.1:8001/mcp",
  computer: process.env.COMPUTER_MCP_URL ?? "http://127.0.0.1:8002/mcp",
};

type Client = Awaited<ReturnType<typeof experimental_createMCPClient>>;
interface ToolConnection {
  client: Client;
  tools: ToolSet;
}

const connections = new Map<string, Promise<ToolConnection | null>>();
const retryAfter = new Map<string, number>();

async function connect(name: string, url: string): Promise<ToolConnection | null> {
  let transport: StreamableHTTPClientTransport | undefined;
  let client: Client | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    transport = new StreamableHTTPClientTransport(new URL(url));
    const activeTransport = transport;
    const setup = async () => {
      const connected = await experimental_createMCPClient({ transport: activeTransport });
      client = connected;
      return { client: connected, tools: await connected.tools() };
    };
    return await Promise.race([
      setup(),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error("Discovery timed out")), 1_500);
      }),
    ]);
  } catch (error) {
    console.warn(`MCP server ${name} unavailable`, error instanceof Error ? error.message : error);
    retryAfter.set(name, Date.now() + 15_000);
    if (client) await client.close().catch(() => {});
    else await transport?.close().catch(() => {});
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function getTools(name: string, url: string): Promise<ToolSet> {
  if (!url || (retryAfter.get(name) ?? 0) > Date.now()) return {};

  let pending = connections.get(name);
  if (!pending) {
    pending = connect(name, url);
    connections.set(name, pending);
  }

  const connection = await pending;
  if (!connection && connections.get(name) === pending) connections.delete(name);
  return connection?.tools ?? {};
}

export async function getAllMCPTools(): Promise<ToolSet> {
  const sets = await Promise.all(Object.entries(MCP_SERVERS).map(async ([name, url]) => {
    const tools = await getTools(name, url);
    return Object.entries(tools).map(([key, tool]) => [`${name}_${key}`, tool] as const);
  }));
  return Object.fromEntries(sets.flat());
}
