import { createServer } from "node:http";
import { Readable } from "node:stream";

const port = Number(process.env.E2E_SERVICES_PORT ?? 4316);
const restUrl = process.env.E2E_POSTGREST_URL ?? "http://127.0.0.1:54381";

createServer(async (request, response) => {
  try {
    const path = request.url ?? "/";
    if (path === "/health") {
      const health = await fetch(restUrl, { signal: AbortSignal.timeout(1000) });
      response.writeHead(health.ok ? 200 : 503).end();
      return;
    }

    if (path.startsWith("/rest/v1/")) {
      const headers = new Headers();
      for (const name of ["accept", "content-type", "prefer", "range", "range-unit"]) {
        if (typeof request.headers[name] === "string") headers.set(name, request.headers[name]);
      }
      const body = ["GET", "HEAD"].includes(request.method) ? undefined : await readBody(request);
      const upstream = await fetch(restUrl + path.slice("/rest/v1".length), {
        method: request.method,
        headers,
        body,
      });
      response.setHeader("Access-Control-Allow-Origin", "*");
      response.setHeader("Access-Control-Allow-Headers", "*");
      response.setHeader("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS");
      if (request.method === "OPTIONS") { response.writeHead(204).end(); return; }
      if (upstream.headers.has("content-type")) response.setHeader("Content-Type", upstream.headers.get("content-type"));
      response.writeHead(upstream.status);
      if (upstream.body) Readable.fromWeb(upstream.body).pipe(response);
      else response.end();
      return;
    }

    if (path === "/accelerated_chart_data") {
      await readBody(request);
      response.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({
        chartType: "line",
        data: [{ label: "2025-11-28", value: 30 }, { label: "2025-11-29", value: 45 }],
        title: "Fixture company performance", description: "Two local data points",
        xAxis: "label", yAxis: ["value"],
      }));
      return;
    }

    if (path === "/v1/chat/completions") {
      const { model, messages } = JSON.parse(await readBody(request));
      const content = messages.filter((message) => message.role === "user").at(-1)?.content;
      const prompt = Array.isArray(content) ? content.map((part) => part.text).join("") : content;
      if (prompt === "Fixture provider error") {
        response.writeHead(400, { "Content-Type": "application/json" });
        response.end(JSON.stringify({ error: { message: "Fixture provider rejected the request" } }));
        return;
      }
      response.writeHead(200, { "Content-Type": "text/event-stream" });
      const send = (delta, finish_reason = null) => {
        response.write(`data: ${JSON.stringify({
          id: "chatcmpl-finsync", object: "chat.completion.chunk", created: 1764417600, model,
          choices: [{ index: 0, delta, finish_reason }],
        })}\n\n`);
      };
      const words = ["Fixture response. ", "The sample policy has ", "**documented exclusions**."];
      send({ role: "assistant" });
      if (prompt === "Chart fixture company" && !messages.some((message) => message.role === "tool")) {
        send({ tool_calls: [{
          index: 0, id: "call_chart", type: "function",
          function: { name: "generateChart", arguments: '{"company_name":"Fixture Company"}' },
        }] });
        send({}, "tool_calls");
        response.end("data: [DONE]\n\n");
        return;
      }
      for (const word of words) {
        await new Promise((resolve) => setTimeout(resolve, 75));
        send({ content: word });
      }
      send({}, "stop");
      response.end("data: [DONE]\n\n");
      return;
    }

    if (path === "/v1/audio/transcriptions") {
      await readBody(request);
      response.writeHead(200, { "Content-Type": "text/plain" }).end("Fixture voice transcription");
      return;
    }

    response.writeHead(404).end();
  } catch (error) {
    console.error(error);
    response.writeHead(503).end();
  }
}).listen(port, "127.0.0.1");

async function readBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return Buffer.concat(chunks);
}
