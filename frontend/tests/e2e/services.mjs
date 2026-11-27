import { createServer } from "node:http";
import { Readable } from "node:stream";
import { setTimeout as delay } from "node:timers/promises";

const port = Number(process.env.E2E_SERVICES_PORT ?? 4316);
const restUrl = process.env.E2E_POSTGREST_URL ?? "http://127.0.0.1:54381";
const answer = ["Fixture response. ", "The sample policy has ", "**documented exclusions**."];
const chartCall = { name: "generateChart", args: { company_name: "Fixture Company" } };

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
      const prompt = messages.findLast((message) => message.role === "user")?.content;
      const example = typeof prompt === "string" && /^Policy verification (\d+) for the fixture$/.exec(prompt);
      if (model !== "gpt-4o-mini" || !example) return reject(response, 400, "Fixture title unavailable");
      response.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({
        id: "title_fixture", created: 1764417600, model,
        choices: [{ index: 0, message: { role: "assistant", content: "Coverage Review " + example[1] + " for Policy Details" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 10, completion_tokens: 7, total_tokens: 17 },
      }));
      return;
    }

    if (path === "/v1/responses") {
      const { model, input, tools, store, temperature } = JSON.parse(await readBody(request));
      if (model !== "gpt-6-luna" || store !== false || temperature !== undefined || tools.some((tool) => tool.strict === true)) {
        return reject(response, 400, "Invalid OpenAI configuration");
      }
      const prompt = input.filter((message) => message.role === "user").at(-1)?.content[0]?.text;
      if (prompt === "Fixture provider error") return reject(response, 400, "Fixture provider rejected the request");
      const send = eventStream(response);
      send({ type: "response.created", response: { id: "resp_fixture", created_at: 1764417600, model } });
      if (prompt === "Chart fixture company" && !input.some((item) => item.type === "function_call_output")) {
        const item = { type: "function_call", id: "fc_fixture", call_id: "call_chart", name: chartCall.name };
        send({ type: "response.output_item.added", output_index: 0, item: { ...item, arguments: "" } });
        send({ type: "response.output_item.done", output_index: 0, item: {
          ...item, arguments: JSON.stringify(chartCall.args), status: "completed",
        } });
      } else {
        await sendAnswer((delta) => send({ type: "response.output_text.delta", delta }));
      }
      send({ type: "response.completed", response: { usage: { input_tokens: 10, output_tokens: 20 } } });
      response.end();
      return;
    }

    if (path === "/gemini/models/gemini-3.8-flash:streamGenerateContent?alt=sse") {
      const { contents, tools } = JSON.parse(await readBody(request));
      const apiTool = tools?.[0]?.functionDeclarations.find((tool) => tool.name === "makeApiRequest");
      if (!apiTool?.parametersJsonSchema?.properties?.body?.additionalProperties) {
        return reject(response, 400, "Missing Gemini tool schema");
      }
      const parts = contents.flatMap((content) => content.parts);
      const needsChart = parts.some((part) => part.text === "Chart fixture company");
      const toolResults = parts.some((part) => part.functionResponse);
      if (toolResults && !parts.some((part) => part.functionCall?.name === chartCall.name && part.thoughtSignature === "fixture-signature")) {
        return reject(response, 400, "Missing Gemini thought signature");
      }
      const send = eventStream(response);
      if (needsChart && !toolResults) {
        send({ candidates: [{ content: { parts: [{ functionCall: chartCall, thoughtSignature: "fixture-signature" }] } }] });
      } else {
        await sendAnswer((text) => send({ candidates: [{ content: { parts: [{ text }] } }] }));
      }
      send({ candidates: [{ finishReason: "STOP" }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 20 } });
      response.end();
      return;
    }

    if (path === "/groq/chat/completions") {
      const { model, messages } = JSON.parse(await readBody(request));
      if (model !== "openai/gpt-oss-120b") return reject(response, 404, "Unexpected Groq model");
      const prompt = messages.filter((message) => message.role === "user").at(-1)?.content;
      if (prompt === "Fixture Groq error") return reject(response, 401, "Fixture API key rejected");
      const send = eventStream(response);
      const chunk = (delta, finish_reason = null) => send({
        id: "chatcmpl-sampada", object: "chat.completion.chunk", created: 1764417600, model,
        choices: [{ index: 0, delta, finish_reason }],
      });
      chunk({ role: "assistant" });
      if (prompt === "Chart fixture company" && !messages.some((message) => message.role === "tool")) {
        chunk({ tool_calls: [{
          index: 0, id: "call_chart", type: "function",
          function: { name: chartCall.name, arguments: JSON.stringify(chartCall.args) },
        }] });
        chunk({}, "tool_calls");
      } else {
        await sendAnswer((content) => chunk({ content }));
        chunk({}, "stop");
      }
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

function eventStream(response) {
  response.writeHead(200, { "Content-Type": "text/event-stream" });
  return (data) => response.write(`data: ${JSON.stringify(data)}\n\n`);
}

function reject(response, status, message) {
  response.writeHead(status, { "Content-Type": "application/json" });
  response.end(JSON.stringify({ error: { message } }));
}

async function sendAnswer(send) {
  for (const part of answer) {
    await delay(75);
    send(part);
  }
}
