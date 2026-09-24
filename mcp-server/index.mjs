#!/usr/bin/env node

/**
 * Sference MCP server — delegate tasks to Sference Agent Cloud.
 *
 * Any MCP-compatible harness (Claude Desktop, Goose, Cline, Zed, Hermes)
 * connects to this server. The harness calls `delegate_task` to send work to
 * Sference's GPU agent loop. Sference runs the full agent loop (inference →
 * tools → repeat) and returns the result.
 *
 * Install:
 *   npx @sference/mcp
 *
 * Configure (Claude Desktop claude_desktop_config.json):
 *   {
 *     "mcpServers": {
 *       "sference": {
 *         "command": "npx",
 *         "args": ["-y", "@sference/mcp"],
 *         "env": { "SFERENCE_API_KEY": "sk_..." }
 *       }
 *     }
 *   }
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const API_KEY = process.env.SFERENCE_API_KEY || "";
const BASE_URL = process.env.SFERENCE_BASE_URL || "https://api.sference.com";
const DEFAULT_MODEL = process.env.SFERENCE_DEFAULT_MODEL || "zai-org/GLM-5.2";
const TIMEOUT_S = parseInt(process.env.SFERENCE_TIMEOUT_S || "1860", 10);

if (!API_KEY) {
  console.error("SFERENCE_API_KEY not set — get one at https://app.sference.com/settings");
  process.exit(1);
}

const TOOLS = [
  {
    name: "list_models",
    description:
      "List GPU models available on Sference. " +
      "These are exclusive models running on Sference's bare-metal GPUs " +
      "(B200/B300/AMD BW100) — not available on OpenAI or Anthropic. " +
      "Use this to discover which models you can pass to delegate_task.",
    inputSchema: { type: "object", properties: {}, required: [] },
  },
  {
    name: "delegate_task",
    description:
      "Delegate a task to Sference Agent Cloud — a GPU-powered agent loop. " +
      "The agent runs on Sference's bare-metal GPUs, executes tool calls in a " +
      "loop until done, and returns structured output. Use this for tasks that " +
      "need GPU reasoning, code execution, or multi-step agent loops with models " +
      "like GLM-5.2, MiniMax-M3, or DeepSeek-V4.",
    inputSchema: {
      type: "object",
      properties: {
        task: { type: "string", description: "The task description or question for the agent." },
        model: { type: "string", description: "Model to use (default: zai-org/GLM-5.2). Call list_models to see options." },
        tools: { type: "string", description: 'Comma-separated tool types (e.g. "code_interpreter"). None = no tools.' },
        flex: { type: "boolean", description: "Use flex tier for discounted pricing (waits for idle GPU, then runs all steps at realtime)." },
        max_steps: { type: "number", description: "Maximum agent loop iterations (default: 25)." },
        timeout_s: { type: "number", description: "Wall-clock timeout in seconds (default: 1800 = 30min)." },
        instructions: { type: "string", description: "Optional system instructions for the agent." },
        mcp_command: { type: "string", description: 'MCP server command to connect to (e.g. "npx -y @modelcontextprotocol/server-github").' },
        mcp_env_file: { type: "string", description: "Path to file containing env var for MCP server." },
      },
      required: ["task"],
    },
  },
];

// --- Tool handlers -------------------------------------------------------

async function listModels() {
  const r = await fetch(`${BASE_URL}/v1/models`, {
    headers: { Authorization: `Bearer ${API_KEY}` },
    signal: AbortSignal.timeout(30000),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await r.json();
  const models = data.data || [];
  if (!models.length) return "No models available.";
  const lines = ["Available Sference models:"];
  for (const m of models) lines.push(`  - ${m.id || "?"}`);
  lines.push("", "Pass any model ID to delegate_task(model=...).");
  return lines.join("\n");
}

async function delegateTask(args) {
  const toolList = [];
  if (args.tools) {
    for (const t of args.tools.split(",")) {
      const trimmed = t.trim();
      if (trimmed) toolList.push({ type: trimmed });
    }
  }

  if (args.mcp_command) {
    const parts = args.mcp_command.split(/\s+/);
    const mcpTool = {
      type: "mcp",
      name: "mcp_server",
      transport: "stdio",
      command: parts[0],
      args: parts.slice(1),
    };
    if (args.mcp_env_file) {
      try {
        const fs = await import("node:fs");
        mcpTool.env = { GITHUB_PERSONAL_ACCESS_TOKEN: fs.readFileSync(args.mcp_env_file, "utf8").trim() };
      } catch {}
    }
    toolList.push(mcpTool);
  }

  const body = {
    model: args.model || DEFAULT_MODEL,
    input: args.task,
    max_steps: args.max_steps ?? 25,
    tools: toolList,
    service_tier: args.flex ? "flex" : "default",
    stream: false,
    timeout_s: args.timeout_s ?? 1800,
  };
  if (args.instructions) body.instructions = args.instructions;

  const r = await fetch(`${BASE_URL}/v1/agents`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(TIMEOUT_S * 1000),
  });

  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(`Sference API error (${r.status}): ${text.slice(0, 500)}`);
  }

  const result = await r.json();
  const status = result.status || "?";
  const steps = result.total_steps ?? "?";
  const totalTokens = result.total_usage?.total_tokens ?? 0;
  const error = result.error;

  const parts = [`Status: ${status}`, `Steps: ${steps}`, `Tokens: ${totalTokens}`];
  if (error) parts.push(`Error: ${JSON.stringify(error)}`);

  for (const item of result.output || []) {
    if (item.type === "message") {
      for (const c of item.content || []) {
        if (c.type === "output_text") {
          parts.push("", c.text);
        }
      }
    } else if (item.type === "reasoning") {
      for (const s of item.summary || []) {
        if (s.type === "summary_text") parts.push(`[reasoning] ${s.text}`);
      }
    }
  }

  return parts.join("\n");
}

// --- Server setup --------------------------------------------------------

const server = new Server(
  { name: "sference", version: "0.1.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, () => ({
  tools: TOOLS,
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  try {
    let text;
    if (name === "list_models") {
      text = await listModels();
    } else if (name === "delegate_task") {
      text = await delegateTask(args || {});
    } else {
      text = `Unknown tool: ${name}`;
    }
    return {
      content: [{ type: "text", text }],
    };
  } catch (err) {
    return {
      content: [{ type: "text", text: `Error: ${err.message}` }],
      isError: true,
    };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
