import { mock } from "bun:test";
import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";

type Handler = (event: unknown, ctx: ExtensionContext) => unknown;

export interface AgentIdentity {
  kind: "main" | "sub";
  id: string;
  name: string;
  depth: number;
  parentId?: string;
}

export interface FakeSession {
  pi: ExtensionAPI;
  ctx: ExtensionContext;
  /** Dispatches like OMP: every handler for the event, awaited in order. Leave un-awaited to interleave. */
  fire: (name: string, event?: Record<string, unknown>) => Promise<void>;
  /** Session transcript the extension reads its user-turn number from. */
  entries: Array<{ type: string; message: Record<string, unknown> }>;
  handlers: Record<string, Handler[]>;
}

export interface FakeSessionOptions {
  agent?: AgentIdentity;
  sessionId?: string;
  cwd?: string;
  model?: { provider: string; id: string; api?: string; contextWindow?: number };
  thinkingLevel?: string;
}

/** A minimal OMP extension host: real handler registry, fake session state. */
export function createFakeSession(options: FakeSessionOptions = {}): FakeSession {
  const handlers: Record<string, Handler[]> = {};
  const entries: FakeSession["entries"] = [];
  const pi = {
    on: (name: string, handler: Handler) => {
      (handlers[name] ??= []).push(handler);
    },
    registerCommand: mock(),
    getThinkingLevel: () => options.thinkingLevel,
    pi: { VERSION: "18.8.9", getAgentDir: () => undefined },
  } as unknown as ExtensionAPI;

  const ctx = {
    ui: { setStatus: mock(), notify: mock() },
    hasUI: true,
    cwd: options.cwd ?? process.cwd(),
    agent: options.agent ?? { kind: "main", id: "Main", name: "main", depth: 0 },
    model: options.model ?? {
      provider: "anthropic",
      id: "claude-opus-4-8",
      api: "anthropic-messages",
      contextWindow: 1_000_000,
    },
    sessionManager: {
      getSessionId: () => options.sessionId ?? "session-1",
      getSessionName: () => "Test session",
      getEntries: () => entries,
      getBranch: () => entries,
    },
    getContextUsage: () => ({ tokens: 1000, contextWindow: 1_000_000, percent: 0.1 }),
  } as unknown as ExtensionContext;

  const dispatch = async (name: string, event: Record<string, unknown> = {}) => {
    for (const handler of handlers[name] ?? []) await handler({ ...event, type: name }, ctx);
  };
  return { pi, ctx, fire: dispatch, entries, handlers };
}

// ---------------------------------------------------------------------------
// Event builders shaped like OMP 18.8 payloads
// ---------------------------------------------------------------------------

export const assistant = (
  content: unknown[],
  extra: Record<string, unknown> = {},
): Record<string, unknown> => ({
  role: "assistant",
  content,
  api: "anthropic-messages",
  provider: "anthropic",
  model: "claude-opus-4-8",
  stopReason: "stop",
  usage: {
    input: 10,
    output: 5,
    cacheRead: 100,
    cacheWrite: 20,
    totalTokens: 135,
    cost: { input: 0.01, output: 0.02, cacheRead: 0.001, cacheWrite: 0.002, total: 0.033 },
  },
  ...extra,
});

export const anthropicPayload = (text: string) => ({
  model: "claude-opus-4-8",
  max_tokens: 32000,
  system: [{ type: "text", text: "You are omp." }],
  messages: [{ role: "user", content: [{ type: "text", text }] }],
  tools: [{ name: "bash", description: "Run a command", input_schema: { type: "object" } }],
});

/**
 * One full LLM call: request, provider response, streaming start, first token,
 * final message. `response: null` skips the provider response, as when the
 * request fails before one arrives.
 */
export async function llmCall(
  session: FakeSession,
  message: Record<string, unknown>,
  prompt = "hi",
  response: Record<string, unknown> | null = { status: 200, requestId: "req-1" },
) {
  await session.fire("before_provider_request", { payload: anthropicPayload(prompt) });
  if (response) await session.fire("after_provider_response", response);
  await session.fire("message_start", { message: { ...message, content: [] } });
  await session.fire("message_update", { message, assistantMessageEvent: { type: "text_delta" } });
  await session.fire("message_end", { message });
}
