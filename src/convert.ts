import { isRecord } from "./types.js";

/**
 * Pure conversions from OMP event payloads to LangSmith run shapes. Nothing in
 * this module touches tracing state, so every function is safe to call from any
 * handler and easy to test in isolation.
 */

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/**
 * OMP represents binary content (images read by the `read` tool or attached by
 * the user) as `{ type, mimeType, data }`. The LangSmith UI renders the
 * LangChain v1 multimodal block (`{ type, mime_type, base64 }`) inline, so that
 * shape is emitted instead. Anything else passes through untouched.
 */
export const normalizeContentPart = (part: unknown): unknown => {
  if (!isRecord(part)) return part;
  if (part.type !== "image") return part;
  if (typeof part.mimeType !== "string" || typeof part.data !== "string") return part;

  const { mimeType, data, ...rest } = part;
  return { ...rest, mime_type: mimeType, base64: data };
};

/** Assistant tool calls become LangSmith `tool_call` parts; images are normalized. Every other field is kept. */
const convertContentPart = (part: unknown): unknown => {
  if (isRecord(part) && part.type === "toolCall") {
    const { arguments: args, type: _, ...rest } = part;
    return { type: "tool_call", args, ...rest };
  }
  return normalizeContentPart(part);
};

const normalizeContent = (content: unknown): unknown =>
  Array.isArray(content) ? content.map(convertContentPart) : content;

/** OMP agent messages -> LangSmith chat messages (`toolResult` becomes `tool`), with every field kept. */
export const convertMessages = (messages: readonly unknown[]): Record<string, unknown>[] =>
  messages.filter(isRecord).map((message) => {
    const { role, content, ...rest } = message;
    return {
      role: role === "toolResult" ? "tool" : role,
      content: normalizeContent(content),
      ...rest,
    };
  });

/** Plain text of a message's content (string or text parts). */
export const extractText = (content: unknown): string => {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) =>
      isRecord(part) && part.type === "text" && typeof part.text === "string" ? part.text : "",
    )
    .join("");
};

/**
 * The messages a prompt produced: everything after the user message that carried
 * it. omp's `agent_end` reports the whole session context, which would repeat the
 * entire conversation on every turn. The last user message whose text equals the
 * prompt wins; a steer that quotes the prompt only contains it, so substring
 * matches are a fallback for user messages omp decorated with extra text. Without
 * any match (a continuation), the messages after the last user message are returned.
 */
export const messagesSincePrompt = (messages: readonly unknown[], prompt: string): unknown[] => {
  const isUser = (message: unknown): message is Record<string, unknown> =>
    isRecord(message) && message.role === "user";
  let start = -1;
  const needle = prompt.trim();
  if (needle) {
    const texts = messages.map((message) =>
      isUser(message) ? extractText(message.content).trim() : undefined,
    );
    start = texts.findLastIndex((text) => text === needle);
    if (start < 0) start = texts.findLastIndex((text) => text?.includes(needle) ?? false);
  }
  if (start < 0) start = messages.findLastIndex(isUser);
  return messages.slice(start + 1);
};

/** Whether streamed assistant content has produced its first visible token. */
export const hasFirstToken = (content: unknown): boolean => {
  if (!Array.isArray(content)) return typeof content === "string" && content.length > 0;
  return content.some(
    (part) =>
      isRecord(part) &&
      ((part.type === "text" && typeof part.text === "string" && part.text.length > 0) ||
        (part.type === "thinking" &&
          typeof part.thinking === "string" &&
          part.thinking.length > 0) ||
        part.type === "redactedThinking" ||
        part.type === "toolCall"),
  );
};

export const convertToolOutputs = (outputs: { result: unknown }): Record<string, unknown> => {
  if (isRecord(outputs.result) && outputs.result.content != null) {
    const { details, ...result } = outputs.result;
    return {
      output: { role: "tool", ...result, content: normalizeContent(result.content) },
      ...(details !== undefined ? { details } : {}),
    };
  }
  return { output: outputs.result };
};

/** First text of a failed tool result, used as the run's error message. */
export const toolErrorMessage = (result: unknown): string => {
  const content = isRecord(result) ? result.content : result;
  const text = (extractText(content) || (typeof content === "string" ? content : "")).trim();
  return text || "Tool execution failed";
};

/** Subagent names a tool call dispatches (OMP's `task` tool takes `tasks[].name`). */
export const spawnedAgentNames = (args: unknown): string[] => {
  if (!isRecord(args) || !Array.isArray(args.tasks)) return [];
  return args.tasks
    .filter(isRecord)
    .map((task) => task.name)
    .filter((name): name is string => typeof name === "string" && name.length > 0);
};

// ---------------------------------------------------------------------------
// Provider payloads
// ---------------------------------------------------------------------------

const omitKeys = <T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  ...keys: K[]
): Omit<T, K> => {
  const result = { ...obj };
  for (const key of keys) delete result[key];
  return result;
};

const systemMessage = (content: unknown): Record<string, unknown> => {
  // Anthropic/Gemini send system prompts as text blocks; a single block reads as plain text.
  const parts = Array.isArray(content) ? content : isRecord(content) ? [content] : [];
  const texts = parts.map((part) =>
    isRecord(part) && typeof part.text === "string" ? part.text : undefined,
  );
  const plain =
    texts.length > 0 && texts.every((text) => text !== undefined) ? texts.join("\n\n") : undefined;
  return { role: "system", content: plain ?? content };
};

const hasSystem = (messages: unknown[]): boolean =>
  messages.some((msg) => isRecord(msg) && (msg.role === "system" || msg.role === "developer"));

/**
 * Responses `input` items -> chat messages. Tool traffic travels as role-less
 * items (`function_call`, `function_call_output`), which LangSmith cannot
 * render as a conversation until they carry a role.
 */
const convertResponsesInput = (input: unknown[]): unknown[] =>
  input.map((maybe) => {
    if (!isRecord(maybe)) return maybe;
    if (maybe.type === "function_call_output" || maybe.type === "custom_tool_call_output") {
      const { type: _, call_id, output, ...rest } = maybe;
      return { role: "tool", tool_call_id: call_id, content: output, ...rest };
    }
    if (maybe.type === "function_call" || maybe.type === "custom_tool_call") {
      const { type: _, call_id, name, arguments: args, input: customInput, ...rest } = maybe;
      return {
        role: "assistant",
        content: "",
        tool_calls: [
          { id: call_id, type: "function", function: { name, arguments: args ?? customInput } },
        ],
        ...rest,
      };
    }
    if (maybe.role !== "user" || !Array.isArray(maybe.content)) return maybe;
    return {
      ...maybe,
      content: maybe.content.map((part) => {
        if (!isRecord(part) || part.type !== "input_text") return part;
        const { type: _, ...rest } = part;
        return { type: "text", ...rest };
      }),
    };
  });

const GEMINI_ROLES: Record<string, string> = { user: "user", model: "assistant" };

const convertGeminiPart = (part: unknown): unknown => {
  if (!isRecord(part)) return part;
  if (typeof part.text === "string") {
    return { type: "text", text: part.text, ...omitKeys(part, "text") };
  }
  if (isRecord(part.functionCall)) {
    return {
      type: "function_call",
      function_call: {
        id: part.functionCall.id ?? null,
        name: part.functionCall.name,
        arguments: part.functionCall.args,
      },
      ...omitKeys(part, "functionCall"),
    };
  }
  if (isRecord(part.functionResponse)) {
    return {
      type: "function_response",
      function_response: part.functionResponse,
      ...omitKeys(part, "functionResponse"),
    };
  }
  if (isRecord(part.imageUrl)) {
    return { type: "image_url", image_url: part.imageUrl, ...omitKeys(part, "imageUrl") };
  }
  return part;
};

const convertAnthropicMessages = (messages: unknown[]): unknown[] =>
  messages.flatMap((maybe) => {
    // Anthropic sends tool results as user messages made only of `tool_result` blocks.
    if (
      isRecord(maybe) &&
      maybe.role === "user" &&
      Array.isArray(maybe.content) &&
      maybe.content.length > 0 &&
      maybe.content.every((part) => isRecord(part) && part.type === "tool_result")
    ) {
      const { role: _, content, ...rest } = maybe;
      return content.filter(isRecord).map(({ type: __, ...part }) => ({
        role: "tool",
        ...rest,
        ...part,
      }));
    }
    return [maybe];
  });

/**
 * Normalizes the provider-native request body into LangSmith LLM inputs
 * (`{ messages, ...rest }`). Tool schemas and sampling settings are split out:
 * LangSmith renders them from the run's invocation params, so they are not
 * repeated in every run's inputs.
 */
export const convertProviderPayload = (
  payload: unknown,
): { inputs: Record<string, unknown>; tools?: unknown[] } => {
  if (!isRecord(payload)) return { inputs: { payload } };
  const tools = Array.isArray(payload.tools) ? payload.tools : undefined;
  const body = omitKeys(payload, "tools");

  // OpenAI Responses: `input` + `instructions`.
  if (Array.isArray(body.input) && body.messages == null) {
    const { input, instructions, ...rest } = body;
    const messages = convertResponsesInput(input as unknown[]);
    if (instructions != null && !hasSystem(messages)) messages.unshift(systemMessage(instructions));
    // A chained Codex websocket turn sends only the items after `previous_response_id`.
    const delta =
      typeof rest.previous_response_id === "string" && rest.previous_response_id.length > 0;
    return { inputs: { messages, ...rest, ...(delta ? { input_is_delta: true } : {}) }, tools };
  }

  // Gemini: `contents` + `systemInstruction`. The REST shape keeps the system
  // instruction, tools and generation settings at the top level; the
  // `@google/genai` SDK shape omp sends nests them under `config`.
  if (Array.isArray(body.contents) && body.messages == null) {
    const { contents, systemInstruction: topSystem, config, ...rest } = body;
    const nested = isRecord(config) ? config : undefined;
    const systemInstruction = topSystem ?? nested?.systemInstruction;
    const geminiTools = tools ?? (Array.isArray(nested?.tools) ? nested.tools : undefined);
    const messages = (contents as unknown[]).map((maybe) => {
      if (!isRecord(maybe) || typeof maybe.role !== "string" || !Array.isArray(maybe.parts)) {
        return maybe;
      }
      return {
        role: GEMINI_ROLES[maybe.role] ?? maybe.role,
        content: maybe.parts.map(convertGeminiPart),
        ...omitKeys(maybe, "role", "parts"),
      };
    });
    if (systemInstruction != null) {
      const parts = isRecord(systemInstruction) ? systemInstruction.parts : systemInstruction;
      messages.unshift(systemMessage(parts));
    }
    // `abortSignal` is a live object the SDK carries on the request, not a setting.
    const settings = nested ? omitKeys(nested, "systemInstruction", "tools", "abortSignal") : {};
    return {
      inputs: { messages, ...rest, ...(Object.keys(settings).length ? { config: settings } : {}) },
      tools: geminiTools,
    };
  }

  // OpenAI chat completions and Anthropic messages.
  if (Array.isArray(body.messages)) {
    const { messages: raw, system, ...rest } = body;
    const messages = convertAnthropicMessages(raw as unknown[]);
    if (system != null && !hasSystem(messages)) {
      messages.unshift({ ...systemMessage(system), _raw: system });
    }
    return { inputs: { messages, ...rest }, tools };
  }

  return { inputs: body, tools };
};

// ---------------------------------------------------------------------------
// Model parameters
// ---------------------------------------------------------------------------

const MAX_TOKENS_KEYS = [
  "max_tokens",
  "max_completion_tokens",
  "max_output_tokens",
  "maxOutputTokens",
];
const THINKING_BUDGET_KEYS = ["budget_tokens", "thinking_budget", "thinkingBudget"];
const SCALAR_PARAM_KEYS = [
  "temperature",
  "top_p",
  "topP",
  "top_k",
  "topK",
  "presence_penalty",
  "frequency_penalty",
  "seed",
  "reasoning_effort",
  "verbosity",
  "service_tier",
  "prompt_cache_retention",
];

/** Positive integer found under `keys`, searching nested objects up to `depth` levels. */
const findInteger = (bag: unknown, keys: string[], depth = 2): number | undefined => {
  if (!isRecord(bag)) return undefined;
  for (const key of keys) {
    const value = bag[key];
    if (typeof value === "number" && Number.isInteger(value) && value > 0) return value;
  }
  if (depth === 0) return undefined;
  for (const value of Object.values(bag)) {
    const found = findInteger(value, keys, depth - 1);
    if (found !== undefined) return found;
  }
  return undefined;
};

/**
 * Sampling and budget settings exactly as the provider dialects spell them on the
 * wire. OMP models do not declare their sampling parameters, so the payload is
 * the source of truth.
 */
export const extractModelParameters = (
  payload: unknown,
  thinkingLevel: string | undefined,
): Record<string, unknown> => {
  const params: Record<string, unknown> = {};
  if (!isRecord(payload)) return params;
  const maxTokens = findInteger(payload, MAX_TOKENS_KEYS);
  if (maxTokens !== undefined) params.max_tokens = maxTokens;
  const thinkingBudget = findInteger(payload, THINKING_BUDGET_KEYS);
  if (thinkingBudget !== undefined) params.thinking_budget_tokens = thinkingBudget;
  if (thinkingLevel && thinkingLevel !== "off") params.thinking_level = thinkingLevel;
  for (const key of SCALAR_PARAM_KEYS) {
    const value = payload[key];
    if (typeof value === "number" || typeof value === "string") params[key] = value;
  }
  const generation = payload.generationConfig ?? payload.config;
  if (isRecord(generation)) {
    for (const key of ["temperature", "topP", "topK", "seed"]) {
      const value = generation[key];
      if (typeof value === "number") params[key] ??= value;
    }
  }
  const stop = payload.stop ?? payload.stop_sequences;
  if (typeof stop === "string") params.stop = [stop];
  else if (Array.isArray(stop)) params.stop = stop.filter((s) => typeof s === "string");
  const toolChoice = payload.tool_choice;
  if (typeof toolChoice === "string") params.tool_choice = toolChoice;
  else if (isRecord(toolChoice) && typeof toolChoice.type === "string") {
    params.tool_choice = toolChoice.type;
  }
  return params;
};

// ---------------------------------------------------------------------------
// Usage and cost
// ---------------------------------------------------------------------------

/** The subset of OMP's `Usage` (`@oh-my-pi/pi-catalog`) the trace reports. */
export interface OmpUsage {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  totalTokens?: number;
  /** Subset of `output`; undefined means the provider did not report it. */
  reasoningTokens?: number;
  /** Anthropic cache-write TTL split; the components sum to `cacheWrite`. */
  cttl?: { ephemeral5m?: number; ephemeral1h?: number };
  /** Server-side tool requests (web search/fetch) billed on this call. */
  server?: { webSearch?: number; webFetch?: number };
  cost?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
    total?: number;
  };
}

export const isUsage = (value: unknown): value is OmpUsage =>
  isRecord(value) && (typeof value.input === "number" || typeof value.output === "number");

const sum = (...values: Array<number | undefined>): number | undefined => {
  let total: number | undefined;
  for (const value of values) {
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    total = (total ?? 0) + value;
  }
  return total;
};

const positive = (value: number | undefined): number | undefined =>
  typeof value === "number" && value > 0 ? value : undefined;

/**
 * OMP usage -> LangSmith `usage_metadata`. LangSmith's cost rollups read this
 * key from LLM runs only; it must never be stamped on parent chains, which
 * would double-count every token.
 */
export const buildUsageMetadata = (usage: OmpUsage): Record<string, unknown> => {
  const reasoning = positive(usage.reasoningTokens);
  const canSplitReasoning = reasoning !== undefined && reasoning <= (usage.output ?? 0);
  const ttl5m = positive(usage.cttl?.ephemeral5m);
  const ttl1h = positive(usage.cttl?.ephemeral1h);

  const inputDetails: Record<string, number> = {};
  if (usage.cacheRead !== undefined) inputDetails.cache_read = usage.cacheRead;
  if (ttl5m !== undefined || ttl1h !== undefined) {
    // LangSmith prices the TTL buckets separately; `cache_creation` would double-count them.
    if (ttl5m !== undefined) inputDetails.ephemeral_5m_input_tokens = ttl5m;
    if (ttl1h !== undefined) inputDetails.ephemeral_1hr_input_tokens = ttl1h;
  } else if (usage.cacheWrite !== undefined) {
    inputDetails.cache_creation = usage.cacheWrite;
  }

  const inputTokens = sum(usage.input, usage.cacheRead, usage.cacheWrite);
  const totalTokens = usage.totalTokens ?? sum(inputTokens, usage.output);
  const metadata: Record<string, unknown> = {
    input_tokens: inputTokens,
    output_tokens: usage.output,
    total_tokens: totalTokens,
    ...(Object.keys(inputDetails).length ? { input_token_details: inputDetails } : {}),
    ...(canSplitReasoning ? { output_token_details: { reasoning } } : {}),
  };

  // Tokens but zero cost means OMP has no pricing; omit cost so LangSmith prices it.
  const totalCost = usage.cost?.total;
  const costUnknown = (totalTokens ?? 0) > 0 && !(totalCost && totalCost > 0);
  if (!costUnknown && usage.cost) {
    const cost = usage.cost;
    metadata.input_cost = sum(cost.input, cost.cacheRead, cost.cacheWrite);
    metadata.output_cost = cost.output;
    metadata.total_cost = totalCost;
    metadata.input_cost_details = { cache_read: cost.cacheRead, cache_creation: cost.cacheWrite };
    if (canSplitReasoning && cost.output && usage.output) {
      metadata.output_cost_details = { reasoning: cost.output * (reasoning / usage.output) };
    }
  }
  return metadata;
};

/** Usage facts LangSmith has no usage key for; they ride along as metadata. */
export const usageExtras = (usage: OmpUsage): Record<string, number> => ({
  ...(usage.server?.webSearch ? { server_web_search_requests: usage.server.webSearch } : {}),
  ...(usage.server?.webFetch ? { server_web_fetch_requests: usage.server.webFetch } : {}),
});

// ---------------------------------------------------------------------------
// Redaction
// ---------------------------------------------------------------------------

const LANGSMITH_KEY = String.raw`\blsv2_[A-Za-z0-9_]+`;
const REDACTED = "[redacted-langsmith-key]";

/**
 * Deep redaction of LangSmith API keys (the configured ones and any `lsv2_`
 * token) so a trace never exports the credential that writes it. Only strings are
 * rewritten: arrays and plain objects are walked, anything else (dates, maps,
 * binary, class instances) passes through untouched. Cycles are cut rather than
 * followed.
 */
export const createRedactor = (...secrets: Array<string | undefined>) => {
  const alternatives = secrets
    .filter((secret): secret is string => typeof secret === "string" && secret.length >= 8)
    .map((secret) => secret.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  alternatives.push(LANGSMITH_KEY);
  const pattern = new RegExp(alternatives.join("|"), "g");
  const isPlainObject = (value: object): value is Record<string, unknown> => {
    const proto = Object.getPrototypeOf(value);
    return proto === Object.prototype || proto === null;
  };
  const walk = (value: unknown, ancestors: readonly object[]): unknown => {
    if (typeof value === "string") return value.replace(pattern, REDACTED);
    if (value === null || typeof value !== "object") return value;
    if (ancestors.includes(value)) return "[circular-ref]";
    const chain = [...ancestors, value];
    if (Array.isArray(value)) return value.map((item) => walk(item, chain));
    if (!isPlainObject(value)) return value;
    const out: Record<string, unknown> = {};
    for (const [key, field] of Object.entries(value)) out[key] = walk(field, chain);
    return out;
  };
  return <T>(value: T): T => walk(value, []) as T;
};
