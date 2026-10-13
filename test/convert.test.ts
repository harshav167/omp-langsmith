import { describe, expect, it } from "bun:test";

import {
  buildUsageMetadata,
  convertMessages,
  convertProviderPayload,
  convertToolOutputs,
  createRedactor,
  extractModelParameters,
  messagesSincePrompt,
  normalizeContentPart,
} from "../src/convert";

// OMP's native binary-content shape and its LangChain v1 equivalent.
const ompImage = { type: "image", mimeType: "image/png", data: "iVBORw0KGgoBASE64" };
const lcImage = { type: "image", mime_type: "image/png", base64: "iVBORw0KGgoBASE64" };

describe("messages", () => {
  it("converts OMP image parts to the LangChain v1 shape and leaves others alone", () => {
    expect(normalizeContentPart(ompImage)).toEqual(lcImage);
    const text = { type: "text", text: "hello" };
    expect(normalizeContentPart(text)).toBe(text);
    expect(normalizeContentPart({ type: "image", mimeType: "image/png" })).toEqual({
      type: "image",
      mimeType: "image/png",
    });
  });

  it("maps toolResult to tool, tool calls to tool_call, and normalizes images", () => {
    const [toolResult, call] = convertMessages([
      { role: "toolResult", toolName: "read", content: [{ type: "text", text: "img" }, ompImage] },
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "1", name: "read", arguments: { path: "a.png" } }],
      },
    ]);
    expect(toolResult!.role).toBe("tool");
    expect(toolResult!.content).toEqual([{ type: "text", text: "img" }, lcImage]);
    expect(call!.content).toEqual([
      { type: "tool_call", id: "1", name: "read", args: { path: "a.png" } },
    ]);
  });

  it("keeps tool details next to the tool message output", () => {
    expect(convertToolOutputs({ result: { content: [ompImage], details: { lines: 3 } } })).toEqual({
      output: { role: "tool", content: [lcImage] },
      details: { lines: 3 },
    });
    expect(convertToolOutputs({ result: "ok" })).toEqual({ output: "ok" });
  });
});

describe("provider payloads", () => {
  it("splits Anthropic system and tools out of the message inputs", () => {
    const { inputs, tools } = convertProviderPayload({
      model: "claude",
      system: [{ type: "text", text: "Be terse." }],
      messages: [
        { role: "user", content: [{ type: "text", text: "hi" }] },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "out" }] },
      ],
      tools: [{ name: "bash" }],
    });
    expect(tools).toEqual([{ name: "bash" }]);
    expect(inputs.tools).toBeUndefined();
    expect(inputs.messages).toEqual([
      { role: "system", content: "Be terse.", _raw: [{ type: "text", text: "Be terse." }] },
      { role: "user", content: [{ type: "text", text: "hi" }] },
      { role: "tool", tool_use_id: "t1", content: "out" },
    ]);
  });

  it("reads OpenAI Responses instructions as the system message", () => {
    const { inputs } = convertProviderPayload({
      instructions: "System rules.",
      input: [{ role: "user", content: [{ type: "input_text", text: "go" }] }],
    });
    expect(inputs.messages).toEqual([
      { role: "system", content: "System rules." },
      { role: "user", content: [{ type: "text", text: "go" }] },
    ]);
  });

  it("gives role-less Responses tool items a chat role and flags chained deltas", () => {
    const { inputs } = convertProviderPayload({
      type: "response.create",
      previous_response_id: "resp_1",
      input: [
        { type: "function_call", call_id: "c1", name: "read", arguments: '{"path":"a"}' },
        { type: "function_call_output", call_id: "c1", output: "file body" },
      ],
    });
    expect(inputs.messages).toEqual([
      {
        role: "assistant",
        content: "",
        tool_calls: [
          { id: "c1", type: "function", function: { name: "read", arguments: '{"path":"a"}' } },
        ],
      },
      { role: "tool", tool_call_id: "c1", content: "file body" },
    ]);
    expect(inputs.input_is_delta).toBe(true);
  });

  it("maps Gemini contents and systemInstruction", () => {
    const { inputs } = convertProviderPayload({
      systemInstruction: { parts: [{ text: "Be kind." }] },
      contents: [
        { role: "user", parts: [{ text: "q" }] },
        { role: "model", parts: [{ functionCall: { name: "ls", args: { dir: "." } } }] },
      ],
    });
    expect(inputs.messages).toEqual([
      { role: "system", content: "Be kind." },
      { role: "user", content: [{ type: "text", text: "q" }] },
      {
        role: "assistant",
        content: [
          {
            type: "function_call",
            function_call: { id: null, name: "ls", arguments: { dir: "." } },
          },
        ],
      },
    ]);
  });

  it("reads the Gemini SDK shape, which nests system, tools and settings under config", () => {
    const tools = [{ functionDeclarations: [{ name: "read", parameters: {} }] }];
    const { inputs, tools: extracted } = convertProviderPayload({
      model: "gemini",
      contents: [{ role: "user", parts: [{ text: "q" }] }],
      config: {
        systemInstruction: "Be kind.",
        tools,
        thinkingConfig: { thinkingBudget: 1024 },
        abortSignal: new AbortController().signal,
      },
    });
    expect(extracted).toBe(tools);
    expect(inputs.messages).toEqual([
      { role: "system", content: "Be kind." },
      { role: "user", content: [{ type: "text", text: "q" }] },
    ]);
    expect(inputs.config).toEqual({ thinkingConfig: { thinkingBudget: 1024 } });
    expect(
      extractModelParameters(
        {
          config: {
            temperature: 0.1,
            maxOutputTokens: 2048,
            thinkingConfig: { thinkingBudget: 1024 },
          },
        },
        undefined,
      ),
    ).toEqual({ max_tokens: 2048, thinking_budget_tokens: 1024, temperature: 0.1 });
  });

  it("starts the prompt's messages at the prompt itself, not at a steer that quotes it", () => {
    const user = (text: string) => ({ role: "user", content: [{ type: "text", text }] });
    const messages = [
      user("fix tests"),
      { role: "assistant", content: "working" },
      user("when you fix tests, keep the API"),
      { role: "assistant", content: "done" },
    ];
    expect(messagesSincePrompt(messages, "fix tests")).toEqual(messages.slice(1));
    // A decorated user message still matches by substring when nothing matches exactly.
    expect(messagesSincePrompt(messages.slice(2), "fix tests")).toEqual(messages.slice(3));
  });

  it("extracts sampling settings as the provider spelled them", () => {
    expect(
      extractModelParameters(
        {
          max_tokens: 4096,
          temperature: 0.2,
          thinking: { type: "enabled", budget_tokens: 2048 },
          stop_sequences: ["END"],
          tool_choice: { type: "auto" },
          messages: [],
        },
        "high",
      ),
    ).toEqual({
      max_tokens: 4096,
      thinking_budget_tokens: 2048,
      thinking_level: "high",
      temperature: 0.2,
      stop: ["END"],
      tool_choice: "auto",
    });
  });
});

describe("usage", () => {
  it("omits cost when OMP has no pricing so LangSmith prices the run", () => {
    const out = buildUsageMetadata({
      input: 2,
      output: 246,
      cacheRead: 2755,
      cacheWrite: 98,
      totalTokens: 3101,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    });
    expect(out).toEqual({
      input_tokens: 2855,
      output_tokens: 246,
      total_tokens: 3101,
      input_token_details: { cache_read: 2755, cache_creation: 98 },
    });
  });

  it("reports reasoning tokens and splits their cost", () => {
    const out = buildUsageMetadata({
      input: 10,
      output: 100,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 110,
      reasoningTokens: 40,
      cost: { input: 0.01, output: 0.1, cacheRead: 0, cacheWrite: 0, total: 0.11 },
    });
    expect(out.output_token_details).toEqual({ reasoning: 40 });
    expect(out.output_cost_details).toEqual({ reasoning: 0.1 * 0.4 });
    expect(out.total_cost).toBe(0.11);
  });

  it("prices Anthropic cache-write TTL buckets separately instead of cache_creation", () => {
    const out = buildUsageMetadata({
      input: 1,
      output: 1,
      cacheRead: 0,
      cacheWrite: 300,
      totalTokens: 302,
      cttl: { ephemeral5m: 100, ephemeral1h: 200 },
    });
    expect(out.input_token_details).toEqual({
      cache_read: 0,
      ephemeral_5m_input_tokens: 100,
      ephemeral_1hr_input_tokens: 200,
    });
  });

  it("omits a zero cost for output-only usage that reports no total", () => {
    const out = buildUsageMetadata({ input: 0, output: 100, cost: { total: 0 } });
    expect(out.total_tokens).toBe(100);
    expect(out.total_cost).toBeUndefined();
  });
});

describe("redaction", () => {
  it("removes LangSmith keys from nested values without following cycles", () => {
    const redact = createRedactor("secret-config-key-123");
    const value: Record<string, unknown> = {
      text: "token lsv2_pt_abc123_def and secret-config-key-123",
      nested: [{ key: "lsv2_sk_zzz" }],
    };
    value.self = value;
    expect(redact(value)).toEqual({
      text: "token [redacted-langsmith-key] and [redacted-langsmith-key]",
      nested: [{ key: "[redacted-langsmith-key]" }],
      self: "[circular-ref]",
    });
  });

  it("leaves non-plain objects alone instead of flattening them", () => {
    const redact = createRedactor();
    const when = new Date("2026-01-01T00:00:00Z");
    const bytes = new Uint8Array([1, 2]);
    expect(redact({ when, bytes, tags: new Set(["a"]) })).toEqual({
      when,
      bytes,
      tags: new Set(["a"]),
    });
  });
});
