import { afterEach, describe, expect, it, setSystemTime, vi } from "bun:test";
import type { Client } from "langsmith";

import extension, { flushTraces } from "../src/index";
import { CONTINUATION_GRACE_MS, RESTART_GRACE_MS } from "../src/tracer";
import { assistant, createFakeSession, type FakeSession, llmCall } from "./utils/harness";
import { mockClient } from "./utils/mock_client";
import { getAssumedTreeFromCalls } from "./utils/tree";

type Run = Record<string, any>;

const config = { enabled: true, project: "omp-test" };
const sessions: FakeSession[] = [];

async function start(options: Parameters<typeof createFakeSession>[0] = {}, client?: Client) {
  const mocked = client ? { client, callSpy: undefined } : mockClient();
  const session = createFakeSession(options);
  await extension(session.pi, { client: mocked.client, config });
  sessions.push(session);
  return { session, ...mocked, client: mocked.client };
}

afterEach(async () => {
  // Module-level registries are shared across sessions, like in OMP.
  for (const session of sessions.splice(0)) await session.fire("session_shutdown");
});

/** Waits for the extension's queued exports, then reads back the exported tree. */
async function treeOf(client: Client, callSpy: { mock: { calls: unknown[][] } }) {
  await flushTraces();
  return getAssumedTreeFromCalls(callSpy.mock.calls, client);
}

const runsNamed = (tree: { data: Record<string, Run> }, name: string): Run[] =>
  Object.entries(tree.data)
    .filter(([key]) => key.split(":")[0] === name)
    .map(([, run]) => run);

const one = (tree: { data: Record<string, Run> }, name: string): Run => {
  const runs = runsNamed(tree, name);
  expect(runs).toHaveLength(1);
  return runs[0]!;
};

/** No run may be left pending: every posted run must be patched with an end time. */
const expectAllEnded = (tree: { data: Record<string, Run> }) => {
  const pending = Object.entries(tree.data)
    .filter(([, run]) => run.end_time == null)
    .map(([key]) => key);
  expect(pending).toEqual([]);
};

async function prompt(session: FakeSession, text: string) {
  session.entries.push({
    type: "message",
    message: { role: "user", content: [{ type: "text", text }] },
  });
  await session.fire("before_agent_start", { prompt: text, systemPrompt: ["You are omp."] });
  await session.fire("agent_start");
}

describe("prompt lifecycle", () => {
  it("traces a prompt as one root with steps, LLM calls and tools, and ends it at agent_end", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "list files");
    await session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    const call = assistant(
      [
        { type: "text", text: "Listing." },
        { type: "toolCall", id: "call_1", name: "bash", arguments: { command: "ls" } },
      ],
      { stopReason: "toolUse", usage: { ...(assistant([]).usage as object), reasoningTokens: 2 } },
    );
    await llmCall(session, call, "list files");
    await session.fire("tool_execution_start", {
      toolCallId: "call_1",
      toolName: "bash",
      args: { command: "ls" },
    });
    await session.fire("tool_execution_update", {
      toolCallId: "call_1",
      toolName: "bash",
      args: {},
      partialResult: { content: [{ type: "text", text: "README.md" }] },
    });
    await session.fire("tool_execution_end", {
      toolCallId: "call_1",
      toolName: "bash",
      result: { content: [{ type: "text", text: "README.md" }] },
      isError: false,
    });
    await session.fire("turn_end", { turnIndex: 0, message: call, toolResults: [] });
    await session.fire("turn_start", { turnIndex: 1, timestamp: Date.now() });
    const final = assistant([{ type: "text", text: "One file." }]);
    await llmCall(session, final);
    await session.fire("turn_end", { turnIndex: 1, message: final, toolResults: [] });
    await session.fire("agent_end", { messages: [call, final] });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const root = one(tree, "omp-agent");
    expect(root.parent_run_id).toBeUndefined();
    expect(root.error).toBeUndefined();
    expect(root.inputs.prompt).toBe("list files");
    expect(root.outputs.output).toBe("One file.");
    expect(runsNamed(tree, "agent-step")).toHaveLength(2);

    const md = root.extra.metadata;
    expect(md).toMatchObject({
      ls_integration: "pi",
      ls_agent_runtime: "omp",
      ls_agent_runtime_version: "18.8.9",
      ls_agent_type: "root",
      ls_trace_schema_version: "coding-agent-v1",
      thread_id: "session-1",
      session_id: "session-1",
      session_name: "Test session",
      turn_number: 1,
      agent_kind: "main",
    });

    const [first] = runsNamed(tree, "anthropic");
    expect(first!.run_type).toBe("llm");
    expect(first!.inputs.messages[0]).toMatchObject({ role: "system", content: "You are omp." });
    expect(first!.inputs.tools).toBeUndefined();
    expect(first!.extra.invocation_params.tools).toHaveLength(1);
    expect(first!.extra.invocation_params.max_tokens).toBe(32000);
    expect(first!.extra.metadata).toMatchObject({
      ls_provider: "anthropic",
      ls_model_name: "claude-opus-4-8",
      ls_max_tokens: 32000,
      stop_reason: "toolUse",
    });
    expect(first!.extra.metadata.usage_metadata).toMatchObject({
      input_tokens: 130,
      output_tokens: 5,
      output_token_details: { reasoning: 2 },
      input_token_details: { cache_read: 100, cache_creation: 20 },
      total_cost: 0.033,
    });
    expect(first!.events.some((event: Run) => event.name === "new_token")).toBe(true);

    const tool = one(tree, "bash");
    expect(tool.run_type).toBe("tool");
    expect(tool.inputs).toEqual({ command: "ls" });
    expect(tool.extra.metadata.tool_call_id).toBe("call_1");
    expect(tool.events.some((event: Run) => event.name === "tool_update")).toBe(true);
    // Usage belongs on LLM runs only; parent chains would double-count it.
    expect(root.extra.metadata.usage_metadata).toBeUndefined();
  });

  it("keeps a mid-run steer inside the running trace instead of replacing it", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "do a long task");
    await session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    await session.fire("before_agent_start", { prompt: "also check the tests", systemPrompt: [] });
    const reply = assistant([{ type: "text", text: "Checked both." }]);
    await llmCall(session, reply);
    await session.fire("turn_end", { turnIndex: 0, message: reply, toolResults: [] });
    await session.fire("agent_end", { messages: [reply] });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const root = one(tree, "omp-agent");
    expect(root.error).toBeUndefined();
    expect(root.extra.metadata.steer_count).toBe(1);
    const steer = one(tree, "steer");
    expect(steer.parent_run_id).toBe(root.id);
    expect(steer.inputs.messages[0].content).toBe("also check the tests");
  });

  it("survives OMP interleaving a steer with turn_end and agent_end", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "first");
    await session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    const reply = assistant([{ type: "text", text: "done" }]);
    await llmCall(session, reply);
    // OMP does not serialize these dispatches; the old extension crashed here
    // reading `active.currentTurn` after another handler cleared it.
    await Promise.all([
      session.fire("turn_end", { turnIndex: 0, message: reply, toolResults: [] }),
      session.fire("before_agent_start", { prompt: "steer", systemPrompt: [] }),
      session.fire("agent_end", { messages: [reply] }),
    ]);

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    expect(runsNamed(tree, "omp-agent")).toHaveLength(1);
  });

  it("records a user interrupt as interrupted, never as an error", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "run the build");
    await session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    const call = assistant([{ type: "toolCall", id: "call_9", name: "bash", arguments: {} }], {
      stopReason: "toolUse",
    });
    await llmCall(session, call);
    await session.fire("tool_execution_start", {
      toolCallId: "call_9",
      toolName: "bash",
      args: { command: "make" },
    });
    vi.useFakeTimers();
    try {
      const aborted = assistant([], { stopReason: "aborted" });
      await session.fire("agent_end", { messages: [call, aborted] });
      // OMP may still restart the loop; with no restart the root settles after the grace.
      vi.advanceTimersByTime(RESTART_GRACE_MS + 1);
    } finally {
      vi.useRealTimers();
    }

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const root = one(tree, "omp-agent");
    expect(root.error).toBeUndefined();
    expect(root.tags).toContain("interrupted");
    expect(root.extra.metadata.interrupted).toBe(true);
    const bash = one(tree, "bash");
    expect(bash.error).toBeUndefined();
    expect(bash.extra.metadata.interrupted).toBe(true);
  });

  it("keeps a steer OMP delivers by aborting and restarting the loop in one trace", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "audit the plugin");
    await session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    const first = assistant([{ type: "text", text: "Reading…" }], { stopReason: "aborted" });
    await llmCall(session, first);
    // The sequence observed live: the steer is dequeued while the loop runs, the
    // restarted loop starts, and only then does the old loop's fire-and-forget end arrive.
    await session.fire("before_agent_start", {
      prompt: "interrupts are not errors",
      systemPrompt: [],
    });
    await session.fire("agent_start");
    await session.fire("agent_end", { messages: [first] });
    await session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    const reply = assistant([{ type: "text", text: "Fixed." }]);
    await llmCall(session, reply);
    await session.fire("agent_end", { messages: [first, reply] });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const root = one(tree, "omp-agent");
    expect(root.error).toBeUndefined();
    expect(root.outputs.output).toBe("Fixed.");
    expect(root.extra.metadata).toMatchObject({ steer_count: 1, interrupted: true });
    expect(runsNamed(tree, "anthropic")).toHaveLength(2);
    expect(runsNamed(tree, "anthropic").every((run) => run.parent_run_id !== undefined)).toBe(true);
    expect(runsNamed(tree, "anthropic").every((run) => run.error === undefined)).toBe(true);
  });

  it("keeps the restarted loop's tools under its step when the aborted turn_end arrives late", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "check traces");
    await session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    const aborted = assistant([], { stopReason: "aborted" });
    await llmCall(session, aborted);
    await session.fire("before_agent_start", { prompt: "see if it updated", systemPrompt: [] });
    await session.fire("agent_start");
    await session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    // Order observed live: the old loop's turn_end and agent_end land after the new step opened.
    await session.fire("turn_end", { turnIndex: 0, message: aborted, toolResults: [] });
    await session.fire("agent_end", { messages: [aborted] });
    await session.fire("tool_execution_start", { toolCallId: "t1", toolName: "eval", args: {} });
    await session.fire("tool_execution_end", {
      toolCallId: "t1",
      toolName: "eval",
      result: {},
      isError: false,
    });
    const reply = assistant([{ type: "text", text: "Updated." }]);
    await llmCall(session, reply);
    await session.fire("turn_end", { turnIndex: 0, message: reply, toolResults: [] });
    await session.fire("agent_end", { messages: [aborted, reply] });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const steps = runsNamed(tree, "agent-step");
    const restarted = steps.find((step) => !step.extra.metadata.superseded);
    expect(one(tree, "eval").parent_run_id).toBe(restarted!.id);
    expect(
      runsNamed(tree, "anthropic").every((run) => run.parent_run_id !== one(tree, "omp-agent").id),
    ).toBe(true);
  });

  it("reports only this prompt's messages, not the whole session, in the root output", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "second question");
    const history = [
      { role: "user", content: [{ type: "text", text: "first question" }] },
      assistant([{ type: "text", text: "first answer" }]),
      { role: "user", content: [{ type: "text", text: "second question" }] },
      assistant([{ type: "text", text: "second answer" }]),
    ];
    await session.fire("agent_end", { messages: history });

    const tree = await treeOf(client, callSpy!);
    const root = one(tree, "omp-agent");
    expect(root.outputs.messages).toHaveLength(1);
    expect(root.outputs.messages[0].content[0].text).toBe("second answer");
  });

  it("joins a steer restart that arrives just after the aborted end", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "first");
    vi.useFakeTimers();
    try {
      await session.fire("agent_end", { messages: [assistant([], { stopReason: "aborted" })] });
      vi.advanceTimersByTime(RESTART_GRACE_MS / 2);
      await session.fire("before_agent_start", { prompt: "steer", systemPrompt: [] });
      await session.fire("agent_start");
    } finally {
      vi.useRealTimers();
    }
    await session.fire("agent_end", { messages: [assistant([{ type: "text", text: "done" }])] });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const root = one(tree, "omp-agent");
    expect(root.extra.metadata).toMatchObject({ steer_count: 1, continuation_count: 1 });
    expect(root.error).toBeUndefined();
  });

  it("joins an automatic continuation (willContinue) to the same root", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "go");
    const failed = assistant([], { stopReason: "error", errorMessage: "overloaded" });
    await llmCall(session, failed);
    await session.fire("agent_end", { messages: [failed], willContinue: true });
    await session.fire("auto_retry_start", {
      attempt: 1,
      maxAttempts: 3,
      delayMs: 10,
      errorMessage: "overloaded",
    });
    await session.fire("agent_start");
    const ok = assistant([{ type: "text", text: "recovered" }]);
    await llmCall(session, ok);
    await session.fire("auto_retry_end", { success: true, attempt: 1 });
    await session.fire("agent_end", { messages: [ok] });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const root = one(tree, "omp-agent");
    expect(root.extra.metadata.continuation_count).toBe(1);
    expect(root.outputs.output).toBe("recovered");
    // A retry that recovered leaves only the failed call marked, not the prompt.
    expect(root.error).toBeUndefined();
    expect(runsNamed(tree, "anthropic")[0]!.error).toBe("overloaded");
    const retry = one(tree, "retry-request");
    expect(retry.error).toBeUndefined();
    expect(retry.inputs.error_message).toBe("overloaded");
    expect(one(tree, "settle-retry").outputs).toEqual({ success: true });
  });

  it("numbers user turns from the session transcript", async () => {
    const { session, client, callSpy } = await start();
    for (const text of ["first", "second"]) {
      await prompt(session, text);
      await session.fire("agent_end", { messages: [] });
    }
    const tree = await treeOf(client, callSpy!);
    expect(runsNamed(tree, "omp-agent").map((run) => run.extra.metadata.turn_number)).toEqual([
      1, 2,
    ]);
  });

  it("numbers user turns on the active branch, not across abandoned ones", async () => {
    const { session, client, callSpy } = await start();
    const user = (text: string) => ({
      type: "message",
      message: { role: "user", content: [{ type: "text", text }] },
    });
    // The session stores an abandoned branch's prompt; only the active path counts.
    session.entries.push(user("first"), user("abandoned"));
    const manager = session.ctx.sessionManager as { getBranch: () => unknown[] };
    manager.getBranch = () => [session.entries[0], ...session.entries.slice(2)];
    await prompt(session, "second");
    await session.fire("agent_end", { messages: [] });
    const tree = await treeOf(client, callSpy!);
    expect(one(tree, "omp-agent").extra.metadata.turn_number).toBe(2);
  });

  it("ends a root whose promised continuation never arrives, at the time its loop ended", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "go");
    const endedAt = Date.now();
    setSystemTime(new Date(endedAt));
    vi.useFakeTimers();
    try {
      await session.fire("agent_end", { messages: [], willContinue: true });
      vi.advanceTimersByTime(CONTINUATION_GRACE_MS + 1);
    } finally {
      vi.useRealTimers();
      setSystemTime();
    }
    // Flushing needs real timers: the client's batch queue drains on them.
    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    expect(one(tree, "omp-agent").extra.metadata.continuation_count).toBeUndefined();
    expect(new Date(one(tree, "omp-agent").end_time).getTime()).toBe(endedAt);
  });
});

describe("LLM call pairing", () => {
  it("binds the reply to its request when an OMP side request overlaps the stream", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "hello");
    await session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    const reply = assistant([{ type: "text", text: "Hi." }]);
    await session.fire("before_provider_request", {
      payload: { model: "m", messages: [{ role: "user", content: "hello" }] },
    });
    await session.fire("message_start", { message: { ...reply, content: [] } });
    // An omp side request (prewalk, memory recall) fires while the main reply streams.
    await session.fire("before_provider_request", {
      payload: { model: "m", messages: [{ role: "user", content: "side request" }] },
    });
    await session.fire("message_end", { message: reply });
    await session.fire("turn_end", { turnIndex: 0, message: reply, toolResults: [] });
    await session.fire("agent_end", { messages: [reply] });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const [main, side] = runsNamed(tree, "anthropic");
    expect(main!.inputs.messages[0].content).toBe("hello");
    expect(main!.outputs.messages[0].content[0].text).toBe("Hi.");
    expect(main!.extra.metadata.usage_metadata).toBeDefined();
    expect(side!.inputs.messages[0].content).toBe("side request");
    expect(side!.outputs).toBeUndefined();
    expect(side!.extra.metadata.usage_metadata).toBeUndefined();
    expect(side!.extra.metadata.ls_omp_unmatched_request).toBe(true);
    expect(main!.extra.metadata.omp_usage).toMatchObject({ input: 10, output: 5 });
  });

  it("binds the reply to its request when a side request arrives before streaming starts", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "hello");
    const sentAt = Date.now();
    setSystemTime(new Date(sentAt));
    try {
      await session.fire("before_provider_request", {
        payload: { model: "m", messages: [{ role: "user", content: "hello" }] },
      });
      setSystemTime(new Date(sentAt + 50));
      await session.fire("before_provider_request", {
        payload: { model: "m", messages: [{ role: "user", content: "side request" }] },
      });
    } finally {
      setSystemTime();
    }
    // omp stamps the assistant message with the time it sent the main request.
    const reply = assistant([{ type: "text", text: "Hi." }], { timestamp: sentAt });
    await session.fire("message_start", { message: { ...reply, content: [] } });
    await session.fire("message_end", { message: reply });
    await session.fire("agent_end", { messages: [reply] });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const main = runsNamed(tree, "anthropic").find(
      (run) => run.inputs.messages[0].content === "hello",
    );
    const side = runsNamed(tree, "anthropic").find(
      (run) => run.inputs.messages[0].content === "side request",
    );
    expect(main!.outputs.messages[0].content[0].text).toBe("Hi.");
    expect(side!.outputs).toBeUndefined();
    expect(side!.extra.metadata.ls_omp_unmatched_request).toBe(true);
  });

  it("traces omp's session-title fork as generate-title without letting it steal the reply", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "hello");
    await session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    await session.fire("before_provider_request", {
      payload: { model: "m", messages: [{ role: "user", content: "hello" }] },
    });
    // The title request lands before the reply starts streaming: the pairing hazard.
    await session.fire("before_provider_request", {
      payload: {
        model: "m",
        messages: [
          { role: "user", content: "hello" },
          { role: "user", content: "<title-request>\nName this session" },
        ],
      },
    });
    const reply = assistant([{ type: "text", text: "Hi." }]);
    await session.fire("message_start", { message: { ...reply, content: [] } });
    await session.fire("message_end", { message: reply });
    await session.fire("turn_end", { turnIndex: 0, message: reply, toolResults: [] });
    await session.fire("agent_end", { messages: [reply] });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const main = one(tree, "anthropic");
    expect(main.inputs.messages).toHaveLength(1);
    expect(main.outputs.messages[0].content[0].text).toBe("Hi.");
    const title = one(tree, "generate-title");
    expect(title.outputs).toEqual({ title: "Test session" });
    expect(title.extra.metadata.ls_omp_side_request).toBe("session_title");
    expect(one(tree, "omp-agent").extra.metadata.session_name).toBe("Test session");
  });

  it("uses the provider-reported time to first token and served model", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "hi");
    await session.fire("before_provider_request", { payload: { messages: [] } });
    await session.fire("message_end", {
      message: assistant([{ type: "text", text: "ok" }], {
        ttft: 120,
        upstreamModel: "claude-opus-4-8-20260101",
      }),
    });
    await session.fire("agent_end", { messages: [] });

    const tree = await treeOf(client, callSpy!);
    const llm = one(tree, "anthropic");
    expect(llm.extra.metadata).toMatchObject({
      ttft_ms: 120,
      ls_model_name: "claude-opus-4-8-20260101",
      requested_model: "claude-opus-4-8",
    });
    expect(llm.events.filter((event: Run) => event.name === "new_token")).toHaveLength(1);
  });

  it("gives an idle cache-warming refresh its own trace", async () => {
    const { session, client, callSpy } = await start();
    await session.fire("cache_warming_decision", {
      warmCost: 0.01,
      missCost: 0.5,
      continuationProbability: 0.7,
      action: "warm",
    });
    await session.fire("before_provider_request", { payload: { messages: [] } });
    await session.fire("after_provider_response", { status: 200, headers: {} });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const warm = one(tree, "warm-cache");
    expect(warm.parent_run_id).toBeUndefined();
    expect(warm.extra.metadata).toMatchObject({ http_status: 200, background_request: true });
    expect(warm.extra.metadata.cache_warming.continuation_probability).toBe(0.7);
  });
});

describe("tools, subagents and maintenance", () => {
  it("nests an in-process subagent under the task tool that spawned it", async () => {
    const main = await start();
    await prompt(main.session, "delegate");
    await main.session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    await main.session.fire("tool_execution_start", {
      toolCallId: "call_task",
      toolName: "task",
      args: { tasks: [{ name: "Explorer", task: "look around" }] },
    });

    const sub = await start(
      {
        sessionId: "session-2",
        agent: { kind: "sub", id: "0-Explorer", name: "explore", depth: 1, parentId: "Main" },
      },
      main.client,
    );
    await prompt(sub.session, "look around");
    const subReply = assistant([{ type: "text", text: "found it" }]);
    await llmCall(sub.session, subReply);
    await sub.session.fire("agent_end", { messages: [subReply] });

    await main.session.fire("tool_execution_end", {
      toolCallId: "call_task",
      toolName: "task",
      result: { content: [{ type: "text", text: "found it" }] },
      isError: false,
    });
    await main.session.fire("agent_end", { messages: [] });

    const tree = await treeOf(main.client, main.callSpy!);
    expectAllEnded(tree);
    const root = one(tree, "omp-agent");
    const task = one(tree, "task");
    const subRoot = one(tree, "explore-agent");
    expect(subRoot.parent_run_id).toBe(task.id);
    expect(subRoot.trace_id).toBe(root.trace_id);
    expect(subRoot.extra.metadata).toMatchObject({
      ls_agent_type: "subagent",
      ls_subagent_id: "0-Explorer",
      ls_subagent_type: "explore",
      thread_id: "session-1",
      agent_session_id: "session-2",
      agent_id: "0-Explorer",
      parent_agent_id: "Main",
    });
    expect(task.extra.metadata.ls_tool_name).toBe("task");
    expect(subRoot.tags).toContain("subagent");
  });

  it("nests a subagent under its task tool even after the async task returned", async () => {
    const main = await start();
    await prompt(main.session, "delegate");
    await main.session.fire("turn_start", { turnIndex: 0, timestamp: Date.now() });
    await main.session.fire("tool_execution_start", {
      toolCallId: "call_task",
      toolName: "task",
      args: { tasks: [{ name: "Reader", task: "read package.json" }] },
    });
    // OMP's task tool returns as soon as the job is spawned.
    await main.session.fire("tool_execution_end", {
      toolCallId: "call_task",
      toolName: "task",
      result: { content: [{ type: "text", text: "Spawned agent `Reader`" }] },
      isError: false,
    });
    await main.session.fire("turn_start", { turnIndex: 1, timestamp: Date.now() });
    await main.session.fire("tool_execution_start", {
      toolCallId: "call_wait",
      toolName: "wait",
      args: {},
    });

    const sub = await start(
      {
        sessionId: "session-2",
        agent: { kind: "sub", id: "Reader", name: "task", depth: 1, parentId: "Main" },
      },
      main.client,
    );
    await prompt(sub.session, "read package.json");
    await sub.session.fire("agent_end", { messages: [] });
    await main.session.fire("tool_execution_end", {
      toolCallId: "call_wait",
      toolName: "wait",
      result: {},
      isError: false,
    });
    await main.session.fire("agent_end", { messages: [] });

    const tree = await treeOf(main.client, main.callSpy!);
    expect(one(tree, "task-agent").parent_run_id).toBe(one(tree, "task").id);
  });

  it("reports a failed tool's message and attributes tool-internal LLM usage", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "search");
    await session.fire("tool_execution_start", {
      toolCallId: "t1",
      toolName: "web_search",
      args: { query: "x" },
    });
    await session.fire("tool_execution_end", {
      toolCallId: "t1",
      toolName: "web_search",
      result: {
        content: [{ type: "text", text: "rate limited by provider" }],
        usage: { input: 50, output: 10, cacheRead: 0, cacheWrite: 0, totalTokens: 60 },
      },
      isError: true,
    });
    await session.fire("agent_end", { messages: [] });

    const tree = await treeOf(client, callSpy!);
    const tool = one(tree, "web_search");
    expect(tool.error).toBe("rate limited by provider");
    const usage = one(tree, "tool-llm-usage");
    expect(usage.parent_run_id).toBe(tool.id);
    expect(usage.run_type).toBe("llm");
    expect(usage.extra.metadata.usage_metadata.input_tokens).toBe(50);
  });

  it("traces auto-compaction with its summarization call and committed summary", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "keep going");
    await session.fire("auto_compaction_start", { reason: "overflow", action: "context-full" });
    await session.fire("before_provider_request", {
      payload: { messages: [{ role: "user", content: "summarize" }] },
    });
    await session.fire("session_compact", {
      compactionEntry: {
        summary: "Earlier work summarized.",
        tokensBefore: 210000,
        tokensAfter: 12000,
      },
      fromExtension: false,
    });
    await session.fire("auto_compaction_end", {
      action: "context-full",
      aborted: false,
      willRetry: true,
    });
    await session.fire("agent_end", { messages: [] });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const compaction = one(tree, "compact-context");
    expect(compaction.parent_run_id).toBe(one(tree, "omp-agent").id);
    expect(compaction.outputs.summary).toBe("Earlier work summarized.");
    expect(compaction.extra.metadata).toMatchObject({
      ls_agent_type: "compaction",
      reason: "overflow",
      tokens_before: 210000,
      tokens_after: 12000,
    });
    const summarizer = one(tree, "anthropic");
    expect(summarizer.parent_run_id).toBe(compaction.id);
    expect(summarizer.outputs.messages[0].content).toBe("Earlier work summarized.");
  });

  it("closes an idle compaction another extension cancelled when the next prompt starts", async () => {
    const { session, client, callSpy } = await start();
    // `/compact` while idle: the hook fires, another extension cancels it, so no session_compact.
    await session.fire("session_before_compact", { preparation: {}, branchEntries: [] });
    await prompt(session, "carry on");
    await llmCall(session, assistant([{ type: "text", text: "ok" }]));
    await session.fire("agent_end", { messages: [] });

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    const compaction = one(tree, "compact-context");
    expect(compaction.extra.metadata.compaction_unresolved).toBe(true);
    // The prompt's own LLM call belongs to the prompt, not to the dead compaction.
    expect(one(tree, "anthropic").parent_run_id).toBe(one(tree, "omp-agent").id);
  });

  it("traces approvals, fallbacks and TTSR as signal runs", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "edit");
    await session.fire("tool_execution_start", { toolCallId: "e1", toolName: "edit", args: {} });
    await session.fire("tool_approval_requested", {
      sessionId: "session-1",
      toolCallId: "e1",
      toolName: "edit",
      approvalMode: "ask",
    });
    await session.fire("tool_approval_resolved", {
      sessionId: "session-1",
      toolCallId: "e1",
      toolName: "edit",
      approved: false,
      reason: "user denied",
    });
    await session.fire("retry_fallback_applied", { from: "a/x", to: "b/y", role: "default" });
    await session.fire("ttsr_triggered", { rules: [{ name: "no-sed", path: "/rules/no-sed.md" }] });
    await session.fire("agent_end", { messages: [] });

    const tree = await treeOf(client, callSpy!);
    const edit = one(tree, "edit");
    expect(one(tree, "request-tool-approval").parent_run_id).toBe(edit.id);
    const resolved = one(tree, "resolve-tool-approval");
    expect(resolved.error).toBeUndefined();
    expect(resolved.extra.metadata).toMatchObject({ denied: true, reason: "user denied" });
    expect(one(tree, "apply-model-fallback").inputs).toEqual({ from: "a/x", to: "b/y" });
    expect(one(tree, "trigger-ttsr-rules").inputs).toEqual({ rules: ["no-sed"] });
  });

  it("closes a live trace as cancelled when the session shuts down", async () => {
    const { session, client, callSpy } = await start();
    await prompt(session, "long");
    await session.fire("before_provider_request", { payload: { messages: [] } });
    await session.fire("session_shutdown");

    const tree = await treeOf(client, callSpy!);
    expectAllEnded(tree);
    expect(one(tree, "omp-agent").outputs.cancelled).toBe(true);
  });
});
