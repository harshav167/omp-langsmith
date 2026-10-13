import { describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import { RunTree } from "langsmith";
import { traceable, withRunTree } from "langsmith/traceable";

import extension, { flushTraces } from "../src/index";
import { mockClient } from "./utils/mock_client";
import { parseRecording, replay, type RecordedEvent } from "./utils/replay";
import { getAssumedTreeFromCalls } from "./utils/tree";

type Run = Record<string, any>;

const recording = (name: string): RecordedEvent[] =>
  toOmpLifecycle(
    parseRecording(fs.readFileSync(new URL(`./recordings/${name}`, import.meta.url), "utf-8")),
  );

/**
 * The recordings come from upstream Pi, which signalled a pending automatic
 * continuation with a later `agent_settled`. OMP marks that `agent_end` with
 * `willContinue` instead (its lifecycle mirror maps `isTerminal: false` to it),
 * so translate: an `agent_end` followed by an `agent_start` with no new prompt in
 * between is a continuation.
 */
function toOmpLifecycle(events: RecordedEvent[]): RecordedEvent[] {
  return events.map((entry, index) => {
    if (entry[1] !== "agent_end") return entry;
    const next = events
      .slice(index + 1)
      .find(([, name]) => name === "agent_start" || name === "before_agent_start");
    return next?.[1] === "agent_start"
      ? [entry[0], entry[1], { ...entry[2], willContinue: true }, entry[3]]
      : entry;
  });
}

const config = { enabled: true };

async function replayed(name: string) {
  const { client, callSpy } = mockClient();
  const events = recording(name);
  await replay((session) => extension(session.pi, { client, config }), events);
  await flushTraces();
  const tree = await getAssumedTreeFromCalls(callSpy.mock.calls, client);
  return { events, runs: Object.values(tree.data) as Run[], callSpy, client };
}

const countEvents = (events: RecordedEvent[], name: string, role?: string) =>
  events.filter(
    ([, event, payload]) =>
      event === name &&
      (role === undefined || (payload.message as { role?: string } | undefined)?.role === role),
  ).length;

describe.each([
  "anthropic-tool-calls.jsonl",
  "oai-responses-tool-calls.jsonl",
  "gemini-tool-calls.jsonl",
  "faux-overflow-compaction.jsonl",
])("replaying %s", (name) => {
  it("ends every run, pairs every assistant message with an LLM run, and traces every tool", async () => {
    const { events, runs } = await replayed(name);

    expect(runs.filter((run) => run.end_time == null).map((run) => run.name)).toEqual([]);

    const assistantMessages = countEvents(events, "message_end", "assistant");
    const answered = runs.filter((run) => run.run_type === "llm" && run.outputs?.messages);
    expect(answered).toHaveLength(assistantMessages);
    for (const run of answered) {
      expect(run.inputs.messages.length).toBeGreaterThan(0);
      expect(run.extra.metadata.usage_metadata).toBeDefined();
    }

    expect(runs.filter((run) => run.run_type === "tool")).toHaveLength(
      countEvents(events, "tool_execution_start"),
    );
    expect(runs.filter((run) => run.name === "omp-agent")).toHaveLength(
      countEvents(events, "before_agent_start"),
    );
  });
});

describe("replay fidelity", () => {
  it("runs each event with the recorded agent identity, cwd and context usage", async () => {
    const { client, callSpy } = mockClient();
    const ctx = {
      cwd: "/work/sub",
      model: { provider: "anthropic", id: "claude-opus-4-8", api: "anthropic-messages" },
      agent: { kind: "sub", id: "Main.worker", name: "worker", depth: 1, parentId: "Main" },
      getContextUsage: { tokens: 321, contextWindow: 1_000_000, percent: 0.03 },
    };
    const events: RecordedEvent[] = [
      [1, "before_agent_start", { prompt: "do it", systemPrompt: [] }, ctx],
      [2, "agent_start", {}, ctx],
      [3, "agent_end", { messages: [] }, ctx],
    ];
    await replay((session) => extension(session.pi, { client, config }), events);
    await flushTraces();
    const tree = await getAssumedTreeFromCalls(callSpy.mock.calls, client);
    const root = Object.values(tree.data).find((run) => run.name === "worker-agent") as Run;
    expect(root.extra.metadata).toMatchObject({ agent_kind: "sub", agent_id: "Main.worker" });
    expect(root.outputs.context_usage).toEqual(ctx.getContextUsage);
  });
});

describe("host embedding", () => {
  const rootOf = (runs: Run[]) => runs.find((run) => run.name === "omp-agent");

  it("applies config defaults to an options-provided config", async () => {
    const { runs } = await replayed("anthropic-tool-calls.jsonl");
    const root = rootOf(runs);
    expect(root?.parent_run_id).toBeUndefined();
    expect(root?.session_name).toBe("pi-coding-agent");
  });

  it("nests agent runs under a host-provided parent run tree", async () => {
    const { client, callSpy } = mockClient();
    const parent = new RunTree({ name: "host run", run_type: "chain", client });
    await parent.postRun();
    await replay(
      (session) => extension(session.pi, { client, config, getCurrentRunTree: () => parent }),
      recording("anthropic-tool-calls.jsonl"),
    );
    await flushTraces();
    const runs = Object.values(
      (await getAssumedTreeFromCalls(callSpy.mock.calls, client)).data,
    ) as Run[];
    const root = rootOf(runs);
    expect(root?.parent_run_id).toBe(parent.id);
    expect(root?.trace_id).toBe(parent.trace_id);
  });

  it("nests agent runs under the ambient traceable run", async () => {
    const { client, callSpy } = mockClient();
    const host = traceable(
      async () => {
        await replay(
          (session) => extension(session.pi, { client, config }),
          recording("anthropic-tool-calls.jsonl"),
        );
      },
      { name: "host run", run_type: "chain", client, tracingEnabled: true },
    );
    await host();
    await flushTraces();
    const runs = Object.values(
      (await getAssumedTreeFromCalls(callSpy.mock.calls, client)).data,
    ) as Run[];
    const parent = runs.find((run) => run.name === "host run");
    expect(rootOf(runs)?.parent_run_id).toBe(parent?.id);
    expect(rootOf(runs)?.trace_id).toBe(parent?.trace_id);
  });

  it("keeps a nested run inside an explicit withRunTree scope", async () => {
    const { client, callSpy } = mockClient();
    const parent = new RunTree({ name: "scope", run_type: "chain", client });
    await parent.postRun();
    await withRunTree(parent, () =>
      replay(
        (session) => extension(session.pi, { client, config }),
        recording("anthropic-tool-calls.jsonl"),
      ),
    );
    await flushTraces();
    const runs = Object.values(
      (await getAssumedTreeFromCalls(callSpy.mock.calls, client)).data,
    ) as Run[];
    expect(rootOf(runs)?.parent_run_id).toBe(parent.id);
  });
});
