import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import { type Client, RunTree, type RunTreeConfig } from "langsmith";

import type { Config } from "./config.js";
import {
  buildUsageMetadata,
  convertMessages,
  convertProviderPayload,
  convertToolOutputs,
  extractModelParameters,
  extractText,
  hasFirstToken,
  isUsage,
  messagesSincePrompt,
  normalizeContentPart,
  spawnedAgentNames,
  toolErrorMessage,
  usageExtras,
} from "./convert.js";
import { codingAgentMetadata, type LSAgentType } from "./metadata.js";
import { isRecord } from "./types.js";

// Run names are an API for filters, dashboards and evaluators: low-cardinality,
// never a per-run value. Per-run detail lives in metadata.
const MAIN_AGENT_RUN = "omp-agent";
const STEP_RUN = "agent-step";
const STEER_RUN = "steer";
const COMPACTION_RUN = "compact-context";
const BRANCH_SUMMARY_RUN = "summarize-branch";
const TOOL_USAGE_RUN = "tool-llm-usage";
const CACHE_WARM_RUN = "warm-cache";
const TITLE_RUN = "generate-title";
const TAGS = ["omp", "coding-agent"];

/** Budget for draining exports when a prompt settles; never blocks OMP for long. */
const SETTLE_FLUSH_MS = 3000;
/** `session_shutdown` handlers get 2s from OMP; stay inside it. */
const SHUTDOWN_FLUSH_MS = 1500;
/**
 * How long an aborted run's root stays open. OMP restarts the loop to deliver a
 * steer by aborting it, and its `agent_end` notification is fire-and-forget, so
 * the restart (or the steer's `before_agent_start`) can land just after it.
 */
export const RESTART_GRACE_MS = 2000;
/**
 * How long a root whose `agent_end` promised an automatic continuation (retry
 * backoff, compaction, stop hook) waits for the restart before settling anyway.
 */
export const CONTINUATION_GRACE_MS = 5 * 60_000;

// A parent OMP process publishes its live root here so an `omp` launched from a
// tool (or any process that sets these) nests under it. Snapshot at load: this
// process overwrites the same variables for its own children.
export const ENV_PARENT_DOTTED_ORDER = "LANGSMITH_PI_PARENT_DOTTED_ORDER";
export const ENV_PARENT_BAGGAGE = "LANGSMITH_PI_PARENT_BAGGAGE";
export const ENV_PARENT_IS_OMP = "LANGSMITH_PI_PARENT_IS_OMP";
const INHERITED_ENV = {
  [ENV_PARENT_DOTTED_ORDER]: process.env[ENV_PARENT_DOTTED_ORDER],
  [ENV_PARENT_BAGGAGE]: process.env[ENV_PARENT_BAGGAGE],
  [ENV_PARENT_IS_OMP]: process.env[ENV_PARENT_IS_OMP],
};

// ---------------------------------------------------------------------------
// Export ordering
// ---------------------------------------------------------------------------

interface EndParams {
  outputs?: Record<string, unknown>;
  error?: string;
  metadata?: Record<string, unknown>;
  endTime?: number;
}

/**
 * Serializes each run's network operations (post, then patch) without making
 * handlers await them. OMP dispatches events concurrently, so a handler that
 * awaited I/O could resume after another event replaced the state it held.
 */
export class RunExporter {
  readonly #chains = new WeakMap<RunTree, Promise<void>>();
  readonly #ended = new WeakSet<RunTree>();
  readonly #inflight = new Set<Promise<void>>();
  /**
   * Every client runs went out through. A run nested under a host-provided
   * parent inherits that parent's client, so flushing only `client` would leave
   * its batch queue pending.
   */
  readonly #clients = new Set<Client>();

  constructor(readonly client: Client) {}

  post(run: RunTree): void {
    this.#enqueue(run, () => run.postRun());
  }

  /** Ends a run exactly once; later calls are no-ops. */
  end(run: RunTree | undefined, params: EndParams = {}): void {
    if (!run || this.#ended.has(run)) return;
    this.#ended.add(run);
    // RunTree.end has no awaits: the fields are set before it returns.
    void run.end(params.outputs, params.error, params.endTime ?? Date.now(), params.metadata);
    this.#enqueue(run, () => run.patchRun());
  }

  isEnded(run: RunTree): boolean {
    return this.#ended.has(run);
  }

  /** Waits for queued exports; with a budget, gives up after `budgetMs` instead. */
  async drain(budgetMs?: number): Promise<void> {
    const work = this.#flush();
    if (budgetMs === undefined) return work;
    let cancelTimer = () => {};
    const timeout = new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, budgetMs);
      timer.unref?.();
      cancelTimer = () => clearTimeout(timer);
    });
    try {
      await Promise.race([work, timeout]);
    } finally {
      cancelTimer();
    }
  }

  async #flush(): Promise<void> {
    try {
      await Promise.all(this.#inflight);
      await Promise.all([...this.#clients].map((client) => client.awaitPendingTraceBatches()));
    } catch {
      // Export failures were reported when they happened; draining never throws.
    }
    // Nothing left to flush: let a disposed session's exporter (and its client) go.
    if (this.#inflight.size === 0) exporters.delete(this);
  }

  #enqueue(run: RunTree, operation: () => Promise<void>): void {
    this.#clients.add(run.client);
    exporters.add(this);
    const next = (this.#chains.get(run) ?? Promise.resolve())
      .then(operation)
      .catch((error: unknown) => {
        console.error(`[langsmith-omp] export failed for run ${run.id}:`, error);
      });
    this.#chains.set(run, next);
    this.#inflight.add(next);
    void next.finally(() => this.#inflight.delete(next));
  }
}

/** Exporters with work queued or batches not yet flushed. */
const exporters = new Set<RunExporter>();

/**
 * Waits until every run this extension queued has been sent to LangSmith. SDK
 * hosts call this before exiting; OMP sessions already flush on settle and shutdown.
 */
export async function flushTraces(): Promise<void> {
  await Promise.all([...exporters].map((exporter) => exporter.drain()));
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface LlmRequest {
  run: RunTree;
  startedAt: number;
  sawToken: boolean;
}

interface ToolState {
  run: RunTree;
  spawnNames: string[];
  updates: number;
}

interface CompactionState {
  run: RunTree;
  requests: RunTree[];
  /** True when the compaction opened its own trace (idle `/compact`). */
  standalone: boolean;
}

/** One OMP prompt (or continuation) and everything it does until it settles. */
interface RootState {
  root: RunTree;
  agentId: string;
  isMain: boolean;
  prompt: string;
  steps: Map<number, RunTree>;
  currentStep?: RunTree;
  /** Requests sent but not yet bound to an assistant message, oldest first. */
  requests: LlmRequest[];
  /** The request the assistant message currently streaming answers. */
  bound?: LlmRequest;
  lastRequest?: LlmRequest;
  tools: Map<string, ToolState>;
  /**
   * Tool runs that dispatched named subagents, kept after they end: OMP's `task`
   * returns as soon as it spawns, so the subagent session starts after the run closed.
   */
  spawners: ToolState[];
  /** `agent_start`s without their `agent_end` yet; OMP may deliver a loop's end after the next loop started. */
  openLoops: number;
  started: boolean;
  /** The last loop ended but OMP is expected to start another for this prompt (retry, steer restart). */
  awaitingRestart: boolean;
  /** When the last loop ended; the root's end time if no restart arrives. */
  endedAt?: number;
  /** Finalizes an aborted root if no restart arrives within {@link RESTART_GRACE_MS}. */
  settleTimer?: ReturnType<typeof setTimeout>;
  /** Messages of the last aborted loop, reported if the root settles without a restart. */
  lastMessages?: readonly unknown[];
  lastModel?: string;
  lastAssistantText?: string;
  error?: string;
  /** The user interrupted at least one loop. An interrupt is not an error. */
  interrupted: boolean;
  steers: number;
  continuations: number;
  /** Steps a restarted loop superseded whose own late turn_end has not arrived yet. */
  supersededSteps: number;
  /** omp's session-title fork requests, ended once the title they produced is known. */
  titleRequests: RunTree[];
}

/**
 * Live roots per OMP agent registry id. OMP rebinds the factory into every
 * in-process subagent session and module state is shared across them, so a
 * subagent finds the parent turn (and its dispatching tool run) here.
 */
const liveRoots = new Map<string, RootState>();

/** The session's omp-generated (or user-set) title, if any. */
const readTitle = (ctx: ExtensionContext): string | undefined => {
  try {
    return ctx.sessionManager.getSessionName?.()?.trim() || undefined;
  } catch {
    return undefined;
  }
};

/** omp's title fork is a side turn whose last user message is its `<title-request>` prompt. */
const isTitleRequest = (inputs: Record<string, unknown>): boolean => {
  const last = Array.isArray(inputs.messages) ? inputs.messages.at(-1) : undefined;
  return (
    isRecord(last) && last.role === "user" && extractText(last.content).includes("<title-request>")
  );
};

/**
 * The `streamSimple` attempt a provider request belongs to. Newer omp builds put
 * it on `before_provider_request`; 18.8 does not, so it is read defensively.
 */
const attemptMetadata = (event: unknown): Record<string, unknown> => {
  const attempt = isRecord(event) && isRecord(event.attempt) ? event.attempt : undefined;
  if (!attempt) return {};
  return {
    ...(typeof attempt.attemptNumber === "number" ? { attempt_number: attempt.attemptNumber } : {}),
    ...(typeof attempt.attemptId === "string" ? { attempt_id: attempt.attemptId } : {}),
    ...(typeof attempt.credentialId === "number" ? { credential_id: attempt.credentialId } : {}),
  };
};

export interface TracingDeps {
  config: Config;
  exporter: RunExporter;
  /** Host-provided parent (SDK embedding); undefined starts a new trace. */
  getParentRunTree: () => RunTree | undefined;
  runtimeVersion?: string;
}

type Ctx = ExtensionContext;

/**
 * Registers every OMP lifecycle handler. All handlers are synchronous with
 * respect to tracing state; only the settle and shutdown paths await, and only
 * after they have finished mutating state.
 */
export function registerTracing(pi: ExtensionAPI, deps: TracingDeps): void {
  const { config, exporter } = deps;
  const client = exporter.client;

  let state: RootState | undefined;
  let compaction: CompactionState | undefined;
  /** Requests sent while no prompt was live (cache warming, idle maintenance). */
  const orphans = new Map<RunTree, number>();
  let pendingWarm: Record<string, unknown> | undefined;
  let fallbackTurn = 0;

  const status = (ctx: Ctx, text: string | undefined) => {
    if (ctx.hasUI) ctx.ui.setStatus("langsmith", text);
  };

  const child = (parent: RunTree, runConfig: RunTreeConfig): RunTree => {
    const run = parent.createChild(runConfig);
    exporter.post(run);
    return run;
  };

  /** A zero-duration run marking a point-in-time OMP signal inside the trace. */
  const signal = (
    name: string,
    attrs: Omit<EndParams, "endTime"> & { inputs?: Record<string, unknown>; parent?: RunTree },
  ) => {
    const parent = attrs.parent ?? state?.currentStep ?? state?.root;
    if (!parent) return;
    const run = child(parent, {
      name,
      run_type: "chain",
      inputs: attrs.inputs ?? {},
      metadata: { ls_omp_signal: true, ...attrs.metadata },
    });
    exporter.end(run, { outputs: attrs.outputs, error: attrs.error });
  };

  const turnNumberFor = (ctx: Ctx, prompt: string | undefined): number => {
    try {
      // Only the active branch counts: after navigating back and prompting again,
      // the abandoned branch's prompts are still stored but no longer in the conversation.
      const manager = ctx.sessionManager;
      const entries =
        typeof manager.getBranch === "function" ? manager.getBranch() : manager.getEntries();
      const users = entries.flatMap((entry) =>
        entry.type === "message" && entry.message.role === "user" ? [entry.message] : [],
      );
      let count = users.length;
      // before_agent_start can fire before or after the prompt is appended.
      const last = users.at(-1);
      const lastText = isRecord(last) ? extractText(last.content) : undefined;
      if (prompt !== undefined && lastText !== prompt) count += 1;
      fallbackTurn = Math.max(count, 1);
    } catch {
      fallbackTurn += 1;
    }
    return fallbackTurn;
  };

  const parentFromEnv = (): RunTree | undefined => {
    const dottedOrder = INHERITED_ENV[ENV_PARENT_DOTTED_ORDER]?.trim();
    if (!dottedOrder) return undefined;
    try {
      const headers: Record<string, string> = { "langsmith-trace": dottedOrder };
      const baggage = INHERITED_ENV[ENV_PARENT_BAGGAGE];
      if (baggage) headers.baggage = baggage;
      return RunTree.fromHeaders(headers, { client, project_name: config.project });
    } catch (error) {
      console.error(`[langsmith-omp] ignoring malformed ${ENV_PARENT_DOTTED_ORDER}:`, error);
      return undefined;
    }
  };

  /** The run an in-process subagent nests under: the tool call that spawned it, else the parent's step. */
  const dispatchParent = (parent: RootState, agentId: string): RunTree => {
    const spawns = (tool: ToolState) =>
      tool.spawnNames.some(
        (name) => agentId === name || agentId.endsWith(`-${name}`) || agentId.endsWith(`.${name}`),
      );
    const open = [...parent.tools.values()];
    const dispatch =
      open.find(spawns) ??
      parent.spawners.findLast(spawns) ??
      (open.length === 1 ? open[0] : undefined);
    return dispatch?.run ?? parent.currentStep ?? parent.root;
  };

  const publishParentEnv = (root: RunTree) => {
    try {
      process.env[ENV_PARENT_DOTTED_ORDER] = root.dotted_order;
      process.env[ENV_PARENT_BAGGAGE] = root.toHeaders().baggage;
      process.env[ENV_PARENT_IS_OMP] = "1";
    } catch {
      // Propagation is best effort; tracing itself is unaffected.
    }
  };

  // A child must not attach to a root that has ended; restore what this process
  // inherited. Another live main session in this process may have published its
  // own root since, which stays.
  const withdrawParentEnv = (root: RunTree) => {
    if (process.env[ENV_PARENT_DOTTED_ORDER] !== root.dotted_order) return;
    for (const [key, value] of Object.entries(INHERITED_ENV)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  };

  const userContent = (prompt: string, images: unknown): unknown => {
    const parts = Array.isArray(images) ? images.map(normalizeContentPart) : [];
    return parts.length ? [{ type: "text", text: prompt }, ...parts] : prompt;
  };

  const openRoot = (
    ctx: Ctx,
    turn: { prompt: string; images?: unknown; trigger: "prompt" | "continuation" },
  ): RootState => {
    const agent = ctx.agent;
    const sessionId = ctx.sessionManager.getSessionId();
    const isMain = agent?.kind !== "sub";
    const turnNumber = turnNumberFor(ctx, turn.trigger === "prompt" ? turn.prompt : undefined);
    // An idle `/compact` another extension cancelled (or that failed) sends no
    // `session_compact`; a new prompt proves it is over.
    if (compaction?.standalone) closeCompaction({ metadata: { compaction_unresolved: true } });

    let parent: RunTree | undefined;
    let agentType: LSAgentType = "root";
    let nested = false;
    const extra: Record<string, unknown> = {};
    if (!isMain && agent.parentId) {
      const parentState = liveRoots.get(agent.parentId);
      if (parentState) {
        parent = dispatchParent(parentState, agent.id);
        agentType = "subagent";
        nested = true;
      }
    }
    if (!parent && isMain) parent = deps.getParentRunTree();
    if (!parent) {
      const fromEnv = parentFromEnv();
      if (fromEnv) {
        parent = fromEnv;
        nested = INHERITED_ENV[ENV_PARENT_IS_OMP] === "1";
        if (nested) {
          agentType = "subagent";
          extra.cross_process = true;
        }
      }
    }
    if (!isMain && !nested) agentType = "subagent";

    const title = readTitle(ctx);
    const thinkingLevel = pi.getThinkingLevel?.();
    const metadata: Record<string, unknown> = {
      // Nested agents keep the parent trace's thread and user turn; their own
      // session id rides along separately.
      ...codingAgentMetadata({
        agentType,
        cwd: ctx.cwd,
        runtimeVersion: deps.runtimeVersion,
        ...(nested ? {} : { threadId: sessionId, turnNumber }),
      }),
      ...(nested
        ? { agent_session_id: sessionId, agent_turn_number: turnNumber }
        : { session_id: sessionId }),
      ...(title ? { session_name: title } : {}),
      agent_id: agent?.id,
      agent_name: agent?.name,
      agent_kind: agent?.kind,
      agent_depth: agent?.depth,
      ...(agent?.parentId ? { parent_agent_id: agent.parentId } : {}),
      // coding-agent-v1: subagent runs always carry these two.
      ...(agentType === "subagent" && agent
        ? { ls_subagent_id: agent.id, ls_subagent_type: agent.name }
        : {}),
      ...(turn.trigger === "continuation" ? { turn_trigger: "continuation" } : {}),
      ...(ctx.model ? { model: ctx.model.id, provider: ctx.model.provider } : {}),
      ...(thinkingLevel ? { thinking_level: thinkingLevel } : {}),
      ...extra,
      ...config.metadata,
    };

    const runConfig = {
      name: isMain ? MAIN_AGENT_RUN : `${agent.name || "sub"}-agent`,
      run_type: "chain",
      project_name: config.project,
      client,
      inputs:
        turn.trigger === "continuation"
          ? { trigger: "continuation" }
          : {
              prompt: turn.prompt,
              messages: [{ role: "user", content: userContent(turn.prompt, turn.images) }],
              ...(Array.isArray(turn.images) && turn.images.length
                ? { image_count: turn.images.length }
                : {}),
            },
      metadata,
      tags: agentType === "subagent" ? [...TAGS, "subagent"] : TAGS,
      replicas: config.replicas?.map((replica) => ({
        ...(replica.api_url ? { apiUrl: replica.api_url } : {}),
        ...(replica.api_key ? { apiKey: replica.api_key } : {}),
        ...(replica.project ? { projectName: replica.project } : {}),
        ...(replica.updates ? { updates: replica.updates } : {}),
      })),
    };
    const root = parent ? parent.createChild(runConfig) : new RunTree(runConfig);
    exporter.post(root);

    const next: RootState = {
      root,
      agentId: agent?.id ?? "Main",
      isMain,
      prompt: turn.prompt,
      steps: new Map(),
      requests: [],
      tools: new Map(),
      spawners: [],
      openLoops: 0,
      awaitingRestart: false,
      started: false,
      interrupted: false,
      steers: 0,
      continuations: 0,
      supersededSteps: 0,
      titleRequests: [],
    };
    liveRoots.set(next.agentId, next);
    if (isMain) publishParentEnv(root);
    status(ctx, "LangSmith: tracing run");
    return next;
  };

  /** Ends every open child of a root; `reason` explains why each was cut short. None of these is an error. */
  const closeDangling = (target: RootState, reason: "interrupted" | "superseded" | "shutdown") => {
    for (const tool of target.tools.values()) {
      exporter.end(tool.run, { outputs: { [reason]: true }, metadata: { [reason]: true } });
    }
    target.tools.clear();
    if (target.bound) {
      exporter.end(target.bound.run, { metadata: { [reason]: true } });
      target.bound = undefined;
    }
    closeUnmatched(target);
    for (const step of target.steps.values()) {
      exporter.end(step, { outputs: { incomplete: true }, metadata: { [reason]: true } });
    }
    target.steps.clear();
    target.currentStep = undefined;
  };

  /**
   * Requests no assistant message answered: OMP side requests (session title,
   * prewalk) share the session's provider hooks but never produce `message_end`,
   * and a request a retry replaced never finishes.
   */
  const closeUnmatched = (target: RootState) => {
    for (const request of target.requests) {
      exporter.end(request.run, { metadata: { ls_omp_unmatched_request: true } });
    }
    target.requests = [];
  };

  /**
   * Ends omp's session-title fork requests. A side turn yields no message here, so
   * its result is the title omp then sets on the session; without one yet, the
   * request stays open until the root finalizes.
   */
  const settleTitleRequests = (ctx: Ctx, target: RootState, final: boolean) => {
    if (!target.titleRequests.length) return;
    const title = readTitle(ctx);
    if (!title && !final) return;
    for (const run of target.titleRequests) {
      exporter.end(run, title ? { outputs: { title } } : { metadata: { title_not_set: true } });
    }
    target.titleRequests = [];
  };

  const finalizeRoot = (
    ctx: Ctx,
    target: RootState,
    result: { messages?: readonly unknown[]; cancelled?: boolean },
  ) => {
    clearTimeout(target.settleTimer);
    closeDangling(target, result.cancelled ? "shutdown" : "interrupted");
    // A compaction this root opened and never saw end (cancelled or failed mid-run).
    if (compaction && !compaction.standalone) {
      closeCompaction({ metadata: { compaction_unresolved: true } });
    }
    let contextUsage: unknown;
    try {
      contextUsage = ctx.getContextUsage();
    } catch {
      contextUsage = undefined;
    }
    if (target.interrupted) target.root.tags = [...(target.root.tags ?? []), "interrupted"];
    const title = readTitle(ctx);
    exporter.end(target.root, {
      outputs: {
        ...(target.lastAssistantText !== undefined ? { output: target.lastAssistantText } : {}),
        // omp's agent_end carries the whole session context; the trace shows only
        // what this prompt produced, so turns do not repeat the conversation.
        ...(result.messages
          ? { messages: convertMessages(messagesSincePrompt(result.messages, target.prompt)) }
          : {}),
        ...(contextUsage !== undefined ? { context_usage: contextUsage } : {}),
        ...(result.cancelled ? { cancelled: true } : {}),
        ...(target.interrupted ? { interrupted: true } : {}),
      },
      error: target.error,
      // A root that was waiting for a restart ended when its last loop did.
      ...(target.awaitingRestart && target.endedAt !== undefined
        ? { endTime: target.endedAt }
        : {}),
      metadata: {
        ...(target.interrupted ? { interrupted: true } : {}),
        ...(target.steers ? { steer_count: target.steers } : {}),
        ...(target.continuations ? { continuation_count: target.continuations } : {}),
        // omp names a session while its first prompt runs, after the root was posted.
        ...(title ? { session_name: title } : {}),
      },
    });
    settleTitleRequests(ctx, target, true);
    if (liveRoots.get(target.agentId) === target) liveRoots.delete(target.agentId);
    if (target.isMain) withdrawParentEnv(target.root);
    if (state === target) state = undefined;
  };

  const closeOrphans = () => {
    for (const run of orphans.keys())
      exporter.end(run, { metadata: { ls_omp_unmatched_request: true } });
    orphans.clear();
  };

  // -------------------------------------------------------------------------
  // Session lifecycle
  // -------------------------------------------------------------------------

  pi.on("session_start", (_event, ctx) => {
    status(ctx, "LangSmith: tracing");
  });

  const endSession = (ctx: Ctx) => {
    if (state) finalizeRoot(ctx, state, { cancelled: true });
    if (compaction) {
      for (const request of compaction.requests) exporter.end(request);
      exporter.end(compaction.run, { metadata: { shutdown: true } });
      compaction = undefined;
    }
    closeOrphans();
  };

  pi.on("session_switch", (_event, ctx) => endSession(ctx));
  pi.on("session_branch", (_event, ctx) => endSession(ctx));

  pi.on("session_shutdown", async (_event, ctx) => {
    endSession(ctx);
    status(ctx, undefined);
    await exporter.drain(SHUTDOWN_FLUSH_MS);
  });

  // -------------------------------------------------------------------------
  // Prompt lifecycle
  // -------------------------------------------------------------------------

  pi.on("before_agent_start", (event, ctx) => {
    if (state?.started && (state.openLoops > 0 || state.awaitingRestart)) {
      // OMP fires before_agent_start for every dequeued steer/follow-up batch,
      // including one it delivers by aborting and restarting the loop. It belongs
      // to the running trace, not a new one.
      state.steers += 1;
      signal(STEER_RUN, {
        parent: state.root,
        inputs: {
          messages: [{ role: "user", content: userContent(event.prompt, event.images) }],
        },
        metadata: { steer_index: state.steers },
      });
      if (state.settleTimer) armSettle(ctx, state, undefined, RESTART_GRACE_MS * 5);
      return;
    }
    if (state && !state.started) {
      // A source-base retry re-runs the chain for the same submission.
      if (state.prompt === event.prompt) return;
      finalizeRoot(ctx, state, { cancelled: true });
    } else if (state) {
      finalizeRoot(ctx, state, {});
    }
    closeOrphans();
    state = openRoot(ctx, { prompt: event.prompt, images: event.images, trigger: "prompt" });
  });

  pi.on("agent_start", (_event, ctx) => {
    // A run OMP starts without a prompt (background job wake, async delivery)
    // still gets its own trace.
    state ??= openRoot(ctx, { prompt: "", trigger: "continuation" });
    clearTimeout(state.settleTimer);
    state.settleTimer = undefined;
    if (state.awaitingRestart) state.continuations += 1;
    state.awaitingRestart = false;
    state.endedAt = undefined;
    state.openLoops += 1;
    state.started = true;
  });

  pi.on("turn_start", (event, ctx) => {
    if (!state) return;
    const step = child(state.root, {
      name: STEP_RUN,
      run_type: "chain",
      inputs: { step_index: event.turnIndex },
      metadata: {
        step_index: event.turnIndex,
        ...(ctx.model ? { ls_provider: ctx.model.provider, ls_model_name: ctx.model.id } : {}),
      },
    });
    // A restarted loop numbers its steps from 0 again; the old loop's step is done,
    // and its own (aborted) turn_end may still arrive late.
    const superseded = state.steps.get(event.turnIndex);
    if (superseded) {
      exporter.end(superseded, { metadata: { superseded: true } });
      state.supersededSteps += 1;
    }
    state.steps.set(event.turnIndex, step);
    state.currentStep = step;
  });

  pi.on("turn_end", (event, ctx) => {
    if (!state) return;
    const message: unknown = event.message;
    if (state.supersededSteps > 0 && isRecord(message) && message.stopReason === "aborted") {
      // The aborted loop's late turn_end: its step already ended, and the step now
      // at this index belongs to the restarted loop.
      state.supersededSteps -= 1;
      return;
    }
    closeUnmatched(state);
    settleTitleRequests(ctx, state, false);
    const step = state.steps.get(event.turnIndex) ?? state.currentStep;
    if (!step) return;
    state.steps.delete(event.turnIndex);
    exporter.end(step, {
      outputs: {
        messages: convertMessages([event.message]),
        tool_result_count: event.toolResults.length,
      },
    });
    if (state.currentStep === step) state.currentStep = undefined;
  });

  /** Arms (or re-arms) finalization of an aborted root that OMP may still restart. */
  const armSettle = (
    ctx: Ctx,
    target: RootState,
    messages: readonly unknown[] | undefined,
    delayMs: number,
  ) => {
    clearTimeout(target.settleTimer);
    const timer = setTimeout(() => {
      try {
        if (state !== target || target.openLoops > 0) return;
        finalizeRoot(ctx, target, { messages: messages ?? target.lastMessages });
        status(ctx, "LangSmith: traced");
        void exporter.drain(SETTLE_FLUSH_MS);
      } catch (error) {
        // A raw timer runs outside OMP's handler isolation; never let it throw.
        console.error("[langsmith-omp] failed to finalize an interrupted trace:", error);
      }
    }, delayMs);
    timer.unref?.();
    target.settleTimer = timer;
    if (messages) target.lastMessages = messages;
  };

  pi.on("agent_end", async (event, ctx) => {
    const target = state;
    if (!target) return;
    target.openLoops = Math.max(0, target.openLoops - 1);
    // OMP sends this notification without awaiting it, so it can arrive after
    // the next loop of the same prompt already started; that loop owns the root.
    if (target.openLoops > 0) return;
    const last = event.messages.at(-1);
    const aborted = last?.role === "assistant" && last.stopReason === "aborted";
    if (aborted) target.interrupted = true;
    // `willContinue`: OMP already scheduled an automatic continuation (retry,
    // compaction, stop-hook). The next agent_start rejoins this root; if it never
    // comes (the continuation failed silently), the root still settles.
    if (event.willContinue === true) {
      closeDangling(target, "superseded");
      target.awaitingRestart = true;
      target.endedAt = Date.now();
      armSettle(ctx, target, event.messages, CONTINUATION_GRACE_MS);
      return;
    }
    if (aborted) {
      // An interrupt, or OMP aborting to deliver a steer and restarting. Keep the
      // root briefly so the restart lands in the same trace.
      closeDangling(target, "interrupted");
      target.awaitingRestart = true;
      target.endedAt = Date.now();
      armSettle(ctx, target, event.messages, RESTART_GRACE_MS);
      return;
    }
    // Only the final loop decides the outcome: an earlier failed attempt that a
    // retry recovered is not the prompt's error.
    target.error =
      last?.role === "assistant" && last.stopReason === "error"
        ? last.errorMessage || "Assistant message ended with an error"
        : undefined;
    finalizeRoot(ctx, target, { messages: event.messages });
    status(ctx, "LangSmith: traced");
    await exporter.drain(SETTLE_FLUSH_MS);
  });

  // -------------------------------------------------------------------------
  // LLM calls
  // -------------------------------------------------------------------------

  pi.on("before_provider_request", (event, ctx) => {
    const { inputs, tools } = convertProviderPayload(event.payload);
    const params = extractModelParameters(event.payload, pi.getThinkingLevel?.());
    const modelLabel = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : undefined;
    const switched =
      state?.lastModel !== undefined && modelLabel !== undefined && state.lastModel !== modelLabel
        ? state.lastModel
        : undefined;
    const runConfig = {
      name: ctx.model?.provider ?? "llm",
      run_type: "llm",
      inputs,
      extra: { invocation_params: { ...params, ...(tools ? { tools } : {}) } },
      metadata: {
        ls_model_type: "chat",
        ...(ctx.model
          ? {
              ls_provider: ctx.model.provider,
              ls_model_name: ctx.model.id,
              api: ctx.model.api,
              context_window: ctx.model.contextWindow,
            }
          : {}),
        ...(typeof params.max_tokens === "number" ? { ls_max_tokens: params.max_tokens } : {}),
        ...(typeof params.temperature === "number" ? { ls_temperature: params.temperature } : {}),
        ...(Array.isArray(params.stop) ? { ls_stop: params.stop } : {}),
        ...(switched ? { model_switch_from: switched, model_switch_to: modelLabel } : {}),
        ...attemptMetadata(event),
        ...(ctx.agent && state && ctx.agent.id !== state.agentId ? { agent_id: ctx.agent.id } : {}),
      },
    };

    if (compaction) {
      compaction.requests.push(child(compaction.run, runConfig));
      return;
    }
    if (!state) {
      // Idle requests (cache warming refreshes the real request) get their own trace.
      const warm = pendingWarm;
      pendingWarm = undefined;
      const run = new RunTree({
        ...runConfig,
        name: warm ? CACHE_WARM_RUN : runConfig.name,
        project_name: config.project,
        client,
        metadata: {
          ...codingAgentMetadata({
            agentType: "root",
            cwd: ctx.cwd,
            runtimeVersion: deps.runtimeVersion,
            threadId: ctx.sessionManager.getSessionId(),
          }),
          ...runConfig.metadata,
          background_request: true,
          ...(warm ? { cache_warming: warm } : {}),
          ...config.metadata,
        },
        tags: TAGS,
      });
      exporter.post(run);
      orphans.set(run, Date.now());
      return;
    }
    if (isTitleRequest(inputs)) {
      // omp's session-title fork: a side turn that never yields a message here, so it
      // must not take part in reply pairing. Its result is the title omp then sets.
      state.titleRequests.push(
        child(state.root, {
          ...runConfig,
          name: TITLE_RUN,
          metadata: {
            ...runConfig.metadata,
            ls_omp_side_request: "session_title",
            usage_missing: true,
          },
        }),
      );
      return;
    }
    if (switched) {
      signal("switch-model", { inputs: { from: switched, to: modelLabel } });
    }
    if (modelLabel) state.lastModel = modelLabel;
    const request: LlmRequest = {
      run: child(state.currentStep ?? state.root, runConfig),
      startedAt: Date.now(),
      sawToken: false,
    };
    state.requests.push(request);
    state.lastRequest = request;
  });

  pi.on("after_provider_response", (event) => {
    const metadata = {
      http_status: event.status,
      ...(event.requestId ? { provider_request_id: event.requestId } : {}),
    };
    if (state?.lastRequest && !exporter.isEnded(state.lastRequest.run)) {
      state.lastRequest.run.metadata = metadata;
      return;
    }
    // Idle requests never produce an assistant message; the response settles them.
    const orphan = [...orphans.keys()].at(-1);
    if (orphan) {
      orphans.delete(orphan);
      exporter.end(orphan, { metadata });
    }
  });

  /**
   * The request an assistant message answers. omp stamps the message with the
   * time it sent the request, so the request that started nearest to it wins: a
   * side request (prewalk, memory recall) omp sends while the reply is pending,
   * before or after streaming starts, stays queued for `closeUnmatched`. Without
   * a timestamp the newest request is assumed.
   */
  const takeRequest = (target: RootState, message: unknown): LlmRequest | undefined => {
    const sentAt = isRecord(message) ? message.timestamp : undefined;
    let index = target.requests.length - 1;
    if (typeof sentAt === "number") {
      let nearest = Number.POSITIVE_INFINITY;
      target.requests.forEach((request, i) => {
        const distance = Math.abs(request.startedAt - sentAt);
        // Ties go to the newer request, as before omp stamped messages.
        if (distance <= nearest) {
          nearest = distance;
          index = i;
        }
      });
    }
    return index >= 0 ? target.requests.splice(index, 1)[0] : undefined;
  };

  pi.on("message_start", (event) => {
    if (!state || event.message.role !== "assistant") return;
    const request = takeRequest(state, event.message);
    if (request) state.bound = request;
  });

  pi.on("message_update", (event) => {
    const bound = state?.bound;
    if (!bound || bound.sawToken || event.message.role !== "assistant") return;
    if (!hasFirstToken(event.message.content)) return;
    bound.sawToken = true;
    bound.run.addEvent({ name: "new_token", time: new Date().toISOString() });
  });

  pi.on("message_end", (event, ctx) => {
    const message = event.message;
    if (!state || message.role !== "assistant") return;
    const request =
      state.bound ??
      takeRequest(state, message) ??
      // A transport that never calls the request hook still yields a message.
      ({
        run: child(state.currentStep ?? state.root, {
          name: message.provider || ctx.model?.provider || "llm",
          run_type: "llm",
          inputs: {},
          metadata: { request_missing: true },
        }),
        startedAt: Date.now() - (typeof message.duration === "number" ? message.duration : 0),
        sawToken: false,
      } satisfies LlmRequest);
    state.bound = undefined;

    const fields: Record<string, unknown> = isRecord(message) ? message : {};
    const ttft = typeof fields.ttft === "number" ? fields.ttft : undefined;
    if (!request.sawToken && ttft !== undefined) {
      request.run.addEvent({
        name: "new_token",
        time: new Date(request.startedAt + ttft).toISOString(),
      });
    }
    const usage = isUsage(message.usage) ? message.usage : undefined;
    const upstreamModel =
      typeof fields.upstreamModel === "string" ? fields.upstreamModel : undefined;
    const servedModel = upstreamModel || message.model;
    // A user abort is not a failure; a provider error is.
    const error =
      message.stopReason === "error"
        ? message.errorMessage || "Assistant message ended with an error"
        : undefined;
    if (message.stopReason === "aborted") state.interrupted = true;
    const text = extractText(message.content);
    if (text) state.lastAssistantText = text;

    exporter.end(request.run, {
      outputs: { messages: convertMessages([message]) },
      error,
      metadata: {
        stop_reason: message.stopReason,
        ...(message.stopReason === "aborted" ? { aborted: true } : {}),
        ...(servedModel ? { ls_model_name: servedModel } : {}),
        ...(servedModel && servedModel !== message.model ? { requested_model: message.model } : {}),
        ...(message.provider ? { ls_provider: message.provider } : {}),
        ...(typeof fields.upstreamProvider === "string"
          ? { upstream_provider: fields.upstreamProvider }
          : {}),
        ...(typeof fields.responseId === "string" ? { response_id: fields.responseId } : {}),
        ...(typeof fields.errorStatus === "number" ? { error_status: fields.errorStatus } : {}),
        ...(typeof fields.credentialId === "number" ? { credential_id: fields.credentialId } : {}),
        ...(ttft !== undefined ? { ttft_ms: ttft } : {}),
        ...(typeof fields.duration === "number" ? { duration_ms: fields.duration } : {}),
        ...(usage
          ? { usage_metadata: buildUsageMetadata(usage), omp_usage: usage, ...usageExtras(usage) }
          : { usage_missing: true }),
      },
    });
  });

  // -------------------------------------------------------------------------
  // Tools
  // -------------------------------------------------------------------------

  pi.on("tool_execution_start", (event, ctx) => {
    if (!state) return;
    const run = child(state.currentStep ?? state.root, {
      name: event.toolName,
      run_type: "tool",
      inputs: isRecord(event.args) ? event.args : { input: event.args },
      metadata: {
        ls_tool_name: event.toolName,
        tool_name: event.toolName,
        tool_call_id: event.toolCallId,
        ...(event.intent ? { intent: event.intent } : {}),
        ...(ctx.agent && ctx.agent.id !== state.agentId ? { agent_id: ctx.agent.id } : {}),
      },
    });
    const tool = { run, spawnNames: spawnedAgentNames(event.args), updates: 0 };
    state.tools.set(event.toolCallId, tool);
    if (tool.spawnNames.length) state.spawners.push(tool);
  });

  pi.on("tool_execution_update", (event) => {
    const tool = state?.tools.get(event.toolCallId);
    if (!tool) return;
    tool.updates += 1;
    // Every streamed update, with the tool's complete partial result.
    tool.run.addEvent({ name: "tool_update", kwargs: { partialResult: event.partialResult } });
  });

  pi.on("tool_execution_end", (event, ctx) => {
    const tool = state?.tools.get(event.toolCallId);
    if (!state || !tool) return;
    state.tools.delete(event.toolCallId);
    const result = isRecord(event.result) ? event.result : undefined;
    if (result && isUsage(result.usage)) {
      // OMP does not report which model a tool used internally; the session
      // model keeps the run priceable and the flag marks the attribution.
      const usageRun = child(tool.run, {
        name: TOOL_USAGE_RUN,
        run_type: "llm",
        inputs: {},
        metadata: {
          ...(ctx.model ? { ls_provider: ctx.model.provider, ls_model_name: ctx.model.id } : {}),
          model_is_session_model: true,
          tool_call_id: event.toolCallId,
          usage_metadata: buildUsageMetadata(result.usage),
          omp_usage: result.usage,
          ...usageExtras(result.usage),
        },
      });
      exporter.end(usageRun);
    }
    exporter.end(tool.run, {
      outputs: convertToolOutputs(event),
      error: event.isError ? toolErrorMessage(event.result) : undefined,
      metadata: {
        is_error: event.isError,
        ...(tool.updates ? { update_count: tool.updates } : {}),
      },
    });
  });

  pi.on("tool_approval_requested", (event) => {
    signal("request-tool-approval", {
      parent: state?.tools.get(event.toolCallId)?.run,
      inputs: { tool_name: event.toolName, approval_mode: event.approvalMode },
      metadata: {
        tool_call_id: event.toolCallId,
        ...(event.reason ? { reason: event.reason } : {}),
      },
    });
  });

  pi.on("tool_approval_resolved", (event) => {
    signal("resolve-tool-approval", {
      parent: state?.tools.get(event.toolCallId)?.run,
      inputs: { tool_name: event.toolName },
      outputs: { approved: event.approved },
      metadata: {
        tool_call_id: event.toolCallId,
        ...(event.approved
          ? {}
          : { denied: true, ...(event.reason ? { reason: event.reason } : {}) }),
      },
    });
  });

  pi.on("before_subagent_spawn", (event) => {
    signal("spawn-subagent", {
      inputs: {
        agent: event.agent,
        invocation_kind: event.invocationKind,
        patterns: event.patterns,
        ...(event.modelRole ? { model_role: event.modelRole } : {}),
      },
    });
  });

  // -------------------------------------------------------------------------
  // Reliability and runtime signals
  // -------------------------------------------------------------------------

  pi.on("auto_retry_start", (event) => {
    signal("retry-request", {
      inputs: {
        attempt: event.attempt,
        max_attempts: event.maxAttempts,
        delay_ms: event.delayMs,
        error_message: event.errorMessage,
      },
      // A retry starting is not a failure; settle-retry reports the outcome.
      metadata: event.errorId !== undefined ? { error_id: event.errorId } : {},
    });
  });

  pi.on("auto_retry_end", (event) => {
    signal("settle-retry", {
      inputs: { attempt: event.attempt },
      outputs: { success: event.success },
      error: event.success ? undefined : event.finalError || "Retries exhausted",
    });
  });

  pi.on("retry_fallback_applied", (event) => {
    signal("apply-model-fallback", {
      inputs: { from: event.from, to: event.to },
      metadata: { role: event.role, ...(event.reason ? { reason: event.reason } : {}) },
    });
  });

  pi.on("retry_fallback_succeeded", (event) => {
    signal("confirm-model-fallback", {
      outputs: { model: event.model },
      metadata: { role: event.role },
    });
  });

  pi.on("ttsr_triggered", (event) => {
    signal("trigger-ttsr-rules", {
      inputs: { rules: event.rules.map((rule) => rule.name) },
      metadata: { rule_paths: event.rules.map((rule) => rule.path) },
    });
  });

  pi.on("todo_reminder", (event) => {
    signal("remind-todos", {
      inputs: { todos: event.todos, attempt: event.attempt, max_attempts: event.maxAttempts },
    });
  });

  pi.on("goal_updated", (event) => {
    signal("update-goal", { inputs: { goal: event.goal, state: event.state } });
  });

  pi.on("credential_disabled", (event) => {
    signal("disable-credential", {
      inputs: { provider: event.provider },
      error: event.disabledCause,
      metadata: Object.fromEntries(Object.entries(event).filter(([key]) => key !== "type")),
    });
  });

  // Transport events newer omp builds emit (absent from 18.8, so typed loosely):
  // per-attempt HTTP failures and in-call credential rotation.
  // The installed overloads do not know these names; the runtime registry accepts any.
  const onTransport = pi.on as unknown as (
    event: string,
    handler: (event: Record<string, unknown>) => void,
  ) => void;
  onTransport("provider_http_error", (event) => {
    const willRetry = event.willRetry === true;
    signal("provider-http-error", {
      parent:
        state?.lastRequest && !exporter.isEnded(state.lastRequest.run)
          ? state.lastRequest.run
          : undefined,
      inputs: {
        status: event.status,
        // Some providers (Gemini) put the API key in the query string.
        url: typeof event.url === "string" ? event.url.split("?")[0] : event.url,
        http_attempt: event.httpAttempt,
        ...(typeof event.bodyExcerpt === "string" ? { body_excerpt: event.bodyExcerpt } : {}),
      },
      outputs: {
        will_retry: willRetry,
        ...(event.retryDelayMs !== undefined ? { retry_delay_ms: event.retryDelayMs } : {}),
      },
      // A response omp retries is not the prompt's failure.
      error: willRetry ? undefined : `HTTP ${String(event.status)}`,
    });
  });
  onTransport("provider_credential_switch", (event) => {
    signal("switch-credential", {
      inputs: { provider: event.provider, from: event.from, to: event.to },
      metadata: {
        ...(event.status !== undefined ? { status: event.status } : {}),
        ...(typeof event.errorMessage === "string" ? { error_message: event.errorMessage } : {}),
      },
    });
  });

  pi.on("cache_warming_decision", (event) => {
    // The refresh replays the real request through the provider hooks while idle.
    if (event.action === "warm" && !state) {
      pendingWarm = {
        warm_cost: event.warmCost,
        miss_cost: event.missCost,
        continuation_probability: event.continuationProbability,
      };
    }
  });

  // -------------------------------------------------------------------------
  // Compaction and branch summaries
  // -------------------------------------------------------------------------

  const openCompaction = (ctx: Ctx, inputs: Record<string, unknown>) => {
    if (compaction) return;
    const metadata = { ls_agent_type: "compaction", ...inputs };
    if (state) {
      compaction = {
        run: child(state.currentStep ?? state.root, {
          name: COMPACTION_RUN,
          run_type: "chain",
          inputs,
          metadata,
        }),
        requests: [],
        standalone: false,
      };
      return;
    }
    // Idle `/compact` runs outside any prompt and becomes its own trace in the thread.
    const run = new RunTree({
      name: COMPACTION_RUN,
      run_type: "chain",
      project_name: config.project,
      client,
      inputs,
      metadata: {
        ...codingAgentMetadata({
          agentType: "compaction",
          cwd: ctx.cwd,
          runtimeVersion: deps.runtimeVersion,
          threadId: ctx.sessionManager.getSessionId(),
        }),
        ...metadata,
        ...config.metadata,
      },
      tags: TAGS,
    });
    exporter.post(run);
    compaction = { run, requests: [], standalone: true };
  };

  const closeCompaction = (params: EndParams, summary?: string) => {
    const current = compaction;
    if (!current) return;
    compaction = undefined;
    current.requests.forEach((request, index) => {
      // The last summarization call produced the committed summary.
      const isLast = index === current.requests.length - 1;
      exporter.end(request, {
        outputs:
          isLast && summary ? { messages: [{ role: "assistant", content: summary }] } : undefined,
        error: params.error,
        metadata: { usage_missing: true },
      });
    });
    exporter.end(current.run, params);
  };

  pi.on("auto_compaction_start", (event, ctx) => {
    openCompaction(ctx, {
      reason: event.reason,
      action: event.action,
      trigger: "auto",
      ...("method" in event && typeof event.method === "string" ? { method: event.method } : {}),
    });
  });

  pi.on("session_before_compact", (event, ctx) => {
    openCompaction(ctx, {
      trigger: "manual",
      ...(event.customInstructions ? { custom_instructions: event.customInstructions } : {}),
    });
  });

  pi.on("session_compact", (event) => {
    const entry = event.compactionEntry;
    closeCompaction(
      {
        outputs: {
          summary: entry.summary,
          ...(entry.shortSummary ? { short_summary: entry.shortSummary } : {}),
        },
        metadata: {
          tokens_before: entry.tokensBefore,
          ...(entry.tokensAfter !== undefined ? { tokens_after: entry.tokensAfter } : {}),
          ...(entry.method ? { method: entry.method } : {}),
          from_extension: event.fromExtension,
        },
      },
      entry.summary,
    );
  });

  pi.on("auto_compaction_end", (event) => {
    closeCompaction({
      outputs: {
        aborted: event.aborted,
        will_retry: event.willRetry,
        ...(event.skipped ? { skipped: true } : {}),
      },
      // Esc during compaction is the user's choice, not a failure.
      error: event.aborted ? undefined : event.errorMessage,
      metadata: { action: event.action },
    });
  });

  pi.on("session_tree", (event, ctx) => {
    const entry = event.summaryEntry;
    if (!entry) return; // plain navigation, no summarization call
    const outputs = { summary: entry.summary };
    const metadata = {
      from_entry_id: entry.fromId,
      ...(event.fromExtension ? { from_extension: true } : {}),
    };
    if (state) {
      signal(BRANCH_SUMMARY_RUN, { parent: state.root, outputs, metadata });
      return;
    }
    const run = new RunTree({
      name: BRANCH_SUMMARY_RUN,
      run_type: "chain",
      project_name: config.project,
      client,
      inputs: { from_entry_id: entry.fromId },
      metadata: {
        ...codingAgentMetadata({
          agentType: "compaction",
          cwd: ctx.cwd,
          runtimeVersion: deps.runtimeVersion,
          threadId: ctx.sessionManager.getSessionId(),
        }),
        ...metadata,
        ...config.metadata,
      },
      tags: TAGS,
    });
    exporter.post(run);
    exporter.end(run, { outputs });
  });

  pi.registerCommand("langsmith-tracing", {
    description: "Show LangSmith tracing status for this omp session",
    handler: async (_args, ctx) => {
      const active = state ? ` Active trace: ${state.root.trace_id}.` : "";
      ctx.ui.notify(`LangSmith tracing is enabled (project "${config.project}").${active}`, "info");
    },
  });
}
