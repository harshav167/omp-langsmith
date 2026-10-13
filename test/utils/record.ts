import * as fs from "node:fs/promises";
import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";

/**
 * Records every OMP lifecycle event to `langsmith.jsonl` at the repo root, in
 * the `[time, event, payload, ctx]` shape `test/utils/replay.ts` reads. Load it
 * next to the extension (`omp -e test/utils/record.ts`) to capture a fixture.
 */
const EVENTS = [
  "session_start",
  "session_switch",
  "session_branch",
  "session_before_compact",
  "session_compact",
  "session_tree",
  "session_shutdown",
  "input",
  "before_agent_start",
  "agent_start",
  "agent_end",
  "turn_start",
  "turn_end",
  "context",
  "before_provider_request",
  "after_provider_response",
  "message_start",
  // message_update is only useful for streaming and bloats fixtures.
  "message_end",
  "tool_execution_start",
  "tool_execution_update",
  "tool_execution_end",
  "tool_approval_requested",
  "tool_approval_resolved",
  "before_subagent_spawn",
  "auto_compaction_start",
  "auto_compaction_end",
  "auto_retry_start",
  "auto_retry_end",
  "retry_fallback_applied",
  "retry_fallback_succeeded",
  "ttsr_triggered",
  "todo_reminder",
  "goal_updated",
  "cache_warming_decision",
  "credential_disabled",
] as const;

export default function record(pi: ExtensionAPI) {
  const output = new URL("../../langsmith.jsonl", import.meta.url);
  for (const name of EVENTS) {
    // The overloads are keyed by literal event name; the loop name is one of them.
    (
      pi.on as (
        event: string,
        handler: (event: unknown, ctx: ExtensionContext) => Promise<void>,
      ) => void
    )(name, async (event, ctx) => {
      await fs.appendFile(
        output,
        `${JSON.stringify([
          Date.now(),
          name,
          event,
          {
            cwd: ctx.cwd,
            model: ctx.model,
            agent: ctx.agent,
            getContextUsage: ctx.getContextUsage?.(),
          },
        ])}\n`,
      );
    });
  }
}
