import { createFakeSession, type FakeSession, type FakeSessionOptions } from "./harness";

export type RecordedEvent = [
  time: number,
  name: string,
  event: Record<string, unknown>,
  ctx: Record<string, unknown>,
];

export const parseRecording = (contents: string): RecordedEvent[] =>
  contents
    .split(/\r?\n/)
    .map((line, index) => [line.trim(), index + 1] as const)
    .filter(([line]) => line !== "")
    .map(([line, lineNumber]) => {
      try {
        return JSON.parse(line) as RecordedEvent;
      } catch (error) {
        throw new Error(`Invalid recording line ${lineNumber}: ${(error as Error).message}`);
      }
    });

/**
 * Replays a recording (written by `test/utils/record.ts`) into a fake OMP host.
 * Each event runs with the context the recording captured (model, cwd, agent
 * identity, context usage), so provider-specific run names, payloads, agent
 * nesting and the final usage match the real session.
 */
export async function replay(
  init: (session: FakeSession) => Promise<void>,
  events: RecordedEvent[],
  options: FakeSessionOptions = {},
): Promise<FakeSession> {
  const session = createFakeSession(options);
  await init(session);
  for (const [, name, event, recordedCtx] of events) {
    for (const key of ["model", "cwd", "agent"] as const) {
      if (recordedCtx[key] !== undefined) Object.assign(session.ctx, { [key]: recordedCtx[key] });
    }
    if (recordedCtx.getContextUsage !== undefined) {
      const usage = recordedCtx.getContextUsage;
      Object.assign(session.ctx, { getContextUsage: () => usage });
    }
    if (name === "before_agent_start" && typeof event.prompt === "string") {
      session.entries.push({ type: "message", message: { role: "user", content: event.prompt } });
    }
    await session.fire(name, event);
  }
  return session;
}
