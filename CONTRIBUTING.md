# Contributing

Thanks for helping improve `@langchain/langsmith-pi-extension`, the LangSmith tracing plugin for oh-my-pi (omp).

## Development Setup

Dependencies, scripts and tests all run on Bun, the runtime omp loads plugins with.

```sh
bun install
bun test            # test suite
bun run lint        # oxfmt check + tsc
bun run format      # format files
```

There is no build step: omp loads `src/index.ts` directly, and the package publishes `src/`.

To try changes in a real session, link the checkout as a plugin (uninstall an npm copy first so the plugin does not load twice), then start a new omp session:

```sh
omp plugin uninstall @langchain/langsmith-pi-extension
omp plugin link .
```

For a one-off run that loads only this checkout:

```sh
omp --no-extensions -e src/index.ts -p "your prompt"
```

## Project Layout

- `src/index.ts` — extension factory, options, shared LangSmith clients.
- `src/tracer.ts` — omp lifecycle event handlers and the run-tree state machine.
- `src/convert.ts` — pure conversions: provider payloads, messages, usage, model parameters, redaction.
- `src/config.ts`, `src/metadata.ts` — configuration discovery and the `coding-agent-v1` metadata contract.
- `test/` — Bun tests, recordings and utilities.

## Making Changes

Handlers in `src/tracer.ts` must stay synchronous with respect to tracing state: omp dispatches events concurrently, so a handler that awaits and then reads shared state can see another prompt's state. Network I/O goes through `RunExporter`, which orders each run's post and patch without blocking handlers.

When changing tracing behavior, configuration parsing, or run serialization, add or update tests that assert the exported run tree. `test/lifecycle.test.ts` drives omp-shaped event sequences through a fake host (`test/utils/harness.ts`); `test/replay.test.ts` replays real recordings.

## Record and Replay

To capture a real event stream, load the recorder next to (or instead of) the plugin; events append to `langsmith.jsonl` at the repo root:

```sh
omp --no-extensions -e test/utils/record.ts -p "your prompt"
```

Before committing a recording under `test/recordings/`, inspect it for secrets, local paths, prompts, user identifiers, and API keys. Keep the shortest recording that exercises the behavior.

Replay it with `replay` and a mocked client, then assert on the tree:

```ts
import { expect, it } from "bun:test";
import extension, { flushTraces } from "../src/index";
import { mockClient } from "./utils/mock_client";
import { parseRecording, replay } from "./utils/replay";
import { getAssumedTreeFromCalls } from "./utils/tree";

it("my scenario", async () => {
  const { client, callSpy } = mockClient();
  const events = parseRecording(await Bun.file("test/recordings/my-scenario.jsonl").text());
  await replay((session) => extension(session.pi, { client, config: { enabled: true } }), events);
  await flushTraces();
  const tree = await getAssumedTreeFromCalls(callSpy.mock.calls, client);
  expect(tree.nodes.some((node) => node.startsWith("omp-agent:"))).toBe(true);
});
```

## Changesets

User-facing package changes should include a changeset (`bun run changeset`):

- `patch` for bug fixes and internal improvements.
- `minor` for backward-compatible features.
- `major` for breaking changes.

Documentation-only changes generally do not need a changeset.

## Secrets and Local Configuration

Do not commit LangSmith API keys, `.env` files, or local omp configuration. For manual testing, put credentials in your omp agent directory's `.env`:

```sh
TRACE_TO_LANGSMITH=true
LANGSMITH_PI_API_KEY=<your-langsmith-api-key>
```

## Pull Request Checklist

- Tests cover the changed behavior, or the PR explains why tests are not needed.
- `bun run lint` and `bun test` pass locally.
- User-facing changes are reflected in `README.md`.
- A changeset is included for package behavior changes.
- No secrets or local-only configuration files are committed.
