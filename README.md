# @langchain/langsmith-pi-extension

Trace [oh-my-pi (omp)](https://github.com/can1357/oh-my-pi) sessions to [LangSmith](https://smith.langchain.com): every prompt, agent step, LLM call, tool call, subagent, steer, retry and compaction, with token usage, cost, time to first token and the exact provider request.

This is an omp-native plugin. It loads from TypeScript source under Bun, follows omp's lifecycle events, and requires `@oh-my-pi/pi-coding-agent` 18.7 or later. Upstream Pi is no longer supported; use version 0.2.x of this package there.

## Installation

```sh
omp plugin install @langchain/langsmith-pi-extension
```

For a local checkout, link it instead (and uninstall the npm copy first so the plugin does not load twice):

```sh
omp plugin uninstall @langchain/langsmith-pi-extension
omp plugin link /path/to/langsmith-pi-extension
```

## Quick Start

Tracing is disabled by default. Enable it and supply a key, for example in the `.env` omp loads from its agent directory (`~/.omp/agent/.env`, or `~/.omp/profiles/<name>/agent/.env` for a profile):

```sh
TRACE_TO_LANGSMITH=true
LANGSMITH_PI_API_KEY=<your-langsmith-api-key>
```

Restart omp. The status line shows `LangSmith: tracing`, and `/langsmith-tracing` reports the project and the active trace. Traces go to the `pi-coding-agent` project unless you set one.

## What Gets Traced

Each user prompt becomes one trace in a LangSmith thread keyed by the omp session id:

```text
omp-agent                      chain  prompt → final answer; thread_id, turn_number, model, agent identity
├─ agent-step                  chain  one agent-loop iteration
│  ├─ anthropic / openai-codex llm    provider request → assistant message; usage, cost, TTFT, tools, params
│  ├─ bash, read, edit, …      tool   arguments → result; streamed updates as events, errors with their message
│  │  ├─ tool-llm-usage        llm    tokens a tool spent internally (web search, …)
│  │  └─ request/resolve-tool-approval
│  └─ task                     tool
│     └─ <agent>-agent         chain  in-process subagent (task, eval agent(), /tan), same trace and thread
├─ steer                       chain  a message you sent while the agent was running
├─ compact-context             chain  auto or manual compaction, with its summarization call and summary
└─ retry-request, settle-retry, apply-model-fallback, switch-model, trigger-ttsr-rules, spawn-subagent, …
```

- **Steering and follow-ups** stay inside the running trace as `steer` runs; they do not end it.
- **Interrupts** end the trace with `Interrupted (aborted)` and close every cut-short tool and LLM call.
- **Automatic continuations** (retry, overflow compaction) rejoin the same trace.
- **LLM calls** carry the provider-native request normalized to chat messages (Anthropic, OpenAI Chat and Responses including chained Codex deltas, Gemini), tool schemas and sampling settings as invocation params, the served model, `usage_metadata` with cache reads, Anthropic cache-write TTL buckets and reasoning tokens, provider cost (omitted when omp has no price so LangSmith prices the run), time to first token, response and request ids, and HTTP status.
- **Side requests** omp sends while a reply streams (session-title generation) are kept separate from the reply and marked `ls_omp_unmatched_request`.
- **Idle work** such as prompt-cache warming refreshes, `/compact` and branch summaries get their own traces in the thread.
- **Nested omp processes**: a `omp` started from a tool nests under the live trace through `LANGSMITH_PI_PARENT_DOTTED_ORDER` and `LANGSMITH_PI_PARENT_BAGGAGE`, which the plugin publishes while a prompt runs. Set them yourself to attach omp to an external trace.

Every run carries the `coding-agent-v1` metadata contract (`ls_integration: pi`, `ls_agent_runtime: omp`, `ls_agent_type`, `thread_id`, `turn_number`, git repository, branch and commit; `ls_tool_name` on tool runs, `ls_subagent_id`/`ls_subagent_type` on subagent runs). LangSmith API keys are redacted from everything the plugin exports.

Run names are deliberately low-cardinality so they work as filters: `eq(name, "omp-agent")` lists prompts, `eq(name, "steer")` finds steered runs, `eq(name, "compact-context")` finds compactions.

## Configuration

Values merge in this order, later sources winning:

1. Defaults
2. `<agentDir>/langsmith.json` — the active profile's agent directory (`~/.omp/agent` by default)
3. `<project>/.omp/langsmith.json`
4. Environment variables, including the agent directory's `.env`

### Environment Variables

| Variable                      | Description                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `TRACE_TO_LANGSMITH`          | Enables tracing when set to `true`, `1`, `yes`, or `on`. Disables tracing when set to `false`, `0`, `no`, or `off`. |
| `LANGSMITH_PI_API_KEY`        | LangSmith API key. Falls back to `LANGSMITH_API_KEY`.                                                               |
| `LANGSMITH_PI_ENDPOINT`       | LangSmith API URL for self-hosted or custom deployments. Falls back to `LANGSMITH_ENDPOINT`.                        |
| `LANGSMITH_PI_PROJECT`        | LangSmith project name. Falls back to `LANGSMITH_PROJECT`. Defaults to `pi-coding-agent`.                           |
| `LANGSMITH_PI_METADATA`       | JSON object added to root run metadata. Falls back to `LANGSMITH_METADATA`.                                         |
| `LANGSMITH_PI_RUNS_ENDPOINTS` | JSON array of replica run destinations. Falls back to `LANGSMITH_RUNS_ENDPOINTS`.                                   |

### Config File

```json
{
  "enabled": true,
  "api_key": "<your-langsmith-api-key>",
  "api_url": "https://api.smith.langchain.com",
  "project": "pi-coding-agent",
  "metadata": { "environment": "local" },
  "replicas": [
    {
      "api_url": "https://replica-langsmith.example.com",
      "api_key": "<replica-api-key>",
      "project": "pi-coding-agent-replica",
      "updates": { "tags": ["replica"] }
    }
  ]
}
```

| Field      | Required | Default               | Description                                                                      |
| ---------- | -------- | --------------------- | -------------------------------------------------------------------------------- |
| `enabled`  | Yes      | `false`               | Set to `true` to enable tracing from the config file.                            |
| `api_key`  | No\*     | -                     | LangSmith API key. Required unless provided by environment variable or replicas. |
| `api_url`  | No       | LangSmith SDK default | LangSmith API URL, usually `https://api.smith.langchain.com`.                    |
| `project`  | No       | `pi-coding-agent`     | LangSmith project name.                                                          |
| `metadata` | No       | -                     | Object merged into root trace metadata.                                          |
| `replicas` | No       | -                     | Additional LangSmith destinations runs are replicated to.                        |

## Programmatic Usage

A Bun host embedding omp through its SDK can pass the plugin as an inline extension factory:

```ts
import { createAgentSession, SessionManager } from "@oh-my-pi/pi-coding-agent";
import langsmith, { flushTraces } from "@langchain/langsmith-pi-extension";

const { session } = await createAgentSession({
  sessionManager: SessionManager.inMemory(),
  extensions: [
    (pi) =>
      langsmith(pi, {
        // Used instead of env vars and langsmith.json files; defaults still apply.
        config: { enabled: true, api_key: "...", project: "my-app-agents" },
        // Optional: nest every omp run under a run your host manages. Defaults to
        // the ambient `traceable` run, if any.
        // getCurrentRunTree: () => myRunTree,
      }),
  ],
});

await session.prompt("List files in the current directory.");
await session.dispose();
await flushTraces(); // wait until every queued run reached LangSmith
```

Options: `config` (skip discovery), `client` (your own LangSmith `Client`), and `getCurrentRunTree` (parent run for each omp run; its client and project then decide where the runs land, while `config.replicas` still applies).

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup, testing, and pull request guidance.
