---
"@langchain/langsmith-pi-extension": minor
---

Make the extension an omp-native plugin. It now targets `@oh-my-pi/pi-coding-agent` 18.7+ (upstream Pi is no longer supported; stay on 0.2.x there), ships TypeScript source loaded by Bun through `omp.extensions`, and reads `langsmith.json` from the omp agent directory and `<project>/.omp/`.

Tracing fixes found in live omp traces:

- Steering or follow-up messages sent while the agent runs stay in the running trace as `steer` runs instead of ending it with "Trace replaced by a new Pi run".
- Concurrent omp events no longer crash the extension (`undefined is not an object (evaluating 'active.currentTurn')`).
- The last prompt of a session is no longer left pending forever; traces end at omp's terminal `agent_end`, and automatic continuations rejoin the same trace.
- A reply is bound to its own request, so overlapping side requests (session-title generation) no longer steal its output and usage.
- `ls_model_name` is the model id (and the served upstream model), not the display name.

Newly traced: interrupts, in-process subagents nested under the tool that spawned them, nested omp processes, compaction with its summarization call, retries, model fallbacks and switches, TTSR, tool approvals, todo reminders, goals, credential events, subagent spawns, cache-warming refreshes and branch summaries. LLM runs add time to first token, reasoning tokens, Anthropic cache-write TTL buckets, invocation params with tool schemas, response and request ids, and normalized OpenAI Responses (including chained Codex deltas) and Gemini inputs. Runs are renamed to low-cardinality names (`omp-agent`, `agent-step`, …), `ls_integration`/`ls_agent_runtime` are `omp`, LangSmith keys are redacted from exports, and `flushTraces()` lets SDK hosts wait for delivery.
