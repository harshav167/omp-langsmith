import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { Client, type RunTree } from "langsmith";
import { getCurrentRunTree } from "langsmith/singletons/traceable";

import { type Config, ConfigSchema, DEFAULT_PROJECT, getConfig } from "./config.js";
import { createRedactor } from "./convert.js";
import { RunExporter, registerTracing } from "./tracer.js";

export { flushTraces } from "./tracer.js";

export type { Config } from "./config.js";
export {
  buildUsageMetadata,
  convertMessages,
  convertProviderPayload,
  convertToolOutputs,
  normalizeContentPart,
} from "./convert.js";

export interface LangSmithExtensionOptions {
  /**
   * LangSmith client to use instead of one built from the resolved config. The
   * extension installs its LangSmith-key redactor only on clients it builds, so
   * a host-supplied client exports inputs, outputs and errors exactly as given:
   * configure its `anonymizer` if traces may carry credentials.
   */
  client?: Client;
  /**
   * Extension config to use instead of discovering one from the environment and
   * `langsmith.json` files. Lets a host that embeds omp through its SDK configure
   * tracing without touching the process environment.
   */
  config?: Config;
  /**
   * When set, each omp agent run is created as a child of the returned run
   * instead of starting a new trace. Called at the start of each run; returning
   * `undefined` falls back to a new root trace. Defaults to the ambient
   * `traceable` run, if any.
   *
   * The returned run's client and project are inherited, so `config.project`,
   * `config.api_key`, `config.api_url` and `client` no longer decide where those
   * traces land. `config.replicas` still applies.
   */
  getCurrentRunTree?: () => RunTree | undefined;
}

/**
 * One client per destination and one exporter per client, shared by every
 * session in the process: omp rebinds this factory into each in-process
 * subagent, and a shared batch queue keeps parent and child runs flushing together.
 */
const clients = new Map<string, Client>();
const exporters = new WeakMap<Client, RunExporter>();

const clientFor = (config: Config): Client => {
  // The redactor is built from every key the config carries, so a config that
  // differs only in replica keys needs its own client.
  const replicaKeys = (config.replicas ?? []).map((replica) => replica.api_key ?? "");
  const key = JSON.stringify([config.api_url ?? "", config.api_key ?? "", replicaKeys]);
  let client = clients.get(key);
  if (!client) {
    client = new Client({
      apiKey: config.api_key,
      apiUrl: config.api_url,
      anonymizer: createRedactor(config.api_key, ...(config.replicas ?? []).map((r) => r.api_key)),
    });
    clients.set(key, client);
  }
  return client;
};

const exporterFor = (client: Client): RunExporter => {
  let exporter = exporters.get(client);
  if (!exporter) {
    exporter = new RunExporter(client);
    exporters.set(client, exporter);
  }
  return exporter;
};

const agentDirOf = (pi: ExtensionAPI): string | undefined => {
  try {
    return pi.pi?.getAgentDir?.();
  } catch {
    return undefined;
  }
};

export default async function langsmithExtension(
  pi: ExtensionAPI,
  options?: LangSmithExtensionOptions,
): Promise<void> {
  const config =
    options?.config != null
      ? ConfigSchema.parse({
          ...options.config,
          project: options.config.project || DEFAULT_PROJECT,
        })
      : await getConfig({ agentDir: agentDirOf(pi), cwd: process.cwd(), env: process.env });

  if (!config.enabled) {
    pi.on("session_start", (_event, ctx) => {
      if (ctx.hasUI) ctx.ui.setStatus("langsmith", "LangSmith: disabled");
    });
    pi.registerCommand("langsmith-tracing", {
      description: "Show LangSmith tracing status for this omp session",
      handler: async (_args, ctx) => {
        ctx.ui.notify(
          "LangSmith tracing is disabled. Set TRACE_TO_LANGSMITH=true and LANGSMITH_PI_API_KEY (for example in the omp agent directory's .env), or enable it in langsmith.json.",
          "warning",
        );
      },
    });
    return;
  }

  const exporter = exporterFor(options?.client ?? clientFor(config));
  registerTracing(pi, {
    config,
    exporter,
    getParentRunTree: options?.getCurrentRunTree ?? (() => getCurrentRunTree(true)),
    runtimeVersion: pi.pi?.VERSION,
  });
}
