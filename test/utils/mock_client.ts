import { mock } from "bun:test";
import { Client, type ClientConfig } from "langsmith";

/** A LangSmith client whose HTTP layer records every request instead of sending it. */
export const mockClient = (config?: Omit<ClientConfig, "autoBatchTracing">) => {
  const callSpy = mock(
    async (_url: string | URL | Request, _init?: RequestInit) =>
      new Response("{}", { status: 200, headers: { "content-type": "application/json" } }),
  );
  // Bun's `typeof fetch` also carries `preconnect`; the client only ever calls it.
  const fetchImplementation = callSpy as unknown as typeof fetch;
  const client = new Client({
    apiKey: "test-api-key",
    ...config,
    autoBatchTracing: false,
    fetchImplementation,
  });
  return { client, callSpy };
};
