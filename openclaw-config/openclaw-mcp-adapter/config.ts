export interface ServerConfig {
  name: string;
  transport: "stdio" | "http";
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  headers?: Record<string, string>;
}

export interface McpAdapterConfig {
  servers: ServerConfig[];
  toolPrefix: boolean;
}

function interpolateEnv(obj: Record<string, string>): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [k, v] of Object.entries(obj)) {
    result[k] = v.replace(/\$\{([^}]+)\}/g, (_, name) => process.env[name] ?? "");
  }
  return result;
}

export function parseConfig(raw: unknown): McpAdapterConfig {
  const cfg = (raw ?? {}) as Record<string, unknown>;
  const servers: ServerConfig[] = [];

  // Accepte deux formats :
  // - tableau : [{name, command, args}, ...]  (format original)
  // - objet   : {"nom": {command, args}, ...} (format produit par openclaw mcp set)
  const serversRaw = cfg.servers;
  const serverEntries: Array<Record<string, unknown>> = [];
  if (Array.isArray(serversRaw)) {
    serverEntries.push(...(serversRaw as Record<string, unknown>[]));
  } else if (serversRaw && typeof serversRaw === "object") {
    for (const [name, srv] of Object.entries(serversRaw as Record<string, unknown>)) {
      serverEntries.push({ name, ...(srv as Record<string, unknown>) });
    }
  }

  for (const s of serverEntries) {
    const srv = s as Record<string, unknown>;
    if (!srv.name) throw new Error("Server missing 'name'");

    let transport = (srv.transport as string) ?? (srv.url ? "http" : "stdio");
    // "streamable-http" est le terme standard MCP ; l'adaptateur le traite
    // comme "http" (StreamableHTTPClientTransport).
    if (transport === "streamable-http") transport = "http";
    if (transport === "stdio" && !srv.command) throw new Error(`Server "${srv.name}" missing 'command'`);
    if (transport === "http" && !srv.url) throw new Error(`Server "${srv.name}" missing 'url'`);

    servers.push({
      name: srv.name as string,
      transport: transport as "stdio" | "http",
      command: srv.command as string | undefined,
      args: srv.args as string[] | undefined,
      env: srv.env ? interpolateEnv(srv.env as Record<string, string>) : undefined,
      url: srv.url as string | undefined,
      headers: srv.headers ? interpolateEnv(srv.headers as Record<string, string>) : undefined,
    });
  }

  return {
    servers,
    toolPrefix: cfg.toolPrefix === true,
  };
}
