function interpolateEnv(obj) {
    const result = {};
    for (const [k, v] of Object.entries(obj)) {
        result[k] = v.replace(/\$\{([^}]+)\}/g, (_, name) => process.env[name] ?? "");
    }
    return result;
}
export function parseConfig(raw) {
    const cfg = (raw ?? {});
    const servers = [];
    // Accepte deux formats :
    // - tableau : [{name, command, args}, ...]  (format original)
    // - objet   : {"nom": {command, args}, ...} (format produit par openclaw mcp set)
    const serversRaw = cfg.servers;
    const serverEntries = [];
    if (Array.isArray(serversRaw)) {
        serverEntries.push(...serversRaw);
    }
    else if (serversRaw && typeof serversRaw === "object") {
        for (const [name, srv] of Object.entries(serversRaw)) {
            serverEntries.push({ name, ...srv });
        }
    }
    for (const s of serverEntries) {
        const srv = s;
        if (!srv.name)
            throw new Error("Server missing 'name'");
        const transport = srv.transport ?? "stdio";
        if (transport === "stdio" && !srv.command)
            throw new Error(`Server "${srv.name}" missing 'command'`);
        if (transport === "http" && !srv.url)
            throw new Error(`Server "${srv.name}" missing 'url'`);
        servers.push({
            name: srv.name,
            transport: transport,
            command: srv.command,
            args: srv.args,
            env: srv.env ? interpolateEnv(srv.env) : undefined,
            url: srv.url,
            headers: srv.headers ? interpolateEnv(srv.headers) : undefined,
        });
    }
    return {
        servers,
        toolPrefix: cfg.toolPrefix !== false,
    };
}
