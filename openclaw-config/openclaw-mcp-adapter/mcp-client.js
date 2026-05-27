import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
export class McpClientPool {
    clients = new Map();
    async connect(config) {
        const client = new Client({ name: "openclaw-mcp-adapter", version: "0.1.0" });
        const transport = this.createTransport(config);
        await client.connect(transport);
        if (transport instanceof StdioClientTransport) {
            // unref() évite que le sous-processus MCP bloque la sortie du processus parent
            // (notamment lors de `openclaw plugins install` qui charge le plugin en in-process)
            const proc = transport._process ?? transport.process;
            if (proc?.unref)
                proc.unref();
            transport.onerror = () => this.markDisconnected(config.name);
            transport.onclose = () => this.markDisconnected(config.name);
        }
        this.clients.set(config.name, { config, client, transport, connected: true });
        return client;
    }
    createTransport(config) {
        if (config.transport === "http") {
            return new StreamableHTTPClientTransport(new URL(config.url), {
                requestInit: { headers: config.headers },
            });
        }
        return new StdioClientTransport({
            command: config.command,
            args: config.args,
            env: { ...process.env, ...config.env },
        });
    }
    async listTools(serverName) {
        const entry = this.clients.get(serverName);
        if (!entry)
            throw new Error(`Unknown server: ${serverName}`);
        const result = await entry.client.listTools();
        return result.tools;
    }
    async callTool(serverName, toolName, args) {
        const entry = this.clients.get(serverName);
        if (!entry)
            throw new Error(`Unknown server: ${serverName}`);
        try {
            return await entry.client.callTool({ name: toolName, arguments: args });
        }
        catch (err) {
            if (!entry.connected || this.isConnectionError(err)) {
                await this.reconnect(serverName);
                const newEntry = this.clients.get(serverName);
                return await newEntry.client.callTool({ name: toolName, arguments: args });
            }
            throw err;
        }
    }
    async reconnect(serverName) {
        const entry = this.clients.get(serverName);
        if (!entry)
            return;
        try {
            await entry.transport.close?.();
        }
        catch { }
        await this.connect(entry.config);
    }
    markDisconnected(serverName) {
        const entry = this.clients.get(serverName);
        if (entry)
            entry.connected = false;
    }
    isConnectionError(err) {
        const msg = String(err);
        return msg.includes("closed") || msg.includes("ECONNREFUSED") || msg.includes("EPIPE");
    }
    getStatus(serverName) {
        const entry = this.clients.get(serverName);
        return { connected: entry?.connected ?? false };
    }
    async close(serverName) {
        const entry = this.clients.get(serverName);
        if (!entry)
            return;
        try {
            await entry.transport.close?.();
        }
        catch {
            // Ignore close errors
        }
        this.clients.delete(serverName);
    }
    async closeAll() {
        for (const name of this.clients.keys()) {
            await this.close(name);
        }
    }
}
