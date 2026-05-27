import { parseConfig } from "./config.js";
import { McpClientPool } from "./mcp-client.js";
export default function (api) {
    // api.pluginConfig contient la config spécifique au plugin (servers en tableau).
    // api.config.mcp contient le bloc mcp racine (servers en objet, format de openclaw mcp set).
    // On fusionne les deux pour accepter les deux sources.
    const pluginCfg = api.pluginConfig ?? {};
    const mcpCfg = api.config?.mcp ?? {};
    const merged = Object.keys(pluginCfg.servers ?? {}).length > 0 || Array.isArray(pluginCfg.servers) && pluginCfg.servers.length > 0
        ? pluginCfg
        : mcpCfg;
    const config = parseConfig(merged);
    if (config.servers.length === 0) {
        return;
    }
    const pool = new McpClientPool();
    // registerService.start() n'est pas appelé lors d'un démarrage à froid pour les
    // plugins non-bundlés (bug lifecycle OpenClaw). On initialise directement.
    (async () => {
        for (const server of config.servers) {
            try {
                console.log(`[mcp-adapter] Connecting to ${server.name}...`);
                await pool.connect(server);
                const tools = await pool.listTools(server.name);
                console.log(`[mcp-adapter] ${server.name}: found ${tools.length} tools`);
                for (const tool of tools) {
                    const toolName = config.toolPrefix ? `${server.name}_${tool.name}` : tool.name;
                    api.registerTool({
                        name: toolName,
                        description: tool.description ?? `Tool from ${server.name}`,
                        parameters: tool.inputSchema ?? { type: "object", properties: {} },
                        async execute(_id, params) {
                            const result = await pool.callTool(server.name, tool.name, params);
                            const text = result.content
                                ?.map((c) => c.text ?? c.data ?? "")
                                .join("\n") ?? "";
                            return {
                                content: [{ type: "text", text }],
                                isError: result.isError,
                            };
                        },
                    });
                    console.log(`[mcp-adapter] Registered: ${toolName}`);
                }
            }
            catch (err) {
                console.error(`[mcp-adapter] Failed to connect to ${server.name}:`, err);
            }
        }
    })();
    // Nettoyage sur arrêt du gateway
    api.registerService({
        id: "mcp-adapter-cleanup",
        async start() { },
        async stop() {
            console.log("[mcp-adapter] Shutting down...");
            await pool.closeAll();
            console.log("[mcp-adapter] All connections closed");
        },
    });
}
