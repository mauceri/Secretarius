import { defineChannelPluginEntry } from "openclaw/plugin-sdk/channel-core";
import { sessionPlugin } from "./src/channel.js";
export default defineChannelPluginEntry({
    id: "session",
    name: "Session",
    description: "Canal Session Messenger (broker local de Secretarius)",
    plugin: sessionPlugin,
});
