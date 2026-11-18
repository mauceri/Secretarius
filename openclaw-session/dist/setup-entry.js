import { defineSetupPluginEntry } from "openclaw/plugin-sdk/channel-core";
import { sessionPlugin } from "./src/channel.js";
export default defineSetupPluginEntry(sessionPlugin);
