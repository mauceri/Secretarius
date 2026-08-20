import {
  App,
  CachedMetadata,
  Notice,
  Plugin,
  PluginSettingTab,
  Setting,
  requestUrl,
} from "obsidian";
import { buildCaptureText } from "./capture-text";

interface WikilmCaptureSettings {
  serverUrl: string;
}

const DEFAULT_SETTINGS: WikilmCaptureSettings = {
  serverUrl: "http://sanroque:5051",
};

function stripFrontmatter(raw: string, cache: CachedMetadata | null): string {
  const pos = cache?.frontmatterPosition;
  if (!pos) return raw;
  return raw.slice(pos.end.offset).replace(/^\s*\n/, "");
}

function extractTags(cache: CachedMetadata | null): string[] {
  const tags = cache?.frontmatter?.tags;
  if (!tags) return [];
  return Array.isArray(tags) ? tags.map(String) : [String(tags)];
}

export default class WikilmCapturePlugin extends Plugin {
  settings: WikilmCaptureSettings = DEFAULT_SETTINGS;

  async onload() {
    await this.loadSettings();
    this.addSettingTab(new WikilmCaptureSettingTab(this.app, this));
    this.addRibbonIcon("upload", "Capturer dans Wiki_LM", () => this.captureCurrentNote());
    this.addCommand({
      id: "capture-current-note",
      name: "Capturer la note courante dans Wiki_LM",
      callback: () => this.captureCurrentNote(),
    });
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }

  async captureCurrentNote(): Promise<void> {
    const file = this.app.workspace.getActiveFile();
    if (!file) {
      new Notice("Aucune note ouverte");
      return;
    }

    const cache = this.app.metadataCache.getFileCache(file);
    const raw = await this.app.vault.read(file);
    const body = stripFrontmatter(raw, cache);
    const tags = extractTags(cache);
    const text = buildCaptureText({ body, title: file.basename, path: file.path });

    if (cache?.frontmatter?.wiki_capture) {
      new Notice(`Déjà capturée le ${cache.frontmatter.wiki_capture} — nouvelle capture en cours…`);
    }

    try {
      const response = await requestUrl({
        url: `${this.settings.serverUrl}/capture`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ text, tags }),
      });
      const data = response.json as { filename: string };
      await this.app.fileManager.processFrontMatter(file, (fm) => {
        fm.wiki_capture = new Date().toISOString();
      });
      new Notice(`Capturée : ${data.filename}`);
    } catch (err) {
      new Notice(`Erreur de capture : ${err}`);
    }
  }
}

class WikilmCaptureSettingTab extends PluginSettingTab {
  plugin: WikilmCapturePlugin;

  constructor(app: App, plugin: WikilmCapturePlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    new Setting(containerEl)
      .setName("URL du serveur Wiki_LM")
      .setDesc("Adresse du serveur wiki-lm-server (ex. http://sanroque:5051)")
      .addText((text) =>
        text
          .setPlaceholder("http://sanroque:5051")
          .setValue(this.plugin.settings.serverUrl)
          .onChange(async (value) => {
            this.plugin.settings.serverUrl = value.trim();
            await this.plugin.saveSettings();
          })
      );
  }
}
