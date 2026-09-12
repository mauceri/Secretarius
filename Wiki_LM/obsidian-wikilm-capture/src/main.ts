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
import {
  applyWikiResults,
  extractWikiBlocks,
  formatWikiResult,
  SUPPORTED_COMMANDS,
  WikiBlockMatch,
} from "./run-commands";

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
    this.addRibbonIcon("play", "Exécuter les commandes wiki de la note", () =>
      this.runWikiCommands()
    );
    this.addCommand({
      id: "run-wiki-commands",
      name: "Exécuter les commandes wiki de la note",
      callback: () => this.runWikiCommands(),
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
        body: JSON.stringify({ text, tags, title: file.basename }),
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

  async runWikiCommands(): Promise<void> {
    const file = this.app.workspace.getActiveFile();
    if (!file) {
      new Notice("Aucune note ouverte");
      return;
    }

    const raw = await this.app.vault.read(file);
    const blocks = extractWikiBlocks(raw);
    if (blocks.length === 0) {
      new Notice("Aucune commande wiki trouvée dans la note");
      return;
    }

    const resultTexts: string[] = [];
    for (const block of blocks) {
      resultTexts.push(await this.runOneCommand(block));
    }

    const newContent = applyWikiResults(raw, blocks, resultTexts);
    await this.app.vault.modify(file, newContent);
    new Notice(`Lot exécuté : ${blocks.length} commande(s)`);
  }

  async runOneCommand(block: WikiBlockMatch): Promise<string> {
    if (!(SUPPORTED_COMMANDS as readonly string[]).includes(block.command)) {
      return formatWikiResult(block.command, {}, true);
    }
    try {
      const response = await requestUrl({
        url: `${this.settings.serverUrl}/run`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ command: block.command, arg: block.arg }),
        throw: false,
      });
      const data = (response.json as Record<string, unknown>) ?? {};
      return formatWikiResult(block.command, data, response.status === 200);
    } catch (err) {
      return formatWikiResult(block.command, { error: String(err) }, false);
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
