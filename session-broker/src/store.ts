import { Database } from "bun:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { InboundEvent } from "./inbound";

// Dédoublonnage des identifiants de message et file des événements non acquittés.
export class Store {
  private db: Database;
  constructor(path: string) {
    mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
    this.db = new Database(path, { create: true });
    this.db.run("CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY)");
    this.db.run("CREATE TABLE IF NOT EXISTS pending (id TEXT PRIMARY KEY, evt TEXT NOT NULL, ts INTEGER NOT NULL)");
  }
  /** Lecture seule : l'id a-t-il déjà été mis en file ? */
  seen(id: string): boolean {
    return this.db.query("SELECT 1 FROM seen WHERE id = ?").get(id) != null;
  }
  /** Met en file ET marque vu, dans une même transaction. */
  enqueue(evt: InboundEvent): void {
    this.db.transaction(() => {
      this.db.query("INSERT OR IGNORE INTO seen (id) VALUES (?)").run(evt.id);
      this.db.query("INSERT OR REPLACE INTO pending (id, evt, ts) VALUES (?, ?, ?)").run(evt.id, JSON.stringify(evt), Date.now());
    })();
  }
  pending(): InboundEvent[] {
    return (this.db.query("SELECT evt FROM pending ORDER BY ts").all() as { evt: string }[]).map((r) => JSON.parse(r.evt));
  }
  ack(id: string): void {
    this.db.query("DELETE FROM pending WHERE id = ?").run(id);
  }
}
