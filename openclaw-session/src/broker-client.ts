// Client HTTP + SSE du broker Session (session-broker, 127.0.0.1).
export type InboundAttachment = { path: string; name: string; contentType: string; size: number; error?: "trop-grande" };
export type InboundEvent = { id: string; from: string; timestamp: number; text: string; attachments: InboundAttachment[] };

export class BrokerNotAllowedError extends Error {}

type Fetch = typeof fetch;

export class BrokerClient {
  constructor(private url: string, private token: string, private fetchImpl: Fetch = fetch) {}

  private headers(json = false): Record<string, string> {
    return { Authorization: `Bearer ${this.token}`, ...(json ? { "Content-Type": "application/json" } : {}) };
  }

  private async post(path: string, body: unknown): Promise<Response> {
    const r = await this.fetchImpl(`${this.url}${path}`, { method: "POST", headers: this.headers(true), body: JSON.stringify(body) });
    if (r.status === 403) throw new BrokerNotAllowedError(`${path} : destinataire non autorisé`);
    if (!r.ok) throw new Error(`${path} : HTTP ${r.status}`);
    return r;
  }

  async *events(signal: AbortSignal): AsyncIterable<InboundEvent> {
    const r = await this.fetchImpl(`${this.url}/events`, { headers: this.headers(), signal });
    if (r.status !== 200 || !r.body) throw new Error(`/events : HTTP ${r.status}`);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) return;
        buf += dec.decode(value, { stream: true });
        let i: number;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const bloc = buf.slice(0, i);
          buf = buf.slice(i + 2);
          const data = bloc.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
          if (data) yield JSON.parse(data) as InboundEvent;
        }
      }
    } catch (e) {
      if (signal.aborted) return;
      throw e;
    }
  }

  async ack(id: string): Promise<void> { await this.post("/ack", { id }); }

  async send(to: string, text: string): Promise<{ messageId: string }> {
    const j: any = await (await this.post("/send", { to, text })).json();
    return { messageId: String(j.messageId) };
  }

  async typing(to: string, on: boolean): Promise<void> { await this.post("/typing", { to, on }); }

  async health(): Promise<any> {
    const r = await this.fetchImpl(`${this.url}/health`, { headers: this.headers() });
    if (!r.ok) throw new Error(`/health : HTTP ${r.status}`);
    return r.json();
  }
}
