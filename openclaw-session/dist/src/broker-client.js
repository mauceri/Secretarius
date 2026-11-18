export class BrokerNotAllowedError extends Error {
}
export class BrokerClient {
    url;
    token;
    fetchImpl;
    constructor(url, token, fetchImpl = fetch) {
        this.url = url;
        this.token = token;
        this.fetchImpl = fetchImpl;
    }
    headers(json = false) {
        return { Authorization: `Bearer ${this.token}`, ...(json ? { "Content-Type": "application/json" } : {}) };
    }
    async post(path, body) {
        const r = await this.fetchImpl(`${this.url}${path}`, { method: "POST", headers: this.headers(true), body: JSON.stringify(body) });
        if (r.status === 403)
            throw new BrokerNotAllowedError(`${path} : destinataire non autorisé`);
        if (!r.ok)
            throw new Error(`${path} : HTTP ${r.status}`);
        return r;
    }
    async *events(signal) {
        const r = await this.fetchImpl(`${this.url}/events`, { headers: this.headers(), signal });
        if (r.status !== 200 || !r.body)
            throw new Error(`/events : HTTP ${r.status}`);
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        try {
            while (true) {
                const { value, done } = await reader.read();
                if (done)
                    return;
                buf += dec.decode(value, { stream: true });
                let i;
                while ((i = buf.indexOf("\n\n")) >= 0) {
                    const bloc = buf.slice(0, i);
                    buf = buf.slice(i + 2);
                    const data = bloc.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
                    if (data)
                        yield JSON.parse(data);
                }
            }
        }
        catch (e) {
            if (signal.aborted)
                return;
            throw e;
        }
    }
    async ack(id) { await this.post("/ack", { id }); }
    async send(to, text) {
        const j = await (await this.post("/send", { to, text })).json();
        return { messageId: String(j.messageId) };
    }
    async typing(to, on) { await this.post("/typing", { to, on }); }
    async health() {
        const r = await this.fetchImpl(`${this.url}/health`, { headers: this.headers() });
        if (!r.ok)
            throw new Error(`/health : HTTP ${r.status}`);
        return r.json();
    }
}
