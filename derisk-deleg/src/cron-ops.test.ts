import { afterEach, describe, expect, it, vi } from "vitest";
import { execFile } from "node:child_process";

vi.mock("node:child_process", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:child_process")>();
  return { ...actual, execFile: vi.fn() };
});

afterEach(() => { vi.mocked(execFile).mockReset(); vi.unstubAllEnvs(); });

function stdoutIs(out: string) {
  vi.mocked(execFile).mockImplementation(((_c: any, _a: any, _o: any, cb: any) =>
    cb(null, { stdout: out, stderr: "" })) as any);
}

describe("runTironCron", () => {
  it("relaie le JSON de la CLI et lance python3 -m tiron_cron dans SECRETARIUS_DIR", async () => {
    vi.stubEnv("SECRETARIUS_DIR", "/srv/sec");
    stdoutIs('{"ok": true, "text": "Chaque lundi…", "args": "add …"}');
    const { runTironCron } = await import("./cron-ops.js");
    expect(await runTironCron(["preview", "--origin", "telegram", "--args", "add …"]))
      .toEqual({ ok: true, text: "Chaque lundi…", args: "add …" });
    const [cmd, args, opts] = vi.mocked(execFile).mock.calls[0] as any[];
    expect(cmd).toBe("python3");
    expect(args).toEqual(["-m", "tiron_cron", "preview", "--origin", "telegram", "--args", "add …"]);
    expect(opts.cwd).toBe("/srv/sec");
  });

  it("échec du processus -> ok:false avec message", async () => {
    vi.mocked(execFile).mockImplementation(((_c: any, _a: any, _o: any, cb: any) =>
      cb(new Error("ENOENT python3"))) as any);
    const { runTironCron } = await import("./cron-ops.js");
    const r = await runTironCron(["manage", "--args", "list"]);
    expect(r.ok).toBe(false);
    expect(r.text).toMatch(/^Planificateur indisponible : .*ENOENT/);
  });
});
