// Run docs/search-worker.js under Node worker_threads with minimal browser shims (self, importScripts, fetch)
// and solve one test cube with a given config. Usage: node experiments/verify_worker.cjs <cfg>
const { Worker } = require("worker_threads");
const path = require("path"), fs = require("fs");
const docs = path.join(__dirname, "../docs");
const shim = `
const { parentPort } = require("worker_threads"); const fs = require("fs"), path = require("path"), vm = require("vm");
const docs = ${JSON.stringify(docs)};
globalThis.self = globalThis;
self.postMessage = m => parentPort.postMessage(m);
globalThis.importScripts = f => vm.runInThisContext(fs.readFileSync(path.join(docs, f), "utf8"));
globalThis.fetch = async u => ({ ok: true, json: async () => JSON.parse(fs.readFileSync(path.join(docs, u), "utf8")) });
vm.runInThisContext(fs.readFileSync(path.join(docs, "search-worker.js"), "utf8"));
parentPort.on("message", d => self.onmessage({ data: d }));`;
const w = new Worker(shim, { eval: true });
const cfg = process.argv[2] || "stages_nonn_lib_s0";
const start = JSON.parse(fs.readFileSync(path.join(__dirname, "e2e_cubes.json")))[0];
w.on("message", m => {
  if (m.type === "done") { console.log(JSON.stringify(m.result).slice(0, 400)); process.exit(m.result.ok ? 0 : 2); }
  else if (m.type === "error") { console.log("ERROR", m.error); process.exit(1); }
  else console.log(m.type, m.stage || m.n);
});
w.postMessage({ id: 1, cfg, kind: cfg.split("_")[0], start, opts: { maxPrograms: 50000 } });
