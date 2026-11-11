// Web Worker: runs the DreamCoder search (search.js) off the UI thread.
// in:  {id, cfg: "<kind>_<arm>_s<seed>", kind, start: 24 ints, opts: {maxPrograms?, timeMs?}}
// out: {id, type: "stage", stage} | {id, type: "progress", n} | {id, type: "done", result} | {id, type: "error", error}
importScripts("search.js");
const models = new Map();
let tablesP = null;

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
async function model(cfg) {
  if (!tablesP) tablesP = getJSON("models/tables.json").then(t => RubikSearch.setTables(t));
  await tablesP;
  if (!models.has(cfg)) models.set(cfg, getJSON(`models/${cfg}.json`).then(RubikSearch.loadModel));
  return models.get(cfg);
}

self.onmessage = async e => {
  const { id, cfg, kind, start, opts } = e.data;
  try {
    const m = await model(cfg);
    const t0 = Date.now();
    const result = RubikSearch.solve(m, kind, start, Object.assign({}, opts, {
      onStage: stage => self.postMessage({ id, type: "stage", stage }),
      onProgress: n => self.postMessage({ id, type: "progress", n }),
    }));
    result.ms = Date.now() - t0;
    for (const p of result.parts) delete p.state;
    self.postMessage({ id, type: "done", result });
  } catch (err) {
    self.postMessage({ id, type: "error", error: String(err && err.message || err) });
  }
};
