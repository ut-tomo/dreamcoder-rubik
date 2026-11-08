// JS side of the in-browser search check (docs/search.js).
//  node experiments/validate_live.cjs compare /tmp/py_live.json
//      same cubes, same deterministic budget (programs per stage) as validate_live.py;
//      every config/cube/stage must find the identical program (or both fail).
//  node experiments/validate_live.cjs rate <n cubes> <seconds per stage | programs per stage + "p"> [configs]
//      wall-clock budget as in the browser (full strategy gets 5x), on the fixed test cubes.
// Every returned solution is replayed move by move from the user's start and must reach the goal.
const fs = require("fs"), path = require("path");
const S = require("../docs/search.js");
const M = process.env.MODELS || path.join(__dirname, "../docs/models");
const T = JSON.parse(fs.readFileSync(path.join(M, "tables.json")));
S.setTables(T);
const TEST = JSON.parse(fs.readFileSync(path.join(__dirname, "e2e_cubes.json")));
const models = fs.readdirSync(M).filter(f => /_s\d\.json$/.test(f)).map(f => f.slice(0, -5)).sort();

function replay(start, res, kind) {
  // apply the face turns of every stage, re-normalising after each stage, check each goal
  let s = S.normalize(start);
  for (const p of res.parts) {
    if (p.skipped || !p.ok) continue;
    for (const m of p.moves) s = S.compose(s, T.face_turns[m]);
    s = S.normalize(s);
    if (!S.goalFn(p.stage === "full" ? "5-solved" : p.stage)(s)) return false;
  }
  return !res.ok || s.every((v, i) => v === i);
}

const [mode, a1, a2, a3] = process.argv.slice(2);
if (mode === "compare") {
  const py = JSON.parse(fs.readFileSync(a1));
  let same = 0, diff = 0;
  for (const [cfg, rows] of Object.entries(py.results)) {
    const model = S.loadModel(JSON.parse(fs.readFileSync(path.join(M, cfg + ".json"))));
    const kind = cfg.split("_")[0];
    let ok = 0;
    rows.forEach((pyParts, i) => {
      const r = S.solve(model, kind, TEST[i], { maxPrograms: py.max_programs });
      if (!replay(TEST[i], r, kind)) { console.log("REPLAY FAIL", cfg, i); process.exit(1); }
      const js = r.parts.map(p => (p.skipped ? "skip" : p.ok ? p.program : null));
      if (JSON.stringify(js) === JSON.stringify(pyParts)) same++; else { diff++; console.log("DIFF", cfg, i, JSON.stringify(js), JSON.stringify(pyParts)); }
      if (r.ok) ok++;
    });
    console.log(cfg.padEnd(22), `JS solved ${ok}/${rows.length}`);
  }
  console.log(`identical ${same}, different ${diff}`);
  process.exit(diff ? 1 : 0);
} else if (mode === "rate") {
  const n = +a1 || 100, sec = +a2 || 5, only = a3 ? a3.split(",") : null;
  for (const cfg of models) {
    if (only && !only.includes(cfg)) continue;
    const model = S.loadModel(JSON.parse(fs.readFileSync(path.join(M, cfg + ".json"))));
    const kind = cfg.split("_")[0];
    let ok = 0, len = 0, stuck = {}, t0 = Date.now();
    for (let i = 0; i < n; i++) {
      const mul = kind === "full" ? 5 : 1;
      const r = S.solve(model, kind, TEST[i], String(a2).endsWith("p") ? { maxPrograms: parseInt(a2) * mul } : { timeMs: 1000 * sec * mul });
      if (!replay(TEST[i], r, kind)) { console.log("REPLAY FAIL", cfg, i); process.exit(1); }
      if (r.ok) { ok++; len += r.parts.reduce((a, p) => a + (p.moves ? p.moves.length : 0), 0); }
      else stuck[r.stuck] = (stuck[r.stuck] || 0) + 1;
    }
    console.log(cfg.padEnd(22), `${ok}/${n} solved`, ok ? `mean ${(len / ok).toFixed(1)} moves` : "", "stuck:", JSON.stringify(stuck), `${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
}
