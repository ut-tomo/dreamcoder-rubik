// Solve rate of every exported config on random face-turn scrambles (seeded), as the website does it.
//  node experiments/rate_random.cjs <n cubes=20> <programs per stage=50000> <scramble length=25> [rng seed=1]
const fs = require("fs"), path = require("path");
const S = require("../docs/search.js");
const M = path.join(__dirname, "../docs/models");
const T = JSON.parse(fs.readFileSync(path.join(M, "tables.json"))); S.setTables(T);
const [n = 20, P = 50000, L = 25, seed = 1] = process.argv.slice(2).map(Number);
let x = seed >>> 0; const rnd = () => ((x = (Math.imul(x, 1664525) + 1013904223) >>> 0) / 2 ** 32);
const MOVES = Object.keys(T.face_turns);
const cubes = [];
for (let i = 0; i < n; i++) {
  let s = [...Array(24).keys()], last = "";
  for (let k = 0; k < L;) { const m = MOVES[Math.floor(rnd() * MOVES.length)]; if (m[0] === last) continue; s = S.compose(s, T.face_turns[m]); last = m[0]; k++; }
  cubes.push(s);
}
const models = fs.readdirSync(M).filter(f => /_s\d\.json$/.test(f)).map(f => f.slice(0, -5)).sort();
const agg = {};
for (const cfg of models) {
  const model = S.loadModel(JSON.parse(fs.readFileSync(path.join(M, cfg + ".json"))));
  const kind = cfg.split("_")[0]; let ok = 0, t0 = Date.now();
  for (const c of cubes) {
    const r = S.solve(model, kind, c, { maxPrograms: P * (kind === "full" ? 5 : 1) });
    let s = S.normalize(c);
    for (const p of r.parts) if (p.ok && !p.skipped) { for (const m of p.moves) s = S.compose(s, T.face_turns[m]); s = S.normalize(s); }
    if (r.ok && !s.every((v, i) => v === i)) throw new Error("replay fail " + cfg);
    if (r.ok) ok++;
  }
  const arm = cfg.replace(/_s\d$/, ""); (agg[arm] = agg[arm] || []).push(ok);
  console.log(cfg.padEnd(22), `${ok}/${n}`, `${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
console.log("\nper arm (sum over 3 seeds):");
for (const [a, v] of Object.entries(agg)) console.log(a.padEnd(20), v.join("+"), "=", `${v.reduce((p, q) => p + q, 0)}/${v.length * n}`);
