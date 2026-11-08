import * as THREE from "three";

// ---------------------------------------------------------------- cube model
// Same geometry/conventions as src/rubik/cube2.py: sticker = (cubie position in {-1,1}^3, outward normal),
// axis 0 = x (R), 1 = y (U), 2 = z (F). state[slot] = home index of the sticker currently in that slot.
const FACES = { R: [0, 1], L: [0, -1], U: [1, 1], D: [1, -1], F: [2, 1], B: [2, -1] };
const COLORS = { U: 0xffffff, D: 0xffd500, F: 0x009b48, B: 0x0046ad, R: 0xb71234, L: 0xff5800 };
let STICKERS = [], INDEX = new Map(), FACE_OF = [];
const key = (p, n) => p.join(",") + "|" + n.join(",");

function rot(axis, sign) {
  const s = -sign, m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const i = (axis + 1) % 3, j = (axis + 2) % 3;
  m[i][i] = 0; m[i][j] = -s; m[j][i] = s; m[j][j] = 0;
  return m;
}
const mv = (m, v) => [0, 1, 2].map(r => m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2]);
const matmul = (a, b) => [0, 1, 2].map(i => [0, 1, 2].map(j => a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]));

function applyMatrix(state, m, filter) {
  const out = state.slice();
  STICKERS.forEach(([p, n], src) => {
    if (filter && !filter(p)) return;
    out[INDEX.get(key(mv(m, p), mv(m, n)))] = state[src];
  });
  return out;
}

function parseMove(s) {
  const f = s[0], suf = s.slice(1);
  const [axis, sign] = FACES[f];
  const q = suf === "2" ? 2 : suf === "'" ? 3 : 1;
  return { f, axis, sign, q };
}

function applyMove(state, s) {
  const { axis, sign, q } = parseMove(s);
  const m = rot(axis, sign);
  for (let k = 0; k < q; k++) state = applyMatrix(state, m, p => p[axis] === sign);
  return state;
}

let ROTATIONS = [];
function buildRotations() {
  const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const gens = [rot(0, 1), rot(1, 1), rot(2, 1)];
  const seen = new Map([[JSON.stringify(I), I]]);
  let frontier = [I];
  while (frontier.length) {
    const nxt = [];
    for (const a of frontier) for (const g of gens) {
      const b = matmul(g, a), k = JSON.stringify(b);
      if (!seen.has(k)) { seen.set(k, b); nxt.push(b); }
    }
    frontier = nxt;
  }
  ROTATIONS = [...seen.values()];
}

const DLB = [-1, -1, -1];
function normalizingRotation(state) {
  const dlb = STICKERS.map((s, i) => [s, i]).filter(([[p]]) => p.join() === DLB.join()).map(([, i]) => i);
  for (const m of ROTATIONS) {
    const s = applyMatrix(state, m);
    if (dlb.every(i => s[i] === i)) return m;
  }
  throw new Error("no normalizing rotation");
}
const isIdentity = m => m.every((r, i) => r.every((v, j) => v === (i === j ? 1 : 0)));

// ---------------------------------------------------------------- 3D view
class CubeView {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1.25, 0.1, 100);
    this.camera.position.set(4.2, 3.4, 5.4);
    this.camera.lookAt(0, 0, 0);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);
    this.scene.add(new THREE.AmbientLight(0xffffff, 2.2));
    const dl = new THREE.DirectionalLight(0xffffff, 1.2); dl.position.set(3, 5, 4); this.scene.add(dl);
    this.root = new THREE.Group();
    this.root.rotation.set(0.0, 0.0, 0.0);
    this.scene.add(this.root);
    this.items = []; // {mesh, basePos, baseQuat, pos (cubie position)}
    const body = new THREE.BoxGeometry(0.96, 0.96, 0.96);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.6 });
    const corners = [...new Set(STICKERS.map(([p]) => p.join()))].map(s => s.split(",").map(Number));
    for (const p of corners) {
      const m = new THREE.Mesh(body, bodyMat);
      this.addItem(m, new THREE.Vector3(...p).multiplyScalar(0.5), new THREE.Quaternion(), p);
    }
    const plane = new THREE.PlaneGeometry(0.82, 0.82);
    this.stickerMeshes = STICKERS.map(([p, n]) => {
      const mat = new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.45, side: THREE.DoubleSide });
      const m = new THREE.Mesh(plane, mat);
      const nv = new THREE.Vector3(...n);
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), nv);
      this.addItem(m, new THREE.Vector3(...p).multiplyScalar(0.5).addScaledVector(nv, 0.482), q, p);
      return m;
    });
    this.yaw = 0; this.pitch = 0;
    this.enableDrag();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }
  addItem(mesh, basePos, baseQuat, pos) {
    mesh.position.copy(basePos); mesh.quaternion.copy(baseQuat);
    this.root.add(mesh);
    this.items.push({ mesh, basePos: basePos.clone(), baseQuat: baseQuat.clone(), pos });
  }
  enableDrag() {
    const el = this.renderer.domElement;
    let down = null;
    el.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY, this.yaw, this.pitch]; el.setPointerCapture(e.pointerId); });
    el.addEventListener("pointermove", e => {
      if (!down) return;
      this.yaw = down[2] + (e.clientX - down[0]) * 0.01;
      this.pitch = Math.max(-1.4, Math.min(1.4, down[3] + (e.clientY - down[1]) * 0.01));
      this.root.rotation.set(this.pitch, this.yaw, 0, "XYZ");
      this.render();
    });
    el.addEventListener("pointerup", () => { down = null; });
  }
  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.render();
  }
  setState(state) {
    state.forEach((home, slot) => this.stickerMeshes[slot].material.color.setHex(COLORS[FACE_OF[home]]));
    this.resetTransforms(); this.render();
  }
  resetTransforms() {
    for (const it of this.items) { it.mesh.position.copy(it.basePos); it.mesh.quaternion.copy(it.baseQuat); }
  }
  // rotate the items selected by filter by quaternion q (relative to base)
  pose(q, filter) {
    for (const it of this.items) {
      if (filter && !filter(it.pos)) continue;
      it.mesh.position.copy(it.basePos).applyQuaternion(q);
      it.mesh.quaternion.copy(q).multiply(it.baseQuat);
    }
    this.render();
  }
  animate(qEnd, filter, ms) {
    const q0 = new THREE.Quaternion();
    return new Promise(res => {
      const t0 = performance.now();
      const tick = now => {
        const t = Math.min(1, (now - t0) / ms), e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        this.pose(q0.clone().slerp(qEnd, e), filter);
        if (t < 1) requestAnimationFrame(tick); else res();
      };
      requestAnimationFrame(tick);
    });
  }
  render() { this.renderer.render(this.scene, this.camera); }
}

function moveQuat({ axis, sign, q }) {
  const ax = new THREE.Vector3(); ax.setComponent(axis, 1);
  const quarters = q === 3 ? -1 : q;
  return new THREE.Quaternion().setFromAxisAngle(ax, -sign * quarters * Math.PI / 2);
}
function matrixQuat(m) {
  const M = new THREE.Matrix4().set(m[0][0], m[0][1], m[0][2], 0, m[1][0], m[1][1], m[1][2], 0, m[2][0], m[2][1], m[2][2], 0, 0, 0, 0, 1);
  return new THREE.Quaternion().setFromRotationMatrix(M);
}

// ---------------------------------------------------------------- one side
class Side {
  constructor(el) {
    this.el = el;
    this.view = new CubeView(el.querySelector(".canvas"));
    this.statusEl = el.querySelector(".status");
    this.movesEl = el.querySelector(".moves");
    this.progEl = el.querySelector(".prog");
  }
  load(start, sol, opts = {}) {
    this.start = start; this.sol = sol; this.i = 0; this.state = start.slice(); this.reoriented = false; this.preoriented = false;
    this.failMsg = opts.fail || null;
    this.view.setState(this.state);
    const ov = this.el.querySelector(".overlay"); if (ov) ov.remove();
    if (opts.idle) { this.sol = null; this.statusEl.textContent = opts.idle; this.movesEl.textContent = ""; this.progEl.textContent = ""; return; }
    if (sol && opts.fail) {
      this.statusEl.innerHTML = `<span class="no">✗ stuck</span> · ${opts.fail}`;
      this.movesEl.innerHTML = sol.moves.map(m => `<span>${m}</span>`).join(" ") || "<i>(no moves before getting stuck)</i>";
      this.progEl.textContent = sol.program;
      return;
    }
    if (!sol) {
      this.statusEl.innerHTML = `<span class="no">✗ not solved</span>`;
      this.movesEl.textContent = "";
      this.progEl.textContent = "(no program found)";
      const d = document.createElement("div"); d.className = "overlay"; d.textContent = "No solution found within the search budget";
      this.el.querySelector(".canvas").appendChild(d);
      return;
    }
    const n = sol.moves.length;
    this.statusEl.innerHTML = `<span class="ok">✓ solved</span> · ${n} move${n === 1 ? "" : "s"}`;
    this.movesEl.innerHTML = sol.moves.map(m => `<span>${m}</span>`).join(" ") || "<i>(no moves)</i>";
    this.progEl.textContent = sol.program;
  }
  note(t) {
    let n = this.el.querySelector(".note");
    if (!n) { n = document.createElement("div"); n.className = "note small"; this.statusEl.after(n); }
    n.textContent = t; clearTimeout(this._nt); this._nt = setTimeout(() => { n.textContent = ""; }, 2500);
  }
  get done() { return !this.sol || (this.i >= this.sol.moves.length && this.reoriented); }
  highlight() {
    [...this.movesEl.children].forEach((s, k) => { s.className = k < this.i - 1 ? "done" : k === this.i - 1 ? "cur" : ""; });
  }
  async step(speed) {
    if (this.done) return;
    if (!this.preoriented) {
      // the solver works on the cube held with its back-left-bottom corner in place; turn the whole cube
      // there visibly instead of silently recolouring it
      this.preoriented = true;
      const M = normalizingRotation(this.state);
      if (!isIdentity(M)) {
        this.note("turning the whole cube (not a move) so the back-left-bottom corner is in its home spot");
        await this.view.animate(matrixQuat(M), null, 700 / speed);
        this.state = applyMatrix(this.state, M);
        this.view.setState(this.state);
      }
      return;
    }
    if (this.i < this.sol.moves.length) {
      const m = this.sol.moves[this.i++], pm = parseMove(m);
      this.highlight();
      await this.view.animate(moveQuat(pm), p => p[pm.axis] === pm.sign, (pm.q === 2 ? 480 : 330) / speed);
      this.state = applyMove(this.state, m);
      if ((this.sol.boundaries || []).includes(this.i)) {
        const M = normalizingRotation(this.state);
        this.view.setState(this.state);
        if (!isIdentity(M)) {
          this.note("stage done; turning the whole cube (not a move) back to the reference orientation");
          await this.view.animate(matrixQuat(M), null, 600 / speed);
          this.state = applyMatrix(this.state, M);
        }
      }
      this.view.setState(this.state);
    } else {
      this.reoriented = true;
      this.i++; this.highlight();
      const M = normalizingRotation(this.state);
      if (!isIdentity(M)) {
        await this.view.animate(matrixQuat(M), null, 600 / speed);
        this.state = applyMatrix(this.state, M);
        this.view.setState(this.state);
      }
      if (this.failMsg) {
        const d = document.createElement("div"); d.className = "overlay"; d.textContent = "Stuck: " + this.failMsg;
        this.el.querySelector(".canvas").appendChild(d);
      } else this.statusEl.innerHTML += " · goal reached";
    }
  }
}

// ---------------------------------------------------------------- page
const SHORT = { nn_lib: "NN + library", nn_nolib: "NN only", nonn_lib: "Library only", nonn_nolib: "Neither" };
const STRAT = { stages: "Subtask curriculum", full: "Full task only" };
const STAGE_TEXT = { "1-corner": "stage 1 (one bottom corner)", "2-corners": "stage 2 (second bottom corner)", "3-Dlayer": "stage 3 (bottom layer)",
  "4-Uorient": "stage 4 (top face one colour)", "5-solved": "stage 5 (solve the cube)", "full": "the full solve" };
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
let DATA, run, ARMS, side, playing = false, busy = false, cube, scr = [];

// stage goals (same as src/rubik/cube2.py + tasks.py), checked on the orientation-normalized cube
const posKey = p => p.join(",");
const cornerSolved = (s, pos) => STICKERS.every(([p], i) => posKey(p) !== posKey(pos) || s[i] === i);
const DRF = [1, -1, 1], DRB = [1, -1, -1], DLF = [-1, -1, 1];
const dLayer = s => STICKERS.every(([p], i) => p[1] !== -1 || s[i] === i);
const uOriented = s => STICKERS.every(([, n], i) => n[1] !== 1 || FACE_OF[s[i]] === "U");
const solvedS = s => s.every((v, i) => v === i);
const GOAL = {
  "1-corner": s => cornerSolved(s, DRF), "2-corners": s => cornerSolved(s, DRF) && cornerSolved(s, DRB),
  "3-Dlayer": dLayer, "4-Uorient": s => dLayer(s) && uOriented(s), "5-solved": solvedS, "full": solvedS };
const normalize = s => applyMatrix(s, normalizingRotation(s));
const applySeq = (s, mv) => { for (const m of mv) s = applyMove(s, m); return s; };

// Use this run's own programs: per stage, the shortest stored program that reaches the stage goal.
function solveWith(kind, armId, seed, start) {
  const pools = DATA.runs.find(r => r.id === `${kind}_s${seed}`).arms[armId].pools;
  let s = normalize(start); const moves = [], bounds = [], parts = [];
  for (const g of Object.keys(pools)) {
    if (GOAL[g](s)) { parts.push(`${g}: already done`); continue; }
    const hit = pools[g].find(p => GOAL[g](normalize(applySeq(s, p.split(" ")))));
    if (hit === undefined) return { moves, bounds, parts, fail: STAGE_TEXT[g], stuck: g, tried: pools[g].length };
    const mv = hit.split(" ");
    s = normalize(applySeq(s, mv)); moves.push(...mv); bounds.push(moves.length);
    parts.push(`${g}: ${hit}`);
  }
  return { moves, bounds, parts };
}

const FACE_MOVES = ["R", "L", "U", "D", "F", "B"].flatMap(f => [f, f + "'", f + "2"]);
function setScramble(list) {
  scr = list; cube = applySeq(SOLVED24(), scr);
  $("scrShow").textContent = scr.length ? `Scramble (${scr.length}): ${scr.join(" ")}` : "Solved cube. Add some moves.";
  stop(); side.view.root.rotation.set(side.view.pitch, side.view.yaw, 0, "XYZ");
  side.load(cube, null, { idle: "Scramble the cube, then press Solve." });
  $("tTitle").textContent = "Your cube"; drawAll8();
}
const SOLVED24 = () => STICKERS.map((_, i) => i);
function randomScramble() {
  const out = []; let last = "";
  while (out.length < 25) { const m = FACE_MOVES[Math.floor(Math.random() * FACE_MOVES.length)]; if (m[0] === last) continue; out.push(m); last = m[0]; }
  return out;
}
function solveNow() {
  stop();
  const k = $("tStrat").value, a = $("tArm").value, seed = $("tSeed").value;
  const r = solveWith(k, a, seed, cube);
  $("tTitle").innerHTML = `<span class="c-${a}">${SHORT[a]}</span> · ${STRAT[k]} · seed ${seed}`;
  const start = cube.slice();
  const sol = { moves: r.moves, boundaries: r.bounds, program: r.parts.join("\n") };
  side.load(start, sol, r.stuck ? { fail: `none of its ${r.tried} stored programs for ${r.fail} works on this cube` } : {});
}
function drawAll8() {
  if (!scr.length) { $("all8").innerHTML = "<tr><td>Scramble the cube first.</td></tr>"; return; }
  const seed = $("tSeed").value;
  const rows = ["stages", "full"].flatMap(k => ARMS.map(a => {
    const r = solveWith(k, a.id, seed, cube);
    const res = r.stuck ? `<span class="no">✗ stuck at ${r.stuck === "full" ? "full solve" : "stage " + r.stuck[0]}</span>` : `<span class="ok">✓ ${r.moves.length} moves</span>`;
    return `<tr><td>${STRAT[k]}</td><td><b class="c-${a.id}">${SHORT[a.id]}</b></td><td>${res}</td></tr>`;
  })).join("");
  $("all8").innerHTML = `<tr><th>Strategy</th><th>Arm</th><th>Seed ${seed}</th></tr>` + rows;
}
const runOf = (k, seed) => DATA.runs.find(r => r.id === `${k}_s${seed}`);
function fillGroups() {
  $("pstage").innerHTML = Object.keys(run.arms[ARMS[0].id].policy || {}).map(g => `<option>${g}</option>`).join("");
}
async function stepAll() {
  if (busy) return; busy = true;
  await side.step(+$("speed").value);
  busy = false;
}
async function play() {
  if (playing) { stop(); return; }
  if (!side.sol) solveNow();
  if (side.done) solveNow();
  playing = true; $("play").textContent = "⏸ Pause";
  while (playing && !side.done) { await stepAll(); await new Promise(r => setTimeout(r, 120 / +$("speed").value)); }
  stop();
}
function stop() { playing = false; $("play").textContent = "▶ Play"; }

// ---------------- results
function drawSummary() {
  const rows = ["stages", "full"].flatMap(k => ARMS.map(a => {
    const rs = [0, 1, 2].map(s => runOf(k, s).arms[a.id]);
    const pct = rs.map(A => 100 * A.test_solved / A.test_n), mean = pct.reduce((x, y) => x + y, 0) / pct.length;
    const sols = rs.flatMap(A => Object.entries(A.solutions).filter(([n]) => n.startsWith("test:")).map(([, v]) => v.moves.length));
    const len = sols.length ? (sols.reduce((x, y) => x + y, 0) / sols.length).toFixed(0) : "–";
    return `<tr><td>${STRAT[k]}</td><td><b class="c-${a.id}">${SHORT[a.id]}</b></td><td><b>${mean.toFixed(0)}%</b> <span class="small">(${pct.map(p => p.toFixed(0)).join(" / ")})</span></td><td>${len}</td></tr>`;
  })).join("");
  $("summary").innerHTML = `<tr><th>Training strategy</th><th>Arm</th><th>Test cubes solved</th><th>Avg. solution length (moves)</th></tr>${rows}`;
}
function drawCurve() {
  $("curve-run").textContent = "(" + run.label.split("·").pop().trim() + ")";
  const svg = $("curve"), W = 420, H = 250, L = 40, B = 30, T = 14, R = 100;
  const n = run.tasks.filter(t => t.split === "train").length, its = run.arms[ARMS[0].id].history.length;
  const x = i => L + (W - L - R) * i / Math.max(1, its - 1), y = v => H - B - (H - B - T) * v / n;
  let s = `<line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#9ca3af"/>`;
  for (const v of [0, n / 2, n]) s += `<text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${Math.round(v)}</text><line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#d5d9de"/>`;
  for (let i = 0; i < its; i++) s += `<text x="${x(i)}" y="${H - B + 16}" text-anchor="middle">${i + 1}</text>`;
  s += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">round</text>`;
  const ends = [];
  ARMS.forEach(a => {
    const h = run.arms[a.id].history, c = `var(--${a.id})`;
    s += `<polyline points="${h.map((p, i) => `${x(i)},${y(p.solved)}`).join(" ")}" fill="none" stroke="${c}" stroke-width="2.5"/>` +
      h.map((p, i) => `<circle cx="${x(i)}" cy="${y(p.solved)}" r="3" fill="${c}"><title>${SHORT[a.id]}, round ${i + 1}: ${p.solved}/${n}</title></circle>`).join("");
    ends.push([y(h[h.length - 1].solved), a, h[h.length - 1].solved]);
  });
  ends.sort((p, q) => p[0] - q[0]);
  let last = -99;
  for (const e of ends) { const yy = Math.max(e[0] + 4, last + 13); last = yy; s += `<text x="${W - R + 8}" y="${yy}" style="fill:var(--${e[1].id});font-weight:600">${SHORT[e[1].id]} ${e[2]}</text>`; }
  svg.innerHTML = s;
}
function drawBars() {
  $("bars-run").textContent = "(" + run.label.split("·").pop().trim() + ")";
  const svg = $("bars"), W = 420, H = 250, L = 34, B = 40, T = 14;
  const groups = Object.keys(run.arms[ARMS[0].id].test_per_group);
  const max = Math.max(...groups.map(g => run.arms[ARMS[0].id].test_per_group[g][1]));
  const gw = (W - L - 10) / groups.length, bw = Math.min(14, gw / 5);
  const y = v => H - B - (H - B - T) * v / max;
  let s = `<line x1="${L}" y1="${H - B}" x2="${W - 10}" y2="${H - B}" stroke="#9ca3af"/>`;
  for (const v of [0, max]) s += `<text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  groups.forEach((g, i) => {
    const cx = L + gw * (i + 0.5);
    ARMS.forEach((a, j) => {
      const v = run.arms[a.id].test_per_group[g][0], bx = cx + (j - 2) * bw;
      s += `<rect x="${bx}" y="${y(v)}" width="${bw - 2}" height="${H - B - y(v)}" fill="var(--${a.id})"><title>${SHORT[a.id]} · ${g}: ${v}/${max}</title></rect>`;
    });
    s += `<text x="${cx}" y="${H - B + 14}" text-anchor="middle">${g}</text>`;
  });
  ARMS.forEach((a, j) => { s += `<rect x="${L + 6 + j * 96}" y="${H - 14}" width="10" height="10" fill="var(--${a.id})"/><text x="${L + 20 + j * 96}" y="${H - 5}">${SHORT[a.id]}</text>`; });
  svg.innerHTML = s;
}
function drawPolicy() {
  const st = $("pstage").value;
  $("policy-grid").innerHTML = ARMS.map(a => {
    const pol = (run.arms[a.id].policy || {})[st] || [];
    const max = Math.max(...pol.map(p => p[1]), 1e-9);
    const rows = pol.map(([k, v]) => `<div class="pbar"><code title="${esc(k)}">${esc(k)}</code>
      <div class="bar"><span style="width:${(100 * v / max).toFixed(1)}%;background:var(--${a.id})"></span></div><span class="v">${(100 * v).toFixed(1)}%</span></div>`).join("");
    return `<div class="card"><h3 class="c-${a.id}">${SHORT[a.id]}</h3>${rows}</div>`;
  }).join("");
}
function drawLibrary() {
  const libArms = ARMS.filter(a => a.id.endsWith("_lib"));
  if (!$("libarm").options.length) $("libarm").innerHTML = libArms.map(a => `<option value="${a.id}">${SHORT[a.id]}</option>`).join("");
  const lib = run.arms[$("libarm").value].library;
  $("library").innerHTML = lib.map(r => `<li>${esc(r.program)} <span class="mv">: ${esc(r.type)}${r.moves ? " = " + esc(r.moves) : ""}</span></li>`).join("") || "<li>(none)</li>";
}
function selectRun() {
  run = DATA.runs[+$("run").value];
  fillGroups(); drawCurve(); drawBars(); drawPolicy(); drawLibrary();
}

async function main() {
  DATA = await (await fetch("data.json")).json();
  STICKERS = DATA.stickers;
  STICKERS.forEach(([p, n], i) => INDEX.set(key(p, n), i));
  FACE_OF = STICKERS.map(([, n]) => { const ax = n.findIndex(v => v); return Object.keys(FACES).find(f => FACES[f][0] === ax && FACES[f][1] === n[ax]); });
  buildRotations();
  ARMS = DATA.arms;
  side = new Side($("tside"));
  $("tArm").innerHTML = ARMS.map(a => `<option value="${a.id}">${SHORT[a.id]}</option>`).join("");
  $("pad").innerHTML = FACE_MOVES.map(m => `<button data-m="${m}">${m}</button>`).join("");
  $("pad").onclick = e => { const m = e.target.dataset && e.target.dataset.m; if (m) setScramble([...scr, m]); };
  $("randomScr").onclick = () => setScramble(randomScramble());
  $("undoScr").onclick = () => setScramble(scr.slice(0, -1));
  $("resetScr").onclick = () => setScramble([]);
  $("applyScr").onclick = () => {
    const toks = $("scrText").value.replace(/’/g, "'").trim().split(/\s+/).filter(Boolean);
    const bad = toks.filter(t => !FACE_MOVES.includes(t));
    if (bad.length) { $("scrShow").textContent = `Unknown move(s): ${bad.join(" ")}. Use R L U D F B with ' or 2.`; return; }
    setScramble(toks);
  };
  $("solve").onclick = solveNow;
  $("tSeed").onchange = drawAll8;
  $("reset").onclick = () => side.sol ? solveNow() : null;
  $("play").onclick = play;
  $("step").onclick = () => { stop(); if (!side.sol) solveNow(); stepAll(); };
  $("run").innerHTML = DATA.runs.map((r, i) => `<option value="${i}">${r.label}</option>`).join("");
  $("run").onchange = selectRun;
  $("pstage").onchange = drawPolicy;
  $("libarm").onchange = drawLibrary;
  drawSummary();
  selectRun();
  $("pstage").value = "5-solved"; drawPolicy();
  setScramble(randomScramble());
}
main();
