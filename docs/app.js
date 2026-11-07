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
  load(start, sol) {
    this.start = start; this.sol = sol; this.i = 0; this.state = start.slice(); this.reoriented = false;
    this.view.setState(this.state);
    const ov = this.el.querySelector(".overlay"); if (ov) ov.remove();
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
  get done() { return !this.sol || (this.i >= this.sol.moves.length && this.reoriented); }
  highlight() {
    [...this.movesEl.children].forEach((s, k) => { s.className = k < this.i - 1 ? "done" : k === this.i - 1 ? "cur" : ""; });
  }
  async step(speed) {
    if (this.done) return;
    if (this.i < this.sol.moves.length) {
      const m = this.sol.moves[this.i++], pm = parseMove(m);
      this.highlight();
      await this.view.animate(moveQuat(pm), p => p[pm.axis] === pm.sign, (pm.q === 2 ? 480 : 330) / speed);
      this.state = applyMove(this.state, m);
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
      this.statusEl.innerHTML += " · goal reached";
    }
  }
}

// ---------------------------------------------------------------- page
const GOALS = {
  "1-corner": "Stage 1: solve the DRF corner (down-right-front).",
  "2-corners": "Stage 2: DRF is solved. Also solve the DRB corner.",
  "3-Dlayer": "Stage 3: DRF and DRB are solved. Solve the whole D (bottom) layer.",
  "4-Uorient": "Stage 4: the D layer is solved. Also make the U (top) face one colour.",
  "5-solved": "Stage 5: the D layer is solved and the U face is one colour. Solve the cube.",
};
const SHORT = { nn_lib: "NN + library", nn_nolib: "NN only", nonn_lib: "Library only", nonn_nolib: "Neither" };
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
let DATA, run, ARMS, sides, playing = false, busy = false;

function buildSides() {
  $("duel").innerHTML = ARMS.map(a => `
    <div class="side ${a.id}" id="side-${a.id}">
      <h3 class="c-${a.id}">${SHORT[a.id]}<span class="tag">${a.label}</span></h3>
      <div class="canvas"></div><div class="status"></div><div class="moves"></div>
      <details><summary>Program</summary><code class="prog"></code></details>
    </div>`).join("");
  sides = ARMS.map(a => new Side($("side-" + a.id)));
}
const solOf = (armId, name) => run.arms[armId].solutions[name];

function taskLabel(t) {
  const marks = ARMS.map(a => { const s = solOf(a.id, t.name); return s ? s.moves.length : "✗"; }).join(" / ");
  return `${t.name.replace("test:", "")} — ${marks}`;
}
function fillRuns() { $("run").innerHTML = DATA.runs.map((r, i) => `<option value="${i}">${r.label}</option>`).join(""); }
function fillGroups() {
  const gs = [...new Set(run.tasks.map(t => t.group))];
  $("group").innerHTML = `<option value="">all</option>` + gs.map(g => `<option>${g}</option>`).join("");
  $("pstage").innerHTML = gs.map(g => `<option>${g}</option>`).join("");
}
function fillTasks() {
  const g = $("group").value, only = $("onlyDiff").checked, split = $("split").value;
  const ts = run.tasks.filter(t => t.split === split && (!g || t.group === g) &&
    (!only || new Set(ARMS.map(a => !!solOf(a.id, t.name))).size > 1));
  $("task").innerHTML = ts.map(t => `<option value="${t.name}">${taskLabel(t)}</option>`).join("");
  loadTask();
}
function loadTask() {
  stop();
  const t = run.tasks.find(t => t.name === $("task").value);
  if (!t) { $("goal").textContent = "No tasks match this filter."; return; }
  $("goal").innerHTML = GOALS[t.group] + (t.split === "test" ? " <i>(held-out task)</i>" : "") +
    `<br><span class="small">Moves per arm in the list: ${ARMS.map(a => SHORT[a.id]).join(" / ")}</span>`;
  ARMS.forEach((a, i) => sides[i].load(t.start, solOf(a.id, t.name)));
}
async function stepAll() {
  if (busy) return; busy = true;
  const sp = +$("speed").value;
  await Promise.all(sides.map(s => s.step(sp)));
  busy = false;
}
async function play() {
  if (playing) { stop(); return; }
  if (sides.every(s => s.done)) loadTask();
  playing = true; $("play").textContent = "⏸ Pause";
  while (playing && !sides.every(s => s.done)) { await stepAll(); await new Promise(r => setTimeout(r, 120 / +$("speed").value)); }
  stop();
}
function stop() { playing = false; $("play").textContent = "▶ Play"; }

// ---------------- results
function drawSummary() {
  const rows = ARMS.map(a => {
    const cells = DATA.runs.map(r => {
      const A = r.arms[a.id];
      const tr = Object.values(A.per_group).reduce((s, v) => s + v[0], 0), trn = Object.values(A.per_group).reduce((s, v) => s + v[1], 0);
      const sols = Object.entries(A.solutions).filter(([k]) => k.startsWith("test:"));
      const len = sols.length ? (sols.reduce((s, [, v]) => s + v.moves.length, 0) / sols.length).toFixed(1) : "–";
      return `<td>${tr}/${trn}</td><td><b>${A.test_solved}/${A.test_n}</b></td><td>${len}</td><td>${A.library.length}</td>`;
    }).join("");
    return `<tr><td><b class="c-${a.id}">${SHORT[a.id]}</b></td>${cells}</tr>`;
  }).join("");
  const head1 = DATA.runs.map(r => `<th colspan="4">${r.label.split("·").pop().trim()}</th>`).join("");
  const head2 = DATA.runs.map(() => `<th>train</th><th>held-out</th><th>avg moves</th><th>routines</th>`).join("");
  $("summary").innerHTML = `<tr><th>Arm</th>${head1}</tr><tr><th></th>${head2}</tr>${rows}`;
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
  fillGroups(); fillTasks(); drawCurve(); drawBars(); drawPolicy(); drawLibrary();
}

async function main() {
  DATA = await (await fetch("data.json")).json();
  STICKERS = DATA.stickers;
  STICKERS.forEach(([p, n], i) => INDEX.set(key(p, n), i));
  FACE_OF = STICKERS.map(([, n]) => { const ax = n.findIndex(v => v); return Object.keys(FACES).find(f => FACES[f][0] === ax && FACES[f][1] === n[ax]); });
  buildRotations();
  ARMS = DATA.arms;
  buildSides();
  fillRuns();
  $("run").onchange = selectRun;
  $("split").onchange = fillTasks;
  $("group").onchange = fillTasks;
  $("onlyDiff").onchange = fillTasks;
  $("task").onchange = loadTask;
  $("pstage").onchange = drawPolicy;
  $("libarm").onchange = drawLibrary;
  $("random").onclick = () => { const o = $("task").options; if (o.length) { $("task").selectedIndex = Math.floor(Math.random() * o.length); loadTask(); } };
  $("reset").onclick = loadTask;
  $("play").onclick = play;
  $("step").onclick = () => { stop(); stepAll(); };
  drawSummary();
  selectRun();
  // open on an informative case: held-out stage 5 tasks where the arms disagree
  $("group").value = "5-solved"; $("onlyDiff").checked = true; fillTasks();
  $("pstage").value = "5-solved"; drawPolicy();
}
main();
