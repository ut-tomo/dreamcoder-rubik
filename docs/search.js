// In-browser DreamCoder search for the 2x2x2 (v5), a JavaScript port of
//   src/rubik/{perm,ec_domain}.py                       (DSL semantics, goals, features, orientation normalisation)
//   pydreamcoder/pyec/enumerator.py = OCaml solver      (depth-first enumeration in 1.5-nat bands, contextual grammars)
//   official recognition.py RecognitionModel.forward    (features -> tanh MLP -> bigram transition matrix)
//   held-out testing semantics with maximumFrontier = 1 (best-prior hit of the band in which the first hit occurs)
// Works as a classic script (Web Worker importScripts) and as a Node module (validation).
(function (root) {
  "use strict";
  let T = null; // tables from docs/models/tables.json
  const ID = Array.from({ length: 24 }, (_, i) => i);

  // ------------------------------------------------------------ group
  const compose = (p, q) => { const r = new Array(24); for (let i = 0; i < 24; i++) r[i] = p[q[i]]; return r; }; // p then q
  const inverse = p => { const r = new Array(24); for (let i = 0; i < 24; i++) r[p[i]] = i; return r; };
  const conj = (g, a) => compose(compose(inverse(g), a), g);
  function normalize(s) {
    for (const r of T.rotations) {
      const c = compose(s, r);
      if (T.dlb.every(i => c[i] === i)) return c;
    }
    throw new Error("no rotation fixes DLB");
  }
  const AMT = { "": 1, "2": 2, "'": 3 }, SUF = { 1: "", 2: "2", 3: "'" };
  function simplify(ms) {
    const out = [];
    for (const m of ms) {
      if (out.length && out[out.length - 1][0] === m[0]) {
        const k = (AMT[out[out.length - 1].slice(1)] + AMT[m.slice(1)]) % 4;
        out.pop();
        if (k) out.push(m[0] + SUF[k]);
      } else out.push(m);
    }
    return out;
  }
  const invMoves = ms => ms.slice().reverse().map(m => m[0] + ({ "": "'", "'": "", "2": "2" })[m.slice(1)]);

  // Two value domains: perms only (fast, for search) and (perm, moves) (for display), like perm.Alg.
  const OPS = {
    fast: {
      move: m => T.face_turns[m],
      seq: (a, b) => compose(a, b),
      inv: a => inverse(a),
      sym: (s, a) => conj(T.sym[s], a),
      perm: a => a,
    },
    full: {
      move: m => ({ p: T.face_turns[m], m: [m] }),
      seq: (a, b) => ({ p: compose(a.p, b.p), m: simplify(a.m.concat(b.m)) }),
      inv: a => ({ p: inverse(a.p), m: invMoves(a.m) }),
      sym: (s, a) => ({ p: conj(T.sym[s], a.p), m: simplify(a.m.map(x => T.rename[s][x])) }),
      perm: a => a.p,
    },
  };
  function primValue(name, O) {
    if (name === "R" || name === "U" || name === "F") return O.move(name);
    const c2 = f => a => b => f(a, b);
    switch (name) {
      case "seq": return c2(O.seq);
      case "inv": return O.inv;
      case "setup": return c2((a, b) => O.seq(O.seq(a, b), O.inv(a)));
      case "comm": return c2((a, b) => O.seq(O.seq(a, b), O.seq(O.inv(a), O.inv(b))));
      case "x": case "y": case "z": case "mirror": return a => O.sym(name, a);
    }
    throw new Error("unknown primitive " + name);
  }

  // ------------------------------------------------------------ programs
  // nodes: {k:"prim",name} {k:"inv",body,key} {k:"lam",body} {k:"var",i} {k:"app",f,x}
  function parse(src) {
    let i = 0;
    const ws = () => { while (src[i] === " ") i++; };
    function atom() {
      ws();
      if (src[i] === "#") { const start = i; i++; const body = atom(); return { k: "inv", body, key: src.slice(start, i) }; }
      if (src[i] === "$") { i++; let j = i; while (/[0-9]/.test(src[i])) i++; return { k: "var", i: +src.slice(j, i) }; }
      if (src[i] === "(") {
        i++; ws();
        if (src.startsWith("lambda ", i)) { i += 7; const body = atom(); ws(); i++; return { k: "lam", body }; }
        let f = atom(); ws();
        while (src[i] !== ")") { f = { k: "app", f, x: atom() }; ws(); }
        i++;
        return f;
      }
      let j = i; while (i < src.length && !" ()".includes(src[i])) i++;
      return { k: "prim", name: src.slice(j, i) };
    }
    const p = atom();
    return p;
  }
  function show(p) {
    switch (p.k) {
      case "prim": return p.name;
      case "inv": return p.key;
      case "var": return "$" + p.i;
      case "lam": return "(lambda " + show(p.body) + ")";
      case "app": { const args = []; let f = p; while (f.k === "app") { args.push(f.x); f = f.f; } return "(" + [f, ...args.reverse()].map(show).join(" ") + ")"; }
    }
  }
  function evaluate(p, env, O, cache) {
    switch (p.k) {
      case "prim": return primValue(p.name, O);
      case "var": return env[p.i];
      case "lam": return x => evaluate(p.body, [x].concat(env), O, cache);
      case "app": return evaluate(p.f, env, O, cache)(evaluate(p.x, env, O, cache));
      case "inv": {
        if (!cache.has(p.key)) cache.set(p.key, evaluate(p.body, [], O, cache));
        return cache.get(p.key);
      }
    }
  }

  // ------------------------------------------------------------ grammar (v5: official DreamCoder)
  // Every type in this DSL is alg -> ... -> alg and programs are closed, so the environment is always empty:
  // unifying_expressions = all library productions, normalised with log-sum-exp (no variables).
  const arity = t => (t.match(/->/g) || []).length;
  function logsumexp(xs) {
    const m = Math.max(...xs);
    if (m === -Infinity) return m;
    let s = 0; for (const x of xs) s += Math.exp(x - m);
    return m + Math.log(s);
  }
  // web model JSON: {grammar:{logVariable, productions:[[l, type, program]]},
  //                  recognition: null | {productions:[[program,type]], library_rows:{program:[rows]}, n_grammars,
  //                                       W1,b1 (64x149), M:{W,b} or A/B (LowRank), activation:"tanh"}}
  function loadModel(m) {
    const mk = (key, t) => ({ key, arity: arity(t), prog: parse(key) });
    const prods = m.grammar.productions.map(([l, t, s]) => Object.assign(mk(s, t), { l }));
    for (const p of prods) if (show(p.prog) !== p.key) throw new Error("parse mismatch " + p.key);
    let rec = null;
    if (m.recognition) {
      const r = m.recognition;
      rec = Object.assign({}, r, { prods: r.productions.map(([s, t]) => mk(s, t)) });
    }
    return { prods, rec, raw: m };
  }
  function features(start, stageIdx) {
    const f = new Array(24 * 6 + 5).fill(0);
    const FI = { R: 0, L: 1, U: 2, D: 3, F: 4, B: 5 };
    for (let slot = 0; slot < 24; slot++) f[slot * 6 + FI[T.face_of[start[slot]]]] = 1;
    f[24 * 6 + stageIdx] = 1;
    return f;
  }
  const linear = (W, b, x) => b.map((bo, o) => { let s = bo; const w = W[o]; for (let k = 0; k < x.length; k++) s += w[k] * x[k]; return s; });
  // official RecognitionModel.forward: _MLP (Linear + tanh) -> ContextualGrammarNetwork_LowRank.transitionMatrix
  function transitionMatrix(rec, x) {
    const h = linear(rec.W1, rec.b1, x).map(Math.tanh);
    const m = rec.n_grammars, n = rec.prods.length + 1;
    const rows = [];
    if (rec.M) {
      const y = linear(rec.M.W, rec.M.b, h);
      for (let i = 0; i < m; i++) rows.push(y.slice(i * n, (i + 1) * n));
    } else {
      const a = linear(rec.A.W, rec.A.b, h), bb = linear(rec.B.W, rec.B.b, h), r = rec.rank;
      for (let i = 0; i < m; i++) { const row = new Array(n).fill(0); for (let j = 0; j < n; j++) for (let k = 0; k < r; k++) row[j] += a[i * r + k] * bb[k * n + j]; rows.push(row); }
    }
    return rows;
  }
  // A contextual grammar: rows of normalised candidates. Candidate order = library stably sorted by arity
  // (OCaml strip_recursion), each with its normalised log-probability and the rows used for its arguments.
  function contextualGrammar(model, start, stageIdx) {
    const order = (prods) => prods.map((p, i) => ({ p, i })).sort((a, b) => a.p.arity - b.p.arity || a.i - b.i);
    const row = (prods, ls, argRows) => {
      const z = logsumexp(ls);
      return order(prods).map(({ p, i }) => ({ l: ls[i] - z, prog: p.prog, arity: p.arity, key: p.key, args: argRows(p) }));
    };
    if (!model.rec) {
      const g = { cands: null };
      g.cands = row(model.prods, model.prods.map(p => p.l), p => new Array(p.arity).fill(g));
      return g;
    }
    const rec = model.rec, tm = transitionMatrix(rec, features(start, stageIdx));
    const G = rec.prods.map(() => null);
    const grammars = tm.map(() => ({ cands: null }));
    const argRows = p => (rec.library_rows[p.key] || []).map(j => grammars[j]);
    tm.forEach((r, j) => { grammars[j].cands = row(rec.prods, r.slice(0, rec.prods.length), argRows); });
    return grammars[rec.n_grammars - 1]; // noParent
  }

  // ------------------------------------------------------------ enumeration (port of the OCaml solver)
  // enumerate_programs' / enumerate_contextual_applications: depth-first, all programs with lb <= -log P < ub.
  class Stop extends Error {}
  function programs(g, lb, ub, k, tick) {
    tick();
    if (ub < 0) return;
    for (const c of g.cands) {
      if (-c.l >= ub) continue;
      applications(c.args, 0, c.prog, lb + c.l, ub + c.l, (p, al) => k(p, c.l + al), tick);
    }
  }
  function applications(args, i, f, lb, ub, k, tick) {
    if (ub < 0) return;
    if (i === args.length) { if (lb <= 0 && 0 < ub) k(f, 0.0); return; }
    programs(args[i], 0.0, ub, (a, ll) => applications(args, i + 1, { k: "app", f, x: a }, lb + ll, ub + ll, (p, al) => k(p, al + ll), tick), tick);
  }

  // search one goal from one (normalised) start state, like a held-out testing job with maximumFrontier = 1:
  // bands of 1.5 nats; after the first hit the current band is finished and the best (highest prior) hit wins.
  function searchStage(model, start, stageIdx, goal, opts) {
    const g = contextualGrammar(model, start, stageIdx);
    const cache = new Map();
    const t0 = Date.now(), deadline = opts.timeMs != null ? t0 + opts.timeMs : Infinity;
    const maxP = opts.maxPrograms != null ? opts.maxPrograms : Infinity;
    let n = 0, calls = 0, best = null;
    const tick = () => { if ((++calls & 1023) === 0 && Date.now() > deadline) throw new Stop(); };
    const emit = (prog, l) => {
      if (n >= maxP) throw new Stop();
      n++;
      if ((n & 4095) === 0) { if (Date.now() > deadline) throw new Stop(); if (opts.onProgress && (n & 32767) === 0) opts.onProgress(n); }
      let s;
      try { s = normalize(compose(start, evaluate(prog, [], OPS.fast, cache))); } catch (e) { return; }
      if (goal(s) && (!best || l > best.l)) best = { prog, l, state: s };
    };
    let lb = 0.0;
    try {
      while (!best && lb + 1.5 <= 99) { programs(g, lb, lb + 1.5, emit, tick); lb += 1.5; }
    } catch (e) { if (!(e instanceof Stop)) throw e; }
    const ms = Date.now() - t0;
    if (!best) return { ok: false, nEnum: n, ms };
    const full = evaluate(best.prog, [], OPS.full, new Map());
    return { ok: true, program: show(best.prog), logp: best.l, moves: full.m, state: best.state, nEnum: n, ms };
  }

  function goalFn(name) {
    const g = T.goals[name];
    return s => g.home.every(i => s[i] === i) && (!g.up || g.up.every(i => T.face_of[s[i]] === "U"));
  }
  // kind "stages": chain stages 1..5 (a stage already satisfied is skipped); kind "full": one search for "solved".
  function solve(model, kind, start, opts) {
    let s = normalize(start);
    const parts = [];
    const stages = kind === "stages" ? [0, 1, 2, 3, 4] : [4];
    for (const k of stages) {
      const name = T.stages[k], goal = goalFn(name);
      if (goal(s)) { parts.push({ stage: name, skipped: true }); continue; }
      if (opts.onStage) opts.onStage(name);
      const r = searchStage(model, s, k, goal, opts);
      parts.push(Object.assign({ stage: kind === "full" ? "full" : name }, r));
      if (!r.ok) return { ok: false, parts, stuck: kind === "full" ? "full" : name };
      s = r.state;
    }
    return { ok: true, parts };
  }

  const api = { setTables: t => { T = t; }, loadModel, solve, searchStage, normalize, compose, goalFn, features, parse, show, contextualGrammar, programs, evaluate, OPS };
  root.RubikSearch = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : this);
