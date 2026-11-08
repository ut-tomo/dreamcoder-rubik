"""Apply the (only) modifications made to the vendored official DreamCoder Python frontend.
Each edit just registers the pure-Python backends next to the OCaml ones; nothing else changes."""
import sys, os
root = sys.argv[1]


def edit(path, old, new, count=1):
    p = os.path.join(root, path)
    s = open(p).read()
    if new in s:
        return
    assert s.count(old) == count, (path, old, s.count(old))
    s = s.replace(old, new)
    open(p, "w").write(s)


# 1. solver registry: add "pyfast" (pyec.enumerator, faithful port of the OCaml solver)
edit("dreamcoder/enumeration.py",
     '''    solvers = {"ocaml": solveForTask_ocaml,''',
     '''    from pyec.backend import solveForTask_pyfast, start_server
    if solver == "pyfast":
        start_server(g, tasks, evaluationTimeout)  # persistent, JIT-warmed PyPy fork-server (no per-call interpreter start-up)
    solvers = {"ocaml": solveForTask_ocaml,
               "pyfast": solveForTask_pyfast,''')
# pyfast receives no likelihood model (like ocaml), so nothing else changes in multicoreEnumeration
# 2. compressor registry: add "pyvs" (pyec.vs_compression, faithful port of compression.ml/versions.ml)
edit("dreamcoder/compression.py",
     '''        elif backend == "ocaml" or backend == "vs_factored":''',
     '''        elif backend == "pyvs":
            from pyec.backend import pyvsInduce
            kwargs.pop('iteration')
            kwargs.pop('topk_use_only_likelihood')
            kwargs['topI'] = 300
            kwargs['bs'] = 1000000
            g, newFrontiers = pyvsInduce(*args, **kwargs)
        elif backend == "ocaml" or backend == "vs_factored":''')
# 3. command line choices
edit("dreamcoder/dreamcoder.py",
     '''            "bottom",
            "python"],''',
     '''            "bottom",
            "python",
            "pyfast"],''')
edit("dreamcoder/dreamcoder.py",
     '''choices=["pypy","rust","vs","pypy_vs","ocaml","vs_factored","memorize"])''',
     '''choices=["pypy","rust","vs","pypy_vs","ocaml","vs_factored","memorize","pyvs"])''')
# 4. Helmholtz (bias-optimal) dreams: pyec.helmholtz, port of the OCaml ./helmholtz binary (opt-in)
edit("dreamcoder/dreaming.py",
     '''    """Returns json (as text)"""
    message = {"request": request.json(),''',
     '''    """Returns json (as text)"""
    if os.environ.get("PYEC_HELMHOLTZ"):
        from pyec.helmholtz import helmholtzEnumeration as pyecHelmholtzEnumeration
        return pyecHelmholtzEnumeration(g, request, inputs, timeout, special=special,
                                        evaluationTimeout=evaluationTimeout)
    message = {"request": request.json(),''')
print("patched", root)
