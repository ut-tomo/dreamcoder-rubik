"""Drop-in replacement for the official ./compression binary: reads the same JSON message on stdin,
writes the same JSON response on stdout. Run with PyPy for speed.
Primitive registries are populated by importing the modules listed in env PYEC_PRIMITIVE_MODULES
(comma separated "module:function" entries, default: list domain)."""
import sys, os, json, time, importlib


def load_primitives():
    spec = os.environ.get("PYEC_PRIMITIVE_MODULES") or (
                          # list domain ("common" primitive set first: first registration wins)
                          "dreamcoder.domains.list.listPrimitives:bootstrapTarget_extra,"
                          "dreamcoder.domains.list.listPrimitives:McCarthyPrimitives,"
                          "dreamcoder.domains.list.listPrimitives:primitives,"
                          "dreamcoder.domains.list.listPrimitives:basePrimitives")
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        mod, fn = item.split(":")
        getattr(importlib.import_module(mod), fn)()


def main():
    load_primitives()
    from pyec.vs_compression import run_message
    j = json.loads(sys.stdin.read())
    t = time.time()
    resp = run_message(j, log=lambda *a: print(*a, file=sys.stderr, flush=True))
    print("pyec compression took %.1fs" % (time.time() - t), file=sys.stderr)
    sys.stdout.write(json.dumps(resp))


if __name__ == "__main__":
    main()
