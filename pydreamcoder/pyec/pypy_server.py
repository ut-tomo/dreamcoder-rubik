"""Persistent PyPy fork-server for the enumerator: imports (and JIT warm-up of the import machinery) are paid
once; every request is served by a forked child, exactly like a fresh ./solver process but without startup cost.
Protocol over a UNIX socket: 8-byte big-endian length + payload (pickled request, see backend._dumps)."""
import os, sys, socket, signal, struct, time


def recv_all(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise EOFError()
        buf.extend(chunk)
    return bytes(buf)


def send_msg(conn, data):
    conn.sendall(struct.pack(">Q", len(data)) + data)


def recv_msg(conn):
    n = struct.unpack(">Q", recv_all(conn, 8))[0]
    return recv_all(conn, n)


def handle(conn, data):
    from pyec.backend import _loads, _dumps
    from pyec.enumerator import enumerate_for_tasks
    req = _loads(data)
    timeout = req["timeout"] - (time.time() - req["sent_at"])
    hits, n = enumerate_for_tasks(req["g"], req["tasks"], req["maximumFrontiers"], req["lowerBound"],
                                  req["upperBound"], req["budgetIncrement"], max(0., timeout), nc=req["nc"],
                                  evaluationTimeout=req["evaluationTimeout"])
    send_msg(conn, _dumps((hits, n)))


def main(path):
    from pyec.compress_cli import load_primitives
    load_primitives()
    import pyec.backend, pyec.enumerator  # noqa: preload
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # auto-reap children
    if os.path.exists(path):
        os.unlink(path)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(path)
    s.listen(1024)
    parent = int(os.environ.get("PYEC_SERVER_PARENT", "0"))
    s.settimeout(5.0)
    while True:
        try:
            conn, _ = s.accept()
        except socket.timeout:
            if parent:
                try:
                    os.kill(parent, 0)
                except OSError:
                    break  # owner died
            continue
        # warm-up requests are served in the server process itself, so that the JIT-compiled code
        # is inherited by every subsequently forked request handler
        try:
            conn.settimeout(None)
            from pyec.backend import _loads, _dumps
            data = recv_msg(conn)
        except Exception:
            conn.close()
            continue
        if data[:6] == b"WARMUP":
            req = _loads(data[6:])
            from pyec.enumerator import enumerate_for_tasks
            t0 = time.time()
            for g, ts in req["jobs"]:
                if time.time() - t0 > req["seconds"]:
                    break
                try:
                    enumerate_for_tasks(g, ts, [10 ** 9] * len(ts), 0., 99., 1.5,
                                        req["seconds"] / max(1, len(req["jobs"])), nc=1,
                                        evaluationTimeout=req["evaluationTimeout"])
                except Exception:
                    import traceback
                    traceback.print_exc()
            send_msg(conn, b"ok")
            conn.close()
            continue
        pid = os.fork()
        if pid == 0:
            s.close()
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            try:
                handle(conn, data)
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                conn.close()
                os._exit(0)
        conn.close()
    os.unlink(path)


if __name__ == "__main__":
    main(sys.argv[1])
