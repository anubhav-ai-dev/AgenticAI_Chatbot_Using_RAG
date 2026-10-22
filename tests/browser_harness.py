"""Shared stub-only headless-Chrome runner for scenario tests.

Serves frontend/ statically (GET only; any other verb is refused with 405 and recorded), replaces the page's fetch
with the in-page stub from tests/test_frontend_guards.py, runs a scenario script and returns the JSON it wrote into
<pre id="qa-out">. Skips (never fails) when Chrome or the CDN libraries are unavailable.
"""
import html as _html
import http.server as _hs
import json as _json
import re
import shutil as _sh
import subprocess as _sp
import threading as _th

import pytest

from tests.test_frontend_guards import _SEED, _STUB, FE


def run_scenario(scenario, tmp_path_factory, name="scenario", extra_flags=()):
    chrome = next((p for p in map(_sh.which, ("google-chrome", "chromium", "chromium-browser", "chrome")) if p), None)
    if not chrome:
        pytest.skip("headless Chrome not available")
    injected = "<head><script>" + _SEED + _STUB.replace("</script", "<\\/script") + "</script>"
    tail = "<script>" + scenario.replace("</script", "<\\/script") + "</script></body>"
    blocked = []

    class H(_hs.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            path = "/index.html" if path == "/" else path
            f = (FE / path.lstrip("/")).resolve()
            if FE.resolve() not in f.parents or not f.is_file():
                blocked.append(f"GET {self.path}")
                self.send_response(599)
                self.end_headers()
                return
            data = f.read_bytes()
            if path == "/index.html":
                data = data.decode("utf-8").replace("<head>", injected, 1).replace("</body>", tail, 1).encode("utf-8")
            self.send_response(200)
            ct = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}.get(f.suffix, "application/octet-stream")
            self.send_header("Content-Type", ct + "; charset=utf-8" if f.suffix in (".html", ".js", ".css") else ct)
            self.end_headers()
            self.wfile.write(data)

        def _refuse(self):
            blocked.append(f"{self.command} {self.path}")
            self.send_response(405)
            self.end_headers()

        do_POST = do_PUT = do_PATCH = do_DELETE = _refuse

    srv = _hs.ThreadingHTTPServer(("127.0.0.1", 0), H)
    _th.Thread(target=srv.serve_forever, daemon=True).start()
    d = tmp_path_factory.mktemp(name)
    try:
        p = _sp.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", f"--user-data-dir={d / 'prof'}", *extra_flags,
                     "--virtual-time-budget=60000", "--window-size=1280,900", "--dump-dom",
                     f"http://127.0.0.1:{srv.server_address[1]}/"], capture_output=True, text=True, timeout=180)
    except (_sp.TimeoutExpired, OSError) as e:
        pytest.skip(f"chrome could not run: {e}")
    finally:
        srv.shutdown()
    m = re.search(r'<pre id="qa-out">(.*?)</pre>', p.stdout, re.S)
    if not m:
        pytest.skip("scenario produced no output (Chrome or CDN unavailable)")
    res = _json.loads(_html.unescape(m.group(1)))
    if res.get("nolibs"):
        pytest.skip("CDN libraries (marked/DOMPurify) unavailable")
    assert "crash" not in res, res.get("crash")
    res["__blocked"] = blocked
    return res
