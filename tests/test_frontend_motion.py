"""Guards for the motion layer: GSAP is vendored + pinned, loaded lazily, and never runs under reduced motion.

Static checks are offline. The browser checks are STUB-ONLY (page fetch replaced, static server refuses non-GET)
and skip cleanly when headless Chrome or the CDN libraries are unavailable.
"""
import hashlib
import html as _html
import http.server as _hs
import json as _json
import re
import shutil as _sh
import subprocess as _sp
import threading as _th

import pytest

from tests.test_frontend_guards import _SEED, _STUB, FE, INDEX, _read

GSAP = FE / "js" / "vendor" / "gsap.min.js"
GSAP_SHA256 = "92bb9a96476f983d212a2bc4f54c889039c1696dd4461d40a736860938570fbb"   # gsap@3.15.0 dist/gsap.min.js


def _code(rel):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", _read(rel), flags=re.S)


# ── static guards ─────────────────────────────────────────────────────────────────────────────

def test_vendored_gsap_is_pinned_and_documented():
    assert sorted(p.name for p in (FE / "js" / "vendor").iterdir()) == ["NOTICE-gsap.txt", "gsap.min.js"]
    digest = hashlib.sha256(GSAP.read_bytes()).hexdigest()
    assert digest == GSAP_SHA256, "gsap.min.js changed: replace it with an official build and update the pin + notice together"
    notice = _read("js/vendor/NOTICE-gsap.txt")
    assert GSAP_SHA256 in notice and "3.15.0" in notice and "gsap.com/standard-license" in notice
    assert "NOT an OSI" in notice, "the notice must say the license is not open source"


def test_gsap_has_no_network_or_dynamic_code_surface():
    src = GSAP.read_text(encoding="utf-8")
    for bad in ("XMLHttpRequest", "fetch(", "eval(", "new Function", "sendBeacon", "WebSocket", "document.write"):
        assert bad not in src, bad


def test_gsap_is_loaded_lazily_from_the_same_origin_only():
    assert "gsap" not in INDEX.lower().replace("js/components/motion.js", ""), "no static <script> or CDN for gsap in index.html"
    motion = _code("js/components/motion.js")
    assert '"/js/vendor/gsap.min.js"' in motion and "s.async = true" in motion
    assert not re.search(r"https?://", motion)
    assert INDEX.index("/js/components/motion.js") < INDEX.index('src="/js/ui.js"')


def test_motion_is_view_only_and_animates_only_transform_and_opacity():
    motion = _code("js/components/motion.js")
    assert not re.search(r"\bfetch\s*\(|\bAPI\.|\bState\.|localStorage|XMLHttpRequest|innerHTML", motion)
    tweens = re.findall(r"gsap\.(?:set|to|fromTo)\([^;]*", motion, re.S)
    assert tweens
    for t in tweens:
        assert not re.search(r"\b(width|height|top|left|right|bottom|margin|padding|fontSize|scale)\b\s*:", t), t
    assert "clearProps" in motion, "inline styles must be removed after the tween"


def test_motion_honours_reduced_motion_and_cleans_up():
    motion = _code("js/components/motion.js")
    assert "(prefers-reduced-motion: reduce)" in motion
    assert motion.count("reduce.matches") >= 3
    assert "_ctx.revert()" in motion and "gsap.context(" in motion, "tweens are scoped to a context and reverted"
    assert '"pagehide", _revert' in motion
    # reduced motion returns before the library is requested
    enter = motion[motion.index("function enter()"):motion.index("function workspaceChanged")]
    assert enter.index("reduce.matches") < enter.index("_load()")


def test_first_paint_guard_has_a_hard_failsafe_and_skips_reduced_motion():
    head = INDEX[:INDEX.index("</head>")]
    assert 'prefers-reduced-motion: reduce' in head and 'classList.add("motion-pending")' in head
    assert re.search(r'setTimeout\(function \(\) \{ r\.classList\.remove\("motion-pending"\) \}, 2500\)', head), \
        "content must reappear even if every script fails"
    css = _read("css/workspace.css")
    assert re.search(r"html\.motion-pending[^{]*\{\s*opacity: 0;", css)


def test_css_entrance_keyframes_that_motion_replaces_are_gone():
    anim = _read("css/animations.css")
    for old in (".welcome > .welcome__title", ".welcome__starters .feature-card:nth-child(1)", ".input-dock .composer        { animation"):
        assert old not in anim, old
    comp = re.sub(r"/\*.*?\*/", "", _read("css/components.css"), flags=re.S)
    assert not re.search(r"\.welcome\s*\{[^}]*animation:", comp), "the welcome block must not also fade in with CSS"


def test_vendor_code_is_exempt_from_the_first_party_origin_guard_only_by_directory():
    guards = _read("../tests/test_frontend_guards.py") if False else (FE.parent / "tests" / "test_frontend_guards.py").read_text(encoding="utf-8")
    assert '"vendor"' in guards or "vendor" in guards.split("def test_no_new_external_origins")[1].split("def ")[0]


# ── browser: reduced motion never loads the library; normal motion ends with everything visible ─

_SCENARIO = r"""
(function(){
var R={},$=function(i){return document.getElementById(i)},tick=function(ms){return new Promise(function(r){setTimeout(r,ms)})};
async function until(f,ms){var t=0;while(!f()&&t<(ms||3000)){await tick(20);t+=20}return !!f()}
// Headless virtual time makes GSAP's frames advance unevenly, so timing is not asserted here. Instead every active tween is
// driven to its END state, which verifies the contract that matters: clearProps leaves no inline style behind, and nothing is
// ever orphaned half-faded after rapid switching. Real-time durations are checked separately (see the notes file).
function finish(){var n=0;if(window.gsap)window.gsap.globalTimeline.getChildren(true,true,false).forEach(function(t){n++;t.progress(1)});return n}
function op(sel){var e=document.querySelector(sel);return e?getComputedStyle(e).opacity:null}
function inline(sel){var e=document.querySelector(sel);return e?e.style.opacity+'|'+e.style.transform:null}
async function pick(m){$('modeSelect').click();await tick(40);document.querySelector('.menu__item[data-menu-id="'+m+'"]').click()}
async function run(){
 if(!window.marked||!window.DOMPurify){R.nolibs=true;return}
 R.reducedQuery=matchMedia('(prefers-reduced-motion: reduce)').matches;
 await until(function(){return !!window.gsap||R.reducedQuery},6000);
 R.tweensEnter=finish();await tick(150);
 R.afterEnter={gsap:typeof window.gsap,pending:document.documentElement.classList.contains('motion-pending'),
   title:op('.welcome__title'),titleInline:inline('.welcome__title'),composer:op('.input-dock .composer'),composerInline:inline('.input-dock .composer'),
   card:op('.feature-card'),cardInline:inline('.feature-card')};
 // interactive immediately: the mode dropdown works while/after the entrance
 await pick('document');await tick(60);R.clickable=$('modeSelectLabel').textContent;
 R.tweensSwitch=finish();await tick(150);
 R.afterSwitch={panelKids:[].slice.call(document.querySelectorAll('#wsDocument > *')).map(function(e){return getComputedStyle(e).opacity+'|'+e.style.opacity+'|'+e.style.transform})};
 // rapid switching must not leave anything half-faded
 var order=['research','code','document','general','code','research','general'];
 for(var i=0;i<order.length;i++){await pick(order[i]);await tick(40)}
 R.tweensRapid=finish();await tick(150);
 var visible=[].slice.call(document.querySelectorAll('.welcome__general > *, #wsDocument > *, #wsResearch > *, #wsCode > *')).filter(function(e){return e.getClientRects().length>0});
 R.afterRapid={shown:visible.length,stuck:visible.filter(function(e){return getComputedStyle(e).opacity!=='1'||e.style.opacity!==''||e.style.transform!==''}).length};
 R.errs=window.__errs.slice();
}
run().catch(function(e){R.crash=String(e&&e.stack||e)}).then(function(){var o=document.createElement('pre');o.id='qa-out';o.textContent=JSON.stringify(R);document.body.appendChild(o)});
})();
"""


def _run_browser(tmp_path_factory, extra_flags):
    chrome = next((p for p in map(_sh.which, ("google-chrome", "chromium", "chromium-browser", "chrome")) if p), None)
    if not chrome:
        pytest.skip("headless Chrome not available")
    injected = "<head><script>" + _SEED + _STUB.replace("</script", "<\\/script") + "</script>"
    tail = "<script>" + _SCENARIO.replace("</script", "<\\/script") + "</script></body>"
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
    d = tmp_path_factory.mktemp("motion")
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


@pytest.fixture(scope="module")
def reduced(tmp_path_factory):
    return _run_browser(tmp_path_factory, ["--force-prefers-reduced-motion"])


@pytest.fixture(scope="module")
def normal(tmp_path_factory):
    return _run_browser(tmp_path_factory, [])


def test_browser_reduced_motion_never_loads_the_library_and_shows_everything(reduced):
    assert reduced["__blocked"] == [] and reduced["errs"] == []
    assert reduced["reducedQuery"] is True
    a = reduced["afterEnter"]
    assert a["gsap"] == "undefined", "reduced motion must not even load gsap"
    assert a["pending"] is False and a["title"] == "1" and a["composer"] == "1" and a["card"] == "1"
    assert reduced["afterRapid"]["stuck"] == 0 and reduced["clickable"] == "Documents"


def test_browser_normal_motion_ends_with_clean_visible_content_and_stays_interactive(normal):
    assert normal["__blocked"] == [] and normal["errs"] == []
    assert normal["tweensEnter"] > 0, "the entrance must have created tweens (GSAP ran)"
    a = normal["afterEnter"]
    assert a["gsap"] == "object", "gsap loads lazily when motion is allowed"
    assert a["pending"] is False
    assert a["title"] == "1" and a["composer"] == "1" and a["card"] == "1"
    # clearProps removed every inline style the tween wrote
    assert a["titleInline"] == "|" and a["composerInline"] == "|" and a["cardInline"] == "|", a
    assert normal["clickable"] == "Documents", "the dropdown works during / right after the entrance"
    assert all(k == "1||" for k in normal["afterSwitch"]["panelKids"]), normal["afterSwitch"]
    assert normal["afterRapid"]["shown"] > 0 and normal["afterRapid"]["stuck"] == 0, normal["afterRapid"]
