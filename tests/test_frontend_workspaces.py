"""Guards for the workspace panels (Documents / Research / Code), My Documents, Settings and uploads.

Static checks are offline and fast. The browser scenario is STUB-ONLY (the page's fetch is replaced and the static
server answers 405 to any non-GET), so no backend starts and nothing here proves real-backend behaviour.
Browser tests skip cleanly when headless Chrome or the CDN libraries are unavailable.
"""
import html as _html
import http.server as _hs
import json as _json
import re
import shutil as _sh
import subprocess as _sp
import threading as _th

import pytest

from tests.test_frontend_guards import _SEED, _STUB, FE, INDEX, _read

NEW_JS = ["components/radiogroup.js", "components/documents.js", "components/workspaces.js",
          "components/settings.js", "components/prefs.js"]


def _strip_comments(src):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", src, flags=re.S)


# ── static guards ─────────────────────────────────────────────────────────────────────────────

def test_three_workspace_panels_exist_and_each_has_a_labelled_heading():
    for pid, mode in (("wsDocument", "document"), ("wsResearch", "research"), ("wsCode", "code")):
        m = re.search(rf'<section class="workspace" id="{pid}" data-mode="{mode}" aria-labelledby="(\w+)">', INDEX)
        assert m, pid
        assert f'<h1 class="workspace__title" id="{m.group(1)}">' in INDEX
    assert 'id="wsGeneral"' in INDEX, "the General hero must stay"


def test_panel_controls_are_buttons_that_only_fill_the_composer():
    panels = INDEX[INDEX.index('id="wsDocument"'):INDEX.index("</section>\n      </div>\n    </section>")]
    controls = re.findall(r"<button[^>]*(?:data-fill|data-template)[^>]*>", panels)
    assert len(controls) >= 12, len(controls)
    for c in controls:
        assert 'type="button"' in c, c
    ws = _strip_comments(_read("js/components/workspaces.js"))
    assert not re.search(r"sendBtn|chatStream|\bfetch\s*\(|\bAPI\.|\bState\.|\.requestSubmit|\.submit\(", ws)
    assert "never sends" in _read("js/components/workspaces.js").lower()


def test_document_actions_are_explained_not_hidden_when_unavailable():
    assert INDEX.count("data-needs-docs") == 4
    ws = _read("js/components/workspaces.js")
    assert 'setAttribute("aria-disabled", "true")' in ws and "docActionsNote" in ws
    assert "Upload a document to enable these actions." in ws and "Upload a document to enable these actions." in INDEX


def test_research_locks_the_web_switch_because_the_server_always_searches():
    ws = _read("js/components/workspaces.js")
    assert 'mode === "research"' in ws and "Web search is always on in Research" in ws
    assert "Web search is always on in Research" in _read("js/app.js")
    # the README documents the server behaviour this relies on
    assert "always in *Research* mode" in (FE.parent / "README.md").read_text(encoding="utf-8")


def test_settings_tabs_and_panes_are_wired_with_aria():
    tabs = re.findall(r'<button class="settings__tab"[^>]*id="(setTab-\w+)"[^>]*aria-controls="(setPane-\w+)"', INDEX)
    assert [t for t, _ in tabs] == ["setTab-profile", "setTab-appearance", "setTab-models", "setTab-notifications",
                                    "setTab-integrations", "setTab-privacy", "setTab-help"]
    for tab, pane in tabs:
        assert re.search(rf'role="tabpanel" id="{pane}" aria-labelledby="{tab}"', INDEX), pane
    assert 'role="tablist" aria-orientation="vertical"' in INDEX
    assert len(re.findall(r'aria-selected="true"', INDEX.split('id="settingsDialog"')[1].split("</dialog>")[0])) == 1


def test_settings_only_offers_controls_with_a_real_effect():
    st = _read("js/components/settings.js")
    app = _read("js/app.js")
    for needle in ("Theme.set(", 'Prefs.set("fontSize"', 'Prefs.set("density"', 'Prefs.set("toasts"',
                   'main.dispatchEvent(new Event("change"', "Notifications.clear()", "Utils.copyText("):
        assert needle in st, needle
    assert 'UI.onSettingsPane((pane) => { if (pane === "integrations") _loadServices(); })' in app
    assert "API.getHealth()" in app
    # no dead placeholder controls
    settings_html = INDEX.split('id="settingsDialog"')[1].split("</dialog>")[0]
    assert "disabled" not in settings_html.replace('aria-disabled', "")
    assert "coming soon" not in settings_html.lower()


def test_privacy_and_help_copy_makes_only_verifiable_statements():
    settings_html = INDEX.split('id="settingsDialog"')[1].split("</dialog>")[0].lower()
    for word in ("100% private", "end-to-end", "encrypted", "never stored", "we never", "guarantee", "unlimited"):
        assert word not in settings_html, word
    for fact in ("groq or google", "tavily", "no accounts or passwords"):
        assert fact in settings_html, fact


@pytest.mark.parametrize("rel", NEW_JS)
def test_new_components_are_view_only_and_never_inject_html(rel):
    code = _strip_comments(_read("js/" + rel))
    assert not re.search(r"\bfetch\s*\(|\bAPI\.|XMLHttpRequest|sendBeacon|WebSocket|\bState\.", code), rel
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function\(", code), rel
    if rel.endswith("prefs.js"):   # the only one that persists, and only under its own "pref." namespace
        assert 'KEY = (name) => "pref." + name' in code
        assert not re.search(r'getItem\(\s*"(userId|mode|model|theme)"', code)
    else:
        assert "localStorage" not in code and "sessionStorage" not in code, f"{rel} must not persist anything itself"


def test_new_components_are_loaded_before_ui_js_in_dependency_order():
    order = [INDEX.index(f'src="/js/{p}"') for p in
             ("components/prefs.js", "components/radiogroup.js", "components/documents.js",
              "components/workspaces.js", "components/settings.js", "ui.js", "app.js")]
    assert order == sorted(order)


def test_workspace_css_obeys_the_project_css_rules():
    css = re.sub(r"/\*.*?\*/", "", _read("css/workspace.css"), flags=re.S)
    assert "/css/workspace.css" in INDEX and INDEX.index("/css/reference.css") < INDEX.index("/css/workspace.css")
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css), "raw hex outside main.css"
    assert "@keyframes" not in css
    for decl in re.findall(r"(?<![-\w])transition\s*:\s*([^;]+);", css):
        props = set(re.findall(r"(?:^|,)\s*([a-z-]+)", decl))
        assert props <= {"transform", "opacity", "background-color", "color", "border-color", "box-shadow"}, decl


def test_text_is_rem_based_so_the_text_size_setting_scales_everything():
    for f in ("components.css", "reference.css", "workspace.css", "main.css"):
        for i, line in enumerate(_read(f"css/{f}").splitlines(), 1):
            if re.search(r"(font-size|line-height):\s*[0-9.]+px", line):
                # allowed: the root size, and the hero orb whose em-based geometry uses font-size as a unit
                root = line.strip() == "font-size: 16px;" or line.lstrip().startswith("html[data-font=")
                assert "ai-viz--hero" in line or root, f"{f}:{i}: {line.strip()}"
    assert 'html[data-font="large"]' in _read("css/workspace.css")


def test_preference_markup_matches_the_allowed_values():
    prefs = _read("js/components/prefs.js")
    spec = {k: re.findall(r'"([a-z]+)"', v) for k, v in
            re.findall(r"(\w+):\s*\{ values: \[([^\]]+)\]", prefs)}
    groups = {"fontSize": "setFont", "density": "setDensity", "toasts": "setToasts", "codeLang": "codeLang"}
    for pref, gid in groups.items():
        block = INDEX.split(f'id="{gid}"')[1].split("</div>")[0]
        offered = re.findall(r'data-value="([a-z]+)"', block)
        assert offered == spec[pref], (pref, offered, spec[pref])
    # first paint reads the same keys the module writes
    assert '"pref.fontSize"' in INDEX and '"pref.density"' in INDEX and 'KEY = (name) => "pref." + name' in prefs


def test_uploads_validate_on_the_client_and_guard_duplicates_without_trusting_it():
    app = _read("js/app.js")
    fn = app[app.index("async function _handleFileUpload"):app.index("async function _onUrlSubmit")]
    for needle in ("State.limits.extensions || ALLOWED", "file.size === 0", "State.limits.maxMb", "file.size >",
                   "State.documents.some((d) => d.filename === file.name)", "UI.confirm("):
        assert needle in fn, needle
    assert "the server validates type and size again" in app.lower()   # documented on the function
    handle_files = app[app.index("async function _handleFiles"):app.index("async function _handleFileUpload")]
    assert "await _handleFileUpload(list[i])" in handle_files and "State.uploading" in handle_files  # one at a time


def test_config_is_a_read_only_get_and_its_failure_is_tolerated():
    api = _read("js/api.js")
    assert 'function getConfig() {\n    return request("/config");' in api
    app = _read("js/app.js")
    cfg = app[app.index("async function _loadConfig"):app.index("/** Upload several files")]
    assert "catch (_)" in cfg and "older backend" in cfg


def test_history_and_documents_have_loading_and_error_states():
    app, ui = _read("js/app.js"), _read("js/ui.js")
    assert 'UI.setSessionsStatus("loading")' in app and 'UI.setSessionsStatus("error"' in app
    assert 'UI.setDocsStatus("loading")' in app and 'UI.setDocsStatus("error"' in app
    assert "function setSessionsStatus" in ui and "Couldn’t load your chats" in ui
    assert "ArrowDown" in ui[ui.index("History: ArrowUp"):ui.index("Skip link: move focus")]


# ── stub-only browser scenario ────────────────────────────────────────────────────────────────

_SCENARIO = r"""
(function(){
var R={},$=function(i){return document.getElementById(i)},tick=function(ms){return new Promise(function(r){setTimeout(r,ms)})};
async function until(f,ms){var t=0;while(!f()&&t<(ms||3000)){await tick(20);t+=20}return !!f()}
var vis=function(id){var e=$(id);return !!e&&e.getClientRects().length>0&&getComputedStyle(e).display!=='none'};
var nonGet=function(){return window.__calls.filter(function(c){return c.method!=='GET'}).map(function(c){return c.method+' '+c.path})};
async function pick(m){$('modeSelect').click();await tick(40);var it=document.querySelector('.menu__item[data-menu-id="'+m+'"]');it.click();await tick(60)}
function key(el,k){el.dispatchEvent(new KeyboardEvent('keydown',{key:k,bubbles:true,cancelable:true}))}
function files(list){var dt=new DataTransfer();list.forEach(function(f){dt.items.add(f)});$('fileInput').files=dt.files;$('fileInput').dispatchEvent(new Event('change',{bubbles:true}))}
async function run(){
 if(!window.marked||!window.DOMPurify){R.nolibs=true;return}
 await tick(400);
 // 1. the dropdown is the only mode control and sits above New Chat
 var a=$('modeSelect').getBoundingClientRect(),b=$('newChatBtn').getBoundingClientRect();
 R.layout={above:a.bottom<=b.top,tabs:document.querySelectorAll('.mode-tab,.top-tab').length};
 // 2. keyboard opens it; 4 radios; the selected one is marked
 $('modeSelect').focus();key($('modeSelect'),'ArrowDown');await tick(60);
 var items=[].slice.call(document.querySelectorAll('.menu--mode [role="menuitemradio"]'));
 R.menu={labels:items.map(function(i){return i.textContent.trim()}),checked:items.filter(function(i){return i.getAttribute('aria-checked')==='true'}).length,
   expanded:$('modeSelect').getAttribute('aria-expanded'),focus:document.activeElement.textContent.trim()};
 key(document.activeElement,'Escape');await tick(60);
 R.escape={closed:!document.querySelector('.menu'),focus:document.activeElement.id,expanded:$('modeSelect').getAttribute('aria-expanded')};
 // 3. every mode shows exactly its own panel and composer hint
 R.modes={};
 var ids={general:'wsGeneral',document:'wsDocument',research:'wsResearch',code:'wsCode'};
 for(var m in ids){await pick(m);var shown=[];for(var k in ids){var probe=k==='general'?document.querySelector('#wsGeneral .welcome__title'):$(ids[k]);if(probe&&probe.getClientRects().length>0)shown.push(k)}
   R.modes[m]={shown:shown,label:$('modeSelectLabel').textContent,workspace:document.documentElement.dataset.workspace,stored:localStorage.getItem('mode'),ph:$('chatInput').placeholder,title:$('contextName').textContent}}
 // 4. Documents with no files: actions explained, a click fills nothing; the drop zone is a button
 await pick('document');
 R.docEmpty={disabled:[].slice.call(document.querySelectorAll('[data-needs-docs]')).every(function(c){return c.getAttribute('aria-disabled')==='true'}),note:$('docActionsNote').textContent,dz:$('docDropzone').tagName};
 $('chatInput').value='';document.querySelector('[data-needs-docs]').click();await tick(60);
 R.docEmpty.afterClick={value:$('chatInput').value,focus:document.activeElement.id};
 // 5. Code: a template puts the caret inside the fence and sends nothing
 await pick('code');
 [].slice.call(document.querySelectorAll('#codeLang [role="radio"]')).filter(function(r){return r.dataset.value==='python'})[0].click();
 document.querySelector('[data-template="explain"]').click();await tick(60);
 var ci=$('chatInput');R.code={value:ci.value,inFence:/```python\n$/.test(ci.value.slice(0,ci.selectionStart)),lang:localStorage.getItem('pref.codeLang')};
 // 6. Research: the switch is locked on, with feedback; unlocked again elsewhere
 await pick('research');
 var t0=$('searchToggle');t0.click();await tick(100);
 R.research={disabled:t0.getAttribute('aria-disabled'),pressed:t0.getAttribute('aria-pressed'),toast:(document.querySelector('.toast')||{}).textContent};
 document.querySelector('.topic[data-select="[topic]"]').click();await tick(60);
 R.research.selected=ci.value.slice(ci.selectionStart,ci.selectionEnd);
 await pick('general');R.research.unlocked=!$('searchToggle').hasAttribute('aria-disabled');
 // 7. My Documents panel opens and refreshes from the server; Esc returns focus
 window.__calls.length=0;$('myDocsBtn').focus();$('myDocsBtn').click();await tick(200);
 R.docsPanel={open:$('docsDialog').open,refreshed:window.__calls.some(function(c){return c.path==='/documents'}),focus:document.activeElement.id,mode:localStorage.getItem('mode')};
 $('docsDialog').close();await tick(100);R.docsPanel.focusBack=document.activeElement.id;
 // 8. Settings: each control has a real effect
 $('settingsBtn').focus();$('settingsBtn').click();await tick(150);
 document.querySelector('#setTheme [data-value="dark"]').click();await tick(60);
 R.theme={resolved:document.documentElement.dataset.theme,stored:localStorage.getItem('theme')};
 document.querySelector('#setFont [data-value="large"]').click();document.querySelector('#setDensity [data-value="compact"]').click();await tick(60);
 R.prefs={font:document.documentElement.dataset.font,density:document.documentElement.dataset.density,rootPx:getComputedStyle(document.documentElement).fontSize};
 document.querySelector('#setTheme [data-value="system"]').click();document.querySelector('#setFont [data-value="medium"]').click();document.querySelector('#setDensity [data-value="comfortable"]').click();await tick(60);
 R.reset={pref:document.documentElement.dataset.themePref,font:document.documentElement.dataset.font||null,density:document.documentElement.dataset.density||null};
 var tabs=[].slice.call(document.querySelectorAll('.settings__tab'));tabs[1].focus();key(tabs[1],'ArrowDown');await tick(40);
 R.tabs={selected:document.querySelector('.settings__tab[aria-selected="true"]').textContent.trim(),panes:[].slice.call(document.querySelectorAll('.pane')).filter(function(p){return !p.hidden}).length};
 key(document.activeElement,'End');await tick(40);R.tabs.end=document.querySelector('.settings__tab[aria-selected="true"]').textContent.trim();
 $('settingsDialog').close();await tick(100);R.settingsFocusBack=document.activeElement.id;
 // 9. uploads: client checks, duplicate guard, one at a time. /config is absent here (older backend) and must not matter.
 window.__calls.length=0;
 files([new File(['x'],'bad.exe')]);await tick(100);
 R.badType={posts:nonGet(),toast:[].slice.call(document.querySelectorAll('.toast')).pop().textContent};
 files([new File([''],'empty.txt')]);await tick(100);
 R.empty={posts:nonGet(),toast:[].slice.call(document.querySelectorAll('.toast')).pop().textContent};
 window.__S.docs.push({id:'dx',filename:'dup.txt',file_type:'txt',chunk_count:1,indexed_at:new Date().toISOString()});
 $('myDocsBtn').click();await tick(200);$('docsDialog').close();await tick(60);
 // NOTE: waits below are bounded by outcome, generously: headless virtual time can delay a dialog's async close event by seconds
 window.__calls.length=0;files([new File(['hello'],'dup.txt')]);await until(function(){return document.querySelector('.modal--confirm')},30000);
 R.dup={confirm:!!document.querySelector('.modal--confirm'),title:(document.querySelector('.modal--confirm .modal__title')||{}).textContent,focus:document.activeElement.textContent,posts:nonGet()};
 document.querySelector('.modal--confirm .btn--secondary').click();await until(function(){return !document.querySelector('.modal--confirm')},30000);await tick(100);R.dup.afterSkip=nonGet();
 files([new File(['hello'],'dup.txt')]);await until(function(){return document.querySelector('.modal--confirm')},30000);
 document.querySelector('.modal--confirm .btn--primary').click();await until(function(){return nonGet().length>0},30000);await until(function(){return !document.querySelector('.modal--confirm')},30000);await tick(200);R.dup.afterConfirm=nonGet();
 await until(function(){return $('uploadToast').hidden},30000);window.__calls.length=0;files([new File(['1'],'one.txt'),new File(['2'],'two.txt')]);await until(function(){return nonGet().length>=2},30000);await tick(200);
 R.multi={posts:nonGet()};
 // 10. history: arrow keys move between chats
 $('newChatBtn').click();await tick(200);$('newChatBtn').click();await tick(200);
 var rows=[].slice.call(document.querySelectorAll('.session-item__main'));
 if(rows.length>=2){rows[0].focus();key(rows[0],'ArrowDown');R.history={moved:document.activeElement===rows[1],rows:rows.length};key(rows[1],'ArrowUp');R.history.back=document.activeElement===rows[0]}
 else R.history={rows:rows.length};
 R.errs=window.__errs.slice();
}
run().catch(function(e){R.crash=String(e&&e.stack||e)}).then(function(){var o=document.createElement('pre');o.id='qa-out';o.textContent=JSON.stringify(R);document.body.appendChild(o)});
})();
"""


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
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
    d = tmp_path_factory.mktemp("workspaces")
    try:
        p = _sp.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", f"--user-data-dir={d / 'prof'}",
                     "--virtual-time-budget=120000", "--window-size=1280,900", "--dump-dom",
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


def test_stub_run_reached_no_server_and_threw_nothing(ws):
    assert ws["__blocked"] == [] and ws["errs"] == [], (ws["__blocked"], ws["errs"])


def test_stub_dropdown_is_the_only_mode_control_and_is_keyboard_operable(ws):
    assert ws["layout"] == {"above": True, "tabs": 0}
    assert ws["menu"]["labels"] == ["General", "Documents", "Research", "Code"]
    assert ws["menu"]["checked"] == 1 and ws["menu"]["expanded"] == "true" and ws["menu"]["focus"] == "General"
    assert ws["escape"] == {"closed": True, "focus": "modeSelect", "expanded": "false"}


def test_stub_each_mode_shows_only_its_own_panel(ws):
    expect = {"general": ("General", "Ask anything..."), "document": ("Documents", "Ask anything about your documents..."),
              "research": ("Research", "Ask a research question..."),
              "code": ("Code", "Describe what to build, or paste code to explain or debug...")}
    for mode, (label, ph) in expect.items():
        r = ws["modes"][mode]
        assert r["shown"] == [mode], (mode, r)
        assert r["label"] == label == r["title"] and r["workspace"] == mode and r["ph"] == ph, r
        assert r["stored"] == (mode if mode != "general" else None), r   # choosing the current mode changes nothing


def test_stub_panel_actions_only_fill_the_composer(ws):
    assert ws["docEmpty"]["disabled"] and ws["docEmpty"]["note"] == "Upload a document to enable these actions."
    assert ws["docEmpty"]["dz"] == "BUTTON"
    assert ws["docEmpty"]["afterClick"] == {"value": "", "focus": "docDropzone"}   # unavailable: explained, nothing filled
    assert ws["code"]["inFence"] and ws["code"]["lang"] == "python" and "```python" in ws["code"]["value"]
    assert ws["research"]["selected"] == "[topic]"


def test_stub_research_locks_the_web_switch_and_other_modes_unlock_it(ws):
    r = ws["research"]
    assert r["disabled"] == "true" and r["pressed"] == "true" and r["toast"] == "Web search is always on in Research." and r["unlocked"]


def test_stub_my_documents_panel_refreshes_and_returns_focus(ws):
    p = ws["docsPanel"]
    assert p["open"] and p["refreshed"] and p["focus"] == "docSearch" and p["focusBack"] == "myDocsBtn"
    assert p["mode"] == "general", "opening My Documents must not switch the workspace"


def test_stub_settings_controls_take_effect_and_reset(ws):
    assert ws["theme"] == {"resolved": "dark", "stored": "dark"}
    assert ws["prefs"]["font"] == "large" and ws["prefs"]["density"] == "compact" and ws["prefs"]["rootPx"] == "18px"
    assert ws["reset"] == {"pref": "system", "font": None, "density": None}
    assert ws["tabs"] == {"selected": "Model preferences", "panes": 1, "end": "Help & support"}
    assert ws["settingsFocusBack"] == "settingsBtn"


def test_stub_uploads_validate_guard_duplicates_and_queue(ws):
    assert ws["badType"] == {"posts": [], "toast": "Unsupported file type: .exe"}
    assert ws["empty"] == {"posts": [], "toast": "\"empty.txt\" is empty."}
    dup = ws["dup"]
    assert dup["confirm"] and dup["title"] == "Upload this file again?" and dup["focus"] == "Skip" and dup["posts"] == []
    assert dup["afterSkip"] == [] and dup["afterConfirm"] == ["POST /upload"]
    assert ws["multi"]["posts"] == ["POST /upload", "POST /upload"]


def test_stub_history_arrow_keys_move_between_chats(ws):
    assert ws["history"].get("rows", 0) >= 2 and ws["history"]["moved"] and ws["history"]["back"]
