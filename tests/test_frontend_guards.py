"""Static, offline guards for the frontend redesign (stages 1-2). Stdlib only."""
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

FE = Path(__file__).resolve().parent.parent / "frontend"
INDEX = (FE / "index.html").read_text(encoding="utf-8")
VIEW_JS = ["ui.js", "chat.js", "utils.js", "icons.js", "theme.js", "components/menu.js", "components/dialog.js",
           "components/model-select.js", "components/ai-viz.js"]
ALLOWED_ORIGINS = {
    "https://fonts.googleapis.com", "https://fonts.gstatic.com",
    "https://cdn.jsdelivr.net", "https://cdnjs.cloudflare.com",
    "http://www.w3.org", "https://example.com",
}


def _read(rel):
    return (FE / rel).read_text(encoding="utf-8")


def test_entrypoints_exist():
    assert (FE / "js/app.js").is_file()
    assert (FE / "css/main.css").is_file()
    assert (FE / "assets/LICENSE-lucide.txt").is_file()


def test_view_files_do_not_call_backend():
    for f in VIEW_JS:
        src = _read(f"js/{f}")
        assert not re.search(r"\bfetch\(|\bAPI\.|\bState\.", src), f"{f} touches fetch/API/State"
        assert not re.search(r"\beval\(|document\.write\(|new Function\(", src), f


def test_no_inline_handlers_or_styles_in_markup():
    assert not re.search(r"\son[a-z]+\s*=", INDEX, re.I)
    styles = re.findall(r'\sstyle="([^"]*)"', INDEX)
    assert styles == ["display:none"], styles  # #stopBtn only (app.js toggles style.display)


def test_no_new_external_origins():
    # first-party code only: js/vendor/ is pinned by SHA-256 in tests/test_frontend_motion.py and its banner names gsap.com
    blob = INDEX + "".join(p.read_text(encoding="utf-8") for p in list((FE / "css").glob("*.css")) + list((FE / "js").rglob("*.js"))
                           if "vendor" not in p.parts)
    origins = set(re.findall(r"https?://[A-Za-z0-9.-]+", blob))
    assert origins <= ALLOWED_ORIGINS, origins - ALLOWED_ORIGINS


def test_app_js_ids_exist_in_index():
    ids = set(re.findall(r'\bid="([^"]+)"', INDEX))
    used = set(re.findall(r"getElementById\(\s*[\"']([^\"']+)[\"']", _read("js/app.js")))
    assert used - ids == set()


def test_sprite_references_resolve():
    ids = set(re.findall(r'<symbol[^>]*\bid="i-([a-z0-9-]+)"', INDEX))
    refs = set(re.findall(r'<use href="#i-([a-z0-9-]+)"', INDEX))
    assert refs <= ids, refs - ids
    for f in VIEW_JS:
        for name in re.findall(r"Icons\.(?:html|el)\(\s*\"([a-z0-9-]+)\"", _read(f"js/{f}")):
            assert name in ids, f"{f}: icon {name}"


def test_no_hex_colors_outside_token_files():
    for f in ("components.css", "animations.css"):
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", _read(f"css/{f}")), f


def test_reduced_motion_rule_present():
    assert "prefers-reduced-motion: reduce" in _read("css/animations.css")


def test_no_right_sidebar_region():
    assert not re.search(r'id="right|class="[^"]*right-(panel|sidebar)', INDEX)


def test_buttons_have_accessible_name():
    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack, self.bad = [], []

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == "button":
                self.stack.append([a, ""])

        def handle_data(self, data):
            if self.stack:
                self.stack[-1][1] += data.strip()

        def handle_endtag(self, tag):
            if tag == "button" and self.stack:
                a, text = self.stack.pop()
                if not (text or a.get("aria-label") or a.get("aria-labelledby") or a.get("title")):
                    self.bad.append(a.get("id") or a.get("class"))

    p = P()
    p.feed(INDEX)
    assert p.bad == []


# ── Stage 2 guards (shell, sidebar, welcome cards, composer, model listbox) ──

def test_index_ids_are_unique():
    ids = re.findall(r'\bid="([^"]+)"', INDEX)
    dupes = {i for i in ids if ids.count(i) > 1}
    assert dupes == set(), dupes


def test_ui_ids_exist_in_index_or_are_created_dynamically():
    ids = set(re.findall(r'\bid="([^"]+)"', INDEX))
    dynamic = {"modelTrigger"}  # created by components/model-select.js
    for f in ("ui.js", "components/model-select.js"):
        used = set(re.findall(r"(?:\$|getElementById)\(\s*[\"']([A-Za-z][\w-]*)[\"']\s*\)", _read(f"js/{f}")))
        assert used - ids - dynamic == set(), (f, used - ids - dynamic)
    assert "modelTrigger" in _read("js/components/model-select.js")


def test_model_select_never_uses_html_injection():
    src = _read("js/components/model-select.js")
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write", src)


def test_session_list_interpolations_are_escaped():
    src = _read("js/ui.js")
    start = src.index("function renderSessionList")
    end = src.index("list.innerHTML = html", start)
    body = src[start:end]
    tpl = "\n".join(l for l in body[body.index("let html"):].splitlines() if not l.strip().startswith("const "))
    exprs = set(re.findall(r"\$\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", tpl))
    safe = ("title", "tip", "id", "gid", "gi", "groupName", "icon", "Utils.escapeHtml(stamp)")
    for e in exprs:
        e = e.strip()
        ok = (e in safe or e.startswith('isActive ? "') or e.startswith("isActive ? '")
              or e.startswith("onRename ?") or e.startswith('Icons.html("'))
        assert ok, f"unreviewed interpolation in renderSessionList: {e}"
    assert "const title = Utils.escapeHtml(rawTitle)" in body
    assert "const id = Utils.escapeHtml(s.id)" in body


def test_file_accept_matches_controller_allow_list():
    accept = re.search(r'id="fileInput"[^>]*accept="([^"]+)"', INDEX).group(1)
    allowed = re.search(r"const ALLOWED = \[([^\]]+)\]", _read("js/app.js")).group(1)
    assert {a.strip().lstrip(".") for a in accept.split(",")} == set(re.findall(r'"([a-z0-9]+)"', allowed))


def test_welcome_cards_and_mode_dropdown_are_consistent():
    ui = _read("js/ui.js")
    # The workspaces are the entries of the sidebar dropdown (ui.js MODES), not permanent rows / tabs
    modes = re.findall(r'\{ id: "([a-z]+)",\s+label: "([^"]+)",\s+icon: "[a-z-]+" \}', ui)
    tabs = [m for m, _ in modes]
    assert sorted(tabs) == ["code", "document", "general", "research"]
    cards = re.findall(r'<button class="welcome__starter[^"]*"[^>]*data-mode="([a-z]+)"\s+data-prompt="([^"]+)"', INDEX)
    assert sorted(m for m, _ in cards) == sorted(tabs)
    assert all(p.strip() for _, p in cards)
    labels = dict(modes)
    for mode, p in cards:  # cards only fill the composer: they must say "Starts with", never "Asks"
        assert f"Starts with: \u201c{p}\u201d" in INDEX, p
        # the aria-label must describe what the card really does: switch to the named workspace and fill the prompt
        # WCAG 2.5.3 (label in name): the accessible name must START with the visible card title
        card = re.search(rf'<button class="welcome__starter[^"]*"[^>]*data-mode="{mode}"\s+data-prompt="{re.escape(p)}"\s+aria-label="([^"]+)">(.*?)</button>', INDEX, re.S)
        al, inner = card.group(1), card.group(2)
        title = re.search(r'<span class="feature-card__title">([^<]+)</span>', inner).group(1)
        assert al == f"{title}. Switches to {labels[mode]} and fills the message box with: {p}", al
    assert "Asks:" not in INDEX
    # example prompts live in the workspace panels now; they only ever fill the composer
    assert "never sends" in _read("js/components/workspaces.js").lower()


def test_no_unverifiable_marketing_claims():
    low = (INDEX + _read("js/ui.js")).lower()
    for word in ("free model", "free models", "100% private", "end-to-end", "encrypted", "unlimited", "fastest"):
        assert word not in low, word


def test_shell_breakpoints_match_between_head_snippet_and_ui():
    assert "w < 768" in INDEX and "w < 1024" in INDEX
    ui = _read("js/ui.js")
    assert "(max-width: 767px)" in ui and "(max-width: 1023px)" in ui


def test_sidebar_toggle_controls_existing_region():
    m = re.search(r'id="sidebarToggle"[^>]*aria-controls="([^"]+)"', INDEX)
    assert m and f'id="{m.group(1)}"' in INDEX
    assert re.search(r'id="sidebarToggle"[^>]*aria-expanded="false"', INDEX)


def test_sidebar_is_hidden_from_tab_order_when_closed_drawer():
    css = _read("css/components.css") + _read("css/main.css")
    assert re.search(r"visibility\s*:\s*hidden", css)
    assert "inert" in _read("js/ui.js")


def test_coarse_pointer_targets_are_44px():
    css = _read("css/components.css") + _read("css/main.css")
    assert re.search(r"@media\s*\(\s*pointer:\s*coarse\s*\)", css)
    assert "44px" in css


def test_model_select_keeps_native_select_as_source_of_truth():
    src = _read("js/components/model-select.js")
    assert 'new Event("change", { bubbles: true })' in src
    assert 'role", "listbox"' in src and "aria-activedescendant" in src
    assert "aria-haspopup" in src and "aria-expanded" in src
    # a no-op re-selection must not emit a change event
    assert "if (sel.value !== item.value)" in src
    # selecting an unavailable option is refused
    assert "item.disabled" in src


def _css_rules(css, selector_re):
    """Yield (selector, body) for every simple top-level or @media-nested rule whose selector matches."""
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        sel = m.group(1).strip()
        if re.search(selector_re, sel):
            yield sel, m.group(2)


def test_feature_card_try_line_is_never_hidden():
    css = _read("css/components.css")
    rules = list(_css_rules(css, r"\.feature-card__try"))
    assert rules, "feature-card__try rules missing"
    for sel, body in rules:
        assert not re.search(r"display\s*:\s*none", body), f"{sel} hides the 'Asks:' line"
        assert not re.search(r"visibility\s*:\s*hidden", body), sel
        assert not re.search(r"(^|;)\s*height\s*:\s*0", body), sel


def test_feature_card_border_uses_control_token():
    css = _read("css/components.css")
    rules = [b for s, b in _css_rules(css, r"^\.welcome__starter$")]
    assert rules, ".welcome__starter rule missing"
    borders = [re.search(r"(?<![-\w])border\s*:\s*([^;]+);", b) for b in rules]
    borders = [m.group(1) for m in borders if m]
    assert borders and all("var(--border-control)" in v for v in borders), borders
    assert not any("var(--border)" in v for v in borders)


# ── Stage 3 guards (messages, sources, AI visualization, smart scroll) ──

def _fn_body(src, name):
    """Source text of a top-level (2-space indented) function in chat.js, up to the next sibling function."""
    start = src.index(f"  function {name}(")
    m = re.search(r"\n  (?:async )?function \w+\(|\n  return \{", src[start + 10:])
    return src[start: start + 10 + m.start()] if m else src[start:]


def test_sources_block_never_uses_html_injection():
    body = _fn_body(_read("js/chat.js"), "_fillSources")
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML", body), "_fillSources must use textContent only"
    assert "Utils.safeUrl(c.url)" in body            # only http(s) urls become links
    assert 'target = "_blank"' in body and 'rel = "noopener noreferrer"' in body


def test_every_blank_target_has_noopener_noreferrer():
    for f in ("chat.js", "ui.js", "utils.js"):
        src = _read(f"js/{f}")
        n_target = len(re.findall(r"""target["']?\s*(?:=|,)\s*["']_blank""", src))
        n_rel = len(re.findall(r"noopener noreferrer", src))
        assert n_rel >= n_target, (f, n_target, n_rel)
    assert 'target="_blank"' not in INDEX or 'rel="noopener' in INDEX


def test_chat_innerhtml_templates_interpolate_only_safe_values():
    src = _read("js/chat.js")
    safe = re.compile(r"^(msgId|Icons\.html\(.*\)|Utils\.escapeHtml\(.*\)|from|status === .*|.*Icons\.html.*|parts\.join.*)$", re.S)
    tpls = re.findall(r"innerHTML = `(.*?)`;", src, re.S)
    assert len(tpls) >= 2
    for tpl in tpls:
        for expr in re.findall(r"\$\{(.*?)\}(?=[^}]|$)", tpl, re.S):
            assert safe.match(expr.strip()), f"unreviewed interpolation in chat.js innerHTML: {expr[:80]}"
    # badges: model and fallback list must be escaped
    body = _fn_body(src, "_badgesHtml")
    assert "Utils.escapeHtml(model)" in body and "Utils.escapeHtml(fellBackFrom.join" in body


def test_renderer_and_chat_api_used_by_app_js_still_exists():
    app, chat = _read("js/app.js"), _read("js/chat.js")
    for name in set(re.findall(r"\brenderer\.(\w+)\(", app)):
        assert re.search(rf"\b{name}\s*[(:]", chat), f"renderer.{name} missing from chat.js"
    for name in set(re.findall(r"\bChat\.(\w+)\(", app)):
        assert re.search(rf"^\s+{name},?$", chat[chat.rindex("return {"):], re.M), f"Chat.{name} not exported"
    # additive renderer API promised in view-contract section 13
    for name in ("markPartial", "setVizState"):
        assert name in chat


def test_ai_viz_is_css_only_and_decorative():
    js, css = _read("js/components/ai-viz.js"), _read("css/ai-viz.css")
    code = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    assert not re.search(r"setInterval|setTimeout|requestAnimationFrame|Math\.random", code)
    assert 'setAttribute("aria-hidden", "true")' in js
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css), "ai-viz.css must use tokens only"
    assert not re.search(r"\b(width|height|top|left|margin|stroke-width)\s*:[^;]*;[^}]*\banimation", css)


def test_ai_viz_reduced_motion_and_hidden_tab_rules():
    css = _read("css/ai-viz.css")
    m = re.search(r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\n\}", css, re.S)
    assert m and "animation: none" in m.group(1) and "transition: none" in m.group(1)
    assert "animation-play-state: paused" in css and "data-page-hidden" in css
    for kf in re.findall(r"@keyframes (\w+)\s*\{(.*?)\n\}", css, re.S):
        assert set(re.findall(r"([a-z-]+)\s*:", kf[1])) <= {"transform", "opacity"}, kf[0]


def test_no_interval_timers_or_raf_loops_in_frontend():
    for p in (FE / "js").rglob("*.js"):
        src = p.read_text(encoding="utf-8")
        assert "setInterval" not in src, p.name
    chat = _read("js/chat.js")
    assert len(re.findall(r"requestAnimationFrame", chat)) <= 2  # one-shot scroll follow only


def test_streaming_does_not_announce_per_token():
    chat = _read("js/chat.js")
    body = chat[chat.index("appendToken(token)"): chat.index("setBadges(")]
    assert "_announce" not in body and "setLiveStatus" not in body
    assert "setTimeout(_render, 60)" in body   # throttle preserved


def test_stage3_ids_exist():
    ids = set(re.findall(r'\bid="([^"]+)"', INDEX))
    assert {"welcomeViz", "jumpLatest", "liveStatus", "chatArea"} <= ids
    for name in re.findall(r"getElementById\(\s*[\"']([A-Za-z][\w-]*)[\"']\s*\)", _read("js/chat.js")):
        assert name in ids or name == "thinkingPlaceholder", name   # thinkingPlaceholder is created by showThinking()
    assert 'href="/css/ai-viz.css"' in INDEX and "/js/components/ai-viz.js" in INDEX


def test_sanitizer_policy_forbids_class_role_download_ping_in_utils():
    m = re.search(r"FORBID_ATTR\s*:\s*\[(.*?)\]", _read("js/utils.js"), re.S)
    assert {"class", "role", "download", "ping"} <= set(re.findall(r'"([^"]+)"', m.group(1)))


# ---------------------------------------------------------------------------
# Citation fixes: fallback labels, consistent ids, 44px coarse-pointer target
# ---------------------------------------------------------------------------
def _fn_body(src, name):
    m = re.search(r"function\s+" + name + r"\s*\([^)]*\)\s*\{(.*?)\n  \}", src, re.S)
    assert m, f"{name} not found"
    return m.group(1)


def test_citation_label_falls_back_to_source_n_and_is_used_everywhere():
    chat = _read("js/chat.js")
    body = _fn_body(chat, "_citationLabel")
    assert "`Source ${c.id}`" in body and ".trim()" in body
    # Sources row title and both dialog entry points go through the label helper
    assert "const title = _citationLabel(c, domain);" in chat
    assert chat.count("UI.showCitationModal(_modalCitation(") == 2
    assert "UI.showCitationModal(cit)" not in chat and "UI.showCitationModal(c)" not in chat


def test_citation_ids_resolved_once_for_stream_and_history_paths():
    chat = _read("js/chat.js")
    assert "citations = _resolveCitations(citations);" in chat
    assert "_resolveCitations(msg.citations)" in chat
    assert "Object.assign({}, c, { id })" in _fn_body(chat, "_resolveCitations")   # copies, never mutates input
    assert "Number(id)" in _fn_body(chat, "getCitation")


def test_coarse_pointer_citation_hit_area_rule_present_and_scoped():
    css = _read("css/components.css")
    m = re.search(r"@media \(pointer: coarse\)\s*\{(.*?)\n\}", css, re.S)
    assert m
    rule = re.search(r'\.citation\[role="button"\]::after\s*\{(.*?)\}', m.group(1), re.S)
    assert rule, "coarse-pointer hit-area pseudo-element missing"
    assert "position: absolute" in rule.group(1) and "height: 44px" in rule.group(1)
    assert re.search(r"\.citation\s*\{\s*position:\s*relative", css)   # anchor, outside the media query


_CITE_HARNESS = """<!doctype html><html><head><meta charset="utf-8"><style>%(css)s</style></head>
<body style="margin:0;padding-top:80px"><div id="chatArea"></div><pre id="out"></pre>
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
<script>%(icons)s</script><script>%(utils)s</script><script>%(chat)s</script>
<script>
const R = {libs: [typeof marked, typeof DOMPurify]};
window.UI = {showCitationModal(c) { window.__modal = c; }, setLiveStatus() {}, setVizState() {}};
function ids(cits) {
  Chat.renderHistory([{role: "assistant", content: "x", citations: cits}]);
  return [...document.querySelectorAll(".source")].map(li => ({
    id: li.querySelector(".source__num").textContent,
    title: li.querySelector(".source__title-text").textContent.trim()}));
}
function sweep(sel) {
  const bd = document.querySelector(sel); const r = bd.getBoundingClientRect();
  const cx = r.x + r.width / 2, cy = r.y + r.height / 2; let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  for (let x = cx - 40; x <= cx + 40; x += 0.5) { const e = document.elementFromPoint(x, cy); if (e && e.closest(".citation") === bd) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); } }
  for (let y = cy - 40; y <= cy + 40; y += 0.5) { const e = document.elementFromPoint(cx, y); if (e && e.closest(".citation") === bd) { y0 = Math.min(y0, y); y1 = Math.max(y1, y); } }
  return [x1 - x0 + 0.5, y1 - y0 + 0.5, r.width, r.height];
}
if (!R.libs.includes("undefined")) {
  const C = (s, id) => (id === undefined ? {source: s, type: "document"} : {source: s, type: "document", id});
  const inputs = {
    none: [C("A"), C("B"), C("C")], explicit: [C("A", 5), C("B", 2), C("C", 9)],
    dup: [C("A", 2), C("B", 2), C("C")], strs: [C("A", "3"), C("B", "x"), C("C", " 4 ")],
    odd: [C("A", 0), C("B", -1), C("C", 1.5), C("D", null)],
    ooo: [C("A", 3), C("B"), C("C", 1), C("D")],
    many: Array.from({length: 25}, (_, i) => C("s" + i)),
    blank: [{type: "web", source: "  ", url: ""}, {type: "web", source: null, url: "javascript:alert(1)"}, {type: "document"},
            {type: "web", url: "https://www.example.org/a"}],
  };
  R.mutated = false; R.ids = {};
  for (const [k, v] of Object.entries(inputs)) { const s = JSON.stringify(v); R.ids[k] = ids(v); if (JSON.stringify(v) !== s) R.mutated = true; }
  Chat.renderHistory([{role: "assistant", content: "a [1] b [2]", citations: [{type: "document", source: "  "}, {type: "web", source: "<img src=x onerror=window.__x=1>", url: "https://e.com/a"}]}]);
  const bs = [...document.querySelectorAll(".citation[role=button]")];
  R.badges = bs.map(b => b.dataset.id + ":" + b.getAttribute("aria-label"));
  bs[0].click(); R.modal0 = window.__modal && window.__modal.source;
  bs[1].click(); R.modal1 = window.__modal && window.__modal.source; R.xss = window.__x || 0;
  R.unsafeRows = document.querySelector(".sources").innerHTML.includes("<img");
  Chat.renderHistory([{role: "assistant", content: "Alone [1] end.\\n\\nfiller\\n\\nfiller\\n\\nFact one [1]. Fact two [2]. Fact three [3].\\n\\nfiller\\n\\nfiller\\n\\nTight [1][2].",
                       citations: [{source: "a"}, {source: "b"}, {source: "c"}]}]);
  const all = document.querySelectorAll(".citation[role=button]");
  all.forEach((b, i) => b.dataset.t = "t" + i);
  R.hit = Array.from(all, (_, i) => sweep('[data-t="t' + i + '"]'));
}
document.getElementById("out").textContent = JSON.stringify(R);
</script></body></html>"""


@pytest.fixture(scope="module")
def cite_result(tmp_path_factory):
    import html as _html
    import json as _json
    import shutil as _sh
    import subprocess as _sp
    chrome = next((p for p in map(_sh.which, ("google-chrome", "chromium", "chromium-browser", "chrome")) if p), None)
    if not chrome:
        pytest.skip("headless Chrome not available")
    css = _read("css/main.css") + _read("css/components.css").replace("@media (pointer: coarse)", "@media all")
    d = tmp_path_factory.mktemp("cite")
    page = d / "h.html"
    esc = lambda s: s.replace("</script", "<\\/script")  # noqa: E731
    page.write_text(_CITE_HARNESS % {"css": css, "icons": esc(_read("js/icons.js")), "utils": esc(_read("js/utils.js")),
                                      "chat": esc(_read("js/chat.js"))}, encoding="utf-8")
    try:
        p = _sp.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--window-size=390,900",
                     f"--user-data-dir={d / 'prof'}", "--virtual-time-budget=15000", "--dump-dom", page.as_uri()],
                    capture_output=True, text=True, timeout=60)
    except (_sp.TimeoutExpired, OSError) as e:
        pytest.skip(f"chrome could not run: {e}")
    m = re.search(r'<pre id="out">(.*?)</pre>', p.stdout, re.S)
    if not m or not m.group(1).strip():
        pytest.skip("harness produced no output")
    res = _json.loads(_html.unescape(m.group(1)))
    if "undefined" in res["libs"]:
        pytest.skip(f"CDN libraries unavailable: {res['libs']}")
    return res


def _id_list(r, k):
    return [row["id"] for row in r["ids"][k]]


def test_browser_citation_ids_consistent_unique_and_input_not_mutated(cite_result):
    r = cite_result
    assert r["mutated"] is False
    assert _id_list(r, "none") == ["1", "2", "3"]
    assert _id_list(r, "explicit") == ["5", "2", "9"]            # explicit backend ids respected
    assert _id_list(r, "dup") == ["2", "1", "3"]                 # duplicate gets next free id
    assert _id_list(r, "strs") == ["3", "1", "4"]                # numeric strings kept, junk replaced
    assert _id_list(r, "odd") == ["1", "2", "3", "4"]            # 0 / negative / float / null replaced
    assert _id_list(r, "ooo") == ["3", "2", "1", "4"]
    assert _id_list(r, "many") == [str(i) for i in range(1, 26)]
    for k, rows in r["ids"].items():
        ids = [x["id"] for x in rows]
        assert len(ids) == len(set(ids)), k                      # no collisions


def test_browser_blank_citations_get_accessible_source_n_names(cite_result):
    r = cite_result
    titles = [row["title"] for row in r["ids"]["blank"]]
    assert titles[:3] == ["Source 1", "Source 2", "Source 3"]    # includes unsafe javascript: url
    assert titles[3] == "example.org"
    assert all(row["title"] for rows in r["ids"].values() for row in rows)   # never an empty control
    assert r["badges"] == ["1:Source 1", "2:Source 2"]
    assert r["modal0"] == "Source 1"                             # dialog title agrees with the row
    assert r["modal1"] == "<img src=x onerror=window.__x=1>" and r["xss"] == 0 and r["unsafeRows"] is False


def test_browser_isolated_badge_reaches_44px_on_coarse_pointers(cite_result):
    w, h, vw, vh = cite_result["hit"][0]
    assert vw <= 32 and vh <= 26                                 # visual size stays small
    assert w >= 44 and h >= 44, (w, h)


def test_browser_multi_citation_sentence_badges_reach_44px_wide(cite_result):
    for w, h, *_ in cite_result["hit"][1:4]:                     # Fact one [1]. Fact two [2]. Fact three [3].
        assert w >= 44 and h >= 44, (w, h)


# ---------------------------------------------------------------------------
# Stage 4 guards (controller integration). Static checks + a STUB-ONLY browser run.
# ---------------------------------------------------------------------------
APP = _read("js/app.js")
API_JS = _read("js/api.js")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐⭕️]")


def _code(src):
    """Source without comments (good enough for guards: no string literal in app.js contains '//')."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$|\s//[^\"'`\n]*$", "", src)


def _app_fn(name):
    m = re.search(r"\n  (?:async )?function " + name + r"\([^)]*\)\s*\{", APP)
    assert m, name
    rest = APP[m.end():]
    nxt = re.search(r"\n  (?:async )?function \w+\(|\n  // ──", rest)
    return rest[: nxt.start()] if nxt else rest


def test_app_js_has_no_native_confirm_inline_color_interval_or_emoji():
    code = _code(APP)
    assert not re.search(r"(?<![.\w])confirm\(", code), "use UI.confirm, not window.confirm"
    assert not re.search(r"\.style\.color\b", code), "style the toggle via [aria-pressed] in CSS"
    assert "setInterval" not in code
    assert not _EMOJI.search(APP), "app.js must use icon names, not emoji"
    assert "UI.confirm(" in code and code.count("await UI.confirm(") == 3   # chat delete + document delete + duplicate-upload guard


def test_welcome_card_handler_never_sends():
    m = re.search(r'UI\.onWelcomeCard\(.*?\n    \}\);', APP, re.S)
    assert m, "UI.onWelcomeCard handler missing"
    body = _code(m.group(0))
    for forbidden in ("createSession", "_ensureSession", "chatStream", "_onSend", "API.", "sendBtn", ".click("):
        assert forbidden not in body, f"welcome card handler must not call {forbidden}"
    assert "_setMode(" in body and "_fillComposer(" in body
    picker = _code(_app_fn("_openFilePicker"))   # the picker helper lives outside the handler body
    assert 'getElementById("fileInput")' in picker
    for forbidden in ("createSession", "chatStream", "_onSend", "API.chat"):
        assert forbidden not in picker, f"_openFilePicker must not reference {forbidden}"
    fill = _code(_app_fn("_fillComposer"))
    assert not re.search(r"_onSend|chatStream|createSession|API\.", fill) and ".focus()" in fill
    # ui.js must not re-introduce a send/click path for the cards
    ui = _code(_read("js/ui.js"))
    assert "welcome__starter" not in ui and "starterPrompts" not in ui


def test_allowed_extensions_include_rst_and_match_accept_and_backend():
    allowed = set(re.findall(r'"([a-z0-9]+)"', re.search(r"const ALLOWED = \[([^\]]+)\]", APP).group(1)))
    assert "rst" in allowed
    accept = re.search(r'id="fileInput"[^>]*accept="([^"]+)"', INDEX).group(1)
    assert {a.lstrip(".") for a in accept.split(",")} == allowed
    from backend.services.parser import SUPPORTED_EXTENSIONS as sup
    assert allowed == set(sup), (allowed ^ set(sup))


def test_request_payload_shapes_match_the_backend_contract():
    """Keys sent to /chat/stream and /sessions are exactly those of the HEAD controller / backend models."""
    chat_keys = set(re.findall(r"^\s+(\w+)(?::| *,)", re.search(r"API\.chatStream\(\s*\{(.*?)\n        \},", APP, re.S).group(1), re.M))
    assert chat_keys == {"session_id", "message", "user_id", "mode", "model", "web_search", "regenerate"}, chat_keys
    # createSession(data, signal): the body keys are unchanged; the only addition is a trailing AbortSignal argument
    sess = re.search(r"API\.createSession\(\{(.*?)\}(?:, signal)?\)", APP, re.S).group(1)
    assert set(re.findall(r"(\w+):", sess)) == {"user_id", "title", "model", "mode"}
    assert len(re.findall(r"API\.createSession\(", APP)) == 1
    assert "}, signal);" in APP and "function createSession(data, signal)" in API_JS
    assert 'body: JSON.stringify(data), ...(signal ? { signal } : {})' in API_JS   # body is still exactly `data`
    assert 'API.updateSession(State.sessionId, { mode })' in APP and "{ model: State.model }" in APP
    assert "API.updateSession(sessionId, { title })" in APP
    # api.js: endpoints unchanged
    for frag in ('/chat/stream', 'request("/sessions", { method: "POST"', "/upload?user_id=", "/upload?url=",
                 "/documents/${docId}?user_id=", "/messages/${messageId}/feedback"):
        assert frag in API_JS, frag
    backend = (FE.parent / "backend/models/api.py").read_text(encoding="utf-8")
    cr = re.search(r"class ChatRequest\(BaseModel\):(.*?)\n\n\n", backend, re.S).group(1)
    assert set(re.findall(r"^\s+(\w+):", cr, re.M)) == chat_keys


def test_regenerate_and_retry_send_the_head_shape():
    """Backend: message defaults to "" and pop_last_exchange() keeps the user turn; the controller must send message=""."""
    body = _app_fn("_onSend")
    assert 'const draft = regenerate ? "" : textarea.value;' in body
    assert "message: text," in body and "regenerate," in body
    retry = _app_fn("_retryAfterError")
    assert "_onSend({ regenerate: true })" in retry and "API." not in retry


def test_composer_is_cleared_only_after_the_server_accepted():
    body = _app_fn("_onSend")
    first_clear = body.index('textarea.value = ""')
    assert body.index("const onAccepted") < first_clear < body.index("await API.chatStream("), "composer cleared before accept"
    assert "{ onOpen: onAccepted }" in body
    assert "onOpen" in API_JS and API_JS.index("if (!resp.ok)") < API_JS.index("opts.onOpen()")


def test_no_new_timer_or_listener_patterns_in_controller():
    code = _code(APP)
    assert len(re.findall(r"setTimeout\(", code)) == 1          # the single cooldown timer
    assert "clearTimeout(State.cooldownTimer)" in code
    assert "innerHTML" not in code and "insertAdjacentHTML" not in code and "document.write" not in code


# ── Stub-only browser regression: nothing leaves the page, no backend is started ──────────────

_STUB = r"""
window.__calls=[];window.__streams=[];window.__handlers={};window.__errs=[];window.__files=[];
window.addEventListener('error',e=>window.__errs.push(String(e.message)));
window.addEventListener('unhandledrejection',e=>window.__errs.push('rej:'+String(e.reason&&e.reason.message||e.reason)));
(function(){
 var S=window.__S={n:1,sessions:[],docs:[],models:{models:[{id:'m1',label:'M1',provider:'g',description:'',available:true,reason:null,default:true}],default:'m1'}};
 var mk=function(o){return Object.assign({id:'s'+(S.n++),user_id:'default',title:'New Chat',model:'m1',mode:'general',created_at:new Date().toISOString(),updated_at:new Date().toISOString(),messages:[]},o)};
 if(typeof window.__seedFn==='function')window.__seedFn(S,mk);
 var js=function(status,body,h){return {status:status,json:body,headers:h||{}}};
 function def(r){var m=r.method,p=r.path,x;
  if(m==='GET'&&p==='/models')return js(200,S.models);
  if(m==='GET'&&p==='/sessions')return js(200,S.sessions.map(function(s){return Object.assign({},s,{messages:undefined})}));
  if(m==='GET'&&p==='/documents')return js(200,S.docs);
  if(m==='POST'&&p==='/sessions'){var s=mk({title:(r.json||{}).title});S.sessions.push(s);return js(200,s)}
  if((x=p.match(/^\/sessions\/([^/]+)$/))){var ss=S.sessions.find(function(q){return q.id===x[1]});if(!ss)return js(404,{detail:'Session not found.'});
   if(m==='GET')return js(200,ss);if(m==='PATCH'){Object.assign(ss,r.json);return js(200,ss)}if(m==='DELETE')return {status:204}}
  if(m==='POST'&&p==='/upload')return js(200,{document_id:'d1',filename:r.fileName||'u',file_type:'txt',chunk_count:1,indexing:'full'});
  if(m==='POST'&&p==='/chat/stream')return {stream:true};
  return js(599,{detail:'unstubbed '+m+' '+p})}
 window.__sse=function(n,d){return 'event: '+n+'\ndata: '+JSON.stringify(d)+'\n\n'};
 window.fetch=async function(url,o){o=o||{};var u=new URL(url,location.href),m=(o.method||'GET').toUpperCase();
  var r={method:m,path:u.pathname,search:u.search};
  if(typeof o.body==='string'){try{r.json=JSON.parse(o.body)}catch(e){}}
  else if(o.body instanceof FormData){var f=o.body.get('file');if(f){r.fileName=f.name;window.__files.push(f)}}
  var q=new URLSearchParams(u.search);if(q.get('url'))r.url=q.get('url');
  window.__calls.push(r);
  var h=window.__handlers[m+' '+u.pathname],spec=h?await h(r):undefined;if(spec===undefined)spec=def(r);
  if(spec.reject)throw new TypeError('Failed to fetch');
  if(spec.hang)await new Promise(function(ok,bad){if(o.signal)o.signal.addEventListener('abort',function(){bad(new DOMException('Aborted','AbortError'))})});
  if(spec.stream){var enc=new TextEncoder(),c;var body=new ReadableStream({start:function(k){c=k}});
   var ctl={push:function(s){c.enqueue(enc.encode(s))},close:function(){try{c.close()}catch(e){}}};
   if(o.signal)o.signal.addEventListener('abort',function(){try{c.error(new DOMException('Aborted','AbortError'))}catch(e){}});
   window.__streams.push(ctl);return new Response(body,{status:200})}
  if(spec.status===204)return new Response(null,{status:204});
  return new Response(JSON.stringify(spec.json),{status:spec.status,headers:new Headers(spec.headers)})};
 window.XMLHttpRequest=function(){throw new Error('XHR blocked')};window.WebSocket=function(){throw new Error('WS blocked')};
})();
"""

_SCENARIO = r"""
(function(){
var R={},$=function(i){return document.getElementById(i)},tick=function(ms){return new Promise(function(r){setTimeout(r,ms)})};
async function until(f,ms){var t=0;while(!f()&&t<(ms||3000)){await tick(20);t+=20}return !!f()}
var posts=function(re){return window.__calls.filter(function(c){return c.method==='POST'&&re.test(c.path)})};
var users=function(){return document.querySelectorAll('.message--user').length};
var asst=function(){return document.querySelectorAll('.message--assistant').length};
async function fresh(){window.__handlers={};window.__calls.length=0;$('chatInput').value='';$('newChatBtn').click();await tick(120);window.__calls.length=0}
async function send(t){$('chatInput').value=t;$('sendBtn').click();await tick(120)}
async function pickMode(m){$('modeSelect').click();await tick(30);var it=document.querySelector('.menu__item[data-menu-id="'+m+'"]');if(it)it.click();await tick(40)}
async function finishStream(i,text){window.__streams[i].push(__sse('token',{text:text||'ok'})+__sse('done',{message_id:'m'+i,persisted:true}));window.__streams[i].close();await until(function(){return $('sendBtn').style.display!=='none'})}
async function run(){
 if(!window.marked||!window.DOMPurify){R.nolibs=true;return}
 await tick(300);
 // 1. welcome cards never send (stub only); from every starting mode
 R.cards=[];
 var modes=['general','document','research','code'];
 for(var a=0;a<modes.length;a++){for(var b=0;b<modes.length;b++){
   await pickMode(modes[a]);
   window.__calls.length=0;$('chatInput').value='';
   var card=document.querySelector('.welcome__starter[data-mode="'+modes[b]+'"]');card.focus();card.click();await tick(60);
   R.cards.push({from:modes[a],card:modes[b],calls:window.__calls.map(function(c){return c.method+' '+c.path}),value:$('chatInput').value,
     prompt:card.dataset.prompt,focus:document.activeElement.id,current:$('modeSelect').dataset.mode})}}
 // 2. drafts
 window.__handlers={};window.__calls.length=0;     // no session exists yet: the first send must create one
 window.__handlers['POST /sessions']=function(){return {status:500,json:{detail:'db down'}}};
 await send('D-session');
 R.sessFail={value:$('chatInput').value,users:users(),streamCalls:window.__calls.filter(function(c){return c.path==='/chat/stream'}).length,busy:$('sendBtn').style.display==='none'};
 delete window.__handlers['POST /sessions'];$('sendBtn').click();await until(function(){return window.__streams.length>0});
 R.sessRecover={users:users(),value:$('chatInput').value,msg:(window.__calls.filter(function(c){return c.path==='/chat/stream'})[0]||{json:{}}).json.message};
 await finishStream(0);
 for(var k=0,specs=[['429',{status:429,json:{detail:'slow down'},headers:{'Retry-After':'5'}}],['503',{status:503,json:{detail:'no provider'}}],['404',{status:404,json:{detail:'Session not found.'}}],['net',{reject:true}]];k<specs.length;k++){
   await fresh();var sp=specs[k][1],nm=specs[k][0];var n0=window.__streams.length;
   window.__handlers['POST /chat/stream']=function(){return sp};
   await send('D-'+nm);
   var rec={value:$('chatInput').value,users:users(),asst:asst(),busy:$('sendBtn').style.display==='none',toast:(document.querySelector('.toast')||{}).textContent};
   if(nm==='429'){window.__calls.length=0;$('sendBtn').click();await tick(80);rec.earlySendCalls=window.__calls.length;
     await tick(5300);delete window.__handlers['POST /chat/stream'];$('sendBtn').click();await until(function(){return window.__streams.length>n0});
     rec.afterCooldown=window.__calls.filter(function(c){return c.path==='/chat/stream'}).length;await finishStream(n0)}
   R['http'+nm]=rec}
 // stream error before the first token: draft restored, one user bubble; resend and Retry
 for(var v=0,routes=['resend','retry'];v<routes.length;v++){
   await fresh();var n1=window.__streams.length;await send('D-err');await until(function(){return window.__streams.length>n1});
   window.__streams[n1].push(__sse('meta',{})+__sse('error',{code:'rate_limited',message:'busy',retryable:true,retry_after:1}));window.__streams[n1].close();
   await until(function(){return document.querySelector('.message-error')});await tick(50);
   var e={value:$('chatInput').value,users:users(),retryDisabledAtFirst:(document.querySelector('.message-error .btn--secondary')||{}).disabled};
   await tick(1500);window.__calls.length=0;
   if(routes[v]==='resend'){$('sendBtn').click()}else{document.querySelector('.message-error .btn--secondary').click()}
   await until(function(){return window.__streams.length>n1+1});
   var pl=window.__calls.filter(function(c){return c.path==='/chat/stream'})[0].json;
   e.payload={regenerate:pl.regenerate,message:pl.message};e.usersAfter=users();e.valueAfter=$('chatInput').value;
   await finishStream(n1+1);R['err_'+routes[v]]=e}
 // regenerate failure keeps the old answer
 await tick(10);
 window.__handlers['POST /chat/stream']=function(){return {status:500,json:{detail:'boom'}}};
 document.querySelector('.session-item[data-session-id="sH"] .session-item__main').click();await until(function(){return /OLD ANSWER/.test($('chatArea').textContent)});
 var before=document.querySelector('.message--assistant .md-content').textContent;window.__calls.length=0;
 document.querySelector('.message__regen').click();await tick(150);
 R.regenFail={same:document.querySelector('.message--assistant .md-content').textContent===before,count:asst(),users:users(),btn:!!document.querySelector('.message__regen'),
   payload:(window.__calls.filter(function(c){return c.path==='/chat/stream'})[0]||{json:{}}).json};
 delete window.__handlers['POST /chat/stream'];
 // upload failure: same File object is re-sent on Retry
 window.__handlers['POST /upload']=function(){return {status:500,json:{detail:'upload exploded'}}};
 var dt=new DataTransfer();dt.items.add(new File(['abc'],'notes.rst'));$('fileInput').files=dt.files;$('fileInput').dispatchEvent(new Event('change',{bubbles:true}));
 await until(function(){return /failed/i.test($('uploadToast').textContent)&&!$('uploadToast').hidden});
 delete window.__handlers['POST /upload'];var rb=[].slice.call(document.querySelectorAll('#uploadToast button')).filter(function(b){return /retry/i.test(b.textContent)})[0];
 R.upFail={retry:!!rb,errorText:/upload exploded/.test($('uploadToast').textContent)};
 if(rb){rb.click();await until(function(){return posts(/^\/upload$/).length>=2})}
 R.upRetry={posts:posts(/^\/upload$/).length,sameFile:window.__files.length>=2&&window.__files[0]===window.__files[1],name:(window.__files[0]||{}).name};
 // unsupported types never reach the network
 window.__calls.length=0;
 for(var z=0,names=['a.exe','a.rst.exe','README'];z<names.length;z++){var d2=new DataTransfer();d2.items.add(new File(['x'],names[z]));$('fileInput').files=d2.files;$('fileInput').dispatchEvent(new Event('change',{bubbles:true}));await tick(40)}
 var d3=new DataTransfer();d3.items.add(new File(['x'],'UP.RST'));$('fileInput').files=d3.files;$('fileInput').dispatchEvent(new Event('change',{bubbles:true}));await tick(120);
 R.types={uploads:posts(/^\/upload$/).map(function(c){return c.fileName})};
 R.errs=window.__errs.slice();
}
run().catch(function(e){R.crash=String(e&&e.stack||e)}).then(function(){var o=document.createElement('pre');o.id='qa-out';o.textContent=JSON.stringify(R);document.body.appendChild(o)});
})();
"""

_SEED = ("try{history.replaceState(null,'','#app')}catch(e){}"
         "window.__seedFn=function(S,mk){var now=new Date().toISOString();S.sessions.push(mk({id:'sH',title:'History',messages:["
         "{id:'u1',session_id:'sH',role:'user',content:'Q1',created_at:now},{id:'a1',session_id:'sH',role:'assistant',content:'OLD ANSWER',created_at:now,meta:{model:'m1'}}]}))};")


@pytest.fixture(scope="module")
def stub_run(tmp_path_factory):
    """Serve frontend/ statically (GET only), replace fetch with a stub, run the scenario in headless Chrome."""
    import html as _html
    import http.server as _hs
    import json as _json
    import shutil as _sh
    import subprocess as _sp
    import threading as _th
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
            ctype = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}.get(f.suffix, "application/octet-stream")
            self.send_header("Content-Type", ctype + "; charset=utf-8" if f.suffix in (".html", ".js", ".css") else ctype)
            self.end_headers()
            self.wfile.write(data)

        def _refuse(self):          # any non-GET request reaching the server is a test failure
            blocked.append(f"{self.command} {self.path}")
            self.send_response(405)
            self.end_headers()

        do_POST = do_PUT = do_PATCH = do_DELETE = _refuse

    srv = _hs.ThreadingHTTPServer(("127.0.0.1", 0), H)
    _th.Thread(target=srv.serve_forever, daemon=True).start()
    d = tmp_path_factory.mktemp("stub4")
    try:
        p = _sp.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", f"--user-data-dir={d / 'prof'}",
                     "--virtual-time-budget=60000", "--window-size=1280,900", "--dump-dom",
                     f"http://127.0.0.1:{srv.server_address[1]}/"], capture_output=True, text=True, timeout=120)
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


def test_stub_nothing_reached_a_server(stub_run):
    # only static GETs (scripts/css/assets) were served; no API request ever left the page
    assert stub_run["__blocked"] == [] and stub_run["errs"] == []


def test_stub_welcome_cards_never_submit_and_fill_the_composer(stub_run):
    assert len(stub_run["cards"]) == 16
    for c in stub_run["cards"]:
        tag = f"{c['card']} from {c['from']}"
        assert not [x for x in c["calls"] if x.startswith("POST")], (tag, c["calls"])   # no createSession / chatStream
        assert c["value"] == c["prompt"] and c["focus"] == "chatInput" and c["current"] == c["card"], (tag, c)


def test_stub_failures_before_accept_keep_the_draft_and_add_no_bubble(stub_run):
    r = stub_run
    assert r["sessFail"] == {"value": "D-session", "users": 0, "streamCalls": 0, "busy": False}
    assert r["sessRecover"] == {"users": 1, "value": "", "msg": "D-session"}          # cleared only once accepted
    for k, text in (("429", "D-429"), ("503", "D-503"), ("404", "D-404"), ("net", "D-net")):
        rec = r["http" + k]
        assert rec["value"] == text and rec["users"] == 0 and rec["asst"] == 0 and not rec["busy"], (k, rec)
        assert rec["toast"], k
    assert r["http429"]["earlySendCalls"] == 0       # cooldown: no request while Retry-After is running
    assert r["http429"]["afterCooldown"] == 1        # and a normal send afterwards


def test_stub_stream_error_restores_draft_and_never_duplicates_the_user_bubble(stub_run):
    for route in ("resend", "retry"):
        e = stub_run["err_" + route]
        assert e["value"] == "D-err" and e["users"] == 1 and e["retryDisabledAtFirst"] is True, (route, e)
        assert e["usersAfter"] == 1, (route, e)
    # M1 (intended): the failed attempt's user turn is already stored server-side, so an UNEDITED resend takes the
    # regenerate path (like Retry) and the server never stores the question twice. Edited drafts: see test_frontend_followup.py
    assert stub_run["err_resend"]["payload"] == {"regenerate": True, "message": ""}
    # inline Retry regenerates (same shape as the regenerate button): backend keeps the stored user turn
    assert stub_run["err_retry"]["payload"] == {"regenerate": True, "message": ""}
    assert stub_run["err_retry"]["valueAfter"] == ""


def test_stub_failed_regenerate_keeps_the_previous_answer(stub_run):
    r = stub_run["regenFail"]
    assert r["same"] and r["count"] == 1 and r["users"] == 1 and r["btn"], r
    assert r["payload"]["regenerate"] is True and r["payload"]["message"] == ""


def test_stub_upload_failure_retry_resends_the_same_file(stub_run):
    assert stub_run["upFail"] == {"retry": True, "errorText": True}
    assert stub_run["upRetry"] == {"posts": 2, "sameFile": True, "name": "notes.rst"}
    assert stub_run["types"]["uploads"] == ["UP.RST"]            # .exe / .rst.exe / no extension never uploaded
