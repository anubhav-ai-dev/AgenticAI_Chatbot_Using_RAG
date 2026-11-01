"""Regression guards for the Markdown sanitiser in frontend/js/utils.js.

Static assertions always run. The browser test needs headless Chrome and
network access to the CDN libraries; it skips cleanly when either is missing.
"""
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

FE = Path(__file__).resolve().parent.parent / "frontend"
UTILS = (FE / "js/utils.js").read_text(encoding="utf-8")
ICONS = (FE / "js/icons.js").read_text(encoding="utf-8")
CODEPREVIEW = (FE / "js/components/code-preview.js").read_text(encoding="utf-8")
INDEX = (FE / "index.html").read_text(encoding="utf-8")


def _forbid_attr():
    m = re.search(r"FORBID_ATTR\s*:\s*\[(.*?)\]", UTILS, re.S)
    assert m, "FORBID_ATTR not found in utils.js"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def _hook_body():
    m = re.search(r"function\s+keepLanguageClass\s*\([^)]*\)\s*\{(.*?)\n  \}", UTILS, re.S)
    assert m, "keepLanguageClass not found"
    return m.group(1)


def test_forbid_attr_contains_required_attributes():
    assert {"download", "ping", "class", "role"} <= _forbid_attr()
    assert {"style", "id", "name", "srcset"} <= _forbid_attr()


def test_language_hook_is_scoped_to_code_and_validated():
    body = _hook_body()
    assert 'data.attrName === "class"' in body
    assert 'node.nodeName === "CODE"' in body
    assert "^language-" in body and "$/" in body  # anchored value check
    assert "forceKeepAttr = true" in body
    # no other element or attribute may be force-kept
    assert body.count("forceKeepAttr") == 1


def test_hook_registered_only_around_sanitize_and_always_removed():
    m = re.search(r"function\s+sanitizeMarkdownHtml\s*\([^)]*\)\s*\{(.*?)\n  \}", UTILS, re.S)
    assert m, "sanitizeMarkdownHtml not found"
    body = m.group(1)
    assert "addHook(\"uponSanitizeAttribute\", keepLanguageClass)" in body
    assert re.search(r"try\s*\{.*sanitize\(.*\}\s*finally\s*\{\s*DOMPurify\.removeHook\(", body, re.S)
    # the hook is never registered anywhere else (i.e. not globally)
    assert UTILS.count("addHook(") == 1
    assert "DOMPurify.sanitize(" not in UTILS.replace(body, "")


def test_markdown_goes_through_sanitizer_wrapper():
    assert re.search(r"sanitizeMarkdownHtml\(\s*marked\.parse\(", UTILS)
    assert not re.search(r"marked\.parse\([^)]*\)\s*;?\s*\n?\s*return", UTILS)


def test_artifact_preview_rebuilds_markdown_like_chat():
    assert "function renderArtifactHtml" in UTILS
    assert "renderMarkdown(_artifactMarkdown(doc))" in UTILS
    assert 'block.type === "list"' in UTILS and 'block.type === "code"' in UTILS
    assert 'format === "txt"' in UTILS and 'format === "xlsx"' in UTILS


def test_buildPreviewDoc_is_exported():
    """buildPreviewDoc must appear in the IIFE's return block."""
    # The public return block is the last 'return {' in the file before the closing '})();'
    # Simpler: the name followed by comma appears in the return list.
    assert "buildPreviewDoc," in UTILS, "buildPreviewDoc not found in utils.js exports"


def test_buildPreviewDoc_handles_all_three_branches():
    """Smoke-check that the three code paths are present in buildPreviewDoc."""
    assert "function buildPreviewDoc" in UTILS, "buildPreviewDoc not found in utils.js"
    # Use a window from the function declaration — generous enough to cover the whole body
    start = UTILS.index("function buildPreviewDoc")
    body = UTILS[start:start + 3000]
    assert "isFull" in body,        "full-document path missing"
    assert "isSvg" in body,         "SVG path missing"
    assert "connect-src" in body,   "CSP meta missing"
    assert "preview.html" in body,  "html filename missing"
    assert "preview.svg" in body,   "svg filename missing"


def test_code_preview_js_exports_required_api():
    """code-preview.js must expose decorate / sync / open / close."""
    assert "decorate" in CODEPREVIEW
    assert "sync" in CODEPREVIEW
    assert "_launch" in CODEPREVIEW or "open:" in CODEPREVIEW
    assert "close" in CODEPREVIEW
    assert "PREVIEWABLE" in CODEPREVIEW   # the allowlist Set
    assert '"html"' in CODEPREVIEW and '"svg"' in CODEPREVIEW


def test_code_preview_js_checks_chatArea():
    """decorate() must bail out when the wrapper is not inside #chatArea."""
    m = re.search(r"function\s+decorate\b(.*?)^\s*\}", CODEPREVIEW, re.S | re.M)
    assert m, "decorate() not found in code-preview.js"
    body = m.group(1)
    assert "chatArea" in body        # it looks up #chatArea
    assert "contains" in body        # it calls .contains(wrapper)


def test_code_workspace_note_updated():
    """The Code workspace note must mention Run (not the old 'never run' copy)."""
    assert "Run" in INDEX
    assert "never run" not in INDEX


def test_purify_config_keeps_http_only_and_forbids_active_tags():
    assert "ALLOWED_URI_REGEXP: /^https?:/i" in UTILS
    assert "ALLOW_DATA_ATTR: false" in UTILS
    for tag in ("style", "iframe", "form", "svg", "math", "object", "embed", "link", "base"):
        assert f'"{tag}"' in UTILS


# ---------------------------------------------------------------- browser test
def _chrome():
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    return None


HARNESS = """<!doctype html><html><body><pre id="out"></pre><svg style="display:none"></svg>
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>%(icons)s</script>
<script>%(utils)s</script>
<script>%(code_preview)s</script>
<script>
const R = {libs: [typeof marked, typeof DOMPurify, typeof hljs]};
if (R.libs.includes("undefined")) { document.getElementById("out").textContent = JSON.stringify(R); }
else {
  const r = (s) => Utils.renderMarkdown(s);
  R.a = r('<a href="https://a.test" download="x" ping="//e" class="citation" role="button">l</a>');
  R.div = r('<div class="citation" role="button">d</div><span class="citation" role="tab">s</span>');
  R.sup = r('<sup class="citation" data-id="1"><sup class="citation" data-id="2">z</sup></sup>');
  R.other = r('<p class="x" role="alert">p</p><h1 class="a">h</h1><li class="z" role="b">q</li>');
  R.nonCode = r('<pre class="language-x">p</pre><span class="language-x">s</span><div class="language-js">d</div>');
  R.evil = r('<code class="language-x citation">a</code><code class="citation language-x">b</code>' +
             '<code class="LANGUAGE-x">c</code><code class="language-x&quot; onclick=&quot;1">d</code>' +
             '<code class="language-x\\nfoo">e</code><code class="language-">f</code>');
  R.codeRole = r('<code class="language-x" role="button" id="z" style="x">r</code>');
  R.fence = r('```python\\nprint(1)\\n```');
  R.cite = r('See [1] and `[2]`.');
  R.link = r('[ok](https://ok.test/a)');
  R.after = DOMPurify.sanitize('<code class="language-x">a</code>', {FORBID_ATTR: ["class"]});
  const d = document.createElement("div"); d.innerHTML = R.fence; document.body.appendChild(d);
  Utils.highlightCode(d);
  R.hljs = d.querySelectorAll('[class^="hljs-"]').length;
  R.copy = d.querySelectorAll(".code-block__copy").length;

  // ── buildPreviewDoc ───────────────────────────────────────
  // fragment wrap
  const pHtml = Utils.buildPreviewDoc('<p>Hello</p>', 'html');
  R.prevHtmlTitle    = pHtml.title;
  R.prevHtmlFilename = pHtml.filename;
  R.prevHtmlHasCsp   = pHtml.html.includes('Content-Security-Policy');
  R.prevHtmlHasBody  = pHtml.html.includes('<p>Hello</p>');
  R.prevHtmlIsDoc    = pHtml.html.startsWith('<!DOCTYPE html>') || pHtml.html.startsWith('<!doctype html>');
  // SVG wrap
  const pSvg = Utils.buildPreviewDoc('<svg xmlns="http://www.w3.org/2000/svg"><circle r="5"/></svg>', 'svg');
  R.prevSvgFilename  = pSvg.filename;
  R.prevSvgCentred   = pSvg.html.includes('min-height:100vh');
  // full document — title extraction + CSP injection
  const pFull = Utils.buildPreviewDoc(
    '<!DOCTYPE html><html><head><title>My App</title></head><body>Hello</body></html>', 'html');
  R.prevFullTitle    = pFull.title;
  R.prevFullCsp      = pFull.html.includes('Content-Security-Policy');

  // ── Run-button scoping via CodePreview.decorate ───────────
  function makeBlock(lang, parent) {
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block';
    const header = document.createElement('div');
    header.className = 'code-block__header';
    const span = document.createElement('span');
    span.className = 'code-block__lang';
    span.textContent = lang;
    const copyBtn = document.createElement('button');
    copyBtn.className = 'code-block__copy';
    header.appendChild(span); header.appendChild(copyBtn);
    const pre = document.createElement('pre');
    pre.appendChild(document.createElement('code'));
    wrapper.appendChild(header); wrapper.appendChild(pre);
    if (parent) parent.appendChild(wrapper);
    return wrapper;
  }
  // create a simulated #chatArea
  const chatArea = document.createElement('section');
  chatArea.id = 'chatArea';
  document.body.appendChild(chatArea);
  // html in chatArea → Run button expected
  const htmlInChat = makeBlock('html', chatArea);
  CodePreview.decorate(htmlInChat);
  R.runInChat = htmlInChat.querySelectorAll('.code-block__run').length;
  // html OUTSIDE chatArea → Run button must NOT appear
  const htmlOut = makeBlock('html', document.body);
  CodePreview.decorate(htmlOut);
  R.runOutsideChat = htmlOut.querySelectorAll('.code-block__run').length;
  // python in chatArea → Run button must NOT appear (not previewable)
  const pyInChat = makeBlock('python', chatArea);
  CodePreview.decorate(pyInChat);
  R.runPythonInChat = pyInChat.querySelectorAll('.code-block__run').length;
  // idempotent: calling decorate twice must not add a second Run button
  CodePreview.decorate(htmlInChat);
  R.runIdempotent = htmlInChat.querySelectorAll('.code-block__run').length;

  document.getElementById("out").textContent = JSON.stringify(R);
}
</script></body></html>"""


@pytest.fixture(scope="module")
def browser_result(tmp_path_factory):
    chrome = _chrome()
    if not chrome:
        pytest.skip("headless Chrome not available")
    d = tmp_path_factory.mktemp("san")
    page = d / "h.html"
    page.write_text(HARNESS % {
        "icons":        ICONS.replace("</script", "<\\/script"),
        "utils":        UTILS.replace("</script", "<\\/script"),
        "code_preview": CODEPREVIEW.replace("</script", "<\\/script"),
    }, encoding="utf-8")
    try:
        p = subprocess.run(
            [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", f"--user-data-dir={d / 'prof'}",
             "--virtual-time-budget=15000", "--dump-dom", page.as_uri()],
            capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError) as e:
        pytest.skip(f"chrome could not run: {e}")
    m = re.search(r'<pre id="out">(.*?)</pre>', p.stdout, re.S)
    if not m or not m.group(1).strip():
        pytest.skip("harness produced no output")
    res = json.loads(html.unescape(m.group(1)))
    if "undefined" in res["libs"]:
        pytest.skip(f"CDN libraries unavailable: {res['libs']}")
    return res


def test_browser_attributes_and_classes_stripped(browser_result):
    r = browser_result
    assert r["a"].strip() == '<p><a href="https://a.test/" target="_blank" rel="noopener noreferrer">l</a></p>'
    assert r["div"] == "<div>d</div><span>s</span>"
    assert "citation" not in r["sup"] and "role" not in r["sup"] and "data-id" not in r["sup"]
    assert r["other"] == "<p>p</p><h1>h</h1><li>q</li>"
    assert "class" not in r["nonCode"] and "language" not in r["nonCode"]


def test_browser_language_hook_cannot_be_abused(browser_result):
    r = browser_result
    assert "citation" not in r["evil"] and "onclick" not in r["evil"]
    assert "class=" not in r["evil"], r["evil"]
    assert r["codeRole"].strip() == '<p><code class="language-x">r</code></p>'
    assert r["after"] == "<code>a</code>"  # hook is not left registered


def test_browser_features_still_work(browser_result):
    r = browser_result
    assert 'class="language-python"' in r["fence"]
    assert 'class="citation"' in r["cite"] and 'role="button"' in r["cite"]
    assert "<code>[2]</code>" in r["cite"]
    assert 'rel="noopener noreferrer"' in r["link"] and 'target="_blank"' in r["link"]
    assert r["hljs"] > 0 and r["copy"] == 1


def test_browser_buildPreviewDoc_html_fragment(browser_result):
    r = browser_result
    assert r["prevHtmlTitle"] == "Preview"        # no <title> in fragment → default
    assert r["prevHtmlFilename"] == "preview.html"
    assert r["prevHtmlHasCsp"] is True            # CSP meta injected
    assert r["prevHtmlHasBody"] is True           # source preserved
    assert r["prevHtmlIsDoc"] is True             # wrapped in full document


def test_browser_buildPreviewDoc_svg(browser_result):
    r = browser_result
    assert r["prevSvgFilename"] == "preview.svg"
    assert r["prevSvgCentred"] is True            # centring CSS present


def test_browser_buildPreviewDoc_full_document(browser_result):
    r = browser_result
    assert r["prevFullTitle"] == "My App"         # <title> extracted correctly
    assert r["prevFullCsp"] is True               # CSP injected into existing <head>


def test_browser_run_button_scoping(browser_result):
    r = browser_result
    assert r["runInChat"] == 1,      "html fence in #chatArea must have exactly one Run button"
    assert r["runOutsideChat"] == 0, "html fence outside #chatArea must NOT have a Run button"
    assert r["runPythonInChat"] == 0, "python fence must NOT have a Run button even in #chatArea"
    assert r["runIdempotent"] == 1,  "calling decorate twice must not add a second Run button"
