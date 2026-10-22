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
    page.write_text(HARNESS % {"icons": ICONS.replace("</script", "<\\/script"),
                               "utils": UTILS.replace("</script", "<\\/script")}, encoding="utf-8")
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
