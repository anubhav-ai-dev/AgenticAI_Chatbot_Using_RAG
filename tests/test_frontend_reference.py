"""Static guards for the reference-design pass (reference.css, notifications.js). Stdlib only.

These close gaps in tests/test_frontend_guards.py, whose file lists predate reference.css and
components/notifications.js (hex-colour, no-fetch and 3:1 border checks never looked at them).
"""
import re
from pathlib import Path

import pytest

FE = Path(__file__).resolve().parent.parent / "frontend"


def _read(rel):
    return (FE / rel).read_text(encoding="utf-8")


def _theme_tokens(theme):
    css = _read("css/main.css")
    if theme == "light":
        block = re.search(r':root,\s*:root\[data-theme="light"\]\s*\{(.*?)\n\}', css, re.S).group(1)
    else:
        block = re.search(r':root\[data-theme="dark"\]\s*\{(.*?)\n\}', css, re.S).group(1)
    return dict(re.findall(r"(--[\w-]+)\s*:\s*(#[0-9A-Fa-f]{6})\b", block))


def _lum(hexc):
    r, g, b = (int(hexc.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _contrast(a, b):
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_reference_assets_exist_and_are_linked():
    index = _read("index.html")
    for rel in ("css/reference.css", "js/components/notifications.js"):
        assert (FE / rel).is_file(), rel
        assert rel in index, f"{rel} not referenced by index.html"
    # reference.css must load after components.css (it overrides it)
    assert index.index("/css/components.css") < index.index("/css/reference.css")


def test_reference_css_uses_tokens_only():
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", _read("css/reference.css")), "raw hex in reference.css"


def test_every_component_and_view_js_is_view_only():
    """No network, State or persistence in any view file (VIEW_JS in test_frontend_guards omits notifications.js)."""
    files = list((FE / "js" / "components").glob("*.js")) + [FE / "js" / n for n in ("ui.js", "chat.js", "utils.js", "icons.js", "theme.js")]
    for p in files:
        code = re.sub(r"/\*.*?\*/|//[^\n]*", "", p.read_text(encoding="utf-8"), flags=re.S)
        assert not re.search(r"\bfetch\s*\(|\bAPI\.|XMLHttpRequest|sendBeacon|WebSocket", code), p.name
    notif = re.sub(r"/\*.*?\*/|//[^\n]*", "", _read("js/components/notifications.js"), flags=re.S)
    assert "innerHTML" not in notif and "insertAdjacentHTML" not in notif
    assert "localStorage" not in notif and "sessionStorage" not in notif


def test_reference_css_has_no_unguarded_motion():
    css = _read("css/reference.css")
    assert "@keyframes" not in css, "new keyframes must live in animations.css/ai-viz.css (reduced-motion rules)"
    anim = _read("css/animations.css")
    m = re.search(r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\n\}", anim, re.S)
    assert m and "animation-duration" in m.group(1) and "transition-duration" in m.group(1)
    for decl in re.findall(r"(?<![-\w])transition\s*:\s*([^;]+);", css):
        props = re.findall(r"(?:^|,)\s*([a-z-]+)", decl)
        assert set(props) <= {"transform", "opacity", "background-color", "color", "border-color", "box-shadow"}, decl


def test_notification_hooks_exist_in_markup():
    index = _read("index.html")
    for i in ("notifBtn", "notifDot", "topSearchBtn", "recentToggle", "recentPanel", "sessionFilterBtn", "myDocsBtn", "composerModel"):
        assert f'id="{i}"' in index, i
    assert re.search(r'id="notifBtn"[^>]*aria-label="[^"]+"', index)
    assert re.search(r'id="topAvatar"[^>]*role="img"', index) or re.search(r'role="img"[^>]*id="topAvatar"', index)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_soft_control_outline_meets_3_to_1(theme):
    """Search field, composer, Web Search pill, model chip: --border-ui >= 3:1 on --surface (WCAG 1.4.11)."""
    t = _theme_tokens(theme)
    assert _contrast(t["--border-ui"], t["--surface"]) >= 3.0


def test_welcome_card_borders_are_documented_exemption_and_text_is_readable():
    """m2: tinted card borders stay pastel (cards are text-labelled buttons, the text carries the state).
    The exemption must be written down in view-contract.md, and card TEXT must still be >= 4.5:1."""
    contract = (FE.parent / "docs" / "redesign" / "view-contract.md").read_text(encoding="utf-8")
    assert "Welcome-card border exemption" in contract
    css = _read("css/main.css")
    for theme in ("light", "dark"):
        t = _theme_tokens(theme)
        block = re.search(r':root,\s*:root\[data-theme="light"\]\s*\{(.*?)\n\}' if theme == "light" else r':root\[data-theme="dark"\]\s*\{(.*?)\n\}', css, re.S).group(1)
        for k in ("gen", "doc", "res", "cod"):
            bg = re.search(rf"--card-{k}-bg:\s*linear-gradient\([^#]*(#[0-9A-Fa-f]{{6}})[^#]*(#[0-9A-Fa-f]{{6}})", block)
            assert bg, (theme, k)
            for stop in bg.groups():
                assert _contrast(t[f"--card-{k}-fg"], stop) >= 4.5, (theme, k, stop)
            assert _contrast(t["--text"], bg.group(1)) >= 4.5, (theme, k)


def test_no_unverifiable_privacy_claim_in_capability_strip():
    low = _read("index.html").lower()
    assert "secure &amp; private" not in low and "data stays yours" not in low
    assert "safe uploads" in low and "file type and size are checked" in low
