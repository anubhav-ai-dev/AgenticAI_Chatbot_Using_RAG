"""Static guards for the application shell: the workspace-mode dropdown and its relation to New Chat.

Product requirement under test: the four workspaces (General, Documents, Research, Code) are chosen in ONE
dropdown that sits ABOVE the New Chat button in the sidebar. They must never come back as permanent sidebar
rows or as a segmented toggle in the top header. Stdlib + the project's own config only; no browser.
"""
import re
from pathlib import Path

from backend.core.config import MODES

FE = Path(__file__).resolve().parent.parent / "frontend"


def _read(rel):
    return (FE / rel).read_text(encoding="utf-8")


INDEX = _read("index.html")
UI = _read("js/ui.js")


def _sidebar():
    return re.search(r'<aside class="sidebar".*?</aside>', INDEX, re.S).group(0)


def test_mode_dropdown_precedes_new_chat_inside_the_sidebar():
    sb = _sidebar()
    mode, new = sb.find('id="modeSelect"'), sb.find('id="newChatBtn"')
    assert mode != -1 and new != -1, "both controls must live in the sidebar"
    assert mode < new, "the mode dropdown must come BEFORE (above) New Chat"
    between = sb[mode:new]
    assert "<h2" not in between and "<nav" not in between, "nothing may separate the dropdown from New Chat except its wrapper"


def test_new_chat_follows_the_dropdown_directly_in_the_header_order():
    sb = _sidebar()
    order = [sb.find(k) for k in ('class="sidebar__header"', 'id="modeSelect"', 'id="newChatBtn"',
                                  'id="recentToggle"', 'id="myDocsBtn"', 'class="sidebar__footer"')]
    assert all(i != -1 for i in order), order
    assert order == sorted(order), "sidebar order: brand, mode, new chat, recent, my documents, user"


def test_trigger_is_a_menu_button_with_state_and_a_name():
    btn = re.search(r'<button[^>]*id="modeSelect"[^>]*>', INDEX).group(0)
    for needle in ('aria-haspopup="menu"', 'aria-expanded="false"', 'aria-label="Workspace mode:'):
        assert needle in btn, needle
    assert "mode-select__chevron" in INDEX, "the dropdown needs a visible menu indicator"


def test_dropdown_entries_are_exactly_the_backend_modes_in_order():
    entries = re.findall(r'\{ id: "([a-z]+)",\s+label: "([^"]+)",\s+icon: "[a-z-]+" \}', UI)
    assert [m for m, _ in entries] == ["general", "document", "research", "code"]
    assert [l for _, l in entries] == ["General", "Documents", "Research", "Code"]
    assert {m for m, _ in entries} == set(MODES), "the UI must offer exactly the modes the backend accepts"


def test_modes_are_not_rendered_as_permanent_rows_or_header_tabs():
    css = "".join(_read(f"css/{f}") for f in ("main.css", "components.css", "reference.css", "animations.css"))
    js = "".join(p.read_text(encoding="utf-8") for p in (FE / "js").rglob("*.js"))
    for dead in ("mode-tab", "mode-tabs", "modeTabs", "top-tab", "top-tabs", "topTabs"):
        assert dead not in INDEX, f"{dead} in index.html"
        assert dead not in css, f"{dead} in CSS"
        assert dead not in js, f"{dead} in JS"
    assert not re.search(r'<button[^>]*\bdata-mode="[a-z]+"[^>]*aria-pressed', INDEX), "no mode toggle buttons"
    assert not re.search(r'role="tab(list)?"', _read("index.html").split('<main')[1].split('</header>')[0]), \
        "no tab bar in the header"


def test_header_has_only_a_read_only_workspace_label():
    header = re.search(r"<header class=\"topbar\">.*?</header>", INDEX, re.S).group(0)
    title = re.search(r'<p class="topbar__title"[^>]*>.*?</p>', header, re.S).group(0)
    assert "<button" not in title, "the header label must not be a control"


def test_controller_owns_the_mode_and_the_view_only_reports_it():
    app = _read("js/app.js")
    assert "UI.onModeChange(_setMode)" in app
    ui_fn = UI[UI.index("function toggleModeMenu"):UI.index("function setActiveMode")]
    assert "localStorage" not in ui_fn and "fetch(" not in ui_fn and "API." not in ui_fn
    assert "_modeHandler(id)" in ui_fn, "selection must be reported to the controller"


def test_drawer_focuses_the_mode_dropdown_first():
    open_fn = UI[UI.index("function openSidebar"):UI.index("function closeSidebar")]
    assert '$("modeSelect")' in open_fn


def test_rail_tooltip_for_new_chat_lives_on_the_wrapper_not_the_clipping_button():
    """Regression: .new-chat-btn has overflow:hidden and uses ::after for its sheen, so a tooltip drawn on the button
    itself was clipped to a stray fragment on hover and never showed on keyboard focus in the collapsed rail."""
    sb = _sidebar()
    assert re.search(r'<div class="sidebar__new" data-rail-tip="New chat">', sb)
    button = re.search(r'<button[^>]*id="newChatBtn"[^>]*>', sb).group(0)
    assert "data-rail-tip" not in button
    css = _read("css/components.css")
    assert re.search(r'\.sidebar__new\[data-rail-tip\]:has\(:focus-visible\)::after \{ opacity: 1; visibility: visible; \}', css)
    assert ".sidebar__new[data-rail-tip]::after { top: calc((100% - var(--sp-3)) / 2); }" in css
    # every rail-tip carrier that is visible in the rail must not clip its overflow
    ref = _read("css/reference.css")
    assert re.search(r"\.new-chat-btn \{[^}]*overflow: hidden", ref), "documents why the tip cannot live on the button"


def test_every_control_with_a_rail_tip_also_has_an_accessible_name():
    sb = _sidebar()
    assert 'aria-label="Workspace mode:' in sb                       # dropdown
    assert '<span class="new-chat-btn__label">New Chat</span>' in sb  # label stays in the DOM (visually hidden in the rail)
    assert 'aria-label="Settings"' in sb
    assert "docs-card__title\">My Documents" in sb
