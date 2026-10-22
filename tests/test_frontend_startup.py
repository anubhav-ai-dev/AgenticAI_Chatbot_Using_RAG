"""Regression: a startup failure must never leave the model list stuck on "Loading models…".

Observed bug: static files were served without Cache-Control, so a browser could run a NEW app.js next to an OLD
ui.js. The old file lacked UI.onModeChange, `_bindEvents()` threw, and because data loading only started after
binding, /models was never requested: the selector stayed on "Loading models…" with no message.
"""
import re

import pytest

from tests.browser_harness import run_scenario
from tests.test_frontend_guards import FE

APP = (FE / "js" / "app.js").read_text(encoding="utf-8")


def _fn(name):
    start = APP.index(f"async function {name}(") if f"async function {name}(" in APP else APP.index(f"function {name}(")
    return APP[start:start + 1800]


def test_data_loading_starts_before_event_handlers_are_bound():
    init = _fn("init")
    assert init.index("Promise.all([_loadModels()") < init.index("_bindEvents()"), "loaders must start first"
    assert init.index("_bindEvents()") < init.index("await loading"), "and be awaited after binding"
    for loader in ("_loadModels()", "_loadSessions({ notify: true })", "_loadDocuments()", "_loadConfig()"):
        assert loader in init


def test_startup_errors_are_reported_with_the_fix_not_swallowed():
    assert "const boot = () => { init().catch(_startupFailed); };" in APP
    failed = _fn("_startupFailed")
    assert "console.error" in failed and "Ctrl+Shift+R" in failed and 'UI.showToast(msg, "error")' in failed


_SCENARIO = r"""
(function(){
var R={},$=function(i){return document.getElementById(i)},tick=function(ms){return new Promise(function(r){setTimeout(r,ms)})};
// Simulate a stale cached ui.js that predates UI.onModeChange. This script runs before DOMContentLoaded, i.e. before app.js init().
delete window.UI.onModeChange;
async function run(){
 if(!window.marked||!window.DOMPurify){R.nolibs=true;return}
 await tick(800);
 R.models={options:[].slice.call(document.querySelectorAll('#modelSelect option')).map(function(o){return o.textContent}),
   chipDisabled:$('composerModel').disabled,chip:$('composerModel').textContent.trim()};
 R.requested=window.__calls.map(function(c){return c.method+' '+c.path});
 R.toasts=[].slice.call(document.querySelectorAll('.toast')).map(function(t){return t.textContent});
 R.sessionsRendered=document.querySelectorAll('.session-item,.sidebar__empty,.list-state').length>0;
 R.errs=window.__errs.slice();
}
run().catch(function(e){R.crash=String(e&&e.stack||e)}).then(function(){var o=document.createElement('pre');o.id='qa-out';o.textContent=JSON.stringify(R);document.body.appendChild(o)});
})();
"""


@pytest.fixture(scope="module")
def stale(tmp_path_factory):
    return run_scenario(_SCENARIO, tmp_path_factory, name="startup")


def test_stub_models_still_load_when_a_handler_fails_to_bind(stale):
    assert stale["__blocked"] == []
    assert "GET /models" in stale["requested"], "the model request must not depend on event binding"
    assert stale["models"]["options"] == ["M1"], stale["models"]
    assert stale["models"]["chipDisabled"] is False and stale["models"]["chip"] != "Loading models…"
    assert stale["sessionsRendered"], "the chat list must also be populated"


def test_stub_the_user_is_told_what_happened_and_how_to_fix_it(stale):
    joined = " ".join(stale["toasts"])
    assert "failed to start" in joined and "Ctrl+Shift+R" in joined and "onModeChange" in joined, stale["toasts"]
    assert not re.search(r"Unhandled|rej:", " ".join(stale["errs"])), "the failure is handled, not an unhandled rejection"
