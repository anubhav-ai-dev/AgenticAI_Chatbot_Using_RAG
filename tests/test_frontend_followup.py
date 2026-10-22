"""Stage 4 follow-up regression tests (M1, M2, M3, N1, N2, N3, N4).

Static guards + STUB-ONLY headless-Chrome scenarios: every request is intercepted by an in-page fetch stub and
the static server answers 405 to any non-GET. No backend is started; nothing here proves real-backend streaming.
Browser tests skip cleanly when Chrome or the CDN libraries (marked/DOMPurify) are unavailable.
"""
import html as _html
import http.server as _hs
import json as _json
import re
import shutil as _sh
import subprocess as _sp
import threading as _th

import pytest

from tests.test_frontend_guards import _SEED, _STUB, API_JS, FE, INDEX, _app_fn, _code


# ── static guards ────────────────────────────────────────────────────────────────────────────

def test_m3_card_aria_label_starts_with_the_visible_title():
    cards = re.findall(r'<button class="welcome__starter[^"]*"[^>]*?aria-label="([^"]+)">(.*?)</button>', INDEX, re.S)
    assert len(cards) == 4
    for label, inner in cards:
        title = re.search(r'<span class="feature-card__title">([^<]+)</span>', inner).group(1).strip()
        assert label.startswith(title), (title, label)           # WCAG 2.5.3 Label in Name
        assert label.startswith(title + ". Switches to "), label


def test_n3_show_error_does_not_announce_and_retry_is_announced_once():
    chat = (FE / "js/chat.js").read_text(encoding="utf-8")
    m = re.search(r"\n      showError\(msg, opts = \{\}\) \{(.*?)\n      \},\n", chat, re.S)
    assert m, "showError not found"
    body = _code(m.group(1))
    assert "role" in body and '"alert"' in body                  # the alert box is the announcement channel
    announces = re.findall(r"_announce\(([^)]*)\)", body)
    assert announces == ['"Retry is available."'], announces


def test_n4_sse_dispatch_only_calls_own_function_handlers():
    code = _code(API_JS)
    assert "Object.prototype.hasOwnProperty.call(handlers, event)" in code
    assert not re.search(r"\bhandlers\[event\]\s*\)?\s*\(?\s*handlers|if \(handlers\[event\]\)", code)
    assert "\\r\\n" not in code                                  # LF-only framing is the documented contract


def test_m1_resend_is_decided_in_the_controller_and_only_after_accept_sets_failed():
    body = _app_fn("_onSend")
    assert "prior.sessionId === State.sessionId" in body and "draftText === prior.text" in body
    assert "const text = resend ? \"\" : draftText;" in body
    # State.failed is only ever assigned in the two post-accept places (never before the server accepted)
    assigns = [m.start() for m in re.finditer(r"State\.failed = \{", body)]
    assert len(assigns) == 2
    assert all(a > body.index("const onAccepted") for a in assigns)
    # the cleared states
    for fn in ("_newChat", "_selectSession"):
        assert "State.failed = null" in _app_fn(fn), fn


def test_m2_create_session_receives_the_abort_signal():
    assert "await _ensureSession(signal)" in _app_fn("_onSend")
    assert "}, signal);" in _app_fn("_ensureSession")
    assert "if (signal.aborted) throw" in _app_fn("_onSend")


def test_n1_busy_notice_is_throttled_and_used_by_send_and_session_select():
    assert "if (State.isLoading) { _notifyBusy(); return; }" in _app_fn("_onSend")
    assert "if (State.isLoading) { _notifyBusy(); return; }" in _app_fn("_selectSession")
    nb = _app_fn("_notifyBusy")
    assert "4000" in nb and "A response is in progress. Press Stop to cancel." in nb


# ── stub-only browser scenarios ──────────────────────────────────────────────────────────────

_SCENARIO = r"""
(function(){
var R={},$=function(i){return document.getElementById(i)},tick=function(ms){return new Promise(function(r){setTimeout(r,ms)})};
async function until(f,ms){var t=0;while(!f()&&t<(ms||3000)){await tick(20);t+=20}return !!f()}
var users=function(){return document.querySelectorAll('.message--user').length};
var asst=function(){return document.querySelectorAll('.message--assistant').length};
var chatCalls=function(){return window.__calls.filter(function(c){return c.path==='/chat/stream'})};
var sessPosts=function(){return window.__calls.filter(function(c){return c.method==='POST'&&c.path==='/sessions'}).length};
window.__toasts=[];var _st=UI.showToast;UI.showToast=function(m,t){window.__toasts.push(m);return _st.apply(UI,arguments)};
window.__live=[];var _sl=UI.setLiveStatus;UI.setLiveStatus=function(t){window.__live.push(t);return _sl.apply(UI,arguments)};
var sendShown=function(){return $('sendBtn').style.display!=='none'};
async function fresh(){window.__handlers={};window.__calls.length=0;$('chatInput').value='';$('newChatBtn').click();await tick(150);window.__calls.length=0;window.__toasts.length=0}
async function send(t){$('chatInput').value=t;$('sendBtn').click();await tick(100)}
function key(el,k,o){el.dispatchEvent(new KeyboardEvent('keydown',Object.assign({key:k,bubbles:true,cancelable:true},o||{})))}
function errEv(extra){return __sse('meta',{})+__sse('error',Object.assign({code:'overloaded',message:'model busy',retryable:true},extra||{}))}
function errEvNoMeta(extra){return __sse('error',Object.assign({code:'overloaded',message:'model busy',retryable:true},extra||{}))}
async function nextStream(n0){await until(function(){return window.__streams.length>n0});return window.__streams[n0]}
async function failFirst(text,extra){var n=window.__streams.length;await send(text);var s=await nextStream(n);s.push(errEv(extra));s.close();
 await until(function(){return document.querySelector('.message-error')&&sendShown()});await tick(30);return n}
async function finish(n,text){window.__streams[n].push(__sse('token',{text:text||'ok'})+__sse('done',{message_id:'m'+n,persisted:true}));window.__streams[n].close();await until(sendShown);await tick(30)}
function nullSessionVia404(){window.__handlers['POST /chat/stream']=function(){return {status:404,json:{detail:'Session not found.'}}};}
async function dropSession(){nullSessionVia404();await send('x-drop');delete window.__handlers['POST /chat/stream'];$('chatInput').value='';window.__calls.length=0;window.__toasts.length=0}
async function run(){
 if(!window.marked||!window.DOMPurify){R.nolibs=true;return}
 await tick(300);
 // ── M2: Stop while POST /sessions is pending (no session exists yet) ──
 window.__handlers={};window.__calls.length=0;
 window.__handlers['POST /sessions']=function(){return {hang:true}};
 $('chatInput').value='D-m2';$('sendBtn').click();await tick(100);
 var busyShown=!sendShown();
 $('stopBtn').click();await tick(150);
 R.m2={busyShown:busyShown,sendBack:sendShown(),value:$('chatInput').value,chat:chatCalls().length,users:users(),asst:asst(),
   toast:window.__toasts.slice(),focus:document.activeElement.id,posts:sessPosts()};
 // M2 race: creation completes even though Stop was pressed (fetch ignores abort) -> session is kept, no chat request
 var gate,release=new Promise(function(r){gate=r});var first=true;
 window.__handlers['POST /sessions']=async function(r){if(first){first=false;await release}return undefined};
 window.__calls.length=0;window.__toasts.length=0;
 $('chatInput').value='D-race';$('sendBtn').click();await tick(100);
 $('stopBtn').click();await tick(50);gate();await tick(200);
 R.m2race={value:$('chatInput').value,chat:chatCalls().length,posts:sessPosts(),sendBack:sendShown(),users:users(),toast:window.__toasts.slice()};
 window.__calls.length=0;$('sendBtn').click();await until(function(){return window.__streams.length>0});
 R.m2race.nextSendPostedSession=sessPosts();R.m2race.nextChat=chatCalls().length;
 await finish(window.__streams.length-1);
 // ── New chat while POST /sessions is pending ──
 await dropSession();
 var g2,rel2=new Promise(function(r){g2=r});var f2=true;
 window.__handlers['POST /sessions']=async function(r){if(f2){f2=false;await rel2}return undefined};
 window.__calls.length=0;
 $('chatInput').value='D-newchat';$('sendBtn').click();await tick(100);
 $('newChatBtn').click();await tick(80);g2();await tick(250);
 R.newChatRace={chat:chatCalls().length,sendBack:sendShown(),errs:window.__errs.slice(),posts:sessPosts(),value:$('chatInput').value,users:users()};
 window.__handlers={};window.__calls.length=0;window.__toasts.length=0;$('chatInput').value='';
 // ── M1 ──
 await fresh();
 var n=await failFirst('Q-m1');
 R.m1err={value:$('chatInput').value,users:users(),errBox:!!document.querySelector('.message-error')};
 window.__calls.length=0;$('sendBtn').click();var s1=await nextStream(n+1);
 var pl=chatCalls()[0].json;R.m1resend={keys:Object.keys(pl).sort(),regenerate:pl.regenerate,message:pl.message,session_id_set:!!pl.session_id};
 R.m1mid={users:users(),errBox:!!document.querySelector('.message-error'),value:$('chatInput').value};
 await finish(n+1);
 R.m1ok={users:users(),asst:asst(),errBox:!!document.querySelector('.message-error'),value:$('chatInput').value,userText:document.querySelector('.message--user').textContent};
 // failed state cleared after success: same text again is a NORMAL send
 window.__calls.length=0;await send('Q-m1');var plN=chatCalls()[0].json;R.m1afterOk={regenerate:plN.regenerate,message:plN.message};
 await finish(n+2);
 // edited draft sends normally
 await fresh();n=await failFirst('Q-edit');
 window.__calls.length=0;$('chatInput').value='Q-edit changed';$('sendBtn').click();await nextStream(n+1);
 var pe=chatCalls()[0].json;R.m1edit={regenerate:pe.regenerate,message:pe.message};
 await finish(n+1);R.m1editAfter={users:users(),asst:asst(),texts:[].map.call(document.querySelectorAll('.message--user'),function(u){return u.textContent})};
 // unedited draft in ANOTHER session must not regenerate (failed in A, switch to history session sH)
 await fresh();n=await failFirst('Q-stale');
 document.querySelector('.session-item[data-session-id="sH"] .session-item__main').click();await until(function(){return /OLD ANSWER/.test($('chatArea').textContent)});
 window.__calls.length=0;$('chatInput').value='Q-stale';$('sendBtn').click();await nextStream(window.__streams.length);
 var ps=chatCalls()[0].json;R.m1otherSession={regenerate:ps.regenerate,message:ps.message,session:ps.session_id};
 await finish(window.__streams.length-1);
 // New chat clears failed: same text in a brand-new chat is a normal send
 await fresh();n=await failFirst('Q-nc');$('chatInput').value='Q-nc';$('newChatBtn').click();await tick(150);$('chatInput').value='Q-nc';
 window.__calls.length=0;$('sendBtn').click();await nextStream(window.__streams.length);var pn=chatCalls()[0].json;R.m1newChat={regenerate:pn.regenerate,message:pn.message};
 await finish(window.__streams.length-1);
 // HTTP-level failure before accept never sets resend
 await fresh();window.__handlers['POST /chat/stream']=function(){return {status:503,json:{detail:'no provider'}}};
 await send('Q-503');delete window.__handlers['POST /chat/stream'];window.__calls.length=0;$('sendBtn').click();await nextStream(window.__streams.length);
 var p5=chatCalls()[0].json;R.m1http={regenerate:p5.regenerate,message:p5.message};await finish(window.__streams.length-1);
 // cooldown + unedited resend: no request until it expires, then regenerate shape
 await fresh();n=await failFirst('Q-cd',{retry_after:2});window.__calls.length=0;$('sendBtn').click();await tick(100);
 R.m1cooldown={early:window.__calls.length,value:$('chatInput').value,toast:window.__toasts.slice(-1)[0]};
 await tick(2300);window.__calls.length=0;$('sendBtn').click();await nextStream(n+1);var pc=chatCalls()[0].json;R.m1cooldown.after={regenerate:pc.regenerate,message:pc.message};
 await finish(n+1);
 // Retry button path still regenerates, and a stale Retry cannot exist after a resend succeeded
 await fresh();n=await failFirst('Q-retry');var retryBtn=document.querySelector('.message-error .btn--secondary');
 window.__calls.length=0;retryBtn.click();await nextStream(n+1);var pr=chatCalls()[0].json;R.m1retry={regenerate:pr.regenerate,message:pr.message,value:$('chatInput').value};
 await finish(n+1);R.m1retry.users=users();
 // ── N1: busy toast ──
 await fresh();n=window.__streams.length;await send('Q-n1');await nextStream(n);window.__toasts.length=0;window.__calls.length=0;
 for(var i=0;i<4;i++){$('chatInput').value='more '+i;key($('chatInput'),'Enter')}
 $('sendBtn').click();
 document.querySelector('.session-item[data-session-id="sH"] .session-item__main').click();await tick(120);
 R.n1={toasts:window.__toasts.filter(function(t){return /response is in progress/i.test(t)}),requests:window.__calls.map(function(c){return c.method+' '+c.path}),users:users(),streams:window.__streams.length-n,
   composer:$('chatInput').value};
 await tick(4200);key($('chatInput'),'Enter');R.n1.afterWindow=window.__toasts.filter(function(t){return /response is in progress/i.test(t)}).length;
 await finish(n);
 // ── N2: Stop before the first token ──
 await fresh();n=window.__streams.length;await send('Q-n2');var s2a=await nextStream(n);s2a.push(__sse('meta',{}));await tick(50);$('stopBtn').click();await until(sendShown);await tick(80);
 R.n2={value:$('chatInput').value,asst:asst(),users:users(),toast:window.__toasts.slice(),focus:document.activeElement.id,errBox:!!document.querySelector('.message-error')};
 window.__calls.length=0;$('sendBtn').click();await nextStream(n+1);var p2=chatCalls()[0].json;R.n2.resend={regenerate:p2.regenerate,message:p2.message};
 await finish(n+1);R.n2.after={users:users(),asst:asst(),value:$('chatInput').value};
 // N2 with new text typed in the composer: it is not overwritten
 await fresh();n=window.__streams.length;await send('Q-n2b');var s2b=await nextStream(n);s2b.push(__sse('meta',{}));await tick(50);$('chatInput').value='typed meanwhile';$('stopBtn').click();await until(sendShown);await tick(80);
 R.n2typed={value:$('chatInput').value,asst:asst(),toast:window.__toasts.slice(-1)[0]};
 // Stop AFTER the first token keeps the partial answer and clears the draft; same text then sends normally
 await fresh();n=window.__streams.length;await send('Q-n2c');var sc=await nextStream(n);sc.push(__sse('token',{text:'partial words'}));await until(function(){return /partial/.test($('chatArea').textContent)});
 $('stopBtn').click();await until(sendShown);await tick(80);
 R.n2partial={value:$('chatInput').value,asst:asst(),users:users(),hasText:/partial words/.test($('chatArea').textContent)};
 window.__calls.length=0;await send('Q-n2c');await nextStream(n+1);var pp=chatCalls()[0].json;R.n2partial.next={regenerate:pp.regenerate,message:pp.message};await finish(n+1);
 // Stop racing a stream error: consistent end state, no exceptions
 await fresh();n=window.__streams.length;await send('Q-race');var sr=await nextStream(n);sr.push(errEv());sr.close();$('stopBtn').click();await tick(300);
 R.stopVsError={sendBack:sendShown(),users:users(),asst:asst(),value:$('chatInput').value,errs:window.__errs.slice()};
 // ── N3: error announced once ──
 await fresh();window.__live.length=0;n=window.__streams.length;await send('Q-n3');var s3=await nextStream(n);
 var liveBefore=$('liveStatus').textContent;window.__live.length=0;
 s3.push(errEv({message:'distinctive failure text'}));s3.close();await until(function(){return document.querySelector('.message-error')});await tick(400);
 R.n3={liveBefore:liveBefore,liveAfter:$('liveStatus').textContent,calls:window.__live.slice(),alerts:document.querySelectorAll('.message-error[role="alert"]').length,
   text:document.querySelector('.message-error__text').textContent};
 await fresh();window.__live.length=0;n=window.__streams.length;await send('Q-n3b');s3=await nextStream(n);window.__live.length=0;
 s3.push(errEv({retry_after:1}));s3.close();await until(function(){return document.querySelector('.message-error')});await tick(2500);
 R.n3cd={calls:window.__live.slice(),btnEnabled:!document.querySelector('.message-error .btn--secondary').disabled};
 // ── N4: SSE dispatch ignores inherited / special event names ──
 var hit={proto:0,ctor:0,inh:0,own:0,tostr:0};
 var Hp=Object.create({inherited:function(){hit.inh++},constructor:function(){hit.ctor++},toString:function(){hit.tostr++}});Hp.token=function(){hit.own++};
 await fresh();n=window.__streams.length;
 var prom=API.chatStream({session_id:'x',message:'',user_id:'u',mode:'general',model:null,web_search:true,regenerate:false},Hp,new AbortController().signal);
 var s4=await nextStream(n);
 s4.push('event: __proto__\ndata: {"a":1}\n\nevent: constructor\ndata: {"a":1}\n\nevent: toString\ndata: {"a":1}\n\nevent: inherited\ndata: {"a":1}\n\nevent: token\ndata: {"text":"t"}\n\n');
 s4.close();var thrown=null;try{await prom}catch(e){thrown=String(e)}
 R.n4={hit:hit,thrown:thrown,errs:window.__errs.slice()};
 // ══ Minor-1/2/3 + Nit-1 (persisted semantics) ══
 // zero-event stream close -> not persisted: both bubbles dropped, draft restored, unedited Send is a NORMAL send
 await fresh();n=window.__streams.length;await send('Z-zero');var sz=await nextStream(n);sz.close();await until(sendShown);await tick(80);
 R.zero={value:$('chatInput').value,users:users(),asst:asst(),errBox:!!document.querySelector('.message-error'),retryBtn:!!document.querySelector('.message-error .btn--secondary'),
   toast:window.__toasts.slice(),focus:document.activeElement.id,sendBack:sendShown()};
 window.__calls.length=0;$('sendBtn').click();await nextStream(n+1);var pz=chatCalls()[0].json;R.zero.resend={keys:Object.keys(pz).sort(),regenerate:pz.regenerate,message:pz.message};
 R.zero.mid={users:users(),value:$('chatInput').value};await finish(n+1);R.zero.after={users:users(),asst:asst(),value:$('chatInput').value};
 // error WITHOUT meta -> same
 await fresh();n=window.__streams.length;await send('Z-nometa');var se=await nextStream(n);se.push(errEvNoMeta());se.close();await until(sendShown);await tick(80);
 R.nometa={value:$('chatInput').value,users:users(),asst:asst(),errBox:!!document.querySelector('.message-error'),retryBtn:!!document.querySelector('.message-error .btn--secondary'),
   toast:window.__toasts.slice(),focus:document.activeElement.id};
 window.__calls.length=0;$('sendBtn').click();await nextStream(n+1);var pm=chatCalls()[0].json;R.nometa.resend={regenerate:pm.regenerate,message:pm.message};
 R.nometa.mid={users:users()};await finish(n+1);R.nometa.after={users:users(),asst:asst(),value:$('chatInput').value};
 // error WITHOUT meta carrying retry_after: cooldown is honoured, draft kept, still a normal send afterwards
 await fresh();n=window.__streams.length;await send('Z-nometa-cd');se=await nextStream(n);se.push(errEvNoMeta({retry_after:1}));se.close();await until(sendShown);await tick(80);
 window.__calls.length=0;$('sendBtn').click();await tick(80);R.nometaCd={early:window.__calls.length,value:$('chatInput').value};
 await tick(1200);window.__calls.length=0;$('sendBtn').click();await nextStream(n+1);pm=chatCalls()[0].json;R.nometaCd.after={regenerate:pm.regenerate,message:pm.message};await finish(n+1);
 // Stop before ANY event -> draft restored, no bubbles, normal resend
 await fresh();n=window.__streams.length;await send('Z-stop0');await nextStream(n);$('stopBtn').click();await until(sendShown);await tick(80);
 R.stop0={value:$('chatInput').value,users:users(),asst:asst(),errBox:!!document.querySelector('.message-error'),toast:window.__toasts.slice(),focus:document.activeElement.id};
 window.__calls.length=0;$('sendBtn').click();await nextStream(n+1);pm=chatCalls()[0].json;R.stop0.resend={regenerate:pm.regenerate,message:pm.message};R.stop0.mid={users:users()};await finish(n+1);R.stop0.after={users:users(),asst:asst()};
 // Stop before any event with new text typed meanwhile: merged
 await fresh();n=window.__streams.length;await send('Z-stop0b');await nextStream(n);$('chatInput').value='typed2';$('stopBtn').click();await until(sendShown);await tick(80);
 R.stop0typed={value:$('chatInput').value};
 // persisted: meta then error -> user bubble kept, error bubble + Retry, resend regenerates
 await fresh();n=window.__streams.length;await send('P-meta-err');var sp1=await nextStream(n);sp1.push(errEv());sp1.close();await until(function(){return document.querySelector('.message-error')&&sendShown()});await tick(50);
 R.pErr={value:$('chatInput').value,users:users(),errBox:!!document.querySelector('.message-error'),retryBtn:!!document.querySelector('.message-error .btn--secondary')};
 window.__calls.length=0;$('sendBtn').click();await nextStream(n+1);pm=chatCalls()[0].json;R.pErr.resend={keys:Object.keys(pm).sort(),regenerate:pm.regenerate,message:pm.message};R.pErr.mid={users:users()};await finish(n+1);R.pErr.after={users:users(),asst:asst()};
 // persisted: status event only (no meta) is also treated as persisted (any non-error event)
 // persisted: Stop after meta, before token -> user bubble kept, asst bubble gone, draft restored, resend = regenerate
 await fresh();n=window.__streams.length;await send('P-stop-meta');var sp2=await nextStream(n);sp2.push(__sse('meta',{}));await tick(60);$('stopBtn').click();await until(sendShown);await tick(80);
 R.pStop={value:$('chatInput').value,users:users(),asst:asst(),errBox:!!document.querySelector('.message-error')};
 window.__calls.length=0;$('sendBtn').click();await nextStream(n+1);pm=chatCalls()[0].json;R.pStop.resend={regenerate:pm.regenerate,message:pm.message};R.pStop.mid={users:users()};await finish(n+1);R.pStop.after={users:users(),asst:asst()};
 // regenerate + Stop before first token -> old answer restored in place, working Regenerate button
 await fresh();document.querySelector('.session-item[data-session-id="sH"] .session-item__main').click();await until(function(){return /OLD ANSWER/.test($('chatArea').textContent)});
 window.__calls.length=0;window.__toasts.length=0;n=window.__streams.length;
 document.querySelector('.message__regen').click();var sg=await nextStream(n);await tick(60);
 R.regenStop={midOld:/OLD ANSWER/.test($('chatArea').textContent),midAsst:asst()};
 $('stopBtn').click();await until(sendShown);await tick(80);
 var order=[].map.call(document.querySelectorAll('#chatArea .message'),function(m){return m.classList.contains('message--user')?'U':'A'}).join('');
 R.regenStop.end={asst:asst(),users:users(),old:/OLD ANSWER/.test($('chatArea').textContent),regenBtns:document.querySelectorAll('.message__regen').length,toast:window.__toasts.slice(),order:order,
   errBox:!!document.querySelector('.message-error'),firstPayload:(chatCalls()[0]||{json:{}}).json};
 window.__calls.length=0;var rb2=document.querySelector('.message__regen');if(rb2){rb2.click();var sg2=await nextStream(n+1);var pg=chatCalls()[0].json;R.regenStop.again={regenerate:pg.regenerate,message:pg.message,session:pg.session_id};
   await finish(n+1,'NEW ANSWER');R.regenStop.final={asst:asst(),users:users(),new:/NEW ANSWER/.test($('chatArea').textContent),old:/OLD ANSWER/.test($('chatArea').textContent),regenBtns:document.querySelectorAll('.message__regen').length}}
 // regenerate + Stop AFTER first token keeps the partial (old answer is not restored over it)
 await fresh();document.querySelector('.session-item[data-session-id="sH"] .session-item__main').click();await until(function(){return /OLD ANSWER/.test($('chatArea').textContent)});
 n=window.__streams.length;document.querySelector('.message__regen').click();var sg3=await nextStream(n);sg3.push(__sse('meta',{})+__sse('token',{text:'half-new'}));await until(function(){return /half-new/.test($('chatArea').textContent)});
 $('stopBtn').click();await until(sendShown);await tick(80);
 R.regenPartial={asst:asst(),old:/OLD ANSWER/.test($('chatArea').textContent),part:/half-new/.test($('chatArea').textContent)};
 // regenerate stopped, then New chat BEFORE the abort is processed: the restored node must not leak into the new chat
 await fresh();document.querySelector('.session-item[data-session-id="sH"] .session-item__main').click();await until(function(){return /OLD ANSWER/.test($('chatArea').textContent)});
 n=window.__streams.length;document.querySelector('.message__regen').click();await nextStream(n);await tick(60);
 $('newChatBtn').click();await tick(300);
 R.regenNewChat={old:/OLD ANSWER/.test($('chatArea').textContent),asst:asst(),users:users(),errs:window.__errs.slice()};
 // regenerate + New chat + (later) Stop after the stream started: nothing from the old chat may reappear in the new one
 await fresh();document.querySelector('.session-item[data-session-id="sH"] .session-item__main').click();await until(function(){return /OLD ANSWER/.test($('chatArea').textContent)});
 n=window.__streams.length;window.__toasts.length=0;document.querySelector('.message__regen').click();var sn=await nextStream(n);await tick(60);
 $('newChatBtn').click();await tick(200);
 var afterNew={old:/OLD ANSWER/.test($('chatArea').textContent),asst:asst(),users:users()};
 try{sn.push(__sse('meta',{})+__sse('token',{text:'late token'}))}catch(e){}
 await tick(150);$('stopBtn').click();await until(sendShown,1500);await tick(150);
 R.regenNewStop={afterNew:afterNew,old:/OLD ANSWER/.test($('chatArea').textContent),late:/late token/.test($('chatArea').textContent),asst:asst(),users:users(),sendBack:sendShown(),
   errs:window.__errs.slice(),errBox:!!document.querySelector('.message-error')};
 // regenerate + Delete of the CURRENT session (UI.confirm accepted through the dialog; DELETE is the in-page stub only)
 await fresh();document.querySelector('.session-item[data-session-id="sH"] .session-item__main').click();await until(function(){return /OLD ANSWER/.test($('chatArea').textContent)});
 window.__handlers['DELETE /sessions/sH']=function(){var i=window.__S.sessions.findIndex(function(q){return q.id==='sH'});if(i>=0)window.__S.sessions.splice(i,1);return {status:204}};
 n=window.__streams.length;window.__calls.length=0;window.__toasts.length=0;
 document.querySelector('.message__regen').click();await nextStream(n);await tick(60);
 var delBtn=document.querySelector('.session-item[data-session-id="sH"] .delete-session-btn');delBtn.click();
 await until(function(){return document.querySelector('.modal--confirm')});
 var dlgOpen=!!document.querySelector('.modal--confirm');
 document.querySelector('.modal--confirm .btn--danger').click();await tick(400);
 R.regenDelete={dlgOpen:dlgOpen,old:/OLD ANSWER/.test($('chatArea').textContent),asst:asst(),users:users(),errs:window.__errs.slice(),sendBack:sendShown(),
   deletes:window.__calls.filter(function(c){return c.method==='DELETE'}).map(function(c){return c.path}),toast:window.__toasts.slice(),
   rowGone:!document.querySelector('.session-item[data-session-id="sH"]'),dlgClosed:!document.querySelector('.modal--confirm')};
 window.__handlers={};
 // Nit-1: Retry countdown announces itself exactly once; no second 'You can send again.'
 await fresh();window.__live.length=0;n=window.__streams.length;await send('N1-live');var sl=await nextStream(n);window.__live.length=0;
 sl.push(errEv({retry_after:1}));sl.close();await until(function(){return document.querySelector('.message-error')});await tick(2600);
 R.nit1={calls:window.__live.slice()};
 // Nit-1 counterpart: error without meta + retry_after has NO inline Retry, so the cooldown end IS announced
 await fresh();window.__live.length=0;n=window.__streams.length;await send('N1-live2');sl=await nextStream(n);window.__live.length=0;
 sl.push(errEvNoMeta({retry_after:1}));sl.close();await until(sendShown);await tick(1600);
 R.nit1b={calls:window.__live.slice()};
 // rapid double Send / Enter+click: one request, one user bubble
 await fresh();n=window.__streams.length;window.__calls.length=0;$('chatInput').value='R-double';$('sendBtn').click();$('sendBtn').click();key($('chatInput'),'Enter');await tick(200);
 R.dbl={chat:chatCalls().length,users:users(),streams:window.__streams.length-n};await finish(n);
 // double Stop and Stop vs done race: no exceptions, consistent end state
 await fresh();n=window.__streams.length;await send('R-stop2');var sd=await nextStream(n);sd.push(__sse('meta',{})+__sse('token',{text:'tok'}));await until(function(){return /tok/.test($('chatArea').textContent)});
 $('stopBtn').click();$('stopBtn').click();await tick(300);R.dblStop={sendBack:sendShown(),asst:asst(),users:users(),errs:window.__errs.slice()};
 await fresh();n=window.__streams.length;await send('R-race');sd=await nextStream(n);sd.push(__sse('meta',{})+__sse('token',{text:'tok'})+__sse('done',{message_id:'mr',persisted:true}));sd.close();$('stopBtn').click();await tick(300);
 R.doneRace={sendBack:sendShown(),asst:asst(),users:users(),errs:window.__errs.slice()};
 // payload / sessions-payload shapes unchanged
 window.__calls.length=0;$('newChatBtn').click();await tick(150);
 var sp=window.__calls.filter(function(c){return c.method==='POST'&&c.path==='/sessions'})[0];R.sessionsKeys=sp?Object.keys(sp.json).sort():null;
 R.errs2=window.__errs.slice();

 R.errs=window.__errs.slice();R.calls_total_nonstub=0;
}
run().catch(function(e){R.crash=String(e&&e.stack||e)}).then(function(){var o=document.createElement('pre');o.id='qa-out';o.textContent=JSON.stringify(R);document.body.appendChild(o)});
})();
"""


@pytest.fixture(scope="module")
def fu(tmp_path_factory):
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
    d = tmp_path_factory.mktemp("followup")
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


def test_stub_followup_nothing_reached_a_server(fu):
    assert fu["__blocked"] == [] and fu["errs"] == [], (fu["__blocked"], fu["errs"])


SEVEN = ["mode", "message", "model", "regenerate", "session_id", "user_id", "web_search"]


def test_stub_m1_unedited_resend_uses_regenerate_and_keeps_one_user_bubble(fu):
    assert fu["m1err"] == {"value": "Q-m1", "users": 1, "errBox": True}
    r = fu["m1resend"]
    assert sorted(r["keys"]) == sorted(SEVEN) and r["session_id_set"]            # payload shape unchanged
    assert (r["regenerate"], r["message"]) == (True, "")
    # once the stream was accepted: the failed attempt's error bubble is replaced, the composer is cleared, still 1 user bubble
    assert fu["m1mid"] == {"users": 1, "errBox": False, "value": ""}
    ok = fu["m1ok"]
    assert ok["users"] == 1 and ok["asst"] == 1 and ok["errBox"] is False and ok["value"] == "" and ok["userText"].strip().startswith("Q-m1")


def test_stub_m1_failed_state_is_cleared_after_success_new_chat_session_switch_and_edit(fu):
    assert fu["m1afterOk"] == {"regenerate": False, "message": "Q-m1"}            # cleared after a successful resend
    assert fu["m1edit"] == {"regenerate": False, "message": "Q-edit changed"}      # edited draft: normal send
    assert fu["m1otherSession"]["regenerate"] is False and fu["m1otherSession"]["message"] == "Q-stale"
    assert fu["m1otherSession"]["session"] == "sH"
    assert fu["m1newChat"] == {"regenerate": False, "message": "Q-nc"}


def test_stub_m1_http_level_failure_never_uses_regenerate(fu):
    assert fu["m1http"] == {"regenerate": False, "message": "Q-503"}              # the server never stored that turn


def test_stub_m1_cooldown_blocks_early_resend_then_regenerates(fu):
    c = fu["m1cooldown"]
    assert c["early"] == 0 and c["value"] == "Q-cd" and c["toast"]
    assert c["after"] == {"regenerate": True, "message": ""}


def test_stub_m1_retry_button_regenerates_and_no_duplicate_bubble(fu):
    assert fu["m1retry"] == {"regenerate": True, "message": "", "value": "", "users": 1}


def test_stub_m2_stop_while_session_creation_is_pending_keeps_the_draft(fu):
    r = fu["m2"]
    assert r["busyShown"] and r["sendBack"] and r["value"] == "D-m2"
    assert r["chat"] == 0 and r["users"] == 0 and r["asst"] == 0 and r["posts"] == 1
    assert r["focus"] == "chatInput" and r["toast"], r


def test_stub_m2_creation_finishing_after_stop_keeps_the_session_and_sends_nothing(fu):
    r = fu["m2race"]
    assert r["chat"] == 0 and r["sendBack"] and r["value"] == "D-race" and r["users"] == 0, r
    assert r["nextSendPostedSession"] == 0 and r["nextChat"] == 1                  # the created session is reused, not orphaned


def test_stub_new_chat_while_session_creation_is_pending_sends_nothing(fu):
    r = fu["newChatRace"]
    assert r["chat"] == 0 and r["sendBack"] and r["errs"] == [], r


def test_stub_n1_four_enters_mid_stream_give_one_toast_and_no_extra_requests(fu):
    r = fu["n1"]
    assert len(r["toasts"]) == 1 and r["toasts"][0] == "A response is in progress. Press Stop to cancel."
    assert r["requests"] == [], r                                                  # no POST, no GET /sessions/sH
    assert r["streams"] == 1 and r["users"] == 1
    assert r["afterWindow"] == 2                                                   # throttle window (4 s) elapsed: one more allowed


def test_stub_n2_stop_before_first_token_restores_draft_and_leaves_no_empty_bubble(fu):
    r = fu["n2"]
    assert r["value"] == "Q-n2" and r["asst"] == 0 and r["users"] == 1 and not r["errBox"], r
    assert r["focus"] == "chatInput" and any("Generation stopped" in t for t in r["toast"])
    assert r["resend"] == {"regenerate": True, "message": ""}                      # M1 applies after N2
    assert r["after"] == {"users": 1, "asst": 1, "value": ""}
    assert fu["n2typed"]["value"] == "Q-n2b\n\ntyped meanwhile" and fu["n2typed"]["asst"] == 0   # Minor-3: saved text is MERGED with what was typed; nothing typed is overwritten


def test_stub_n2_stop_after_first_token_keeps_partial_and_state_is_not_failed(fu):
    r = fu["n2partial"]
    assert r["value"] == "" and r["asst"] == 1 and r["users"] == 1 and r["hasText"]
    assert r["next"] == {"regenerate": False, "message": "Q-n2c"}
    e = fu["stopVsError"]
    assert e["sendBack"] and e["errs"] == [] and e["users"] == 1, e


def test_stub_n3_error_is_announced_once_by_the_alert_box_only(fu):
    r = fu["n3"]
    assert r["alerts"] == 1 and r["text"] == "distinctive failure text"
    assert r["liveAfter"] == r["liveBefore"], r                                    # #liveStatus untouched by showError
    assert not [c for c in r["calls"] if "distinctive" in c or c == "Response complete"], r
    cd = fu["n3cd"]
    assert cd["btnEnabled"] and cd["calls"].count("Retry is available.") == 1, cd


def test_stub_n4_sse_dispatch_ignores_inherited_and_special_event_names(fu):
    r = fu["n4"]
    assert r["thrown"] is None and r["errs"] == []
    assert r["hit"] == {"proto": 0, "ctor": 0, "inh": 0, "own": 1, "tostr": 0}, r


# ── Minor-1/2/3 + Nit-1: persisted semantics (stub-only) ─────────────────────────────────────

def test_stub_zero_event_close_is_not_persisted_and_unedited_send_is_a_normal_send(fu):
    z = fu["zero"]
    assert z["value"] == "Z-zero" and z["users"] == 0 and z["asst"] == 0 and not z["errBox"] and not z["retryBtn"], z
    assert z["sendBack"] and z["focus"] == "chatInput" and z["toast"] and "Your message is kept." in z["toast"][-1], z
    assert sorted(z["resend"]["keys"]) == sorted(SEVEN)
    assert (z["resend"]["regenerate"], z["resend"]["message"]) == (False, "Z-zero")
    assert z["mid"] == {"users": 1, "value": ""} and z["after"] == {"users": 1, "asst": 1, "value": ""}      # exactly 1 user bubble


def test_stub_error_without_meta_is_not_persisted_and_unedited_send_is_a_normal_send(fu):
    z = fu["nometa"]
    assert z["value"] == "Z-nometa" and z["users"] == 0 and z["asst"] == 0 and not z["errBox"] and not z["retryBtn"], z
    assert z["focus"] == "chatInput" and z["toast"] and "Your message is kept." in z["toast"][-1], z
    assert z["resend"] == {"regenerate": False, "message": "Z-nometa"}
    assert z["mid"] == {"users": 1} and z["after"] == {"users": 1, "asst": 1, "value": ""}
    c = fu["nometaCd"]
    assert c["early"] == 0 and c["value"] == "Z-nometa-cd" and c["after"] == {"regenerate": False, "message": "Z-nometa-cd"}, c


def test_stub_stop_before_any_event_restores_draft_drops_bubbles_and_resends_normally(fu):
    z = fu["stop0"]
    assert z["value"] == "Z-stop0" and z["users"] == 0 and z["asst"] == 0 and not z["errBox"], z
    assert z["focus"] == "chatInput" and any("Generation stopped" in t for t in z["toast"]), z
    assert z["resend"] == {"regenerate": False, "message": "Z-stop0"}
    assert z["mid"] == {"users": 1} and z["after"] == {"users": 1, "asst": 1}
    assert fu["stop0typed"]["value"] == "Z-stop0b\n\ntyped2"                       # merged, nothing typed is lost


def test_stub_persisted_error_keeps_user_bubble_and_resend_regenerates_with_empty_message(fu):
    p = fu["pErr"]
    assert p["value"] == "P-meta-err" and p["users"] == 1 and p["errBox"] and p["retryBtn"], p
    assert sorted(p["resend"]["keys"]) == sorted(SEVEN)
    assert (p["resend"]["regenerate"], p["resend"]["message"]) == (True, "")
    assert p["mid"] == {"users": 1} and p["after"] == {"users": 1, "asst": 1}
    s = fu["pStop"]                                                                   # Stop after meta, before the first token
    assert s["value"] == "P-stop-meta" and s["users"] == 1 and s["asst"] == 0 and not s["errBox"], s
    assert s["resend"] == {"regenerate": True, "message": ""}
    assert s["mid"] == {"users": 1} and s["after"] == {"users": 1, "asst": 1}


def test_stub_regenerate_stopped_before_first_token_restores_old_answer_in_place(fu):
    r = fu["regenStop"]
    assert r["midOld"] is False and r["midAsst"] == 1                                 # old answer replaced once the stream started
    e = r["end"]
    assert e["asst"] == 1 and e["users"] == 1 and e["old"] and e["regenBtns"] == 1 and e["order"] == "UA" and not e["errBox"], e
    assert any("previous answer is restored" in t for t in e["toast"]), e
    assert (e["firstPayload"]["regenerate"], e["firstPayload"]["message"]) == (True, "")
    assert r["again"]["regenerate"] is True and r["again"]["message"] == "" and r["again"]["session"] == "sH"
    assert r["final"] == {"asst": 1, "users": 1, "new": True, "old": False, "regenBtns": 1}
    p = fu["regenPartial"]                                                            # after the first token the partial is kept
    assert p["asst"] == 1 and p["part"] and not p["old"], p


def test_stub_regenerate_restore_must_not_leak_into_a_different_chat(fu):
    n = fu["regenNewChat"]
    assert not n["old"] and n["asst"] == 0 and n["users"] == 0 and n["errs"] == [], n


def test_stub_regenerate_then_delete_current_session_leaves_an_empty_chat_and_only_a_stubbed_delete(fu):
    d = fu["regenDelete"]
    assert d["dlgOpen"] and d["dlgClosed"], d
    assert d["deletes"] == ["/sessions/sH"], d                                        # the in-page stub saw it; the server saw nothing
    assert not d["old"] and d["asst"] == 0 and d["users"] == 0 and d["rowGone"] and d["errs"] == [] and d["sendBack"], d


def test_stub_regenerate_then_new_chat_then_stop_after_tokens_leaks_nothing(fu):
    n = fu["regenNewStop"]
    assert n["afterNew"] == {"old": False, "asst": 0, "users": 0}, n
    assert not n["old"] and n["asst"] == 0 and n["users"] == 0 and n["sendBack"] and n["errs"] == [] and not n["errBox"], n


def test_stub_nothing_was_ever_sent_to_a_real_server(fu):
    assert fu["__blocked"] == [] or all(b.startswith("GET /") and b.split()[1].endswith(("favicon.ico",)) for b in fu["__blocked"]), fu["__blocked"]


def test_stub_retry_countdown_announces_itself_once_and_no_double_announcement(fu):
    assert fu["nit1"]["calls"] == ["Retry is available."], fu["nit1"]
    assert fu["nit1b"]["calls"] == ["You can send again."], fu["nit1b"]               # no inline Retry => the cooldown end is announced


def test_stub_double_send_double_stop_and_stop_done_race_are_clean(fu):
    assert fu["dbl"] == {"chat": 1, "users": 1, "streams": 1}, fu["dbl"]
    assert fu["dblStop"]["sendBack"] and fu["dblStop"]["errs"] == [] and fu["dblStop"]["asst"] == 1 and fu["dblStop"]["users"] == 1
    assert fu["doneRace"]["sendBack"] and fu["doneRace"]["errs"] == [] and fu["doneRace"]["users"] == 1
    assert fu["errs2"] == []


def test_stub_sessions_post_payload_keeps_its_four_keys(fu):
    assert fu["sessionsKeys"] == ["mode", "model", "title", "user_id"], fu["sessionsKeys"]
