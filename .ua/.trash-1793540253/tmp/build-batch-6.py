# -*- coding: utf-8 -*-
"""Build knowledge-graph fragments for file-analyzer batch 6."""
import json
from pathlib import Path

ROOT = Path(r"F:\Personal Projects\AgenticAI_Chatbot_Using_RAG")
UA = ROOT / ".ua"
EXTRACT = UA / "tmp" / "ua-file-extract-results-6.json"
BATCHES = UA / "intermediate" / "batches.json"
OUT_DIR = UA / "intermediate"

with EXTRACT.open(encoding="utf-8") as f:
    extracted = json.load(f)
with BATCHES.open(encoding="utf-8") as f:
    batches_doc = json.load(f)

# Inventory path -> node id prefix from fileCategory (markup/code -> file, docs -> document).
inventory = {}

def walk(obj):
    if isinstance(obj, dict):
        if "path" in obj and "fileCategory" in obj and isinstance(obj["path"], str):
            cat = obj["fileCategory"]
            prefix = "document" if cat == "docs" else "file"
            if cat == "config":
                prefix = "config"
            inventory[obj["path"]] = prefix
        for v in obj.values():
            walk(v)
    elif isinstance(obj, list):
        for v in obj:
            walk(v)

walk(batches_doc)

def prod_id(path):
    if path not in inventory:
        raise SystemExit(f"production path not in project inventory: {path}")
    disk = ROOT / path
    if not disk.is_file():
        raise SystemExit(f"production path missing on disk: {path}")
    return f"{inventory[path]}:{path}"

def node_fn(path, name, start, end, summary, tags):
    span = end - start + 1
    complexity = "moderate" if span >= 50 else "simple"
    return {
        "id": f"function:{path}:{name}",
        "type": "function",
        "name": name,
        "filePath": path,
        "lineRange": [start, end],
        "summary": summary,
        "tags": tags,
        "complexity": complexity,
    }

# Summaries keyed by (path, name). The later _fn_body definition wins.
S = {}

def s(path, name, summary, tags):
    S[(path, name)] = (summary, tags)

P = "tests/browser_harness.py"
s(P, "run_scenario",
  "Serves the frontend over a GET-only local server, injects the fetch stub and a scenario, and returns the JSON the page writes into #qa-out. Skips when headless Chrome is unavailable.",
  ["test", "browser-harness", "headless-chrome", "stub"])

P = "tests/test_frontend_followup.py"
s(P, "test_m3_card_aria_label_starts_with_the_visible_title",
  "Checks that each welcome-card accessible name starts with the visible title and then describes the workspace switch, matching WCAG 2.5.3 Label in Name.",
  ["test", "accessibility", "welcome-cards"])
s(P, "test_n3_show_error_does_not_announce_and_retry_is_announced_once",
  "Checks that chat.js showError relies on the alert box for the error and calls _announce only once, with the text \"Retry is available.\"",
  ["test", "accessibility", "error-handling"])
s(P, "test_n4_sse_dispatch_only_calls_own_function_handlers",
  "Checks that api.js dispatches SSE events only through hasOwnProperty and keeps LF-only framing, so inherited names are not treated as handlers.",
  ["test", "sse", "api-client"])
s(P, "test_m1_resend_is_decided_in_the_controller_and_only_after_accept_sets_failed",
  "Checks that _onSend chooses regenerate only for an unedited same-session draft and assigns State.failed only after the server has accepted the stream.",
  ["test", "resend", "controller"])
s(P, "test_m2_create_session_receives_the_abort_signal",
  "Checks that _onSend passes the abort signal into _ensureSession and throws if that signal is already aborted.",
  ["test", "abort", "sessions"])
s(P, "test_n1_busy_notice_is_throttled_and_used_by_send_and_session_select",
  "Checks that send and session select both bail through a 4-second _notifyBusy notice while a response is already in progress.",
  ["test", "streaming", "throttle"])
s(P, "fu",
  "Module fixture that serves the frontend on a GET-only port, replaces fetch with the shared stub, and runs the follow-up scenario in headless Chrome.",
  ["test", "fixture", "headless-chrome", "stub"])
s(P, "test_stub_followup_nothing_reached_a_server",
  "Asserts the follow-up browser run recorded no blocked non-GET requests and no page errors.",
  ["test", "stub", "isolation"])
s(P, "test_stub_m1_unedited_resend_uses_regenerate_and_keeps_one_user_bubble",
  "Asserts an unedited resend after a failed accepted stream sends regenerate with an empty message and still shows one user bubble.",
  ["test", "resend", "stub"])
s(P, "test_stub_m1_failed_state_is_cleared_after_success_new_chat_session_switch_and_edit",
  "Asserts a later send is a normal message after success, after an edited draft, after switching to another session, and after New chat.",
  ["test", "resend", "stub"])
s(P, "test_stub_m1_http_level_failure_never_uses_regenerate",
  "Asserts a 503 before the turn is stored resends the original text with regenerate false.",
  ["test", "resend", "error-handling"])
s(P, "test_stub_m1_cooldown_blocks_early_resend_then_regenerates",
  "Asserts a resend during the cooldown is blocked with a toast, then regenerates with an empty message once the cooldown ends.",
  ["test", "cooldown", "resend"])
s(P, "test_stub_m1_retry_button_regenerates_and_no_duplicate_bubble",
  "Asserts the inline Retry control regenerates with an empty message, clears the composer, and keeps a single user bubble.",
  ["test", "retry", "resend"])
s(P, "test_stub_m2_stop_while_session_creation_is_pending_keeps_the_draft",
  "Asserts Stop during a pending session create restores the draft, shows the send button, and does not add a chat bubble.",
  ["test", "abort", "sessions"])
s(P, "test_stub_m2_creation_finishing_after_stop_keeps_the_session_and_sends_nothing",
  "Asserts a session that finishes creating after Stop is reused by the next send and that the stopped draft itself was not posted.",
  ["test", "abort", "sessions"])
s(P, "test_stub_new_chat_while_session_creation_is_pending_sends_nothing",
  "Asserts New chat during a pending session create sends nothing and restores the send button without page errors.",
  ["test", "abort", "sessions"])
s(P, "test_stub_n1_four_enters_mid_stream_give_one_toast_and_no_extra_requests",
  "Asserts four Enter presses during a stream produce one busy toast, no extra requests, and a second notice only after the four-second window.",
  ["test", "throttle", "streaming"])
s(P, "test_stub_n2_stop_before_first_token_restores_draft_and_leaves_no_empty_bubble",
  "Asserts Stop before the first token restores the draft, merges text typed meanwhile, and the following resend uses regenerate.",
  ["test", "stop", "streaming"])
s(P, "test_stub_n2_stop_after_first_token_keeps_partial_and_state_is_not_failed",
  "Asserts Stop after the first token keeps the partial answer and the next send is a normal message rather than a failed regenerate.",
  ["test", "stop", "streaming"])
s(P, "test_stub_n3_error_is_announced_once_by_the_alert_box_only",
  "Asserts a stream error is announced once by the alert box, leaves the live region unchanged, and announces \"Retry is available.\" a single time.",
  ["test", "accessibility", "error-handling"])
s(P, "test_stub_n4_sse_dispatch_ignores_inherited_and_special_event_names",
  "Asserts prototype, constructor, and inherited SSE event names do not run handlers and only an own-property handler fires.",
  ["test", "sse", "stub"])
s(P, "test_stub_zero_event_close_is_not_persisted_and_unedited_send_is_a_normal_send",
  "Asserts a stream that closes with no events restores the draft without bubbles and the next unedited send is a normal message.",
  ["test", "streaming", "persistence"])
s(P, "test_stub_error_without_meta_is_not_persisted_and_unedited_send_is_a_normal_send",
  "Asserts an error that arrives before meta is not persisted, and a resend during the following cooldown stays a normal send.",
  ["test", "error-handling", "persistence"])
s(P, "test_stub_stop_before_any_event_restores_draft_drops_bubbles_and_resends_normally",
  "Asserts Stop before any SSE event drops both bubbles, merges text typed in the meantime, and resends as a normal message.",
  ["test", "stop", "streaming"])
s(P, "test_stub_persisted_error_keeps_user_bubble_and_resend_regenerates_with_empty_message",
  "Asserts an error after meta keeps the user bubble and both resend and stop-then-resend use regenerate with an empty message.",
  ["test", "resend", "error-handling"])
s(P, "test_stub_regenerate_stopped_before_first_token_restores_old_answer_in_place",
  "Asserts stopping a regenerate before the first token restores the previous answer in place, while a partial after the first token is kept.",
  ["test", "regenerate", "stop"])
s(P, "test_stub_regenerate_restore_must_not_leak_into_a_different_chat",
  "Asserts the restored previous answer does not appear after switching to a new chat during regenerate.",
  ["test", "regenerate", "sessions"])
s(P, "test_stub_regenerate_then_delete_current_session_leaves_an_empty_chat_and_only_a_stubbed_delete",
  "Asserts deleting the current session during regenerate leaves an empty chat and the stub records only DELETE /sessions/sH.",
  ["test", "regenerate", "sessions"])
s(P, "test_stub_regenerate_then_new_chat_then_stop_after_tokens_leaks_nothing",
  "Asserts New chat during regenerate, then Stop after tokens, leaves the new chat empty and does not show the old answer.",
  ["test", "regenerate", "stop"])
s(P, "test_stub_nothing_was_ever_sent_to_a_real_server",
  "Asserts the static server saw no non-GET traffic other than an optional favicon request.",
  ["test", "stub", "isolation"])
s(P, "test_stub_retry_countdown_announces_itself_once_and_no_double_announcement",
  "Asserts the retry countdown announces \"Retry is available.\" once, and a cooldown with no retry button announces \"You can send again.\" once.",
  ["test", "accessibility", "cooldown"])
s(P, "test_stub_double_send_double_stop_and_stop_done_race_are_clean",
  "Asserts a double send opens one stream, and double-stop plus a stop-after-done race restore the send button without page errors.",
  ["test", "streaming", "race"])
s(P, "test_stub_sessions_post_payload_keeps_its_four_keys",
  "Asserts the New chat POST /sessions body contains exactly mode, model, title, and user_id.",
  ["test", "sessions", "api-contract"])

P = "tests/test_frontend_guards.py"
s(P, "_read",
  "Reads a path under frontend/ as UTF-8 text for the static guard assertions.",
  ["test", "utility", "filesystem"])
s(P, "test_entrypoints_exist",
  "Checks that frontend/js/app.js, frontend/css/main.css, and the Lucide license file are present.",
  ["test", "frontend", "entry-point"])
s(P, "test_view_files_do_not_call_backend",
  "Checks that view scripts never call fetch, API, or State and never use eval, document.write, or new Function.",
  ["test", "security", "view-layer"])
s(P, "test_no_inline_handlers_or_styles_in_markup",
  "Checks that index.html has no inline event handlers and that the only inline style is display:none on the stop button.",
  ["test", "markup", "security"])
s(P, "test_no_new_external_origins",
  "Checks that first-party HTML, CSS, and JS reference only the allow-listed font and CDN origins, ignoring the vendored directory.",
  ["test", "security", "csp"])
s(P, "test_app_js_ids_exist_in_index",
  "Checks that every getElementById target in app.js is an id present in index.html.",
  ["test", "dom", "controller"])
s(P, "test_sprite_references_resolve",
  "Checks that every icon used in markup or via Icons.html and Icons.el has a matching symbol in the index.html sprite.",
  ["test", "icons", "markup"])
s(P, "test_no_hex_colors_outside_token_files",
  "Checks that components.css and animations.css contain no raw hex colors.",
  ["test", "css", "design-tokens"])
s(P, "test_reduced_motion_rule_present",
  "Checks that animations.css contains a prefers-reduced-motion: reduce rule.",
  ["test", "accessibility", "motion"])
s(P, "test_no_right_sidebar_region",
  "Checks that index.html does not define a right sidebar or right panel region.",
  ["test", "layout", "markup"])
s(P, "test_buttons_have_accessible_name",
  "Parses index.html and checks that every button has visible text, an aria-label, aria-labelledby, or a title.",
  ["test", "accessibility", "markup"])
s(P, "test_index_ids_are_unique",
  "Checks that id attributes in index.html are unique.",
  ["test", "dom", "markup"])
s(P, "test_ui_ids_exist_in_index_or_are_created_dynamically",
  "Checks that ids looked up by ui.js and model-select.js exist in index.html, except modelTrigger which the component creates.",
  ["test", "dom", "ui"])
s(P, "test_model_select_never_uses_html_injection",
  "Checks that model-select.js never assigns innerHTML, outerHTML, insertAdjacentHTML, or document.write.",
  ["test", "security", "xss"])
s(P, "test_session_list_interpolations_are_escaped",
  "Checks that renderSessionList interpolations are escaped or on a reviewed allow-list, including title and session id.",
  ["test", "security", "xss"])
s(P, "test_file_accept_matches_controller_allow_list",
  "Checks that the file input accept attribute lists the same extensions as ALLOWED in app.js.",
  ["test", "uploads", "validation"])
s(P, "test_welcome_cards_and_mode_dropdown_are_consistent",
  "Checks that welcome cards cover the same four modes as the ui.js dropdown, that their names describe a switch-and-fill action, and that workspace prompts never send.",
  ["test", "welcome-cards", "accessibility"])
s(P, "test_no_unverifiable_marketing_claims",
  "Checks that index.html and ui.js do not claim the product is free, fully private, encrypted, unlimited, or the fastest.",
  ["test", "copy", "validation"])
s(P, "test_shell_breakpoints_match_between_head_snippet_and_ui",
  "Checks that the 768px and 1024px width checks in index.html match the 767px and 1023px media queries in ui.js.",
  ["test", "layout", "responsive"])
s(P, "test_sidebar_toggle_controls_existing_region",
  "Checks that the sidebar toggle's aria-controls target exists and that the toggle starts aria-expanded false.",
  ["test", "accessibility", "sidebar"])
s(P, "test_sidebar_is_hidden_from_tab_order_when_closed_drawer",
  "Checks that closed-drawer styles use visibility:hidden and that ui.js sets the sidebar inert.",
  ["test", "accessibility", "sidebar"])
s(P, "test_coarse_pointer_targets_are_44px",
  "Checks that a pointer: coarse media query exists and that the shell styles include a 44px target size.",
  ["test", "accessibility", "touch"])
s(P, "test_model_select_keeps_native_select_as_source_of_truth",
  "Checks that the custom listbox updates the native select, exposes listbox ARIA, and refuses a no-op or disabled choice.",
  ["test", "accessibility", "model-select"])
s(P, "_css_rules",
  "Yields selector and body pairs for simple CSS rules, including rules nested in @media, whose selector matches a pattern.",
  ["test", "css", "utility"])
s(P, "test_feature_card_try_line_is_never_hidden",
  "Checks that .feature-card__try rules never use display:none, visibility:hidden, or height:0.",
  ["test", "css", "welcome-cards"])
s(P, "test_feature_card_border_uses_control_token",
  "Checks that .welcome__starter borders use var(--border-control) and do not fall back to var(--border).",
  ["test", "css", "design-tokens"])
s(P, "_fn_body",
  "Returns the brace body of a named function in a JavaScript source string and fails the test when that function is missing. This definition replaces an earlier helper of the same name.",
  ["test", "utility", "source-parse"])
s(P, "test_sources_block_never_uses_html_injection",
  "Checks that chat.js _fillSources writes text only, builds links with Utils.safeUrl, and sets noopener noreferrer on new tabs.",
  ["test", "security", "citations"])
s(P, "test_every_blank_target_has_noopener_noreferrer",
  "Checks that every _blank target in chat.js, ui.js, utils.js, and index.html is paired with rel=\"noopener\".",
  ["test", "security", "links"])
s(P, "test_chat_innerhtml_templates_interpolate_only_safe_values",
  "Checks that chat.js innerHTML templates interpolate only escaped or static values and that badge text goes through Utils.escapeHtml.",
  ["test", "security", "xss"])
s(P, "test_renderer_and_chat_api_used_by_app_js_still_exists",
  "Checks that every renderer and Chat method called from app.js is still defined in chat.js, including markPartial and setVizState.",
  ["test", "api-contract", "renderer"])
s(P, "test_ai_viz_is_css_only_and_decorative",
  "Checks that the AI visualization is aria-hidden, uses no timers or random motion in script, and animates only through tokenized CSS.",
  ["test", "accessibility", "animation"])
s(P, "test_ai_viz_reduced_motion_and_hidden_tab_rules",
  "Checks that ai-viz.css disables animation under reduced motion, pauses it when the page is hidden, and keyframes only transform and opacity.",
  ["test", "accessibility", "animation"])
s(P, "test_no_interval_timers_or_raf_loops_in_frontend",
  "Checks that no frontend script uses setInterval and that chat.js calls requestAnimationFrame at most twice.",
  ["test", "performance", "frontend"])
s(P, "test_streaming_does_not_announce_per_token",
  "Checks that appendToken does not announce each token and that token rendering stays throttled to 60ms.",
  ["test", "accessibility", "streaming"])
s(P, "test_stage3_ids_exist",
  "Checks that the welcome visualization, jump-latest, live status, and chat area ids exist and that chat.js element lookups resolve.",
  ["test", "dom", "chat"])
s(P, "test_sanitizer_policy_forbids_class_role_download_ping_in_utils",
  "Checks that the utils.js sanitizer FORBID_ATTR list includes class, role, download, and ping.",
  ["test", "security", "sanitizer"])
s(P, "test_citation_label_falls_back_to_source_n_and_is_used_everywhere",
  "Checks that a blank citation title becomes \"Source N\" and that both citation dialogs go through _citationLabel.",
  ["test", "citations", "accessibility"])
s(P, "test_citation_ids_resolved_once_for_stream_and_history_paths",
  "Checks that stream and history citations are resolved once and that _resolveCitations copies objects instead of mutating the input.",
  ["test", "citations", "immutability"])
s(P, "test_coarse_pointer_citation_hit_area_rule_present_and_scoped",
  "Checks that coarse pointers get a 44px absolutely positioned hit area on citation buttons without changing the visual badge.",
  ["test", "accessibility", "citations"])
s(P, "cite_result",
  "Runs chat.js, utils.js, and icons.js in headless Chrome with citation fixtures and returns the JSON the harness writes, skipping when Chrome or the CDN libraries are missing.",
  ["test", "fixture", "headless-chrome", "citations"])
s(P, "_id_list",
  "Returns the citation id column for one named case in the browser citation result.",
  ["test", "utility", "citations"])
s(P, "test_browser_citation_ids_consistent_unique_and_input_not_mutated",
  "Asserts citation ids stay unique across missing, explicit, duplicate, and invalid inputs and that the source array is not mutated.",
  ["test", "citations", "browser"])
s(P, "test_browser_blank_citations_get_accessible_source_n_names",
  "Asserts blank citations are named Source 1, Source 2, and Source 3, a javascript: URL is not linked, and the dialog title matches the row.",
  ["test", "citations", "accessibility"])
s(P, "test_browser_isolated_badge_reaches_44px_on_coarse_pointers",
  "Asserts an isolated citation badge stays visually small while its coarse-pointer hit target is at least 44 by 44 pixels.",
  ["test", "accessibility", "touch"])
s(P, "test_browser_multi_citation_sentence_badges_reach_44px_wide",
  "Asserts adjacent citation badges in one sentence each have a coarse-pointer hit area at least 44 by 44 pixels.",
  ["test", "accessibility", "touch"])
s(P, "_code",
  "Strips block comments and line comments from a source string so later assertions do not match commented-out code.",
  ["test", "utility", "source-parse"])
s(P, "_app_fn",
  "Returns the body of a top-level function in app.js, from its opening brace up to the next sibling function.",
  ["test", "utility", "controller"])
s(P, "test_app_js_has_no_native_confirm_inline_color_interval_or_emoji",
  "Checks that app.js uses UI.confirm four times and does not call window.confirm, set inline colors, use setInterval, or embed emoji.",
  ["test", "controller", "accessibility"])
s(P, "test_welcome_card_handler_never_sends",
  "Checks that the welcome-card handler, file picker, and composer filler only switch mode and fill text, and that ui.js does not send from those cards.",
  ["test", "welcome-cards", "controller"])
s(P, "test_allowed_extensions_include_rst_and_match_accept_and_backend",
  "Checks that app.js allows .rst and that the accept attribute, the client allow-list, and parser.SUPPORTED_EXTENSIONS are the same set.",
  ["test", "uploads", "api-contract"])
s(P, "test_request_payload_shapes_match_the_backend_contract",
  "Checks that chat and session payloads in app.js and api.js match ChatRequest and the session body fields in the backend models.",
  ["test", "api-contract", "controller"])
s(P, "test_regenerate_and_retry_send_the_head_shape",
  "Checks that regenerate and the retry helper send message as an empty string and pass regenerate through _onSend.",
  ["test", "regenerate", "api-contract"])
s(P, "test_composer_is_cleared_only_after_the_server_accepted",
  "Checks that the composer is cleared inside the onAccepted callback, which runs only after api.js has seen a successful response.",
  ["test", "composer", "streaming"])
s(P, "test_no_new_timer_or_listener_patterns_in_controller",
  "Checks that app.js has a single setTimeout, clears the cooldown timer, and never writes HTML.",
  ["test", "controller", "security"])
s(P, "stub_run",
  "Serves the frontend on a GET-only port, injects the shared fetch stub and stage-4 scenario, and returns the JSON from headless Chrome.",
  ["test", "fixture", "headless-chrome", "stub"])
s(P, "test_stub_nothing_reached_a_server",
  "Asserts the stage-4 browser run served only static GETs and recorded no page errors.",
  ["test", "stub", "isolation"])
s(P, "test_stub_welcome_cards_never_submit_and_fill_the_composer",
  "Asserts each welcome card fills the composer with its prompt, focuses the input, and never POSTs.",
  ["test", "welcome-cards", "stub"])
s(P, "test_stub_failures_before_accept_keep_the_draft_and_add_no_bubble",
  "Asserts session-create and HTTP failures before accept keep the draft, add no bubbles, and that a 429 cooldown blocks then allows one later send.",
  ["test", "error-handling", "composer"])
s(P, "test_stub_stream_error_restores_draft_and_never_duplicates_the_user_bubble",
  "Asserts a stream error keeps one user bubble and that both resend and retry regenerate with an empty message.",
  ["test", "error-handling", "resend"])
s(P, "test_stub_failed_regenerate_keeps_the_previous_answer",
  "Asserts a failed regenerate leaves the previous answer in place and sends regenerate with an empty message.",
  ["test", "regenerate", "error-handling"])
s(P, "test_stub_upload_failure_retry_resends_the_same_file",
  "Asserts a failed upload can be retried with the same notes.rst file and that .exe and extensionless names are never posted.",
  ["test", "uploads", "error-handling"])

P = "tests/test_frontend_motion.py"
s(P, "_code",
  "Loads a frontend file through the guards reader and strips comments so motion assertions see executable code only.",
  ["test", "utility", "source-parse"])
s(P, "test_vendored_gsap_is_pinned_and_documented",
  "Checks that the vendor directory contains only gsap.min.js and its notice, and that the file hash and version match the pinned GSAP 3.15.0 build.",
  ["test", "vendor", "integrity"])
s(P, "test_gsap_has_no_network_or_dynamic_code_surface",
  "Checks that the vendored GSAP build contains no fetch, XHR, eval, new Function, sendBeacon, WebSocket, or document.write.",
  ["test", "security", "vendor"])
s(P, "test_gsap_is_loaded_lazily_from_the_same_origin_only",
  "Checks that motion.js loads /js/vendor/gsap.min.js asynchronously from the same origin and is included before ui.js.",
  ["test", "motion", "loading"])
s(P, "test_motion_is_view_only_and_animates_only_transform_and_opacity",
  "Checks that motion.js does not touch the network or app state and that GSAP tweens avoid layout properties and clear inline styles.",
  ["test", "motion", "view-layer"])
s(P, "test_motion_honours_reduced_motion_and_cleans_up",
  "Checks that reduced motion returns before GSAP is loaded and that tweens live in a context reverted on pagehide.",
  ["test", "accessibility", "motion"])
s(P, "test_first_paint_guard_has_a_hard_failsafe_and_skips_reduced_motion",
  "Checks that the index.html head adds motion-pending, removes it after 2.5 seconds, and that workspace.css hides that class with opacity 0.",
  ["test", "motion", "first-paint"])
s(P, "test_css_entrance_keyframes_that_motion_replaces_are_gone",
  "Checks that the old welcome and composer CSS entrance animations are gone so GSAP is the only entrance.",
  ["test", "css", "motion"])
s(P, "test_vendor_code_is_exempt_from_the_first_party_origin_guard_only_by_directory",
  "Checks that the external-origin guard skips files only by a vendor directory path.",
  ["test", "security", "vendor"])
s(P, "_run_browser",
  "Serves the frontend on a GET-only port with the shared fetch stub and runs the motion scenario in headless Chrome, with optional browser flags.",
  ["test", "fixture", "headless-chrome", "motion"])
s(P, "reduced",
  "Browser fixture that reruns the motion scenario with Chrome forced into prefers-reduced-motion.",
  ["test", "fixture", "accessibility"])
s(P, "normal",
  "Browser fixture that reruns the motion scenario with default motion preferences.",
  ["test", "fixture", "motion"])
s(P, "test_browser_reduced_motion_never_loads_the_library_and_shows_everything",
  "Asserts reduced motion never defines window.gsap, leaves title, composer, and cards fully visible, and still lets the mode menu switch to Documents.",
  ["test", "accessibility", "motion"])
s(P, "test_browser_normal_motion_ends_with_clean_visible_content_and_stays_interactive",
  "Asserts normal motion loads GSAP, finishes with no leftover inline styles, and stays interactive across rapid workspace switches.",
  ["test", "motion", "browser"])

P = "tests/test_frontend_startup.py"
s(P, "_fn",
  "Returns a slice of app.js starting at the named function so startup assertions can read init and _startupFailed.",
  ["test", "utility", "source-parse"])
s(P, "test_data_loading_starts_before_event_handlers_are_bound",
  "Checks that init starts model, session, document, and config loads before _bindEvents and awaits them afterward.",
  ["test", "startup", "controller"])
s(P, "test_startup_errors_are_reported_with_the_fix_not_swallowed",
  "Checks that init failures go to _startupFailed, which logs the error and toasts a Ctrl+Shift+R reload hint.",
  ["test", "startup", "error-handling"])
s(P, "stale",
  "Runs a headless scenario that deletes UI.onModeChange before init, simulating a stale cached ui.js beside a new app.js.",
  ["test", "fixture", "startup"])
s(P, "test_stub_models_still_load_when_a_handler_fails_to_bind",
  "Asserts models and the session list still load when event binding throws, and the model chip does not stay on \"Loading models…\".",
  ["test", "startup", "regression"])
s(P, "test_stub_the_user_is_told_what_happened_and_how_to_fix_it",
  "Asserts the startup toast names the failure, mentions onModeChange and Ctrl+Shift+R, and that the error is not an unhandled rejection.",
  ["test", "startup", "error-handling"])

P = "tests/test_frontend_workspaces.py"
s(P, "_strip_comments",
  "Removes block and line comments from a source string before workspace scripts are scanned for network or HTML writes.",
  ["test", "utility", "source-parse"])
s(P, "test_three_workspace_panels_exist_and_each_has_a_labelled_heading",
  "Checks that Documents, Research, and Code each have a labelled workspace section and that the General hero remains.",
  ["test", "workspace", "accessibility"])
s(P, "test_panel_controls_are_buttons_that_only_fill_the_composer",
  "Checks that workspace prompt controls are type=button and that workspaces.js never sends, fetches, or touches API or State.",
  ["test", "workspace", "view-layer"])
s(P, "test_document_actions_are_explained_not_hidden_when_unavailable",
  "Checks that document actions stay visible but aria-disabled, with the note \"Upload a document to enable these actions.\"",
  ["test", "accessibility", "documents"])
s(P, "test_research_locks_the_web_switch_because_the_server_always_searches",
  "Checks that Research mode locks the web-search switch in both workspaces.js and app.js, matching the README note that the server always searches.",
  ["test", "research", "web-search"])
s(P, "test_settings_tabs_and_panes_are_wired_with_aria",
  "Checks that the seven settings tabs control matching tabpanels and that exactly one tab starts selected.",
  ["test", "accessibility", "settings"])
s(P, "test_settings_only_offers_controls_with_a_real_effect",
  "Checks that settings.js only wires theme, font, density, toasts, notifications, copy, and health, with no disabled or \"coming soon\" controls.",
  ["test", "settings", "preferences"])
s(P, "test_privacy_and_help_copy_makes_only_verifiable_statements",
  "Checks that settings copy names Groq or Google, Tavily, and the lack of accounts, and avoids privacy or encryption guarantees.",
  ["test", "copy", "settings"])
s(P, "test_new_components_are_view_only_and_never_inject_html",
  "Checks that the new workspace components do not fetch, touch State, or inject HTML, and that only prefs.js writes localStorage under the pref. prefix.",
  ["test", "security", "view-layer"])
s(P, "test_new_components_are_loaded_before_ui_js_in_dependency_order",
  "Checks that prefs, radiogroup, documents, workspaces, and settings scripts load before ui.js and app.js.",
  ["test", "loading", "dependencies"])
s(P, "test_workspace_css_obeys_the_project_css_rules",
  "Checks that workspace.css loads after reference.css, uses no hex colors or keyframes, and transitions only paint and transform properties.",
  ["test", "css", "workspace"])
s(P, "test_text_is_rem_based_so_the_text_size_setting_scales_everything",
  "Checks that font-size and line-height stay off raw pixels except the root size and the hero orb, and that the large font hook exists.",
  ["test", "css", "typography"])
s(P, "test_code_workspace_is_language_agnostic_and_has_debug_logs",
  "Checks that the code workspace has no language picker and offers a debug-logs template.",
  ["test", "workspace", "code"])
s(P, "test_composer_github_item_and_resizable_preview_exist",
  "Checks that the GitHub connect dialog, the artifact resize handle, and Chat.prepareFilePreview are wired from ui.js, chat.js, and app.js.",
  ["test", "composer", "preview"])
s(P, "test_preference_markup_matches_the_allowed_values",
  "Checks that font, density, and toast radio values in index.html match the values prefs.js allows, including the pref. storage keys.",
  ["test", "preferences", "markup"])
s(P, "test_uploads_validate_on_the_client_and_guard_duplicates_without_trusting_it",
  "Checks that client uploads reject empty and oversized files, confirm duplicate names, and still document that the server validates again.",
  ["test", "uploads", "validation"])
s(P, "test_config_is_a_read_only_get_and_its_failure_is_tolerated",
  "Checks that getConfig is a GET /config and that a failed config load is treated as an older backend rather than a startup failure.",
  ["test", "config", "api-client"])
s(P, "test_history_and_documents_have_loading_and_error_states",
  "Checks that session and document lists have loading and error states and that history rows move with ArrowUp and ArrowDown.",
  ["test", "accessibility", "history"])
s(P, "ws",
  "Serves the frontend on a GET-only port with the shared fetch stub and runs the workspace scenario in headless Chrome.",
  ["test", "fixture", "headless-chrome", "workspace"])
s(P, "test_stub_run_reached_no_server_and_threw_nothing",
  "Asserts the workspace browser run recorded no blocked non-GET requests and no page errors.",
  ["test", "stub", "isolation"])
s(P, "test_stub_dropdown_is_the_only_mode_control_and_is_keyboard_operable",
  "Asserts the mode menu is the only mode control, lists the four workspaces, and returns focus to the trigger on Escape.",
  ["test", "accessibility", "workspace"])
s(P, "test_stub_each_mode_shows_only_its_own_panel",
  "Asserts each mode shows only its panel, updates the label and placeholder, and does not rewrite storage when the current mode is chosen again.",
  ["test", "workspace", "browser"])
s(P, "test_stub_panel_actions_only_fill_the_composer",
  "Asserts empty-document actions explain themselves, code templates fill a fence without naming Python, and research fills the selected topic.",
  ["test", "workspace", "composer"])
s(P, "test_stub_research_locks_the_web_switch_and_other_modes_unlock_it",
  "Asserts Research forces the web-search switch on with an explanation and that leaving Research unlocks it.",
  ["test", "research", "web-search"])
s(P, "test_stub_my_documents_panel_refreshes_and_returns_focus",
  "Asserts opening My Documents refreshes the list, focuses search, returns focus to the composer, and does not change the workspace.",
  ["test", "documents", "accessibility"])
s(P, "test_stub_settings_controls_take_effect_and_reset",
  "Asserts theme, font size, and density apply, reset clears them, and settings tab focus returns to the settings button.",
  ["test", "settings", "preferences"])
s(P, "test_stub_uploads_validate_guard_duplicates_and_queue",
  "Asserts .exe and empty files are rejected, a duplicate upload needs confirmation, and two valid files post one after another.",
  ["test", "uploads", "validation"])
s(P, "test_stub_history_arrow_keys_move_between_chats",
  "Asserts ArrowDown and ArrowUp move focus between at least two chat rows.",
  ["test", "accessibility", "history"])

FILES = {
    "tests/browser_harness.py": {
        "summary": "Shared headless-Chrome runner for frontend scenario tests. It serves frontend/ with GET only, injects the fetch stub from the guard suite, and returns the JSON a scenario writes into the page.",
        "tags": ["test", "browser-harness", "headless-chrome", "stub"],
        "complexity": "moderate",
        "languageNotes": "Any verb other than GET is answered with 405 and recorded, so a scenario cannot silently reach a real backend.",
    },
    "tests/test_frontend_followup.py": {
        "summary": "Stage-4 follow-up tests for resend, abort, busy notices, and SSE error handling. Static checks read the controller and chat renderer; browser checks run against a fetch stub and never start the backend.",
        "tags": ["test", "frontend", "regression", "streaming"],
        "complexity": "complex",
        "languageNotes": "Browser cases skip when headless Chrome or the marked and DOMPurify CDN scripts are unavailable.",
    },
    "tests/test_frontend_guards.py": {
        "summary": "Offline guards for the frontend shell, chat renderer, citations, and controller, plus a stub-only Chrome run of send, retry, and upload failures. View scripts are forbidden from calling the backend or injecting HTML.",
        "tags": ["test", "frontend", "static-analysis", "accessibility", "security"],
        "complexity": "complex",
        "languageNotes": "Static checks use the standard library. The citation and stage-4 fixtures launch headless Chrome and skip, rather than fail, when Chrome or the CDN libraries are missing.",
    },
    "tests/test_frontend_motion.py": {
        "summary": "Guards the GSAP motion layer: the library is vendored and hash-pinned, loaded lazily from the same origin, and skipped entirely under reduced motion. Browser cases confirm content stays visible and interactive.",
        "tags": ["test", "motion", "accessibility", "frontend"],
        "complexity": "moderate",
        "languageNotes": "The vendored build is checked by SHA-256 against GSAP 3.15.0 and is loaded only from /js/vendor/gsap.min.js.",
    },
    "tests/test_frontend_startup.py": {
        "summary": "Regression tests that a startup failure must not leave the model list stuck on \"Loading models…\". Data loading starts before event binding, and a stale ui.js still surfaces a reload hint.",
        "tags": ["test", "startup", "regression", "frontend"],
        "complexity": "moderate",
        "languageNotes": "The browser case deletes UI.onModeChange before init to reproduce a new app.js running beside a cached ui.js.",
    },
    "tests/test_frontend_workspaces.py": {
        "summary": "Guards for the Documents, Research, and Code workspaces, settings, preferences, and uploads. Static checks stay offline; the browser scenario uses the shared fetch stub and refuses every non-GET.",
        "tags": ["test", "workspace", "accessibility", "frontend"],
        "complexity": "complex",
        "languageNotes": "Browser cases skip when headless Chrome or the marked and DOMPurify CDN scripts are unavailable.",
    },
}

# tested_by: production node -> this test file. Paths must be in the project inventory.
TESTED = {
    "tests/browser_harness.py": [
        "frontend/index.html",
    ],
    "tests/test_frontend_followup.py": [
        "frontend/index.html",
        "frontend/js/app.js",
        "frontend/js/api.js",
        "frontend/js/chat.js",
    ],
    "tests/test_frontend_guards.py": [
        "frontend/index.html",
        "frontend/js/app.js",
        "frontend/js/api.js",
        "frontend/js/chat.js",
        "frontend/js/ui.js",
        "frontend/js/utils.js",
        "frontend/js/icons.js",
        "frontend/js/theme.js",
        "frontend/js/components/menu.js",
        "frontend/js/components/dialog.js",
        "frontend/js/components/model-select.js",
        "frontend/js/components/ai-viz.js",
        "frontend/js/components/workspaces.js",
        "frontend/css/main.css",
        "frontend/css/components.css",
        "frontend/css/animations.css",
        "frontend/css/ai-viz.css",
        "frontend/assets/LICENSE-lucide.txt",
        "backend/models/api.py",
        "backend/services/parser.py",
    ],
    "tests/test_frontend_motion.py": [
        "frontend/index.html",
        "frontend/js/components/motion.js",
        "frontend/css/workspace.css",
        "frontend/css/animations.css",
        "frontend/css/components.css",
    ],
    "tests/test_frontend_startup.py": [
        "frontend/index.html",
        "frontend/js/app.js",
        "frontend/js/ui.js",
    ],
    "tests/test_frontend_workspaces.py": [
        "frontend/index.html",
        "frontend/js/app.js",
        "frontend/js/api.js",
        "frontend/js/ui.js",
        "frontend/js/chat.js",
        "frontend/js/components/workspaces.js",
        "frontend/js/components/settings.js",
        "frontend/js/components/prefs.js",
        "frontend/js/components/documents.js",
        "frontend/js/components/radiogroup.js",
        "frontend/css/workspace.css",
        "frontend/css/reference.css",
        "frontend/css/components.css",
        "frontend/css/main.css",
        "README.md",
    ],
}

CALLS = [
    ("tests/test_frontend_followup.py", "test_n3_show_error_does_not_announce_and_retry_is_announced_once", "tests/test_frontend_guards.py", "_code"),
    ("tests/test_frontend_followup.py", "test_n4_sse_dispatch_only_calls_own_function_handlers", "tests/test_frontend_guards.py", "_code"),
    ("tests/test_frontend_followup.py", "test_m1_resend_is_decided_in_the_controller_and_only_after_accept_sets_failed", "tests/test_frontend_guards.py", "_app_fn"),
    ("tests/test_frontend_followup.py", "test_m2_create_session_receives_the_abort_signal", "tests/test_frontend_guards.py", "_app_fn"),
    ("tests/test_frontend_followup.py", "test_n1_busy_notice_is_throttled_and_used_by_send_and_session_select", "tests/test_frontend_guards.py", "_app_fn"),
    ("tests/test_frontend_motion.py", "_code", "tests/test_frontend_guards.py", "_read"),
    ("tests/test_frontend_motion.py", "test_vendored_gsap_is_pinned_and_documented", "tests/test_frontend_guards.py", "_read"),
    ("tests/test_frontend_motion.py", "test_first_paint_guard_has_a_hard_failsafe_and_skips_reduced_motion", "tests/test_frontend_guards.py", "_read"),
    ("tests/test_frontend_motion.py", "test_css_entrance_keyframes_that_motion_replaces_are_gone", "tests/test_frontend_guards.py", "_read"),
    ("tests/test_frontend_startup.py", "stale", "tests/browser_harness.py", "run_scenario"),
]
for name in (
    "test_panel_controls_are_buttons_that_only_fill_the_composer",
    "test_document_actions_are_explained_not_hidden_when_unavailable",
    "test_research_locks_the_web_switch_because_the_server_always_searches",
    "test_settings_only_offers_controls_with_a_real_effect",
    "test_new_components_are_view_only_and_never_inject_html",
    "test_workspace_css_obeys_the_project_css_rules",
    "test_text_is_rem_based_so_the_text_size_setting_scales_everything",
    "test_code_workspace_is_language_agnostic_and_has_debug_logs",
    "test_composer_github_item_and_resizable_preview_exist",
    "test_preference_markup_matches_the_allowed_values",
    "test_uploads_validate_on_the_client_and_guard_duplicates_without_trusting_it",
    "test_config_is_a_read_only_get_and_its_failure_is_tolerated",
    "test_history_and_documents_have_loading_and_error_states",
):
    CALLS.append(("tests/test_frontend_workspaces.py", name, "tests/test_frontend_guards.py", "_read"))

IMPORTS = {
    "tests/browser_harness.py": ["tests/test_frontend_guards.py"],
    "tests/test_frontend_followup.py": ["tests/test_frontend_guards.py"],
    "tests/test_frontend_guards.py": [],
    "tests/test_frontend_motion.py": ["tests/test_frontend_guards.py"],
    "tests/test_frontend_startup.py": ["tests/browser_harness.py", "tests/test_frontend_guards.py"],
    "tests/test_frontend_workspaces.py": ["tests/test_frontend_guards.py"],
}

nodes = []
edges = []
seen_ids = set()

for result in extracted["results"]:
    path = result["path"]
    meta = FILES[path]
    file_node = {
        "id": f"file:{path}",
        "type": "file",
        "name": Path(path).name,
        "filePath": path,
        "summary": meta["summary"],
        "tags": meta["tags"],
        "complexity": meta["complexity"],
    }
    if meta.get("languageNotes"):
        file_node["languageNotes"] = meta["languageNotes"]
    nodes.append(file_node)
    seen_ids.add(file_node["id"])

    # Later definition of a repeated name replaces the earlier one (live binding).
    chosen = {}
    for fn in result["functions"]:
        chosen[fn["name"]] = fn
    export_names = {e["name"] for e in result.get("exports") or []}
    for fn in chosen.values():
        key = (path, fn["name"])
        if key not in S:
            raise SystemExit(f"missing summary for {key}")
        summary, tags = S[key]
        if len(tags) < 3 or len(tags) > 5:
            raise SystemExit(f"tag count for {key}: {tags}")
        n = node_fn(path, fn["name"], fn["startLine"], fn["endLine"], summary, tags)
        if n["id"] in seen_ids:
            raise SystemExit(f"duplicate id {n['id']}")
        nodes.append(n)
        seen_ids.add(n["id"])
        edges.append({
            "source": f"file:{path}",
            "target": n["id"],
            "type": "contains",
            "direction": "forward",
            "weight": 1.0,
        })
        if fn["name"] in export_names:
            edges.append({
                "source": f"file:{path}",
                "target": n["id"],
                "type": "exports",
                "direction": "forward",
                "weight": 0.8,
            })
    for target in IMPORTS[path]:
        edges.append({
            "source": f"file:{path}",
            "target": f"file:{target}",
            "type": "imports",
            "direction": "forward",
            "weight": 0.7,
        })
    for prod in TESTED[path]:
        edges.append({
            "source": prod_id(prod),
            "target": f"file:{path}",
            "type": "tested_by",
            "direction": "forward",
            "weight": 0.5,
        })

for src_path, src_name, dst_path, dst_name in CALLS:
    edges.append({
        "source": f"function:{src_path}:{src_name}",
        "target": f"function:{dst_path}:{dst_name}",
        "type": "calls",
        "direction": "forward",
        "weight": 0.8,
    })

# Summaries must cover every emitted function and nothing extra.
emitted_fns = {(n["filePath"], n["name"]) for n in nodes if n["type"] == "function"}
if emitted_fns != set(S):
    missing = set(S) - emitted_fns
    extra = emitted_fns - set(S)
    raise SystemExit(f"summary mismatch missing={missing} extra={extra}")

import_edges = [e for e in edges if e["type"] == "imports"]
expected_imports = sum(len(v) for v in IMPORTS.values())
if len(import_edges) != expected_imports:
    raise SystemExit(f"import edge count {len(import_edges)} != {expected_imports}")

for e in edges:
    if e["source"] == e["target"]:
        raise SystemExit(f"self edge {e}")
    for end in ("source", "target"):
        ident = e[end]
        if ident in seen_ids:
            continue
        if ident.startswith("file:") or ident.startswith("document:") or ident.startswith("config:"):
            path = ident.split(":", 1)[1]
            if path in inventory:
                continue
        raise SystemExit(f"dangling {end} {ident} on {e['type']}")

# Partition. Guards alone exceeds 60 nodes / 120 edges, so pack alphabetically
# and start a new part before a file would push a part further over those caps.
by_file = {}
for n in nodes:
    by_file.setdefault(n["filePath"], {"nodes": [], "edges": []})
    by_file[n["filePath"]]["nodes"].append(n)
for e in edges:
    # Attribute an edge to the file that owns the source when that source is in this batch;
    # tested_by sources are production files, so attribute those to the test target.
    if e["type"] == "tested_by":
        owner = e["target"].split(":", 1)[1]
    elif e["source"].startswith("file:"):
        owner = e["source"].split(":", 1)[1]
    elif e["source"].startswith("function:"):
        owner = e["source"].split(":")[1]
    else:
        raise SystemExit(f"cannot place edge {e}")
    by_file[owner]["edges"].append(e)

order = sorted(by_file)
parts = []
cur = {"files": [], "nodes": [], "edges": []}

def flush():
    if cur["files"]:
        parts.append({"files": list(cur["files"]), "nodes": list(cur["nodes"]), "edges": list(cur["edges"])})
        cur["files"].clear()
        cur["nodes"].clear()
        cur["edges"].clear()

for path in order:
    bucket = by_file[path]
    next_nodes = len(cur["nodes"]) + len(bucket["nodes"])
    next_edges = len(cur["edges"]) + len(bucket["edges"])
    if cur["files"] and (next_nodes > 60 or next_edges > 120):
        flush()
    cur["files"].append(path)
    cur["nodes"].extend(bucket["nodes"])
    cur["edges"].extend(bucket["edges"])
flush()

# If the whole graph fits one file, write batch-6.json. It does not.
total_nodes = len(nodes)
total_edges = len(edges)
print(f"TOTAL nodes={total_nodes} edges={total_edges} parts={len(parts)}")
for i, part in enumerate(parts, 1):
    print(f"  part {i}: files={part['files']} nodes={len(part['nodes'])} edges={len(part['edges'])}")

for i, part in enumerate(parts, 1):
    payload = {"nodes": part["nodes"], "edges": part["edges"]}
    dest = OUT_DIR / f"batch-6-part-{i}.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    json.loads(dest.read_text(encoding="utf-8"))
    print(f"wrote {dest.name}")
