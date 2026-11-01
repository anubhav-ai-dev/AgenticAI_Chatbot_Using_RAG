const fs = require("fs");
const path = require("path");

const root = "F:/Personal Projects/AgenticAI_Chatbot_Using_RAG";
const outDir = path.join(root, ".ua", "intermediate");

const nodes = [];
const edges = [];

function cx(start, end) {
  const n = end - start + 1;
  if (n >= 50) return "complex";
  if (n >= 20) return "moderate";
  return "simple";
}

function addFile(filePath, summary, tags, complexity, languageNotes) {
  const node = {
    id: "file:" + filePath,
    type: "file",
    name: path.basename(filePath),
    filePath,
    summary,
    tags,
    complexity,
  };
  if (languageNotes) node.languageNotes = languageNotes;
  nodes.push(node);
}

function addFn(filePath, name, start, end, summary, tags, exported) {
  const id = "function:" + filePath + ":" + name;
  nodes.push({
    id,
    type: "function",
    name,
    filePath,
    lineRange: [start, end],
    summary,
    tags,
    complexity: cx(start, end),
  });
  edges.push({ source: "file:" + filePath, target: id, type: "contains", direction: "forward", weight: 1.0 });
  if (exported) {
    edges.push({ source: "file:" + filePath, target: id, type: "exports", direction: "forward", weight: 0.8 });
  }
}

function dep(from, to) {
  edges.push({ source: "file:" + from, target: "file:" + to, type: "depends_on", direction: "forward", weight: 0.6 });
}

function call(fromFile, fromFn, toFile, toFn) {
  edges.push({
    source: "function:" + fromFile + ":" + fromFn,
    target: "function:" + toFile + ":" + toFn,
    type: "calls",
    direction: "forward",
    weight: 0.8,
  });
}

function testedBy(prod, test) {
  edges.push({ source: "file:" + prod, target: "file:" + test, type: "tested_by", direction: "forward", weight: 0.5 });
}

const RG = "frontend/js/components/radiogroup.js";
const ST = "frontend/js/components/settings.js";
const WS = "frontend/js/components/workspaces.js";
const IC = "frontend/js/icons.js";
const IN = "frontend/js/intro.js";
const TH = "frontend/js/theme.js";
const UI = "frontend/js/ui.js";
const UT = "frontend/js/utils.js";
const INIT = "tests/__init__.py";
const ARCH = "tests/test_architecture.py";
const REF = "tests/test_frontend_reference.py";
const SAN = "tests/test_frontend_sanitizer.py";

addFile(RG, "ARIA radio-group helper that moves selection with clicks and arrow keys while keeping all state in DOM attributes.", ["component", "accessibility", "event-handler", "view"], "moderate");
addFn(RG, "set", 16, 23, "Reflects a radio value on aria-checked and tabindex without notifying the change handler.", ["accessibility", "view", "dom"], true);
addFn(RG, "get", 25, 28, "Returns the data-value of the radio that is currently aria-checked, or null.", ["accessibility", "view", "dom"], true);
addFn(RG, "wire", 30, 61, "Binds click and arrow, Home, and End keys so the group selects a radio and reports user changes only.", ["accessibility", "event-handler", "keyboard"], true);

addFile(ST, "Settings dialog view for profile, appearance, models, notifications, integrations, privacy, and help. Controls write through Theme and Prefs and never fetch on their own.", ["component", "settings", "dialog", "view"], "complex");
addFn(ST, "select", 35, 48, "Shows one settings pane, updates tab selection, and notifies the pane handler.", ["settings", "navigation", "view"], true);
addFn(ST, "_wireTabs", 50, 79, "Binds tab clicks and arrow-key roving, and flips aria-orientation when the nav becomes a horizontal strip.", ["settings", "accessibility", "keyboard"], false);
addFn(ST, "syncModels", 91, 115, "Copies options, groups, and disabled reasons from the composer model select into the settings model control.", ["settings", "view", "models"], true);
addFn(ST, "_syncAll", 117, 128, "Refreshes theme, text size, density, toast preference, user id, models, and the notice count from their sources of truth.", ["settings", "view", "sync"], false);
addFn(ST, "_badge", 144, 159, "Maps a health-service value to a text label, tone, and icon name so status is not conveyed by color alone.", ["settings", "view", "status"], false);
addFn(ST, "renderIntegrations", 165, 191, "Renders GET /health service rows with icon badges, or a loading or error note when status is unavailable.", ["settings", "view", "integrations"], true);
addFn(ST, "renderGithubRepos", 193, 223, "Renders connected GitHub repositories and a Disconnect button that calls the supplied handler.", ["settings", "view", "github"], true);
addFn(ST, "open", 227, 234, "Syncs every control, selects a pane, and opens the settings dialog with the active tab focused.", ["settings", "dialog", "view"], true);
addFn(ST, "close", 236, 236, "Closes the settings dialog through the shared Dialog helper.", ["settings", "dialog", "view"], true);
addFn(ST, "setUserId", 237, 237, "Stores the displayed user id and writes it into the profile pane.", ["settings", "profile", "view"], true);
addFn(ST, "onPane", 238, 238, "Registers the controller callback fired when the user changes settings pane.", ["settings", "event-handler", "view"], true);
addFn(ST, "init", 240, 291, "Wires tabs, theme and preference radios, model mirroring, notice clearing, user-id copy, and the integrations refresh.", ["settings", "event-handler", "initialization"], true);

addFile(WS, "Welcome-screen workspace panels for Documents, Research, and Code. Actions only fill the composer and focus it; nothing is sent from a panel.", ["component", "view", "composer", "workspaces"], "moderate");
addFn(WS, "fill", 36, 52, "Writes prompt text into the composer, focuses it, and places the caret in a placeholder or code fence. Never sends.", ["composer", "view", "workspaces"], true);
addFn(WS, "_onClick", 54, 70, "Handles welcome-card fill and template buttons, and explains why document actions are disabled when nothing is indexed.", ["event-handler", "composer", "workspaces"], false);
addFn(WS, "_syncWebToggle", 73, 88, "Locks the web-search switch on in Research and restores the user's pressed state in other workspaces.", ["workspaces", "view", "accessibility"], false);
addFn(WS, "setActive", 90, 94, "Sets the composer placeholder for the active workspace and syncs the web-search switch.", ["workspaces", "composer", "view"], true);
addFn(WS, "setDocCount", 97, 108, "Enables document actions when at least one document is indexed and explains why they are unavailable otherwise.", ["workspaces", "documents", "view"], true);
addFn(WS, "setUploadBusy", 111, 117, "Marks the document drop zone busy and shows the file name while an upload is in progress.", ["workspaces", "upload", "view"], true);
addFn(WS, "setLimits", 121, 127, "Appends the per-file size limit to the drop-zone hint.", ["workspaces", "upload", "view"], true);
addFn(WS, "_wireDropzone", 129, 142, "Highlights the drop zone during a drag and routes a click to the composer's existing attach button.", ["workspaces", "upload", "event-handler"], false);
addFn(WS, "init", 144, 156, "Binds welcome-screen actions, the drop zone, add-URL, and manage-documents, then starts with document actions disabled.", ["workspaces", "initialization", "event-handler"], true);

addFile(IC, "Sprite-icon helper over the inline SVG symbols in index.html. Unknown names, including legacy emoji from the app, fall back to escaped text.", ["utility", "svg", "accessibility", "icons"], "moderate");
addFn(IC, "resolve", 38, 47, "Maps an icon name or legacy emoji to a sprite id that is present in the document, or returns null.", ["icons", "utility", "svg"], false);
addFn(IC, "has", 49, 49, "Reports whether a name resolves to an icon currently in the sprite.", ["icons", "utility", "svg"], true);
addFn(IC, "html", 62, 66, "Returns decorative SVG markup for a known icon, or escaped text when the name is unknown.", ["icons", "svg", "accessibility"], true);
addFn(IC, "el", 69, 81, "Returns an SVG element for a known icon, or a text node for an unknown name.", ["icons", "svg", "dom"], true);

addFile(IN, "Introduction page that reveals the chatbot shell, toggles the mobile nav, and switches theme through the shared Theme store. It makes no network calls.", ["entry-point", "view", "theme", "navigation"], "moderate");
addFn(IN, "showApp", 38, 47, "Switches the page to the chatbot, clears the intro inert state, and can focus the composer.", ["view", "navigation", "entry-point"], false);
addFn(IN, "boot", 55, 100, "Syncs the intro theme button, wires menu, start, Escape, and hash routing between the intro and the app.", ["initialization", "event-handler", "theme"], false);

addFile(TH, "Persists the light, dark, or system theme preference, migrates the legacy stored value, and applies the resolved theme to the document and theme-color meta tag.", ["theme", "persistence", "utility", "accessibility"], "moderate", "Migrates localStorage theme to version 2 so a previously stored dark default becomes system, then follows prefers-color-scheme.");
addFn(TH, "resolved", 49, 51, "Returns the theme currently painted on the document element, light or dark.", ["theme", "view", "utility"], true);
addFn(TH, "get", 53, 53, "Returns the stored preference: light, dark, or system.", ["theme", "persistence", "utility"], true);
addFn(TH, "_apply", 55, 65, "Writes the resolved theme onto the document, color-scheme, theme-color meta, and change listeners.", ["theme", "dom", "utility"], false);
addFn(TH, "set", 68, 73, "Normalizes a preference, stores it, and applies the resolved theme.", ["theme", "persistence", "utility"], true);
addFn(TH, "onChange", 76, 82, "Subscribes to preference changes and returns an unsubscribe function.", ["theme", "event-handler", "utility"], true);
addFn(TH, "init", 85, 104, "Migrates storage, applies the saved preference once, and listens for OS theme changes and other tabs.", ["theme", "initialization", "persistence"], true);

addFile(UI, "Shell renderer for the sidebar, session list, composer menus, upload toasts, citation and URL dialogs, and settings hand-off. It paints DOM only and reaches sibling views through window globals.", ["entry-point", "component", "view", "shell"], "complex", "Assigned to window.UI from an IIFE. Sibling modules are globals rather than ES imports, so the shell can call Theme, Utils, Workspaces, and SettingsView without a module graph.");
addFn(UI, "_syncThemeButton", 18, 27, "Mirrors the resolved theme onto the sidebar light, dark, and quick-toggle buttons.", ["theme", "view", "accessibility"], false);
addFn(UI, "initTheme", 30, 33, "Initializes the Theme store and paints the sidebar theme buttons from the saved preference.", ["theme", "initialization", "view"], true);
addFn(UI, "setTheme", 36, 38, "Persists a light, dark, or system preference by delegating to Theme.set.", ["theme", "view", "utility"], true);
addFn(UI, "isSidebarOpen", 67, 70, "Reports whether the narrow-screen sidebar drawer is open.", ["sidebar", "view", "shell"], true);
addFn(UI, "_syncSidebarChrome", 87, 102, "Updates sidebar toggle labels, icons, and expanded state for the drawer and rail shells.", ["sidebar", "view", "accessibility"], false);
addFn(UI, "openSidebar", 104, 118, "Opens the sidebar drawer, shows the backdrop, and moves focus into the sidebar.", ["sidebar", "view", "accessibility"], true);
addFn(UI, "closeSidebar", 121, 136, "Closes the sidebar drawer, clears inert on the main region, and can restore focus to the toggle.", ["sidebar", "view", "accessibility"], true);
addFn(UI, "setSidebarCollapsed", 142, 148, "Switches the wide layout between the icon rail and the expanded sidebar.", ["sidebar", "view", "shell"], true);
addFn(UI, "isSidebarCollapsed", 150, 150, "Reports whether the shell is currently the collapsed icon rail.", ["sidebar", "view", "shell"], true);
addFn(UI, "toggleSidebar", 152, 155, "Opens or closes the drawer on a narrow viewport, and otherwise toggles the rail.", ["sidebar", "event-handler", "view"], true);
addFn(UI, "_trapSidebarTab", 164, 176, "Keeps Tab focus inside the open sidebar drawer.", ["sidebar", "accessibility", "keyboard"], false);
addFn(UI, "renderSessionList", 206, 301, "Groups chat sessions by date, paints rows with rename and delete actions, and restores focus after a re-render.", ["sessions", "view", "sidebar"], true);
addFn(UI, "setSessionsStatus", 307, 349, "Replaces the session list with a loading skeleton or an error panel that can offer retry.", ["sessions", "view", "status"], true);
addFn(UI, "_beginRename", 352, 399, "Turns a session row into an inline rename field and commits or reverts the title.", ["sessions", "event-handler", "view"], false);
addFn(UI, "renderDocList", 404, 406, "Forwards the document list to DocsView.render.", ["documents", "view", "delegation"], true);
addFn(UI, "setDocsStatus", 409, 411, "Forwards document-list loading and error state to DocsView.", ["documents", "view", "status"], true);
addFn(UI, "renderModelSelect", 415, 448, "Fills the composer model select, surfaces an error or retry, and asks Settings to mirror the same options.", ["models", "view", "settings"], true);
addFn(UI, "_clearUploadError", 454, 471, "Resets the upload toast back to its idle progress chrome.", ["upload", "view", "toast"], false);
addFn(UI, "showUploadToast", 473, 481, "Shows the in-progress upload toast and marks the workspace drop zone busy.", ["upload", "toast", "view"], true);
addFn(UI, "hideUploadToast", 483, 495, "Hides the upload toast, clears a busy drop zone, and restores focus.", ["upload", "toast", "view"], true);
addFn(UI, "showUploadError", 503, 556, "Shows a failed upload in the toast, announces it, and records an error notification.", ["upload", "toast", "error-handling"], true);
addFn(UI, "_toastRegion", 564, 574, "Ensures the live toast stack exists in the document.", ["toast", "view", "dom"], false);
addFn(UI, "showToast", 583, 634, "Pushes a typed toast, honors the errors-only preference, and records it in the notification history.", ["toast", "view", "notifications"], true);
addFn(UI, "setLiveStatus", 641, 647, "Writes a polite live-region message for screen readers.", ["accessibility", "view", "status"], true);
addFn(UI, "showCitationModal", 651, 678, "Opens the citation dialog with a safe link and the source snippet.", ["citations", "dialog", "view"], true);
addFn(UI, "hideCitationModal", 680, 682, "Closes the citation dialog.", ["citations", "dialog", "view"], true);
addFn(UI, "showUrlModal", 686, 692, "Opens the add-URL dialog and clears its fields when it was closed.", ["dialog", "documents", "view"], true);
addFn(UI, "hideUrlModal", 694, 696, "Closes the add-URL dialog.", ["dialog", "documents", "view"], true);
addFn(UI, "showGithubModal", 698, 710, "Opens the GitHub connect dialog and clears URL and token fields when it was closed.", ["dialog", "github", "view"], true);
addFn(UI, "hideGithubModal", 712, 714, "Closes the GitHub connect dialog.", ["dialog", "github", "view"], true);
addFn(UI, "setGithubOauth", 716, 719, "Shows or hides the GitHub OAuth button based on server readiness.", ["github", "view", "settings"], true);
addFn(UI, "confirmDialog", 727, 729, "Asks for confirmation through the shared dialog and resolves true only when the user confirms.", ["dialog", "view", "accessibility"], true);
addFn(UI, "initDropzone", 733, 761, "Tracks page-level file drags and delivers dropped files to the upload handler.", ["upload", "event-handler", "view"], true);
addFn(UI, "onModeChange", 781, 783, "Registers the handler invoked when the user picks a workspace mode.", ["composer", "event-handler", "workspaces"], true);
addFn(UI, "toggleModeMenu", 785, 801, "Opens or closes the workspace mode menu anchored to the composer.", ["composer", "menu", "workspaces"], false);
addFn(UI, "toggleComposerMenu", 811, 837, "Opens or closes the composer plus menu for upload, URL, and documents.", ["composer", "menu", "documents"], false);
addFn(UI, "_wireComposerMenu", 839, 849, "Opens the composer plus menu from the keyboard when it is closed.", ["composer", "keyboard", "menu"], false);
addFn(UI, "_wireModeSelect", 852, 862, "Opens the mode menu from the keyboard when it is closed.", ["composer", "keyboard", "menu"], false);
addFn(UI, "setActiveMode", 874, 905, "Paints the active workspace in the composer and welcome header and notifies Workspaces and code preview.", ["workspaces", "composer", "view"], true);
addFn(UI, "setUserIdDisplay", 910, 919, "Shows the user id in the top bar and forwards it to the settings profile pane.", ["profile", "view", "settings"], true);
addFn(UI, "setVizState", 928, 930, "Sets the welcome illustration state through AIViz.", ["view", "animation", "welcome"], true);
addFn(UI, "_initViewWiring", 934, 1024, "Binds sidebar, welcome cards, Escape, and the first-run wiring for documents, workspaces, and settings.", ["initialization", "event-handler", "shell"], false);
addFn(UI, "onMyDocuments", 1031, 1033, "Registers the handler for the My Documents composer action.", ["documents", "event-handler", "composer"], true);
addFn(UI, "focusSessionSearch", 1036, 1044, "Opens the sidebar on a narrow screen and focuses the session search field.", ["sessions", "accessibility", "sidebar"], true);
addFn(UI, "_wireReferenceShell", 1053, 1091, "Binds recents, the session date filter, theme switches, and search for the reference shell.", ["shell", "event-handler", "theme"], false);
addFn(UI, "_wireHeroLook", 1098, 1122, "Tilts the welcome illustration toward the pointer when reduced motion is off.", ["animation", "welcome", "view"], false);
addFn(UI, "setUploadLimits", 1125, 1127, "Forwards the maximum upload size to the workspace drop zone.", ["upload", "workspaces", "view"], true);
addFn(UI, "openSettings", 1130, 1132, "Opens the settings dialog on an optional pane.", ["settings", "dialog", "view"], true);
addFn(UI, "onSettingsPane", 1135, 1137, "Forwards the settings pane callback to SettingsView.", ["settings", "event-handler", "view"], true);
addFn(UI, "renderIntegrations", 1140, 1142, "Forwards health-service status to the settings integrations pane.", ["settings", "integrations", "view"], true);
addFn(UI, "renderGithubRepos", 1144, 1148, "Forwards connected repositories to the settings GitHub list.", ["settings", "github", "view"], true);
addFn(UI, "openDocuments", 1150, 1152, "Opens the documents dialog.", ["documents", "dialog", "view"], true);
addFn(UI, "notify", 1155, 1157, "Records a message in the notification history.", ["notifications", "view", "utility"], true);
addFn(UI, "clearNotifications", 1159, 1161, "Clears the notification history.", ["notifications", "view", "utility"], true);
addFn(UI, "onWelcomeCard", 1167, 1169, "Registers the handler for welcome-card mode and prompt actions.", ["welcome", "event-handler", "composer"], true);
addFn(UI, "focusComposer", 1172, 1179, "Focuses the chat input when it is visible and enabled.", ["composer", "accessibility", "view"], true);
addFn(UI, "setComposerBusy", 1187, 1197, "Swaps Send and Stop while a request runs and keeps the chat input enabled.", ["composer", "view", "accessibility"], true);

addFile(UT, "Markdown, artifact, and code-preview helpers. Chat HTML is sanitized with DOMPurify before citation badges and highlight.js chrome are added.", ["utility", "sanitization", "markdown", "security"], "complex", "A DOMPurify uponSanitizeAttribute hook keeps only a validated language-* class on CODE elements and is removed in a finally block.");
addFn(UT, "safeUrl", 58, 66, "Allows only http and https URLs and returns null for anything else.", ["security", "sanitization", "utility"], true);
addFn(UT, "applyLinkImagePolicy", 74, 117, "Forces safe link targets, wraps tables, and drops images whose sources are not http(s).", ["security", "sanitization", "dom"], false);
addFn(UT, "decorateCitations", 124, 166, "Turns bracketed citation markers in sanitized HTML into buttons and merges adjacent badges.", ["citations", "markdown", "dom"], false);
addFn(UT, "_artifactMarkdown", 168, 198, "Serializes an artifact document's blocks to Markdown.", ["artifacts", "markdown", "serialization"], false);
addFn(UT, "_artifactPlain", 209, 241, "Serializes an artifact document to plain text with inline markup stripped.", ["artifacts", "serialization", "utility"], false);
addFn(UT, "_artifactSheet", 243, 272, "Serializes an artifact document to an HTML table suitable for a spreadsheet download.", ["artifacts", "serialization", "html"], false);
addFn(UT, "renderArtifactHtml", 279, 290, "Renders an artifact as sanitized Markdown, plain text, or a spreadsheet table.", ["artifacts", "markdown", "sanitization"], true);
addFn(UT, "renderMarkdown", 293, 306, "Parses model Markdown, sanitizes it, and decorates citations, falling back to escaped text.", ["markdown", "sanitization", "citations"], true);
addFn(UT, "highlightCode", 308, 359, "Highlights fenced code, wraps it with a copy button, and lets CodePreview add a Run action.", ["markdown", "code", "dom"], true);
addFn(UT, "buildPreviewDoc", 369, 418, "Builds a sandboxed HTML or SVG preview document, injecting a connect-src-none content security policy.", ["security", "preview", "html"], true);
addFn(UT, "copyText", 421, 436, "Copies text with the async clipboard API and falls back to a hidden textarea.", ["utility", "clipboard", "dom"], true);
addFn(UT, "escapeHtml", 438, 444, "Escapes text for insertion into HTML.", ["security", "sanitization", "utility"], true);
addFn(UT, "parseDate", 453, 460, "Parses an ISO-like timestamp into a Date, treating naive values as UTC.", ["utility", "date", "parsing"], true);
addFn(UT, "formatTime", 462, 466, "Formats a timestamp as a short local time for today's session rows.", ["utility", "date", "formatting"], true);
addFn(UT, "formatDateGroup", 468, 480, "Buckets a timestamp into Today, Yesterday, Previous 7 Days, Previous 30 Days, or Older.", ["utility", "date", "sessions"], true);
addFn(UT, "uuid", 482, 487, "Returns a random RFC 4122 version-4 UUID string.", ["utility", "id", "random"], true);

addFile(INIT, "Empty tests package marker so pytest can import the suite.", ["test", "package", "entry-point"], "simple");

addFile(ARCH, "Guards backend layout: imports may only point at allowed lower layers, module-level imports must be acyclic, and environment reads and model ids stay in core.config.", ["test", "architecture", "validation", "imports"], "moderate", "Parses each backend module with ast and checks a downward package allow-list. The composition root and api layer are the only modules permitted to reach the DI container.");
addFn(ARCH, "_modules", 27, 34, "Maps backend dotted module names to source paths, skipping bytecode caches.", ["test", "architecture", "filesystem"], true);
addFn(ARCH, "_imports", 37, 52, "Collects backend import targets from a file's AST, optionally ignoring imports inside functions.", ["test", "imports", "ast"], true);
addFn(ARCH, "_package", 55, 57, "Returns the top-level backend package name for a dotted module.", ["test", "architecture", "utility"], true);
addFn(ARCH, "test_backend_root_only_contains_main", 60, 63, "Asserts the backend root contains only __init__.py and main.py, and that each layer package exists.", ["test", "architecture", "layout"], true);
addFn(ARCH, "test_layer_direction", 66, 78, "Asserts every import stays inside the package allow-list and that only api imports the DI container.", ["test", "architecture", "imports"], true);
addFn(ARCH, "test_no_module_level_import_cycles", 81, 99, "Depth-first searches module-level backend imports and fails when a cycle is found.", ["test", "imports", "graph"], true);
addFn(ARCH, "test_package_initialisers_do_not_import_siblings", 102, 105, "Asserts layer package __init__ files contain no import statements.", ["test", "imports", "packages"], true);
addFn(ARCH, "test_environment_only_read_in_config", 108, 115, "Asserts os.getenv and os.environ appear only in backend.core.config.", ["test", "configuration", "security"], true);
addFn(ARCH, "test_no_hardcoded_model_ids_outside_config", 118, 122, "Asserts model-id literals such as gpt-oss and gemini appear only in backend.core.config.", ["test", "configuration", "models"], true);
addFn(ARCH, "test_every_module_imports_in_a_clean_interpreter", 126, 129, "Imports each backend module in a fresh interpreter and requires a zero exit code.", ["test", "imports", "smoke"], true);

addFile(REF, "Static checks for the reference-design pass: token-only CSS, view-only frontend scripts, reduced-motion guards, and contrast of control outlines and welcome-card text.", ["test", "accessibility", "validation", "css"], "moderate");
addFn(REF, "_read", 14, 15, "Reads a frontend-relative text file as UTF-8.", ["test", "filesystem", "utility"], true);
addFn(REF, "_theme_tokens", 18, 24, "Parses light or dark color tokens from the :root blocks in main.css.", ["test", "theme", "css"], true);
addFn(REF, "_lum", 27, 30, "Converts a hex color to relative luminance for contrast checks.", ["test", "accessibility", "color"], true);
addFn(REF, "_contrast", 33, 35, "Returns the WCAG contrast ratio between two hex colors.", ["test", "accessibility", "color"], true);
addFn(REF, "test_reference_assets_exist_and_are_linked", 38, 44, "Asserts reference.css and notifications.js exist and that reference.css loads after components.css.", ["test", "css", "assets"], true);
addFn(REF, "test_reference_css_uses_tokens_only", 47, 48, "Asserts reference.css contains no raw hex colors.", ["test", "css", "design-tokens"], true);
addFn(REF, "test_every_component_and_view_js_is_view_only", 51, 59, "Asserts view scripts do not fetch, touch storage, or build notification HTML with innerHTML.", ["test", "view", "security"], true);
addFn(REF, "test_reference_css_has_no_unguarded_motion", 62, 70, "Asserts new motion stays in the reduced-motion stylesheets and reference transitions are limited to paint-safe properties.", ["test", "css", "accessibility"], true);
addFn(REF, "test_notification_hooks_exist_in_markup", 73, 78, "Asserts notification, search, recents, and composer hooks exist with accessible names in index.html.", ["test", "accessibility", "markup"], true);
addFn(REF, "test_soft_control_outline_meets_3_to_1", 82, 85, "Asserts --border-ui contrasts at least 3:1 with --surface in light and dark themes.", ["test", "accessibility", "contrast"], true);
addFn(REF, "test_welcome_card_borders_are_documented_exemption_and_text_is_readable", 88, 102, "Requires the welcome-card border exemption in the view contract and card text contrast of at least 4.5:1.", ["test", "accessibility", "contrast"], true);
addFn(REF, "test_no_unverifiable_privacy_claim_in_capability_strip", 105, 109, "Asserts the capability strip describes upload checks and does not claim unverifiable privacy.", ["test", "content", "validation"], true);

addFile(SAN, "Regression guards for the Markdown sanitizer in utils.js. Static source checks always run; a headless Chrome harness exercises DOMPurify when a browser and CDN libraries are available.", ["test", "sanitization", "security", "markdown"], "complex");
addFn(SAN, "_forbid_attr", 22, 25, "Parses the FORBID_ATTR list from utils.js.", ["test", "sanitization", "parsing"], true);
addFn(SAN, "_hook_body", 28, 31, "Extracts the keepLanguageClass function body from utils.js.", ["test", "sanitization", "parsing"], true);
addFn(SAN, "test_forbid_attr_contains_required_attributes", 34, 36, "Asserts dangerous and layout attributes, including class, style, and srcset, are forbidden.", ["test", "sanitization", "security"], true);
addFn(SAN, "test_language_hook_is_scoped_to_code_and_validated", 39, 46, "Asserts the sanitize hook keeps only an anchored language-* class on CODE elements.", ["test", "sanitization", "security"], true);
addFn(SAN, "test_hook_registered_only_around_sanitize_and_always_removed", 49, 57, "Asserts the language hook is added only around DOMPurify.sanitize and removed in a finally block.", ["test", "sanitization", "security"], true);
addFn(SAN, "test_markdown_goes_through_sanitizer_wrapper", 60, 62, "Asserts marked.parse output is passed through sanitizeMarkdownHtml and not returned raw.", ["test", "markdown", "sanitization"], true);
addFn(SAN, "test_artifact_preview_rebuilds_markdown_like_chat", 65, 69, "Asserts artifact HTML is rebuilt through the same Markdown sanitizer, with text and spreadsheet branches.", ["test", "artifacts", "sanitization"], true);
addFn(SAN, "test_buildPreviewDoc_is_exported", 72, 76, "Asserts buildPreviewDoc is returned from the utils IIFE.", ["test", "preview", "exports"], true);
addFn(SAN, "test_buildPreviewDoc_handles_all_three_branches", 79, 89, "Asserts the preview builder has full-document, SVG, and CSP branches and the expected filenames.", ["test", "preview", "security"], true);
addFn(SAN, "test_code_preview_js_exports_required_api", 92, 99, "Asserts code-preview.js exposes decorate, sync, open, and close and allowlists html and svg.", ["test", "preview", "exports"], true);
addFn(SAN, "test_code_preview_js_checks_chatArea", 102, 108, "Asserts decorate bails out unless the wrapper is inside #chatArea.", ["test", "preview", "security"], true);
addFn(SAN, "test_code_workspace_note_updated", 111, 114, "Asserts the Code workspace copy mentions Run and no longer says code is never run.", ["test", "content", "preview"], true);
addFn(SAN, "test_purify_config_keeps_http_only_and_forbids_active_tags", 117, 121, "Asserts the purifier allows only http(s) URIs and forbids active tags such as script, iframe, and form.", ["test", "sanitization", "security"], true);
addFn(SAN, "_chrome", 125, 130, "Returns a headless Chrome or Chromium binary from PATH, if one is installed.", ["test", "browser", "utility"], true);
addFn(SAN, "browser_result", 225, 249, "Runs the sanitizer harness in headless Chrome and returns the JSON result, skipping when Chrome or the CDN libraries are missing.", ["test", "browser", "sanitization"], true);
addFn(SAN, "test_browser_attributes_and_classes_stripped", 252, 258, "Asserts the live sanitizer strips classes, roles, and data attributes while keeping safe links.", ["test", "browser", "sanitization"], true);
addFn(SAN, "test_browser_language_hook_cannot_be_abused", 261, 266, "Asserts a hostile class cannot survive the language hook and that the hook is not left registered.", ["test", "browser", "security"], true);
addFn(SAN, "test_browser_features_still_work", 269, 275, "Asserts highlighting, citation badges, safe link targets, and the copy button still survive sanitizing.", ["test", "browser", "markdown"], true);
addFn(SAN, "test_browser_buildPreviewDoc_html_fragment", 278, 284, "Asserts an HTML fragment preview is wrapped as a document named preview.html with a CSP meta tag.", ["test", "browser", "preview"], true);
addFn(SAN, "test_browser_buildPreviewDoc_svg", 287, 290, "Asserts an SVG preview uses preview.svg and includes the centring stylesheet.", ["test", "browser", "preview"], true);
addFn(SAN, "test_browser_buildPreviewDoc_full_document", 293, 296, "Asserts a full HTML document keeps its title and receives a CSP injection in the existing head.", ["test", "browser", "preview"], true);
addFn(SAN, "test_browser_run_button_scoping", 299, 304, "Asserts the Run button appears once for HTML inside the chat area and never for Python or blocks outside it.", ["test", "browser", "preview"], true);

dep(ST, RG);
dep(ST, TH);
dep(ST, IC);
dep(ST, UT);
dep(ST, UI);
dep(WS, UI);
dep(IN, TH);
dep(UT, IC);

call(ST, "_syncAll", RG, "set");
call(ST, "_syncAll", TH, "get");
call(ST, "renderIntegrations", IC, "el");
call(ST, "init", RG, "wire");
call(ST, "init", TH, "set");
call(ST, "init", TH, "onChange");
call(ST, "init", RG, "set");
call(ST, "init", UI, "setLiveStatus");
call(ST, "init", UT, "copyText");
call(ST, "init", UI, "showToast");
call(WS, "_onClick", UI, "setLiveStatus");
call(WS, "_wireDropzone", UI, "setLiveStatus");
call(IN, "boot", TH, "onChange");
call(IN, "boot", TH, "set");
call(IN, "boot", TH, "resolved");
call(UT, "highlightCode", IC, "html");

testedBy(RG, REF);
testedBy(ST, REF);
testedBy(WS, REF);
testedBy(IC, REF);
testedBy(TH, REF);
testedBy(UT, REF);
testedBy(UT, SAN);
testedBy(IC, SAN);

function edgesFrom(filePath) {
  return edges.filter((e) => e.source === "file:" + filePath || e.source.startsWith("function:" + filePath + ":")).length;
}

const uiCalls = [
  ["initTheme", TH, "init"],
  ["setTheme", TH, "set"],
  ["_wireReferenceShell", TH, "set"],
  ["renderSessionList", UT, "escapeHtml"],
  ["showCitationModal", UT, "safeUrl"],
  ["renderModelSelect", ST, "syncModels"],
  ["_initViewWiring", ST, "init"],
  ["_initViewWiring", WS, "init"],
  ["setActiveMode", WS, "setActive"],
  ["showUploadToast", WS, "setUploadBusy"],
  ["openSettings", ST, "open"],
  ["setUserIdDisplay", ST, "setUserId"],
  ["renderIntegrations", ST, "renderIntegrations"],
  ["setUploadLimits", WS, "setLimits"],
  ["onSettingsPane", ST, "onPane"],
  ["renderGithubRepos", ST, "renderGithubRepos"],
  ["hideUploadToast", WS, "setUploadBusy"],
];

const uiDeps = [TH, IC, UT, WS, ST];
for (const target of uiDeps) {
  if (edgesFrom(UI) >= 120) break;
  dep(UI, target);
}
if (edgesFrom(UI) < 120) testedBy(UI, REF);
let uiCallsKept = 0;
for (const [fromFn, toFile, toFn] of uiCalls) {
  if (edgesFrom(UI) >= 120) break;
  call(UI, fromFn, toFile, toFn);
  uiCallsKept++;
}

const idSet = new Set();
for (const n of nodes) {
  if (idSet.has(n.id)) throw new Error("duplicate id " + n.id);
  idSet.add(n.id);
  if (!n.summary || !n.summary.trim()) throw new Error("empty summary " + n.id);
  if (!Array.isArray(n.tags) || n.tags.length < 3 || n.tags.length > 5) throw new Error("tags " + n.id + " " + JSON.stringify(n.tags));
  if (!["simple", "moderate", "complex"].includes(n.complexity)) throw new Error("complexity " + n.id);
  for (const t of n.tags) {
    if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(t)) throw new Error("bad tag " + t + " on " + n.id);
  }
}

const importEdges = edges.filter((e) => e.type === "imports");
if (importEdges.length !== 0) throw new Error("expected 0 imports, got " + importEdges.length);

for (const e of edges) {
  if (e.source === e.target) throw new Error("self edge " + e.source);
  if (!idSet.has(e.source)) throw new Error("missing source " + e.source);
  if (!idSet.has(e.target)) throw new Error("missing target " + e.target);
  if (e.direction !== "forward") throw new Error("direction " + e.type);
}

const expectedFiles = [RG, ST, WS, IC, IN, TH, UI, UT, INIT, ARCH, REF, SAN];
for (const f of expectedFiles) {
  if (!nodes.some((n) => n.id === "file:" + f)) throw new Error("missing file node " + f);
}

function groupsFor(partCount) {
  const files = expectedFiles.slice().sort();
  const chunk = Math.ceil(files.length / partCount);
  const groups = [];
  for (let i = 0; i < files.length; i += chunk) groups.push(files.slice(i, i + chunk));
  return groups;
}

function stats(groups) {
  return groups.map((g) => {
    const ns = nodes.filter((n) => g.includes(n.filePath));
    const ids = new Set(ns.map((n) => n.id));
    const es = edges.filter((e) => ids.has(e.source));
    return { files: g, nodes: ns.length, edges: es.length, ns, es };
  });
}

let partCount = Math.ceil(Math.max(nodes.length / 60, edges.length / 120));
let groups = groupsFor(partCount);
while (stats(groups).some((s) => s.nodes > 60 || s.edges > 120) && partCount < expectedFiles.length) {
  partCount += 1;
  groups = groupsFor(partCount);
}
const finalStats = stats(groups);
if (finalStats.some((s) => s.nodes > 60 || s.edges > 120)) {
  console.log(JSON.stringify(finalStats.map((s) => ({ files: s.files, nodes: s.nodes, edges: s.edges })), null, 2));
  throw new Error("a part still exceeds 60 nodes or 120 edges");
}

fs.mkdirSync(outDir, { recursive: true });
const written = [];
finalStats.forEach((s, i) => {
  const file = path.join(outDir, "batch-12-part-" + (i + 1) + ".json");
  const ids = new Set(s.ns.map((n) => n.id));
  for (const e of s.es) {
    const targetOk = ids.has(e.target) || idSet.has(e.target);
    if (!targetOk) throw new Error("bad target in part " + (i + 1) + " " + e.target);
  }
  fs.writeFileSync(file, JSON.stringify({ nodes: s.ns, edges: s.es }, null, 2));
  written.push({ file: path.basename(file), nodes: s.ns.length, edges: s.es.length, files: s.files });
});

console.log(JSON.stringify({
  nodeCount: nodes.length,
  edgeCount: edges.length,
  parts: written.length,
  uiEdges: edgesFrom(UI),
  uiCallsKept,
  written,
}, null, 2));
