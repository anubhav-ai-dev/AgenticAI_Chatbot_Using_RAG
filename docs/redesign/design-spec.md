# AI Assistant Pro — Design Spec (Phase 1)

Author: ui-designer. Date: 2026-10-20. Audience: `frontend-developer` (implements), `integration-developer`
(wires controllers), `qa-reviewer` (checks). Repo: vanilla HTML/CSS/JS, no build step.

> **Reconciliation warnings (read first)**
> 1. `docs/redesign/audit.md` did **not exist** when this spec was written (only `PROJECT_BRIEF.md` and `baseline/`).
>    Everything here was derived from the brief plus a read-only pass over `frontend/index.html`, `css/main.css`,
>    `js/ui.js`, `js/chat.js`, `js/utils.js` and grep of `js/app.js`. Items marked **[verify-with-audit]** are
>    assumptions to reconcile once the audit lands. If the audit disagrees with this spec on a *fact*, the audit wins.
> 2. `docs/redesign/references/` does **not exist**. The master prompt says visual references were supplied; none are in
>    the repo. This design follows the **written spec only**. I did not see any reference images and make no claim of
>    matching them. If the user has references, a short review pass of this spec against them is needed.
> 3. No Figma file was created (see section 11).

---------------------------------------------------------------------------------------------------

## 1. Principles and scope

**Decides:** palette and verified contrast, type/space/radius/motion scales, theme mechanics, layout and responsive
rules, component list with states/ARIA, screen specs, AI visualization behavior, icon plan.

**Leaves to implementation:** exact class names (except where an id/class is a stable contract), file-internal structure,
copy beyond what is specified, pixel nudging within the tokens given.

Principles:
1. **Two regions only**: left sidebar + main workspace. No permanent right panel, no persistent model panel. Everything
   else (model list, menus, theme choice, source detail) is a transient popover or dialog.
2. **Quiet by default**: flat surfaces separated by 1px borders. Shadows only on things that float (menus, popovers,
   dialogs, toasts, mobile drawer). No gradients, blur, glow, or per-card colors. One accent hue (indigo) plus semantic
   success/warning/error. Cyan appears only inside the AI visualization.
3. **Honest data only**: nothing is shown that the API cannot supply (section 10).
4. **Never color alone**: every state (selected, on/off, error, active) also has an icon, text, weight or shape change.
5. **Keep it bootable**: all ids bound by `app.js` survive (section 12 lists them); markup moves, ids stay.
6. **Respect the seam**: views never call `API.*` or touch `State`; where a view needs a controller change this spec
   names it as a **cross-boundary request** (collected in 12.4).

---------------------------------------------------------------------------------------------------

## 2. Tokens

### 2.1 Method for contrast
WCAG 2.x relative luminance: each sRGB channel `c/255`, linearised (`c <= .03928 ? c/12.92 : ((c+.055)/1.055)^2.4`),
`L = .2126R + .7152G + .0722B`, ratio `(L_light+.05)/(L_dark+.05)`. Computed with `python3 -I` on 2026-10-20 for every pair
below. Targets: body text 4.5:1; large text (>=24px, or >=18.66px bold) and UI boundaries / focus rings / meaningful icons 3:1.
Decorative dividers and disabled controls are exempt (WCAG 1.4.3/1.4.11), stated explicitly where used.
Translucent tints were composited to a solid hex first (stated in the table).

### 2.2 Palette changes from the master prompt (and why)

| Token | Suggested | Final | Reason |
|---|---|---|---|
| light `--text-secondary` | `#667085` | **`#586277`** | Confirmed failures: 4.45 on sidebar `#F0F2FA`, 4.15 on hover `#E7EAF6`, 4.04 on active `#E0E7FF`. New value is 5.48 / 5.10 / 4.97 (lowest case 4.97 on active). |
| light placeholder | (none) | **`#6B7488`** | 4.69 on white. **Rule:** placeholders only on white inputs (4.30 on `#F3F5FB` fails). Inputs never use the subtle bg. |
| light control border | `#E1E6F0` | **`#7C8699`** for input/button/chip boundaries | `#E1E6F0` is 1.25:1 on white, fails 3:1 for UI boundaries. Kept `#E1E6F0` only as the *decorative* divider `--border`. |
| light success text | `#087F68` | **`#067059`** | 4.42 on its own 8% tint (fail); new 5.41. |
| light warning text | `#A16207` | **`#8F5706`** | 4.40 on tint (fail); new 5.31. |
| light error | `#B42318` | unchanged | 5.75 on tint. |
| dark `--accent` as button fill | `#8B83FF` with white text | **`#8B83FF` with ink `#0B1020` text** | White on `#8B83FF` is 3.09 (fail). Ink is 6.12 (hover `#A49EFF`: 8.04). |
| dark accent **as text on active sidebar row** | `#8B83FF` | use `--text` for the label; accent only for icon + 3px bar | `#8B83FF` on `#26345B` is 3.94 (fails text, passes 3:1 for icons). |
| dark link/accent text | `#8B83FF` | new token `--accent-text` = **`#A49EFF`** | >= 5.17 on every dark surface incl. the active row. |
| dark control border | `#2B3850` | **`#66779A`** for control boundaries | `#2B3850` is 1.50 on surface. Kept as decorative `--border`. |
| dark placeholder | (none) | **`#8A97B0`** | 5.39 on raised, 6.01 on surface. |
| dark disabled text | (none) | `#66738E` | 3.71 on surface (exempt; kept >= 3:1 on surface). |
| dark status text | (none) | `#34D3A7` / `#F5B942` / `#FF8A80` | 6.16–10.02 on surface/raised/tints. |

Everything else from the master prompt is kept as given (including `#4F46E5`, `#4338CA`, `#F7F8FC`, `#F0F2FA`, `#E7EAF6`,
`#E0E7FF`, `#172033`, `#0B1020`, `#11182A`, `#0F1728`, `#182238`, `#1A2840`, `#26345B`, `#F1F5FF`, `#B0BDD3`).

### 2.3 Color tokens (CSS custom properties)

Declared on `:root` (light) and overridden on `:root[data-theme="dark"]`. Both live in `css/main.css` (the file must keep its name).

| Token | Light | Dark | Use |
|---|---|---|---|
| `--app-bg` | `#F7F8FC` | `#0B1020` | page background behind workspace |
| `--surface` | `#FFFFFF` | `#11182A` | workspace, topbar, cards, dialogs (light), inputs (light) |
| `--surface-raised` | `#FFFFFF` | `#182238` | menus, popovers, dialogs, composer box, toasts, inputs (dark) |
| `--surface-subtle` | `#F3F5FB` | `#162036` | hover on main-area rows, chips, kbd, table header |
| `--sidebar-bg` | `#F0F2FA` | `#0F1728` | sidebar |
| `--sidebar-hover` | `#E7EAF6` | `#1A2840` | sidebar row hover |
| `--sidebar-active` | `#E0E7FF` | `#26345B` | sidebar current row |
| `--accent` | `#4F46E5` | `#8B83FF` | primary fill, selected indicators, focus ring, icon accents |
| `--accent-hover` | `#4338CA` | `#A49EFF` | primary hover |
| `--accent-pressed` | `#3730A3` | `#7A71F5` | primary active |
| `--on-accent` | `#FFFFFF` | `#0B1020` | text/icon on `--accent` fills |
| `--accent-subtle` | `#EEF0FF` | `#25294C` | user bubble, icon tile, selected option bg |
| `--accent-text` | `#4F46E5` | `#A49EFF` | links, accent-colored text |
| `--text` | `#172033` | `#F1F5FF` | primary text |
| `--text-secondary` | `#586277` | `#B0BDD3` | descriptions, metadata, timestamps, icon-button glyphs |
| `--text-placeholder` | `#6B7488` | `#8A97B0` | placeholders (only on `--surface` light / any dark input surface) |
| `--text-disabled` | `#8A94A8` | `#66738E` | disabled labels (exempt from 4.5:1) |
| `--border` | `#E1E6F0` | `#2B3850` | decorative dividers, card outlines, sidebar edge |
| `--border-control` | `#7C8699` | `#66779A` | input/button/checkbox/chip boundaries (3:1) |
| `--ring` | `#4F46E5` | `#8B83FF` | focus ring |
| `--success` / `--success-bg` | `#067059` / `#E8F5F2` | `#34D3A7` / `#16323C` | text+icon / tint |
| `--warning` / `--warning-bg` | `#8F5706` / `#FBF1DF` | `#F5B942` / `#312F2D` | |
| `--error` / `--error-bg` | `#B42318` / `#FDECEA` | `#FF8A80` / `#322836` | |
| `--danger-fill` / `--danger-fill-hover` | `#B42318` / `#912018` | `#F97066` / `#FF8A80` | destructive buttons (text = `--on-danger`) |
| `--on-danger` | `#FFFFFF` | `#0B1020` | |
| `--tooltip-bg` / `--tooltip-text` | `#172033` / `#FFFFFF` | `#F1F5FF` / `#0B1020` | |
| `--scrim` | `rgba(23,32,51,.48)` | `rgba(3,6,14,.64)` | dialog/drawer backdrop |
| `--code-bg` / `--code-header` | `#F3F5FB` / `#EDF0F8` | `#0D1424` / `#131B30` | code blocks |
| `--viz-cyan` | `#0E7490` | `#22D3EE` | AI visualization secondary only |

Syntax tokens (replace the swapped highlight.js CDN theme, see 3.6). Contrast on `--code-bg`:

| Token | Light | ratio | Dark | ratio |
|---|---|---|---|---|
| `--syn-default` | `#172033` | 14.92 | `#E6EBF7` | 15.39 |
| `--syn-keyword` | `#6D28D9` | 6.52 | `#C4B5FD` | 9.95 |
| `--syn-string` | `#0B6B4F` | 5.96 | `#6EE7B7` | 12.06 |
| `--syn-number` | `#B54708` | 4.98 | `#FDBA74` | 10.90 |
| `--syn-comment` | `#5B6578` | 5.38 | `#8A97B0` | 6.24 |
| `--syn-title` (functions/classes) | `#1D4ED8` | 6.15 | `#93C5FD` | 10.19 |
| `--syn-builtin` | `#0E7490` | 4.92 | `#67E8F9` | 12.68 |
| `--syn-attr` | `#9A3412` | 6.70 | `#FCA5A5` | 9.68 |
| `--syn-tag` | `#BE185D` | 5.54 | `#F9A8D4` | 10.13 |
| `--syn-deletion` | `#B42318` | 6.03 | `#FF8A80` | 8.05 |
Additions (`hljs-addition`) use `--syn-string`; `hljs-meta` uses `--text-secondary`.

### 2.4 Contrast table (all computed 2026-10-20; "**FAIL**" rows are intentional rules, not used)

Light theme
| Pair (foreground) | Target | Backgrounds and ratios |
|---|---|---|
| text `#172033` | 4.5 | app 15.33; surface 16.27; sidebar 14.55; side-hover 13.56; side-active 13.20; subtle 14.92; accent-subtle 14.37 — pass |
| text-secondary `#586277` | 4.5 | app 5.77; surface 6.13; sidebar 5.48; side-hover 5.10; side-active 4.97; subtle 5.62; accent-subtle 5.41 — pass |
| text-placeholder `#6B7488` | 4.5 | surface 4.69 pass; subtle 4.30 **FAIL (never used there)** |
| accent-text `#4F46E5` | 4.5 | app 5.92; surface 6.29; sidebar 5.62; side-hover 5.24; side-active 5.10; subtle 5.77; accent-subtle 5.55 — pass |
| accent-hover `#4338CA` as link | 4.5 | 5 surfaces 6.41–7.90 — pass |
| `--on-accent` `#FFFFFF` on accent / hover / pressed | 4.5 | 6.29 / 7.90 / 9.93 — pass |
| white on danger `#B42318` / hover `#912018` | 4.5 | 6.57 / 8.66 — pass |
| border-control `#7C8699` | 3 | app 3.46; surface 3.67; sidebar 3.28; subtle 3.36 — pass |
| ring `#4F46E5` | 3 | app 5.92; surface 6.29; sidebar 5.62; side-hover 5.24; side-active 5.10 — pass |
| success `#067059` | 4.5 | surface 6.05; tint `#E8F5F2` 5.41 — pass |
| warning `#8F5706` | 4.5 | surface 5.94; tint `#FBF1DF` 5.31 — pass |
| error `#B42318` | 4.5 | surface 6.57; app 6.19; tint `#FDECEA` 5.75 — pass |
| tooltip white on `#172033` | 4.5 | 16.27 — pass |
| disabled `#8A94A8` | exempt | surface 3.05 (informational) |
| divider `#E1E6F0` | exempt (decorative) | surface 1.25; sidebar 1.12 |
| viz cyan `#0E7490` (decorative) | 3 | app 5.05; surface 5.36 |
| *Original suggestion* `#667085` | 4.5 | surface 4.97; app 4.69; sidebar **4.45 FAIL**; hover **4.15 FAIL**; active **4.04 FAIL** |
| *Original suggestion* `#E1E6F0` as control border | 3 | surface **1.25 FAIL** |

Dark theme
| Pair (foreground) | Target | Backgrounds and ratios |
|---|---|---|
| text `#F1F5FF` | 4.5 | app 17.35; surface 16.20; sidebar 16.40; raised 14.53; side-hover 13.54; side-active 11.16; subtle 14.87; accent-subtle 12.83 — pass |
| text-secondary `#B0BDD3` | 4.5 | app 9.98; surface 9.32; sidebar 9.43; raised 8.35; side-hover 7.78; side-active 6.42; subtle 8.55; accent-subtle 7.38 — pass |
| text-placeholder `#8A97B0` | 4.5 | surface 6.01; raised 5.39; subtle 5.51 — pass |
| accent-text `#A49EFF` | 4.5 | app 8.04; surface 7.50; sidebar 7.60; raised 6.73; side-hover 6.27; side-active 5.17; subtle 6.89; accent-subtle 5.94 — pass |
| `--accent` `#8B83FF` as text | 4.5 | passes on app/surface/sidebar/raised/hover/subtle (4.78–6.12), accent-subtle 4.52; **FAIL on side-active 3.94 (use as icon/bar only; passes 3:1)** |
| `--on-accent` ink `#0B1020` on accent / hover / pressed | 4.5 | 6.12 / 8.04 / 4.97 — pass |
| white on `#8B83FF` | 4.5 | 3.09 **FAIL (rejected)** |
| ink on danger `#F97066` / hover `#FF8A80` | 4.5 | 6.80 / 8.29 — pass |
| border-control `#66779A` | 3 | app 4.21; surface 3.93; sidebar 3.98; raised 3.53; subtle 3.61 — pass |
| ring `#8B83FF` | 3 | app 6.12; surface 5.71; sidebar 5.78; raised 5.12; side-hover 4.78; side-active 3.94; accent-subtle 4.52 — pass |
| success `#34D3A7` | 4.5 | surface 9.28; raised 8.32; tint `#16323C` 7.09 — pass |
| warning `#F5B942` | 4.5 | surface 10.02; raised 8.98; tint `#312F2D` 7.56 — pass |
| error `#FF8A80` | 4.5 | surface 7.74; raised 6.94; tint `#322836` 6.16 — pass |
| tooltip ink `#0B1020` on `#F1F5FF` | 4.5 | 17.35 — pass |
| disabled `#66738E` | exempt | surface 3.71; raised 3.33 |
| divider `#2B3850` | exempt | surface 1.50; sidebar 1.52 |
| viz cyan `#22D3EE` | 3 | app 10.48; surface 9.78 |
| *Original* `#B0BDD3` on `#11182A` | 4.5 | 9.32 — pass (as brief) |

Dark tints were composited as 14% foreground over `#11182A`. Light tints are fixed hex values chosen to keep the status text >= 5.3.

### 2.5 Typography
Font: **Inter** (400/500/600) for UI, **JetBrains Mono** (400/500) for code — both already in use. Replace the CSS `@import`
in `main.css` with `<link rel="preconnect">` + `<link rel="stylesheet">` in `index.html` `<head>` (removes a render-blocking
chained request) and add `font-display: swap` via the `&display=swap` URL already used. Fallback stack
`Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`. Weight 700/800 are dropped (existing 28px/800 hero is retired).

| Role | Token | Size/Line | Weight | Letter-spacing | Color |
|---|---|---|---|---|---|
| Page heading (welcome h1, empty-state h1) | `--fs-page` | 28/36 (24/32 below 640px) | 600 | -0.01em | `--text` |
| Section heading (dialog title, panel heading) | `--fs-section` | 18/28 | 600 | 0 | `--text` |
| Subheading (card title, sidebar group) | `--fs-sub` | 15/24 | 600 | 0 | `--text` |
| Body (chat text, inputs) | `--fs-body` | 15/24 chat 15/26 | 400 | 0 | `--text` |
| Secondary (descriptions, helper text) | `--fs-secondary` | 14/20 | 400 | 0 | `--text-secondary` |
| Label (buttons, nav items, menu items, tabs) | `--fs-label` | 14/20 | 500 | 0 | `--text` |
| Small label (chips, badges, group captions) | `--fs-caption` | 12/16 | 500 | 0.01em | `--text-secondary` |
| Metadata / timestamp | `--fs-meta` | 12/16 | 400 | 0 | `--text-secondary` |
| Code inline | | 0.9em of context | 400 | | `--syn-default` on `--surface-subtle` |
| Code block | `--fs-code` | 13/20 | 400 | | `--syn-*` |

Minimum rendered size is 12px. Inputs are 16px below 768px width (prevents iOS focus-zoom). Reading width: see 4.4.
Markdown rhythm inside `.md-content`: paragraph margin 0 0 12px; list item gap 4px; h1/h2/h3 = 20/28, 18/28, 16/24 at 600
with 24px top margin; table cell padding 8px 12px; blockquote 3px left border `--border-control`, text `--text-secondary`.

### 2.6 Spacing, sizes, radii
- **Spacing scale (8px base, 4px half-step allowed only inside components):** `--sp-1:4px --sp-2:8px --sp-3:12px --sp-4:16px --sp-5:24px --sp-6:32px --sp-7:40px --sp-8:48px --sp-9:64px`.
- **Control heights:** `--ctl-sm:32px` (row actions, chips, icon buttons in dense areas), `--ctl-md:40px` (buttons, inputs, nav rows, topbar controls),
  `--ctl-lg:48px` (composer send on coarse pointers only). On `@media (pointer:coarse)`: `--ctl-sm` becomes 40px and `--ctl-md` 44px.
  Minimum hit target is 32x32 (WCAG 2.2 AA is 24x24); coarse pointers get >=40x40.
- **Icon sizes:** 16 (rows, chips, meta), 20 (buttons, nav, topbar), 24 (brand mark, card icons). Stroke width **1.75** (`--icon-stroke`).
- **Layout dims:** sidebar expanded `272px`, rail `72px`, drawer `min(296px, 86vw)`, topbar `56px`, composer min-height 56px (1 row).
- **Card padding** 16px; **dialog padding** 24px; **menu padding** 4px, item height 36px (40px on coarse); **sidebar row** 40px (history row 48px incl. meta line? see 7.7: two-line 52px is NOT used; one line 40px with time right-aligned).
- **Radii:** `--r-xs:4px` (kbd, inline code), `--r-sm:6px` (chips, badges), `--r-md:8px` (buttons, inputs, rows, menu items), `--r-lg:12px` (cards, menus, popovers, composer, code block), `--r-xl:16px` (dialogs, user bubble, drawer edge), `--r-pill:999px` (send/stop, toggle pills).
- **Borders:** 1px. `--border` for structure, `--border-control` for interactive boundaries. Hover on a bordered control darkens nothing; it changes background to `--surface-subtle` (main) / `--sidebar-hover` (sidebar), border stays.
- **Elevation (used sparingly):** `--elev-0: none` (default for everything in-flow). `--elev-1` popover/menu/toast: light `0 4px 16px rgba(23,32,51,.10), 0 0 0 1px var(--border)`; dark `0 4px 16px rgba(0,0,0,.40), 0 0 0 1px var(--border)`. `--elev-2` dialog / mobile drawer: light `0 16px 40px rgba(23,32,51,.16)`; dark `0 16px 40px rgba(0,0,0,.55)`. Nothing else has a shadow. No `backdrop-filter` anywhere.
- **z-index:** `--z-base:0, --z-sticky:10 (topbar, composer), --z-rail:20, --z-scrim:90, --z-drawer:100, --z-popover:200, --z-tooltip:300, --z-dialog:400 (native <dialog> top layer ignores this), --z-toast:500, --z-skip:600`.

### 2.7 Motion
- Durations: `--dur-fast:120ms` (hover, color), `--dur-base:200ms` (menus, drawer, collapse), `--dur-slow:320ms` (viz state change, dialog in).
  Easings: `--ease-out: cubic-bezier(.2,0,0,1)` (enter), `--ease-in: cubic-bezier(.4,0,1,1)` (exit), `--ease-inout: cubic-bezier(.4,0,.2,1)`.
- Allowed properties to animate: `transform`, `opacity`, `background-color`, `border-color`, `color`, `stroke-dashoffset` (one-shot only). Never `width/height/top/left` except the
  sidebar width (rail/expanded) at `--dur-base`, and `max-height`-free disclosure via `grid-template-rows` 0fr→1fr.
- Message enter: opacity 0→1 + translateY(4px→0), `--dur-base`. Replace the legacy `stagger` and `float` (4s) and `gradientShift`; delete `bounceIn`, `pulse`.
- **Reduced motion (global):**
  ```css
  @media (prefers-reduced-motion: reduce){
    *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;
      transition-duration:.01ms!important;scroll-behavior:auto!important}
    .ai-viz *{animation:none!important}
  }
  ```
  Also `scroll-behavior:smooth` on `.chat-area` is removed unless `prefers-reduced-motion: no-preference`.

### 2.8 Focus ring
`:focus-visible { outline: 2px solid var(--ring); outline-offset: 2px; }` for every interactive element, in both themes. Never `outline:none` without
replacement. Inside tight containers (sidebar rows, menu items, input in a bordered box) use `outline-offset: -2px` so it is not clipped by `overflow:hidden`.
Text inputs: border becomes `--ring` plus the 2px outline (border color is not the only cue). Mouse clicks do not show the ring (`:focus-visible`).
Filled accent buttons: ring is offset 2px so the gap shows the surface behind (>= 3:1 against it, table 2.4).

### 2.9 Legacy-variable migration map
Mechanical rules for `components.css` / `animations.css` / `main.css` / inline styles in `index.html`, `ui.js`, `chat.js`.

| Legacy | New | Notes |
|---|---|---|
| `--bg-primary` | `--app-bg` | body only |
| `--bg-surface` | `--surface` (topbar, workspace, dock, modal) **or** `--sidebar-bg` (inside `.sidebar*`) | legacy meaning was "panel"; choose by context |
| `--bg-card` | `--surface-subtle` (chips, kbd, hover bg) **or** `--surface-raised` (menus, popovers, composer, `<select>` options) | |
| `--bg-glass`, `--blur-sm/md/lg` | delete | no glass/blur |
| `--bg-hover` | `--surface-subtle` (main) / `--sidebar-hover` (sidebar) | |
| `--accent-blue` | `--accent` (fills/borders/icons) or `--accent-text` (link/text) | |
| `--accent-blue-h` | `--accent-hover` | |
| `--accent-purple` | delete (no replacement) | |
| `--accent-green` | `--success` (text/icon) / `--success-bg` | web-search toggle "on" no longer green; see 6 |
| `--accent-red` | `--error` / `--error-bg` / `--danger-fill` | `.dock-stop` no longer red (see 6.4) |
| `--accent-yellow` | `--warning` / `--warning-bg` | |
| `--gradient-hero/green/card` | delete; use flat `--accent`, `--accent-subtle`, or none | also remove `-webkit-background-clip:text` on `.sidebar__title`, `.welcome__title` |
| `--text-primary` | `--text` | |
| `--text-secondary` | `--text-secondary` (same name, new values) | |
| `--text-muted` | `--text-secondary` for readable text; `--text-placeholder` for placeholders; `--text-disabled` for disabled; `--border-control` for scrollbar thumbs | `--text-muted` (#484F58 dark) failed contrast; never carry it forward |
| `--border` | `--border` (decorative) **or** `--border-control` (inputs, buttons, chips, checkboxes) | choose by whether the element is interactive |
| `--border-focus` | `--ring` | |
| `--shadow-card`, `--shadow-sm` | `--elev-1` (popover/menu/toast) or none | |
| `--shadow-glow` | delete | |
| `--r-sm/md/lg/xl/pill` (6/10/14/20/100) | `--r-sm/md/lg/xl/pill` (6/8/12/16/999) | same names, new values — audit every use |
| `--sidebar-w` | `--sidebar-w` (272px) + `--rail-w` (72px) | |
| `--topbar-h` (60px) | `--topbar-h` (56px) | |
| `--dock-h` (100px) | delete; dock height is content-driven | |
| `--t-fast/mid/slow` (.15/.25/.4s ease) | `--dur-fast/base/slow` + `--ease-*` | |
| `--font-ui`, `--font-mono` | same names | |
| literal `#fff` on logos/buttons | `--on-accent` | |
| literal `rgba(16,185,129,..)`, `rgba(239,68,68,..)`, `rgba(37,99,235,..)` in `.thought-step--*`, `.badge--*` | `--success-bg`/`--error-bg`/`--accent-subtle` + matching text token | |
| `#hljs-theme` stylesheet | removed (see 3.6) | |
| inline `style="color:var(--accent-green)"` etc. | classes + tokens | `index.html`, `ui.js`, `chat.js` contain inline styles; remove |

---------------------------------------------------------------------------------------------------

## 3. Theme mechanics

### 3.1 Model
- Preference `pref ∈ {light, dark, system}` stored in `localStorage["theme"]` (same key).
- Resolved theme `light|dark` written to `document.documentElement.dataset.theme`. CSS only ever reads `[data-theme]`; the preference is mirrored in `data-theme-pref` for the control.
- `<html>` in `index.html` no longer hardcodes `data-theme="dark"`. Add `<meta name="color-scheme" content="light dark">` and set `style.colorScheme` so native scrollbars, form controls and `<dialog>` UA styles follow the theme. Update `<meta name="theme-color">` (light `#F7F8FC`, dark `#0B1020`) via JS when the theme changes.
- Default for a new visitor: **System**.

### 3.2 Migration of the old default
Old code wrote `theme=dark` on every load (`initTheme` → `setTheme` persists), so a stored `dark` is indistinguishable from a deliberate choice. Rule, run once:
```
if localStorage.themeVersion !== "2":
    stored = localStorage.theme
    pref   = (stored === "light") ? "light" : "system"     // "light" can only come from an explicit toggle; dark/absent -> System
    localStorage.theme = pref; localStorage.themeVersion = "2"
```
`themeVersion` is a new visual-preference key (views may own it; it is not `userId/mode/model`).

### 3.3 No-flash snippet (first element inside `<head>`, before any stylesheet; not deferred, not external)
```html
<script>
(function () {
  var r = document.documentElement, p = "system";
  try {
    var s = localStorage.getItem("theme");
    if (localStorage.getItem("themeVersion") !== "2") {
      s = (s === "light") ? "light" : "system";
      localStorage.setItem("theme", s); localStorage.setItem("themeVersion", "2");
    }
    p = (s === "light" || s === "dark") ? s : "system";
  } catch (e) {}
  var dark = p === "dark" || (p === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  r.dataset.theme = dark ? "dark" : "light"; r.dataset.themePref = p;
  r.style.colorScheme = dark ? "dark" : "light";
})();
</script>
```
`theme.js` (new, loaded before `ui.js`) exposes `Theme.get()`, `Theme.set(pref)`, `Theme.resolved()`, and subscribes to
`matchMedia("(prefers-color-scheme: dark)").addEventListener("change", …)` (re-resolve only when pref is `system`) and to the `storage` event (sync other tabs). `UI.initTheme/setTheme/toggleTheme` stay as thin wrappers (`setTheme("light"|"dark"|"system")`; `toggleTheme` opens the switcher menu) so existing callers keep working.
Add `body{transition:none}` for the first frame: the legacy `transition: background .4s` on `body` must not run on load. Theme changes themselves may fade `background-color`/`color` at `--dur-base` on `body`, `.sidebar`, `.surface` only (not `*`), and are suppressed by reduced motion.

### 3.4 Theme control (`ThemeSwitcher`, in the top bar, right side)
- Trigger button `#themeToggle` (id kept: `app.js` binds `click` on it — it must exist or `app.js:87` throws). 40x40, icon-only, 20px icon showing the **current pref**: `sun` (Light), `moon` (Dark), `monitor` (System). Accessible name: `Theme: System` (updates). Tooltip (same text) on hover and focus after 400ms, `Esc` dismisses. `aria-haspopup="menu"`, `aria-expanded`.
- Clicking it (and `UI.toggleTheme`) opens a popover `role="menu"` with three `role="menuitemradio"` items: `Light` (sun), `Dark` (moon), `System` (monitor, caption "Match device"). The selected item has `aria-checked="true"`, a trailing `check` icon and weight 600 — not color only. Choosing closes the menu, returns focus to the trigger, applies instantly (no reload), and persists.
- Keyboard: Enter/Space/ArrowDown on trigger opens and focuses the selected item; ArrowUp/Down/Home/End move; Enter/Space selects; Esc closes; Tab closes and moves on.
- Rationale for a menu over a 3-segment control: one 40px target in the compact topbar at 360px, consistent with the model selector popover.

### 3.5 Where it applies
All surfaces consume tokens only; no component may hardcode a hex. Dark mode is a full token set (section 2.3), including sidebar, inputs, menus, dialogs, toasts, scrollbars (`scrollbar-color: var(--border-control) transparent`), `color-scheme`, selection (`::selection{background:var(--accent-subtle)}` + `--text`) and code blocks.

### 3.6 highlight.js theming without a swap
Remove `<link id="hljs-theme">` and the `href` swapping in `UI.setTheme`. Ship `css/syntax.css` (new) styling `.hljs*` classes with the `--syn-*` tokens (2.3). Both themes are covered by the same stylesheet, so there is **no second download, no flash, no double load**; the browser re-styles instantly when `data-theme` changes. Keep the highlight.js **script** (CDN) as is. Also `Utils.highlightCode` must not set inline colors. Map: `.hljs{color:var(--syn-default);background:transparent}`; `.hljs-keyword,.hljs-selector-tag,.hljs-literal` → keyword; `.hljs-string,.hljs-regexp,.hljs-addition` → string; `.hljs-number,.hljs-symbol,.hljs-bullet` → number; `.hljs-comment,.hljs-quote` → comment (italic); `.hljs-title,.hljs-section,.hljs-function .hljs-title` → title; `.hljs-built_in,.hljs-type,.hljs-class .hljs-title` → builtin; `.hljs-attr,.hljs-attribute,.hljs-variable,.hljs-template-variable` → attr; `.hljs-tag,.hljs-name,.hljs-selector-id,.hljs-selector-class` → tag; `.hljs-deletion` → deletion; `.hljs-meta` → text-secondary.

---------------------------------------------------------------------------------------------------

## 4. Layout and responsive rules

### 4.1 Regions
```
┌──────────────┬───────────────────────────────────────────────┐
│  Sidebar     │  Topbar (56)                                   │
│  272 / rail  ├───────────────────────────────────────────────┤
│  72 / drawer │  Workspace scroll area (.chat-area, flex:1)    │
│              ├───────────────────────────────────────────────┤
│              │  Composer dock (content height, in flow)       │
└──────────────┴───────────────────────────────────────────────┘
```
`body{height:100dvh;overflow:hidden}`; `.app-layout` is `display:flex`; `.main` is a flex column with `min-width:0`; only `.chat-area` and the sidebar history scroll. Add `container-type:inline-size` on `.main` so workspace grids respond to workspace width (sidebar-aware) rather than viewport; keep `@media` fallbacks for the sidebar itself.

### 4.2 Breakpoints
| Name | Viewport | Sidebar | Notes |
|---|---|---|---|
| `xs` | < 480 | drawer | single column, topbar compact |
| `sm` | 480–767 | drawer | 2-col feature grid at container >= 560 only |
| `md` | 768–1023 | **icon rail 72px** by default; expanding opens it as an overlay drawer (does not push content) | |
| `lg` | 1024–1439 | expanded 272px, user may collapse to rail | |
| `xl` | >= 1440 | expanded 272px | content stays centered at max widths |

Collapse choice persists in `localStorage["sidebarState"]` = `expanded|rail` (visual key; applies >= 1024 only). Default expanded.

### 4.3 Sidebar states
1. **Expanded (>= 1024)** top to bottom, `--sidebar-bg`, 1px right border `--border`, padding 16px 12px:
   brand row (logo mark 32px rounded-8 `--accent` fill with `sparkles` 20px in `--on-accent`; name "AI Assistant Pro" 15/600 `--text`; no tagline; right: `panel-left-close` 32px icon button) → **New chat** primary button (full width, 40px, `plus` + "New chat") → nav list (4 rows, 40px: General `message-circle`, Documents `file-text` + count, Research `globe`, Code `code-xml`) → 16px gap → search input (40px, `search` icon, "Search chats") → history list (flex:1, scrolls) → footer (user line).
2. **Rail (72px)**: brand mark only, `panel-left-open` button, New chat as 40px icon-only primary (`plus`, tooltip "New chat", right side), four nav icons (tooltips on the right, 40x40 centered), `history`-free: search/history are not available in rail; a `panel-left-open` expand button is the single path (a chat history icon would be a dead end).
3. **Mobile drawer (< 768; also md "expanded")**: off-canvas from the **left** (replaces the legacy 75dvh bottom sheet), width `min(296px, 86vw)`, `--elev-2`, scrim behind (`--scrim`, replaces `rgba(0,0,0,.55)` inline), slides 200ms `--ease-out`. Behavior: opened by the hamburger `#sidebarToggle` (aria-expanded updated; id kept) ; `role="dialog" aria-modal="true" aria-label="Navigation"` on `#sidebar` while open; initial focus on the first nav item; **focus trap** (Tab/Shift+Tab cycle inside); `Esc` closes; scrim click closes; selecting a session or nav item closes; focus returns to the hamburger; `inert` set on `.main` while open (fallback `aria-hidden`); body scroll lock is already provided by `overflow:hidden`. `#sidebarBackdrop` id kept; its inline `onclick="UI.closeSidebar()"` and inline style are replaced by a class + listener. Drawer width accounts for `env(safe-area-inset-left)`.
4. Collapse/expand animates `width` only (200ms). Labels fade with opacity; `overflow:hidden` on sidebar; no layout jump in the workspace beyond its natural resize.

### 4.4 Top bar (56px, `--surface`, bottom border `--border`, padding 0 16px (0 8px at xs))
Left: sidebar button (hamburger `menu` on < 768, `panel-left-open` on rail) → **context title**: current mode name (15/600) and, if a session is open, `/` + session title (truncate, 14/400 secondary). Right: **ModelSelector** (3.x) → **ThemeSwitcher**. No other controls; New chat lives in the sidebar (and `Ctrl/Cmd+K`-free: no new shortcuts are added). At xs: context title shows mode only; model trigger shows an icon + short label truncated to 12ch.
The old inline mode tabs are removed from the top bar; the `#modeTabs` container and `.mode-tab[data-mode]` buttons move into the sidebar nav (ids/class/`data-mode` kept so `UI.setActiveMode` and `app.js` work), with `role="tablist"/role="tab"/aria-selected` replaced by `<nav aria-label="Workspaces">` and `aria-current="page"` on the active item (`UI.setActiveMode` must set `aria-current` and `.active`).

### 4.5 Workspace widths
| Area | Max content width | Horizontal padding |
|---|---|---|
| Chat/message column and composer | **768px** (`48rem`), centered | 24px (>= 768), 16px (< 768) |
| Welcome | 880px, centered; cards grid max 720px | same |
| Documents workspace | 960px | same |
| Research answer column | 768px (sources block same width) | same |
| Code answer column | **880px** (code blocks benefit; prose lines capped at 68ch inside) | same |
| Dialogs | 480px (confirm 400px), `min(…, 100vw - 32px)` | 24px |
Prose measure inside messages: `max-width: 68ch` for paragraphs; tables/code blocks may fill the column and scroll.

### 4.6 Composer docking, safe areas, keyboard
- The composer is an **in-flow footer** of `.main` (not fixed): `.chat-area` is `flex:1; min-height:0; overflow-y:auto`, so it can never cover messages.
- Dock padding: `12px 24px max(16px, env(safe-area-inset-bottom))`; sides add `env(safe-area-inset-left/right)`. Add `viewport-fit=cover` and `interactive-widget=resizes-content` to the viewport meta so Android Chrome resizes the layout with the keyboard. iOS Safari still overlays; **risk R3** (12.5).
- Composer box: `--surface-raised`, 1px `--border-control`, `--r-lg`, no shadow; focus-within shows the 2px ring. Auto-grow textarea 1→8 rows (24px/row, max 200px, or `30dvh` if smaller), then internal scroll.
- Hint line ("Enter to send · Shift+Enter for newline"): 12px `--text-secondary`, shown only at `(min-width:1024px) and (min-height:720px) and (pointer:fine)`. The drag-and-drop sentence moves out of it (the dropzone overlay explains it).
- **Laptop budget** (1366x768, ~657px of viewport): topbar 56 + dock <= 148 (with hint) → >= 453px for the scroll area. Welcome content height = viz 96 + 16 + h1 36 + 8 + sub 24 + 24 + cards 2x96+16 = 412 (+40 capabilities strip = 452). Therefore: capabilities strip hidden when height < 760; viz shrinks to 64px below height 720 and is hidden below 600; the scroll area scrolls if still short. The composer is always visible because it is in-flow and the page never scrolls.

### 4.7 Behavior per width
| Width | Behavior |
|---|---|
| **360** | Drawer. Topbar: hamburger, mode title, model icon-button, theme. Content padding 16. Feature cards 1 column (each ~88px, text wraps to 2 lines). Composer: attachment/URL/web-search icons in a toolbar row 40x40 (coarse: 44), model text hidden, send 40px. Inputs 16px font. Dialogs full-width minus 32px; confirm buttons stacked, full width. Code blocks scroll horizontally; no horizontal page scroll. Toasts full-width minus 32px, above composer (`bottom: calc(dock height + 8px)` via safe-area aware container). |
| **390** | As 360; model chip shows label up to 14ch. |
| **768** | Rail by default (72). Feature grid 2x2. Message column 100% - 48px. Hint hidden (pointer-dependent). Topbar shows full model label up to 24ch. |
| **1024** | Expanded sidebar 272. Workspace 752. Column 704 inside padding. Hint visible if height >= 720. |
| **1440** | Workspace 1168; content centered at 768/880/960. Nothing stretches. |
| **200% zoom** (equivalent to 720 CSS px wide at 1440 monitor, 384 at 768) | Layout follows the CSS-px breakpoints (so 200% on a 1440 screen behaves like `sm`/`md`: rail or drawer). Text must reflow (WCAG 1.4.10): no two-axis scroll except inside code blocks/tables, no clipped controls, topbar wraps never — it truncates titles; `min-width:0` on all flex children; no fixed pixel widths on composer. |
Long content: tables and code blocks have their own `overflow-x:auto`; URLs/long tokens `overflow-wrap:anywhere` in message text; filenames and titles truncate with `text-overflow:ellipsis` and expose full text via `title`/tooltip.

---------------------------------------------------------------------------------------------------

## 5. Component hierarchy and module ownership

```
AppShell                          index.html + css/main.css (layout); app.js owns State, no markup
├─ SkipLink ("Skip to composer")  index.html (first focusable; z-skip)
├─ Sidebar                        ui.js (drawer/rail/collapse; replaces openSidebar/closeSidebar internals)
│  ├─ Brand (+ collapse btn)      index.html
│  ├─ NewChatButton               index.html (#newChatBtn kept)
│  ├─ Navigation (#modeTabs)      index.html + ui.js setActiveMode (aria-current, doc count)
│  ├─ ConversationHistory         ui.js renderSessionList (extended) + components/menu.js
│  │   ├─ SessionSearch (#sessionSearch)
│  │   ├─ SessionGroup / SessionRow
│  │   └─ SessionMenu (rename/delete) + inline rename field
│  └─ SidebarFooter (user line)   ui.js setUserIdDisplay
├─ Main
│  ├─ TopNavigation               index.html + ui.js (context title)
│  │   ├─ SidebarToggle (#sidebarToggle)
│  │   ├─ ModelSelector           components/model-select.js (wraps #modelSelect)
│  │   └─ ThemeSwitcher (#themeToggle)   theme.js (+ components/menu.js)
│  ├─ Workspace (#chatArea, role=log)
│  │   ├─ WelcomeScreen (#welcomeScreen)  index.html markup; wiring existing starterPrompts handler → [cross-boundary 12.4]
│  │   │   ├─ AIVisualization (lg)        components/ai-viz.js
│  │   │   ├─ FeatureCard x4
│  │   │   └─ CapabilitiesStrip
│  │   ├─ ModeEmptyState (document|research|code)   ui.js
│  │   ├─ DocumentsWorkspace (#docSection/#docList) ui.js renderDocList (moved into workspace)
│  │   ├─ ChatConversation
│  │   │   ├─ ChatMessage (user)          chat.js
│  │   │   └─ ChatMessage (assistant)     chat.js
│  │   │       ├─ AIVisualization (sm avatar)
│  │   │       ├─ ActivitySteps (thought steps)
│  │   │       ├─ MarkdownBody (+ CodeBlock via utils.js highlightCode)
│  │   │       ├─ SourcesBlock (citations)     chat.js
│  │   │       ├─ BadgeRow (model/cached/fallback)
│  │   │       └─ MessageActions (copy, regenerate, feedback, time)
│  │   ├─ LoadingState (skeleton, replaces showThinking) chat.js
│  │   └─ ErrorState (inline, retry)          chat.js showError (extended)
│  └─ MessageComposer (footer)                index.html + ui.js (auto-grow, toggles)
│      ├─ AttachmentList (chips)              ui.js (new renderAttachments) 
│      ├─ Textarea (#chatInput)
│      └─ ComposerToolbar (#attachBtn, #addUrlBtn, #searchToggle, model chip, #sendBtn / #stopBtn)
├─ Overlays
│  ├─ ConfirmationDialog                      components/dialog.js (UI.confirm)
│  ├─ CitationDialog (#citationModal)         ui.js showCitationModal
│  ├─ UrlDialog (#urlModal)                   ui.js showUrlModal
│  ├─ DropzoneOverlay (#dropzoneOverlay)      ui.js initDropzone
│  └─ ToastRegion                             ui.js showToast (extended; replaces #uploadToast visual)
└─ shared: Icons (icons.js + inline sprite), Menu/Popover (components/menu.js), Tooltip (CSS + 10 lines JS or CSS-only)
```
New files (all under served folders): `js/icons.js`, `js/theme.js`, `js/components/ai-viz.js`, `js/components/menu.js` (popover + roving-focus menu used by ThemeSwitcher, SessionMenu, ModelSelector), `js/components/model-select.js`, `js/components/dialog.js`, `css/syntax.css`, optional `css/layout.css`. Load order (classic scripts): `icons → utils → api → chat → theme → components/menu → components/dialog → components/model-select → components/ai-viz → ui → app`. Do not rename `main.css`/`app.js`.

---------------------------------------------------------------------------------------------------

## 6. State matrix and accessibility per interactive component

Global rules: hover (pointer:fine only; wrap in `@media (hover:hover)`), `:focus-visible` = ring (2.8), `:active` = one step darker/pressed token, `:disabled` / `aria-disabled` = `--text-disabled`, no hover, `cursor:not-allowed`, still focusable only when it explains why (via `aria-disabled` + tooltip), loading = spinner (`loader-circle`, 1s linear) + `aria-busy="true"` and the control stays the same width.

### 6.1 Buttons
| Variant | Default | Hover | Active | Disabled | Loading |
|---|---|---|---|---|---|
| Primary (New chat, Send, dialog confirm) | `--accent` fill, `--on-accent` text, `--r-md`, 40h | `--accent-hover` | `--accent-pressed` | fill `--surface-subtle`, text `--text-disabled`, border `--border` | label kept, icon replaced by spinner, `aria-busy` |
| Secondary | `--surface` fill, 1px `--border-control`, `--text` | `--surface-subtle` | bg `--border` | as above | as above |
| Ghost / icon | transparent, glyph `--text-secondary` | `--surface-subtle` (sidebar: `--sidebar-hover`), glyph `--text` | `--border` bg | glyph `--text-disabled` | spinner |
| Danger | `--danger-fill`, `--on-danger` | `--danger-fill-hover` | darken 8% | as primary disabled | spinner |
Icon-only buttons always have `aria-label` and tooltip. Pressed toggles (`#searchToggle`) use `aria-pressed`.

### 6.2 Navigation item (Sidebar)
Default: icon 20 + label 14/500 `--text`, 40h, `--r-md`. Hover `--sidebar-hover`. Active (`aria-current="page"`): `--sidebar-active` bg, label weight 600, icon `--accent`, 3px `--accent` left bar inside the row (inset, radius 0 3 3 0). Focus: inset ring. Disabled: n/a. Documents item shows a count pill (`--surface` bg, 12px, `--text-secondary`, `aria-label="3 documents"`) only when `documents.length > 0`; absent otherwise (no "0"). Keyboard: Tab/Shift+Tab (native buttons), Enter/Space. Pressed = `--accent-subtle` darker step is not needed; use `--sidebar-active`.

### 6.3 Session row
Structure: `<li>` → primary `<button class=session-item aria-current="true|false">` (title 14/400, truncated; time 12px right) + sibling `<button class=session-menu-btn aria-haspopup="menu" aria-expanded aria-label="Chat options: {title}">` (`ellipsis`, 32px). **No nested interactive elements** (the legacy div+button is replaced).
Default: transparent. Hover/focus-within: `--sidebar-hover`, menu button visible. Active session: `--sidebar-active` + weight 600 + left bar + `aria-current="true"`. Menu button visible always on `(hover:none)` and on the active row. Loading (list): 5 skeleton rows 40h. Empty: see 7.7. Error (list fetch fails): inline `circle-alert` + "Couldn't load chats" + `Retry` ghost button **[needs controller callback `onRetry`]**.
Menu (`role="menu"`, popover anchored right-aligned below button, `--surface-raised`, `--elev-1`, `--r-lg`, padding 4): items `Rename` (pencil) and `Delete` (trash-2, `--error` text + icon). Keyboard: Enter/Space/ArrowDown on the button opens and focuses first item; ArrowUp/Down/Home/End/typeahead; Enter activates; Esc closes and refocuses the button; Tab closes. Rename: row title becomes `<input>` (40h, `--r-md`, ring) preselected; Enter commits, Esc cancels, blur commits if non-empty and changed; empty → revert; max length 120 **[verify-with-audit: backend title limit]**; while saving: input disabled + spinner; error → toast + revert. Delete: opens ConfirmationDialog (6.9).

### 6.4 Composer controls
- **Textarea** `#chatInput`: default border `--border-control`; the box (not the textarea) carries the border; box hover = border `--text-secondary` (>= 5:1 for the stroke); focus-within = ring + border `--ring`; disabled (during streaming the textarea stays enabled so users can type the next message; Send is what is gated — **preserve existing behavior [verify-with-audit]**). Placeholder per mode: General "Message AI Assistant Pro…", Document "Ask about your documents…", Research "What do you want to research?", Code "Describe the code you need or paste code to review…". Error (send failed): box border `--error` + inline message below `circle-alert` "Message not sent — your text is kept." + Retry is the Send button itself.
- **Attach** `#attachBtn` (`paperclip`): ghost icon; label "Attach file"; opens file picker (`#fileInput`, accept list: `.pdf,.docx,.csv,.xlsx,.txt,.md` + `.rst` since the backend accepts it **[verify-with-audit]**). Loading (an upload in progress): stays enabled (one upload at a time is the existing behavior **[verify-with-audit]**); if a second is not allowed, `aria-disabled` + tooltip "Wait for the current upload".
- **Add URL** `#addUrlBtn` (`link`): opens UrlDialog.
- **Web search** `#searchToggle`: a **pill toggle**, not a bare icon: `globe` + visible text "Web" (hidden label below 480px, icon remains) ; off: ghost, `aria-pressed="false"`; on: `--accent-subtle` bg, `--accent-text`, 1px `--accent` border and a leading `check` (16) so state is not color-only, `aria-pressed="true"`. In Research mode the server always searches: the pill is `aria-disabled="true"`, shown on, tooltip "Research always searches the web". Replaces the green icon.
- **Model chip** (read-only text): `{model label}` 12px `--text-secondary`, truncated at 20ch, no hover styles, `title` full label. Not interactive (no dead control); selection is in the top bar.
- **Send** `#sendBtn` (`arrow-up`, 40px circle `--accent`, `--on-accent`): disabled when input is empty/whitespace (filled `--surface-subtle`, glyph `--text-disabled`) or a request is in flight; loading not shown here (the Stop button takes its place).
- **Stop** `#stopBtn` (`square`, filled): replaces Send while streaming (same position, same 40px circle); fill `--text`, glyph `--surface`; ring on focus; label "Stop generating". Neutral, not red (red means error). Esc in the textarea does not stop (not a current behavior; do not add).
- Duplicate-submit: Send is disabled and Enter is ignored while a request is in flight (controller guarantee; view disables).

### 6.5 Attachment chip (AttachmentList)
Row above the textarea inside the composer box, wraps, gap 8. Chip: 32h, `--r-sm`, 1px `--border-control`, `--surface-subtle`; contents: file-type icon 16 + name (truncate 20ch, `title` full) + size from the browser `File` (e.g. "1.2 MB", **only at upload time**) + trailing `x` icon button (24x24 target, label "Remove {name}").
States: **uploading** — `loader-circle` spinner, text "Uploading and indexing…", an **indeterminate** 2px bar (no percentage; the API has no progress); the remove button cancels the chip only if the controller supports abort, otherwise it is hidden **[cross-boundary: controller decides]**. **Indexed** (only after the POST resolves) — `circle-check` in `--success`, "Indexed · N chunks" (+ "lexical only" when `indexing=lexical`), × becomes "Dismiss" (it hides the chip; it does **not** delete the document — deleting is in the Documents workspace). **Error** — `circle-alert` `--error` + server message (413: "File is larger than the server limit" using the returned message), actions `Retry` (needs the retained `File`) and `x`. `role="status"` per chip for announcements (errors `role="alert"`). Replaces the legacy `#uploadToast` visual and its fake 2.5s progress animation (`progressFill`).

### 6.6 Model selector
Enhances the existing native `<select id="modelSelect">`, which **remains the source of truth** (hidden visually but in DOM, `app.js` reads `.value` and listens for `change`). The custom control sets `select.value` and dispatches `change`. If JS fails, the native select is shown.
- Trigger: 40h secondary button, `sparkles` 16 + label 14/500 (truncate) + `chevron-down`. `aria-haspopup="listbox"`, `aria-expanded`, `aria-label="AI model: {label}"`.
- Popover: `role="listbox"` `aria-label="AI model"`, width `min(360px, 100vw - 32px)`, right-aligned to trigger, max-height `min(420px, 60dvh)`, scroll inside. Grouped by `provider` with non-selectable group captions (12/16, `role="group"` + `aria-labelledby`); provider display names `groq → "Groq"`, `google → "Google"`, others as returned (drop "(free)" — unverifiable **[verify-with-audit]**).
- Option row (`role="option"`, min 52h, 12px padding): line 1 `label` 14/500; line 2 `description` 12/16 `--text-secondary` (2 lines max); trailing `check` for the selected option (`aria-selected="true"` + `--accent-subtle` bg + weight 600). **Nothing else** (no speed, price, context, capabilities).
- Unavailable (`available:false`): `aria-disabled="true"`, `--text-disabled`, and the `reason` text visible on line 2 prefixed "Unavailable — " (not only a tooltip); not selectable; skipped by typeahead but reachable by arrows so screen reader users hear the reason.
- Search: a filter input (40h, `search` icon, `aria-label="Filter models"`) is rendered **only if models.length > 8**; otherwise typeahead by first letters.
- Keyboard: Enter/Space/ArrowDown opens and focuses the selected option; ArrowUp/Down/Home/End; Enter selects, closes, restores focus to trigger; Esc closes; Tab closes. Use `aria-activedescendant` on the listbox (focus stays on the listbox).
- Loading (`GET /models` pending): trigger shows spinner + "Loading models…", `aria-busy`, disabled. Error: trigger shows `circle-alert` + "Models unavailable" and is disabled with tooltip; retry requires a controller hook **[cross-boundary: `onRetryModels`]** — if absent, show the message only. Empty list: "No models available" (disabled).
- Fallback: automatic fallback is already server-side; the chosen model stays the selection. Fallback is announced per message (6.8 badges) — the selector never changes by itself.

### 6.7 Feature card (`FeatureCard`)
`<button type="button">`, 96h min, padding 16, `--surface`, 1px `--border-control` (interactive boundary, >= 3:1 in both themes; hover only changes bg), `--r-lg`, text-align left. The "Asks:" prompt line is never hidden at any height (compacts to 1 line / smaller padding; the chat area scrolls). Default / hover (bg `--surface-subtle`, border `--border-control`; no lift, no shadow) / focus-visible (ring) / active (bg `--border` at 40%… use `--surface-subtle` darker: `--border`) / disabled n/a. Action: switch to the card's mode and focus the composer (General→`general`, Analyze Documents→`document`, Web Research→`research`, Write & Debug Code→`code`). See cross-boundary 12.4.

### 6.8 Chat message
- **User**: right-aligned, max 80% of column, `--accent-subtle` bg, `--text`, padding 12px 16px, `--r-xl` (bottom-right 4px), text 15/24, `pre-wrap`; no avatar; timestamp 12px `--text-secondary` below, right-aligned.
- **Assistant**: no bubble, no background; left gutter 36px (28px avatar = AIVisualization `sm`, 8px gap) at >= 640px, none below (screen-reader-only "Assistant said"). Content: ActivitySteps (collapsed after `done`), body (MarkdownBody), SourcesBlock, BadgeRow, MessageActions.
- Roles for SR: each message `<article aria-label="You said"/"Assistant said">`; list container stays `role="log"` with `aria-live="off"` (live announcements come from the status region below).
- **Streaming status region**: a visually hidden `<div id="liveStatus" role="status" aria-live="polite" aria-atomic="true">` updated at stage changes ("Searching documents", "Searching the web", "Generating response", "Response complete", "Response failed: {message}") and **never per token**. The streaming bubble carries `aria-busy="true"` until `done`. Cursor `stream-cursor` becomes a 2px x 1em `--text-secondary` caret blink 1s `steps(1)` (disabled under reduced motion, static caret).
- **MessageActions** (assistant, below badges, 32px icon buttons, glyph `--text-secondary`, always visible — no opacity tricks): Copy (`copy`→`check` for 2s, label "Copy response"→"Copied", announces via liveStatus), Regenerate (`rotate-ccw`, label "Regenerate response", only on the latest assistant message and only when `onRegenerate` provided), Thumbs up/down (`thumbs-up`/`thumbs-down`, toggle `aria-pressed`; active = filled glyph + `--accent-text`; a second click clears, per existing logic). Disabled while the message is streaming. Metadata (time) 12px at the right end.
- **BadgeRow**: model chip `{model}` (12px, `--surface-subtle`, `--r-sm`); `Cached` with `zap` (only if `cached`); `Fallback used` with `shuffle` on `--warning-bg`/`--warning`, **visible text** `Fell back from {label}` (full list in `title`); never tooltip-only.
- **ErrorState** (inline in the assistant message): `--error-bg` block, `--r-lg`, padding 12px 16px, `circle-alert` + message (`--error`, 14/20) + `Retry` secondary button if `error.retryable` (calls the send handler with the preserved text) and a `Dismiss` ghost button. The user's input text is preserved in the composer.
- **LoadingState** (before first status/token): 3 skeleton lines (`--surface-subtle` shimmer 1.4s, reduced motion: static) + mini viz in `processing`.

### 6.9 Dialogs (`ConfirmationDialog`, CitationDialog, UrlDialog)
Implementation: native `<dialog>` + `showModal()` (gives top layer, focus trap, `Esc`, inert background); `::backdrop` = `--scrim`. Width 400/480 max, `--surface-raised`, `--r-xl`, padding 24, `--elev-2`, enter: opacity + scale(.98→1) 200ms. `aria-labelledby` (title), `aria-describedby` (body). Focus: confirm dialogs start on **Cancel**; UrlDialog starts on the input; CitationDialog on Close. On close, focus returns to the invoking element. Backdrop click cancels (confirm dialogs: also cancels, never confirms). Buttons: right-aligned, Cancel (secondary) then confirm (primary or danger). API: `UI.confirm({title, message, confirmLabel, danger}) → Promise<boolean>`. The existing `confirm()` calls (`app.js:336` delete chat, `app.js:400` remove document) must switch to it **[cross-boundary]**. Copy: "Delete this chat?" / "This permanently deletes the conversation and its messages." ; "Remove “{filename}”?" / "It will no longer be used to answer questions. This can't be undone." (Do not claim anything about backend storage beyond the action; **[verify-with-audit]** whether `DELETE /documents/{id}` also removes vectors.) Loading state: confirm button spinner, dialog stays until the promise resolves; error: inline `circle-alert` message in the dialog, buttons re-enabled.
UrlDialog: field label "Web page URL", helper "The page is indexed for document Q&A.", `type=url`, validation message under the field (`aria-invalid`, `aria-describedby`): "Enter a full URL starting with https://". Submit disabled while indexing (spinner "Indexing…"). Enter submits.
CitationDialog: title "Source" ; doc citation: `file-text` + `source` + "Page {n}" when present; web: `globe` + domain + link (see 7.4); snippet in a quote block (`--surface-subtle`, max-height 240 scroll); Close button.

### 6.10 Toast
Region fixed bottom-right 16px (desktop), bottom-center above the composer on < 640, max 3 stacked (oldest dropped), width 360 max. Toast: `--surface-raised`, `--elev-1`, `--r-lg`, 12px padding, icon 20 + text 14/20 `--text` + close `x` on errors. Icons: `circle-check` `--success`, `circle-alert` `--error`, `info` `--accent-text`. Text is `--text` (not colored) so contrast is the 16:1 text pair; the icon carries the semantic. `role="status"` (info/success) / `role="alert"` (error). Timing: info/success 4000ms, error 8000ms and stays while hovered/focused. Enter: translateY(8px)→0 + opacity 200ms; exit 120ms. Messages must no longer contain emoji (`app.js:371,391` pass strings with "✅" — **cross-boundary**: drop them).

### 6.11 Dropzone overlay
Full-viewport `--scrim`-based layer + centered `--surface-raised` panel (`upload` 24px icon, "Drop a file to add it to your documents", secondary "PDF, DOCX, CSV, XLSX, TXT, MD" **[verify rst per audit]**). No float animation. Shown on `dragenter` with items. `aria-hidden` when inactive; not focusable. Only the first file is handled (existing behavior) — if several are dropped, toast "Only the first file was added." **[needs controller hook]**. Keyboard users have the Attach button.

### 6.12 Tooltip
Pure CSS `[data-tooltip]` (existing attribute) + `:focus-visible` trigger, 12/16, `--tooltip-bg`/`--tooltip-text` (16.27 / 17.35), padding 6px 8px, `--r-sm`, 400ms delay on hover, immediate on keyboard focus, hidden with Esc (JS one-liner) and never the only carrier of essential information. Not shown on `(hover:none)`. Never on disabled controls without `aria-disabled` (they are focusable with it).

---------------------------------------------------------------------------------------------------

## 7. Screen specs

### 7.1 Welcome (General mode, no messages, `#welcomeScreen`)
Vertical stack, centered, max 880, gap per scale:
1. AIVisualization lg (96px; 64px if height < 720; hidden if height < 600), state `idle`.
2. `<h1>` **"What can I help you with?"** (28/36, 600, `--text`).
3. `<p>` **"Chat, analyze documents, research the web, and write code in one workspace."** (16/24, `--text-secondary`, max-width 560).
4. Feature cards, grid `repeat(2, 1fr)`, gap 16, max-width 720; 1 column when container < 560px.
5. CapabilitiesStrip (optional; hidden at height < 760): single row of three text items with 16px icons, separated by 24px gaps, wraps on narrow, 13/20 `--text-secondary`:
   - `shuffle` "Multiple AI models with automatic fallback" (true: `/models` + fallback)
   - `file-search` "Questions about PDF, DOCX, CSV, XLSX, TXT and MD files" (true: backend list; add RST per audit)
   - `globe` "Web search with cited sources" (true: Research mode/web toggle)
   No privacy, security, speed or accuracy claims.

Cards (all identical layout: 40px icon tile `--accent-subtle`, icon `--accent` 24; title 15/600; description 13/20 `--text-secondary`):
| Card | Icon (Lucide) | Title | Description | Action |
|---|---|---|---|---|
| 1 | `message-circle` | General Chat | Ask questions, brainstorm, and explore ideas. | mode `general`, focus composer |
| 2 | `file-search` | Analyze Documents | Upload supported files, summarize information, and ask questions about them. | mode `document`, focus composer; empty-state upload shown |
| 3 | `globe` | Web Research | Discover information with web search and cited sources. | mode `research` |
| 4 | `code-xml` | Write & Debug Code | Generate, explain, and review code. | mode `code` |
The legacy six emoji "starter prompts" (`#starterPrompts`, `.welcome__starter`, `data-prompt`) are **replaced** by the cards; their prompt-fill behavior has no equivalent, so the `data-prompt` handler in `app.js` becomes dead. Decision to keep suggestions: show 3 text-only suggestion chips per non-General mode in `ModeEmptyState` (7.5). **[open question 3]**.

### 7.2 Chat (General)
Column 768. Messages separated by 24px; between a user message and its answer 16px. Sticky-to-bottom: auto-scroll only while the user is within 80px of the bottom; if they scrolled up, show a floating `arrow-down` pill "Jump to latest" (32h, `--surface-raised`, `--elev-1`, bottom-center of the scroll area, `aria-label="Scroll to latest message"`). This replaces forced `scrollTop = scrollHeight` on every token **[requires `_scrollToBottom` change in chat.js — frontend-developer owns it]**. Streaming: tokens render as today (markdown re-render throttled 60ms); on `done` run full markdown + highlight. Empty restored session: welcome.

### 7.3 Documents (mode `document`)
Header row (workspace top, inside the column 960): `<h1>` "Documents" (page heading is 24/32 here), count text "{n} documents" (secondary), buttons `Upload file` (secondary, `upload`) → `#attachBtn` action, `Add URL` (secondary, `link`) → `#addUrlBtn` action.
- **Empty (n=0)**: bordered dashed zone (`1px dashed --border-control`, `--r-lg`, padding 32, centered, no illustration, one `file-text` 24 icon in `--accent-subtle` tile): title "No documents yet" (18/28), text "Upload a file or add a web page, then ask questions about it. You can also drag and drop a file anywhere." , primary "Upload file", secondary "Add URL", and caption "Accepted: PDF, DOCX, CSV, XLSX, TXT, MD. Large files may be rejected by the server." (No size number — the limit is not exposed.)
- **List (n>0)**: list (not cards) `role="list"`, rows 56h: type icon tile (32), filename (14/500, truncate, full in `title`), second line `{TYPE} · {chunk_count} chunks · Indexed {date from indexed_at}` (12, secondary), trailing `Remove` (`trash-2` ghost, label "Remove {filename}") → ConfirmationDialog. Row hover `--surface-subtle`. No size, no status chip, no progress (not in the API). A document only appears after upload resolved, so every listed row is "indexed" by construction; this is stated by the "Indexed {date}" text, not a status badge.
- In-flight uploads appear as AttachmentList chips (6.5) and, in this workspace, as a pinned row at the top of the list (uploading → indexed/error), removed from the pin when the list refreshes.
- Once messages exist in Documents mode, the panel collapses to a one-line bar "N documents ▾" (`aria-expanded`, `chevron-down`) above the conversation; expanded it pushes the conversation down (no overlay).
- Ask flow = normal chat with `mode=document`; citations as Sources (7.4).
- IDs: `#docSection` becomes the workspace container, `#docList` the list element (`renderDocList` keeps working; extended with `indexed_at`).

### 7.4 Research (mode `research`) — answer vs. sources
Empty state: `<h1>` "Research the web" (24/32), text "Ask a question. The assistant searches the web and answers with sources.", 3 suggestion chips (7.5).
After a question: ActivitySteps shows real stages ("Searching the web…", "Generating response…") with elapsed times from events.
Result structure (assistant message):
1. **Answer** — normal markdown body; inline citation markers `[1]` render as small superscript-like badges (existing `.citation`): 20x20 min target 24 on coarse, `--accent-subtle` bg, `--accent-text`, number inside, `button`, label "Source 1".
2. **Sources** — a separate block under the answer: heading "Sources (n)" (14/600) with `globe`, `1px --border` top rule + 12px padding, list of source rows (not cards-in-cards): number badge matching the markers, **title** (`citation.source`, 14/500, link), domain (`new URL(url).hostname` minus `www.`, 12 secondary), `external-link` 16 icon. Row = `<a href target="_blank" rel="noopener noreferrer">` when `type==="web"` and the URL protocol is `http:` or `https:` (validate; otherwise render as plain text — untrusted data); doc citations (`type==="doc"`) open the CitationDialog showing snippet + page. Visual distinction: answer text is on the plain workspace background; the sources block is on `--surface-subtle` with `--r-lg` and a `--border` outline, label always "Sources", and generated text never appears inside it. Snippets appear only in the dialog/expandable "Show excerpt" (`aria-expanded`), 13/20, `--text-secondary`, never styled as the answer.
3. Empty sources (search returned none): no block; a one-line note "No sources were returned for this answer." (only if the answer completed with zero citations in research mode).
Errors: ErrorState with Retry. Loading: ActivitySteps + skeleton. The `/search` endpoint is not used (see section 10).

### 7.5 Mode empty states (document is 7.3)
Research and Code share `ModeEmptyState`: `<h1>` (24/32) + one-line text + 3 text-only suggestion chips (32h, `--r-pill`, 1px `--border-control`, 13/20; click fills the composer — does not auto-send; labels are example prompts, not data). Research: "Research the web" / "Compare the main approaches to vector search" style examples; Code: "Write & Debug Code" / "Explain this error", "Write a Python function that…", "Review this snippet for bugs". Example prompts are static copy and are allowed; keep ≤ 3 per mode.

### 7.6 Code (mode `code`)
- Same chat column but 880 wide. Prose capped at 68ch; code blocks use the full column.
- **CodeBlock**: `--code-bg`, 1px `--border`, `--r-lg`, overflow hidden. Header 36h `--code-header`: language label left (12/16, 500, `--text-secondary`, lowercase from the fence; if absent show "Code"), `Copy` button right (secondary-ghost 28h hit target 32, `copy` 16 + "Copy"; on success `check` + "Copied" for 2000ms, announced by liveStatus; failure shows "Copy failed"). Body: `pre` with `overflow-x:auto`, padding 16, `--fs-code`, `tab-size:2`, `white-space:pre` (no wrap), `tabindex="0"` and `role="region" aria-label="{lang} code"` so keyboard users can scroll. Scrollbar thin (`scrollbar-width:thin`). Separate from explanation by 16px margins; no line numbers, no run/preview (no execution backend).
- Inline code: `--surface-subtle`, `--r-xs`, padding 2px 4px.
- Long lines never expand the page (`min-width:0` on message/column).

### 7.7 Conversation history
- Search input `#sessionSearch`: 40h, `search` icon left, clear `x` ghost button appears when non-empty, `type=search`, `aria-label="Search chats"`; filters by title (current behavior **[verify-with-audit]**); no results: "No chats match “{q}”".
- Groups (existing `Utils.formatDateGroup`): **Today, Yesterday, Previous 7 Days, Previous 30 Days, Earlier**. Group caption 12/16 500 `--text-secondary`, padding 16px 12px 4px, `<h2 class="sr-visible">`-style semantics: each group is a `<section aria-labelledby>` with `<ul>`.
- Row: 40h, `--r-md`; leading mode icon 16 (`message-circle|file-text|globe|code-xml`); title 14/400 truncate; trailing time 12/16 secondary (hidden when the menu button shows); menu button as 6.3. Model name is dropped from the row (kept in `title`); it was visual noise.
- Empty: `message-circle` 24 muted tile + "No conversations yet" + "Start a new chat to see it here." Loading: skeleton rows. 
- **No Pin** (API has no field). Rename and Delete are the only actions.
- Active-row scroll into view on load.

---------------------------------------------------------------------------------------------------

## 8. AI visualization (`AIVisualization`)

**Purpose:** a small, tasteful status indicator tied to real app events. It is decoration plus state; never the only carrier of information (text is in the activity steps and `#liveStatus`). `aria-hidden="true"` on the SVG.

### 8.1 Placement and size
- `lg` 96x96 (64 when height < 720, hidden < 600) on the Welcome screen, always above the heading.
- `sm` 28x28 as the assistant message avatar. Only the **message currently streaming** animates; past messages render the static `idle` frame (no CSS animation running on history).
- Not used in the topbar or the composer.

### 8.2 Structure (inline SVG, viewBox 0 0 96 96, 7 nodes)
```
<svg class="ai-viz" data-state="idle" viewBox="0 0 96 96" aria-hidden="true">
  <circle class="ai-viz__halo" cx="48" cy="48" r="44"/>            fill --accent-subtle
  <circle class="ai-viz__ring" cx="48" cy="48" r="38"/>            stroke --accent 1.5, opacity .35
  <g class="ai-viz__spin"><circle class="ai-viz__arc" cx="48" cy="48" r="38" pathLength="100"
        stroke-dasharray="28 72"/><circle class="ai-viz__dot" cx="48" cy="10" r="3"/></g>   arc --accent, dot --viz-cyan
  <g class="ai-viz__core"><path d="…four-point sparkle…"/></g>      fill --accent
  <path class="ai-viz__alert" d="…!…"/>                             stroke --error, hidden unless state=error
</svg>
```
Colors come from tokens so it re-themes with `data-theme`. At `sm` the halo and dot are omitted (CSS `display:none`), keeping ring + arc + core.

### 8.3 States, triggers, motion
| State | Triggered by | Visual | Motion |
|---|---|---|---|
| **idle** | initial; stream `done`; user abort (Stop); new chat; mode switch | halo, thin ring (opacity .35), sparkle core | core `scale 1→1.03` and `translateY 0→-2px`, 6000ms `--ease-inout` infinite alternate. Nothing else moves. |
| **processing** | request submitted (before the first event); `meta`; any `status` with `state:"start"` for `rewriting|retrieving|searching|generating` | ring at .6 opacity; arc (28% of circumference) and cyan dot visible | `.ai-viz__spin` rotates 360deg, 1400ms linear infinite; core steady |
| **responding** | first `token` event, or `status` stage `generating` with `state:"first_token"` | arc eases to a full ring (stroke-dasharray `100 0`, 320ms one-shot), dot fades out | core opacity .85↔1 + scale 1→1.04, 2000ms ease-in-out infinite alternate |
| **error** | `error` event; fetch failure | ring and core tinted `--error`; `!` glyph shown in place of the sparkle; halo `--error-bg` | one-shot 240ms scale .96→1 then **static** |
Transitions between states: 320ms (`--dur-slow`) on opacity/stroke-dasharray, never on layout. `error` persists until the next send, new chat or mode switch (the error text itself is in the message); on `done` after an earlier error in a *new* request the state returns to idle. `done` → idle with a 320ms fade of the arc.
Driver API (view side; controllers call it through `Chat`/`UI`, never touching DOM): `AIViz.setState("idle|processing|responding|error", {target: "welcome"|messageId})`. `integration-developer` maps events: `meta|status:start → processing`, `first token → responding`, `done → idle`, `error → error`, abort → idle. Because `app.js` cannot reach DOM, `Chat.startAssistantMessage()` returns an object whose `setVizState(state)` is called from the SSE handlers (new method; cross-boundary 12.4).

### 8.4 Reduced motion
`@media (prefers-reduced-motion: reduce)`: all `.ai-viz` animations are `none`; the visual states remain distinguishable **statically**: idle = ring + core; processing = ring + 28% arc + dot at the top, no rotation; responding = full ring; error = alert glyph. Text status in the activity steps and `#liveStatus` carries the meaning.

### 8.5 Performance budget (hard)
- <= 8 SVG nodes per instance, <= 1.5 KB of markup; inline SVG, no raster/Lottie/canvas/WebGL; no animation library.
- Only `transform` and `opacity` animate continuously; `stroke-dasharray` changes are one-shot transitions; no `filter`, `box-shadow`, `blur`, `mask`, or `clip-path` animation.
- <= 1 running infinite animation per instance and <= 2 instances animating at once (welcome xor streaming message).
- Zero JavaScript in the animation loop: state is an attribute; no `requestAnimationFrame`, no timers.
- `will-change: transform` only on `.ai-viz__spin` and `.ai-viz__core` while `data-state` is animating.
- Animations stop when hidden (`display:none` on welcome hide; `animation-play-state:paused` on `[hidden]`/offscreen history); browsers also throttle background tabs.
- Acceptance: idle + processing add no visible long tasks in the Chrome performance panel and do not push the page above ~1% CPU on a mid-range laptop (QA to measure; this spec does not claim a measurement).

---------------------------------------------------------------------------------------------------

## 9. Icon plan

### 9.1 Library and shipping
**Lucide** (ISC license; portions from Feather are MIT). The project has no npm and re-renders with `innerHTML` everywhere, so a runtime script that scans the DOM (`lucide.createIcons()`) is rejected (CDN dependency, rescan after each render, flash of empty icons).
**Recommendation: a vendored inline SVG sprite.**
1. Download Lucide static SVGs into a fresh scratch directory (untrusted data rule), copy **only the ~40 icons used** into `<svg style="display:none"><symbol id="i-copy" viewBox="0 0 24 24">…</symbol>…</svg>` placed once at the top of `<body>` in `index.html`. Keep Lucide's original path data unmodified.
2. Usage anywhere (static HTML or `innerHTML` strings, no JS rescan needed): `<svg class="icon" aria-hidden="true"><use href="#i-copy"/></svg>`.
3. `css`: `.icon{width:20px;height:20px;flex:none;fill:none;stroke:currentColor;stroke-width:1.75;stroke-linecap:round;stroke-linejoin:round}` with `.icon--16` / `.icon--24`; spinner = `.icon--spin` (1s linear, reduced motion: none).
4. `js/icons.js` exposes `Icons.html(name, {size=20, cls=""}) → string` used by `ui.js`/`chat.js` templates (also accepts the legacy emoji by returning it escaped, for graceful degradation).
5. License text: add `frontend/assets/LICENSE-lucide.txt` (ISC) and a comment above the sprite. File size estimate: 40 symbols x ~250 B = ~10 KB inline, zero requests.
Names change between Lucide releases; **at copy time confirm each name exists in the downloaded version**; aliases below are given so the developer can match either.

### 9.2 Icon list (all 16/20/24 as noted, stroke 1.75)
| Use | Lucide name (current / master-prompt alias) | Size |
|---|---|---|
| Brand mark | `sparkles` (alt `bot`) | 20 in 32 tile |
| New chat | `plus` | 20 |
| Collapse / expand sidebar | `panel-left-close` / `panel-left-open` | 20 |
| Mobile menu | `menu` | 20 |
| Nav: General | `message-circle` | 20 |
| Nav: Documents | `file-text` | 20 |
| Nav: Research | `globe` | 20 |
| Nav: Code | `code-xml` (alias `code-2`) | 20 |
| Search | `search` | 16 |
| Clear | `x` | 16 |
| Session menu | `ellipsis` (alias `more-horizontal`) | 16 |
| Rename / Delete | `pencil` / `trash-2` | 16 |
| Theme | `sun`, `moon`, `monitor`, `check` | 20 / 16 |
| Model selector | `sparkles`, `chevron-down`, `check` | 16 |
| Attach / URL | `paperclip` / `link` | 20 |
| Web search toggle | `globe`, `check` | 16 |
| Send / Stop | `arrow-up` / `square` (filled) | 20 / 16 |
| Copy / copied | `copy` / `check` | 16 |
| Regenerate | `rotate-ccw` | 16 |
| Feedback | `thumbs-up` / `thumbs-down` | 16 |
| Cached / fallback | `zap` / `shuffle` | 16 |
| Sources | `external-link`, `globe`, `file-text` | 16 |
| Activity steps | `pencil`, `file-search`, `globe`, `sparkles`, `circle-check`, `circle-alert`, `loader-circle` | 16 |
| Status / toast | `circle-check` (alias `check-circle-2`), `circle-alert` (alias `alert-circle`), `info`, `triangle-alert` | 20 |
| Upload / dropzone | `upload` | 24 |
| File types | `file-text` (pdf, docx, txt, md, rst), `file-spreadsheet` (csv, xlsx), `globe` (url), `file` (other) | 16/20 |
| Scroll-to-latest | `arrow-down` | 16 |
| User line | `user` | 16 |

### 9.3 Emoji → icon mapping (every emoji used as an icon today)
| Location | Emoji | Replace with |
|---|---|---|
| `index.html` favicon data-URI | 🤖 | `assets/logo.svg` (or a sparkles SVG data-URI with `fill` attribute for both themes) |
| `index.html` dropzone icon | 📂 | `upload` 24 |
| `index.html` upload toast title | 📤 | `loader-circle` (spinning) / `upload` |
| `index.html` citation modal title | 📌 | `file-text` / `globe` by citation type (no pin glyph; pins don't exist) |
| `index.html` URL modal title | 🌐 | `link` |
| `index.html` modal close | ✕ | `x` |
| `index.html` mode tabs | 💬 📄 🌐 🧮 | `message-circle`, `file-text`, `globe`, `code-xml` |
| `index.html` theme toggle (via `ui.js`) | 🌙 ☀️ | `moon`, `sun`, `monitor` |
| `index.html` welcome logo | 🤖 (56px) | AIVisualization `lg` |
| `index.html` starter chips | 📄 🌐 🧮 ⚛️ 🔧 ✉️ | removed (cards + text-only chips) |
| `index.html` hamburger (3 spans) | – | `menu` |
| `index.html` sidebar docs label | 📚 | none (Documents nav + count) |
| `ui.js` session row | 💬 📄 🌐 🧮 | same four nav icons by `s.mode` |
| `ui.js` doc type | 📄 📝 📊 📃 🌐 📎 | `file-text` (pdf/docx/txt/md/rst), `file-spreadsheet` (csv/xlsx), `globe` (url), `file` (fallback) |
| `ui.js` doc/chip remove | ✕ | `x` |
| `ui.js` delete session | inline trash svg | `trash-2` |
| `ui.js` toast icons | ℹ️ ✅ ❌ | `info`, `circle-check`, `circle-alert` |
| `ui.js` citation source | 📌 | `file-text` / `globe` |
| `chat.js` bot avatar | inline bot svg | AIVisualization `sm` |
| `chat.js` copy | inline svg | `copy` |
| `chat.js` thought steps icon param | emoji strings from `app.js` | icon **names**: rewriting ✏️→`pencil`, retrieving 🔍→`file-search`, searching 🌐→`globe`, generating 🤖→`sparkles`, done ✅→`circle-check`, error ⚠️→`circle-alert` (`Chat.addThought/resolveThought` accept a name or legacy text) |
| `chat.js` badges | ⚡ ↪ | `zap`, `shuffle` |
| `chat.js` actions | ↻ 👍 👎 | `rotate-ccw`, `thumbs-up`, `thumbs-down` |
| `chat.js` citation card | 📌 | `file-text` / `globe` |
| `chat.js` error | ⚠️ | `circle-alert` |
| `utils.js` code copy | inline svg, "✓ Copied!" | `copy` / `check` + "Copied" |
| `app.js` toast strings | ✅ in message text | strip (icon comes from toast type) |

---------------------------------------------------------------------------------------------------

## 10. Not designable from current data

| Master-prompt implication | Why not | Honest alternative in this spec |
|---|---|---|
| Model capabilities, latency, price, context window | `GET /models` returns only `id,label,provider,description,available,reason,default` | Row = label, provider group, description, availability/reason, selected check |
| Pin conversations | `PATCH /sessions` has only `title, model, mode` | No pin control anywhere. Menu = Rename, Delete |
| Document size, status, progress, processing/indexing stages | `GET /documents` has `filename,file_type,chunk_count,indexed_at`; upload returns after indexing, no progress events | Show type, chunk count, indexed date; browser-known file size only at upload; **indeterminate** progress, never a percentage; "Indexed" only after POST resolves |
| "Supported file types and limits from the backend" | No endpoint exposes them (`MAX_UPLOAD_MB` only via 413) | Types listed as static copy (backend accepts `pdf docx csv xlsx txt md rst`; frontend whitelist currently omits `rst` — drift, **[open question 4]**); limit shown only as the server's 413 message |
| Settings entry / theme preferences page | No user settings exist | Theme control lives in the top bar; no Settings button (would be dead) |
| User section with account/avatar | There is no authentication, only a client `user_id` | Plain line "User: {id}" with a `user` icon; no avatar, no profile, no sign-out |
| Search across conversation content | Sessions list is title-based **[verify-with-audit]** | Placeholder "Search chats" filters titles only; do not imply message search |
| "Free" model claims, privacy/security badges, accuracy claims | Not verifiable from data | Omitted |
| Code execution / preview, share/export, line numbers | No backend support | Not designed |
| Real upload % progress | Not provided | Indeterminate bar |
| Server-side usage/cost display | `done` carries `usage` and `timings`, but their shape/units are not verified for display | Not shown in v1; step elapsed times only if provided by `status` events **[verify-with-audit]** |
| Research as a separate source-discovery product (`/search`, `/retrieve`) | UI doesn't call them; Research is a chat mode | Research workspace = chat mode with a distinct Sources block |

---------------------------------------------------------------------------------------------------

## 11. Figma

`mcp__claude_ai_Figma__whoami` was run once and **succeeded**: Figma is connected, authenticated as a shared IT-admin account
with access to three teams/plans. That is a different account from the git/session user; confirm with the user which account/team should own any file.

**No Figma file was created.** Creating one is outward-facing and needs the user's consent plus the Figma skills loaded in the main session.
If approved, a Figma build would contain: (1) a **Variables** collection with Light/Dark modes for every token in 2.3 plus spacing/radius/type styles;
(2) a **Components** page: Button (4 variants x 5 states), Icon button, Input, Nav item, Session row (+menu), Chip, Badge, Toast, Dialog, Model option, Feature card, Message (user/assistant), CodeBlock, Source row, AI-viz (4 states, static frames); (3) **Screens** at 1440, 768, 390: Welcome, Chat streaming, Chat error, Documents empty/list, Research answer+sources, Code, drawer open, model popover, delete dialog, in light and dark. The AI-viz motion would be annotated, not prototyped.

---------------------------------------------------------------------------------------------------

## 12. Implementation notes for `frontend-developer`

### 12.1 Order of work
1. **Tokens + theme** (`main.css` tokens per 2.3/2.9, `syntax.css`, no-flash snippet, `theme.js`, remove `#hljs-theme` swap, remove gradients/blur/glow). Keep legacy variable names temporarily aliased (`--bg-primary: var(--app-bg)` …) for one commit, then delete after converting `components.css`.
2. **Icons** (sprite + `icons.js` + CSS) and replace all emoji in markup/templates (9.3).
3. **Shell**: sidebar (expanded/rail/drawer), topbar, nav move (`#modeTabs` into sidebar), skip link, `viewport-fit`, dvh/safe-area dock.
4. **Primitives**: buttons/inputs/chips, tooltip, `components/menu.js`, `components/dialog.js`, toast.
5. **Components**: ThemeSwitcher, ModelSelector (wrapping `#modelSelect`), SessionRow+menu+rename, composer (toolbar, chips, toggle pill), messages (assistant flat layout, actions, badges, sources, code block), AIVisualization.
6. **Screens**: Welcome, ModeEmptyStates, DocumentsWorkspace, Research sources, Code.
7. **Polish**: reduced motion, 200% zoom pass, long-content pass, both themes screenshot pass.
Write `docs/redesign/view-contract.md` before changing signatures.

### 12.2 Ids and functions that must survive
`sidebar, sidebarBackdrop, sidebarToggle, newChatBtn, sessionSearch, sessionList, docSection, docList, userIdDisplay, themeToggle, modeTabs (+ .mode-tab[data-mode]), modelSelect, chatArea, welcomeScreen, chatInput, sendBtn, stopBtn, attachBtn, fileInput, addUrlBtn, searchToggle, uploadToast*, uploadProgress, dropzoneOverlay, citationModal(+Close/Source/Snippet), urlModal(+Close/Cancel/Submit), urlInput`. `#starterPrompts` is removed only if `app.js` guards null (it currently does `querySelector…` **[verify-with-audit]**; otherwise keep an empty hidden element until `integration-developer` updates it). `UI.*`, `Chat.*`, `Utils.*` public names persist; additions only.

### 12.3 Acceptance checks QA should apply
1. `node --check frontend/js/**/*.js`; `venv/bin/pytest -q tests/test_api.py` (serves `/js/app.js`, `/css/main.css`).
2. Both themes: screenshot of Welcome, Chat with code block and table, Documents list, Research with sources, open model popover, open menus, dialog, toast, at 1440, 1024, 768, 390, 360.
3. No hardcoded hex outside the token blocks (grep `#[0-9a-fA-F]{3,8}` in `components.css`, `animations.css`, `index.html`, `js/`); no emoji used as icons (grep ranges); no `gradient`, `blur(`, `box-shadow` except `--elev-*`.
4. Contrast spot-check: computed values for the 6 hardest pairs (secondary on sidebar active both themes, accent text on dark active row not used, placeholder, control border, warning on tint, error on tint) match 2.4.
5. Keyboard: Tab order matches visual; every control reachable; menus/listbox/dialog/drawer behave per 6; focus visible in both themes; drawer traps and restores focus; Esc closes everything it should.
6. Screen reader: icon-only buttons named; `aria-current` on nav; `aria-pressed` on toggles; `liveStatus` announces stages, not tokens.
7. `prefers-reduced-motion` emulation: no running animations, viz static states distinct.
8. No horizontal page overflow at 360 with a 300-char unbroken string, a wide table and a 200-char code line; 200% zoom at 1280 wide reflows.
9. Composer visible without scrolling at 1366x768 and 1280x600; mobile with keyboard open (Android Chrome emulation).
10. Theme: switch Light/Dark/System with no reload; reload shows no wrong-theme flash (record first-frame screenshot); System follows OS change live; legacy `theme=dark` migrates to System.
11. Every visible control performs an action (no dead buttons); no Pin; no fabricated metadata.

### 12.4 Cross-boundary requests (for the orchestrator)
To **integration-developer** (`app.js`, `workflows/**`):
1. Replace both `confirm()` calls (`app.js:336`, `:400`) with `await UI.confirm({...})`.
2. Add Rename: session menu callback `onRename(sessionId, newTitle)` → `PATCH /sessions/{id}`; `renderSessionList` gains `onRename`, `onRetry`.
3. Drive the AI viz: call `msg.setVizState` per 8.3 mapping; call `UI.setLiveStatus(text)` at stage changes/complete/error.
4. Pass icon **names** (not emoji) to `addThought/resolveThought` (`pencil|file-search|globe|sparkles|circle-check|circle-alert`); drop emoji in toast strings (`app.js:371,391`).
5. Feature cards: handle `data-mode` card click → `setMode(mode)` + focus `#chatInput`; remove the `#starterPrompts` `data-prompt` handler or retarget it to the suggestion chips (`data-prompt` kept).
6. Upload flow: retain the `File` for Retry; expose `onUploadRetry`; remove the fake progress; one drop → toast "Only the first file was added." if >1 file.
7. Model fetch retry hook `onRetryModels`; keep reading `#modelSelect.value` and the `change` event.
8. `UI.setActiveMode` now sets `aria-current="page"` (frontend-developer implements in `ui.js`; no app.js change).
9. Documents mode: `renderDocList(documents, onDelete)` also feeds the count badge.
To **frontend-developer**: `app.js:87` needs `#themeToggle` to exist; keep it a `<button>`.
To **orchestrator**: confirm whether `rst` should be added to the accept list; decide on the open questions below.

### 12.5 Risks
- **R1 Markup moves vs. id binding**: mode tabs moved from topbar to sidebar; `querySelectorAll(".mode-tab")` still works only if class+`data-mode` are preserved.
- **R2 Native `<dialog>`** changes how `display:flex` toggling in `ui.js` worked; replace both sides together.
- **R3 iOS Safari keyboard** overlay may hide the composer; fallback: `visualViewport` resize listener sets `--vvh` (not specified further; try `100dvh` first and test).
- **R4 Container queries** (Chrome 105+, Safari 16+, Firefox 110+): provide `@media` fallbacks for the feature grid.
- **R5 Auto-scroll change** (sticky-to-bottom) touches `chat.js` only but alters feel.
- **R6 Font loading**: moving Inter from CSS `@import` to `<link>` must keep `display=swap`.
- **R7 Contrast regressions** when someone adds a color: every new text/background pair must be added to 2.4.
- **R8 Audit drift**: items tagged **[verify-with-audit]** may need changes once `audit.md` exists.

### 12.6 Open questions for the user
1. Visual references: do they exist? None are in `docs/redesign/references/`; this design follows the written brief only.
2. Is **no Settings entry and no Pin** acceptable until the backend adds those? (Pin requires a backend field.)
3. Keep example prompt chips (text-only, per non-General mode) or drop them entirely now that feature cards replace the six starter prompts?
4. Add `rst` to the frontend upload whitelist (backend accepts it)?
5. Should an indexed-document chip's "×" only dismiss the chip (spec) or delete the document?
6. Which Figma account/team should own a Figma file, and do you want one built? (The connected account is a shared IT-admin account, not the session user.)
7. Should the default theme for existing users be **System** (spec) or stay Dark?
