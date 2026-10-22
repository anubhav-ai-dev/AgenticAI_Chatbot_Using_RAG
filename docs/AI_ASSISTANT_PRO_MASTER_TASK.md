# AI ASSISTANT PRO — FIGMA-QUALITY DESIGN AND PRODUCTION FRONTEND IMPLEMENTATION

## Role and mission

You are a senior product designer, professional Figma design-system architect, senior frontend engineer, and SaaS UX specialist.

Your task is to transform my existing AI Assistant Pro application into an industry-ready AI productivity product.

I want a polished, modern, friendly, and professional experience that makes users curious about the capabilities of AI while remaining clean, accessible, and easy to use.

Treat this as a real commercial product that must be maintainable, extensible, and ready for continued product development.

Do not merely create a pretty landing page or an isolated dashboard mockup. Implement the actual interface in the existing project and preserve its working functionality.

### Product vision

AI Assistant Pro brings multiple AI workflows into one workspace:

- General AI chat.
- Document upload, analysis, and question answering.
- Web research and source discovery.
- Code generation, explanation, and debugging.
- AI model selection and existing model fallback.
- Persistent conversation history.
- A unified message composer.

The interface must make these capabilities understandable to new users and efficient for frequent users.

---

## PHASE 1 — INSPECT THE PROJECT BEFORE CHANGING IT

Before editing files:

1. Inspect the repository structure.
2. Identify the frontend framework, language, package manager, build system, routing, and styling architecture.
3. Inspect the current dashboard, conversation components, navigation, model selector, composer, and responsive behavior.
4. Inspect existing API clients, authentication, conversation persistence, document processing, research, code, streaming, and fallback logic.
5. Identify existing UI libraries and reusable components.
6. Run the current application and inspect the existing interface where possible.
7. Identify existing tests and commands for linting, type checking, and building.
8. Create a short implementation plan based on the real project.

Do not assume the project uses React, Bootstrap, Tailwind, or any particular framework.

### Preserve existing functionality

- Do not replace real API calls with mock responses.
- Do not change backend contracts unnecessarily.
- Preserve authentication and authorization.
- Preserve existing environment variables and configuration.
- Preserve model selection and automatic fallback.
- Preserve streaming responses and cancellation.
- Preserve conversation history and persistence.
- Preserve document upload, indexing, and retrieval.
- Preserve existing web-search and research integrations.
- Preserve error handling and existing application behavior.
- Do not expose API secrets in frontend code.
- Do not remove working features simply to simplify the UI.

This is primarily a frontend redesign and UX improvement.

---

## PHASE 2 — DESIGN THE PRODUCT BEFORE IMPLEMENTATION

Think like a Figma designer before writing components.

Define the application structure, design tokens, component hierarchy, interaction states, and responsive layouts.

If Figma access or a suitable Figma integration is available, create or update the actual Figma design file. Otherwise, implement a faithful, documented design system in the existing frontend project. Do not claim to have created a native Figma file unless one was actually created.

Use the supplied AI Assistant Pro visual references as the design direction.

The final design should be:

- Minimal and premium.
- Professional and trustworthy.
- Friendly to first-time users.
- Efficient for experienced users.
- Visually consistent.
- Accessible.
- Responsive.
- Suitable for long-term product growth.

Avoid excessive gradients, neon glows, decorative cards, large static illustrations, unnecessary animations, and competing visual elements.

Use whitespace and typography to establish hierarchy.

---

## PHASE 3 — ESTABLISH THE DESIGN SYSTEM

Create reusable theme tokens using CSS variables or the existing theme system.

Suggested light theme:

- App background: #F7F8FC
- Main surface: #FFFFFF
- Sidebar background: #F0F2FA
- Sidebar hover: #E7EAF6
- Sidebar active: #E0E7FF
- Primary accent: #4F46E5
- Primary hover: #4338CA
- Primary text: #172033
- Secondary text: #667085
- Border: #E1E6F0
- Success: #087F68
- Warning: #A16207
- Error: #B42318

Suggested dark theme:

- App background: #0B1020
- Main surface: #11182A
- Sidebar background: #0F1728
- Elevated surface: #182238
- Sidebar hover: #1A2840
- Sidebar active: #26345B
- Primary accent: #8B83FF
- Primary hover: #A49EFF
- Primary text: #F1F5FF
- Secondary text: #B0BDD3
- Border: #2B3850

Treat these as starting values. Verify contrast and adjust the final palette.

### Typography

Use the existing product font if appropriate. Otherwise, choose a professional interface font such as Inter.

Define a consistent type scale for:
- Page headings.
- Section headings.
- Body text.
- Secondary descriptions.
- Labels.
- Timestamps and metadata.

Use consistent line heights, font weights, and readable content widths.

### Spacing and geometry

Use an 8px spacing system with consistent:
- Button heights.
- Input heights.
- Card padding.
- Sidebar rows.
- Icon dimensions.
- Border radii.
- Section spacing.
- Dropdown alignment.
- Responsive breakpoints.

Use subtle borders and restrained shadows. Avoid applying shadows to every component.

Create reusable buttons, inputs, cards, menus, dialogs, tooltips, and status indicators.

---

## PHASE 4 — APPLICATION LAYOUT

### Important: no right sidebar

The application must have only two primary regions:

1. Left navigation sidebar.
2. Main workspace.

Do not implement a permanent right sidebar.

The central workspace should use all available horizontal space and adapt to the selected mode.

### Left sidebar

Design a soft, light-colored sidebar using muted lavender-gray surfaces.

It must not use a dark background in light mode.

Include:

**Brand**
- AI Assistant Pro logo.
- Product name.
- Optional short tagline.
- Sidebar collapse control.

**Primary action**
- New Chat button.
- Clear primary action styling.
- Accessible hover, focus, pressed, and disabled states.

**Navigation**
- General.
- Documents.
- Research.
- Code.

**Conversation history**
- Search input.
- Today, Yesterday, Previous 7 Days, and older groups when relevant.
- Conversation titles.
- Timestamps when available.
- Rename, pin, and delete where supported by existing functionality.
- Empty and loading states.

**Document access**
- My Documents entry linked to the existing document workflow.
- Actual document status or count only when available.

**User section**
- Existing user information.
- Settings entry.
- Theme preferences.

The sidebar must be collapsible on desktop and become an accessible drawer on mobile.

Keep navigation labels readable and the active item visually clear without oversized gradients.

### Main workspace

Include:

- Compact top navigation.
- Mode-specific content.
- Welcome screen when appropriate.
- Conversation history and responses when a conversation is active.
- A bottom-aligned composer.
- Contextual controls relevant to the current mode.

Avoid wasting screen space on decorative headers.

The layout should work on typical laptop screens without forcing the user to scroll just to reach the composer.

---

## PHASE 5 — LIGHT AND DARK MODE

Implement a complete theme system.

### Theme control

Place an accessible sun/moon control in the top navigation.

Provide:
- Light.
- Dark.
- System preference, if practical.

Use clear tooltips and selected-state indicators.

### Behavior

- Switch themes without reloading the application.
- Update the entire interface, not only the background.
- Preserve text contrast.
- Update sidebar, cards, borders, inputs, dropdowns, menus, dialogs, and code blocks.
- Persist the selected preference using the application's existing settings mechanism or local storage if appropriate.
- Respect the operating system preference when System is selected.
- Avoid a visible flash of the wrong theme during initialization.
- Keep theme state consistent across application routes.

Do not implement dark mode by applying a global black background while leaving text, borders, or form controls unchanged.

### Contrast and accessibility

Validate important text and controls against WCAG AA contrast targets where applicable.

Provide visible keyboard focus states in both themes.

Ensure the selected theme is understandable to users without relying on color alone.

---

## PHASE 6 — ANIMATED AI IDENTITY

Do not use a static hero image as the primary dashboard centerpiece.

Build a lightweight animated AI visualization using CSS, SVG, or an existing animation library where appropriate.

The visualization should be tasteful and purposeful.

### Animation states

Idle:
- Very subtle floating or breathing motion.
- Soft indigo or cyan accents.
- No aggressive pulsing.

Processing:
- A gentle rotating ring, orbit, or progress indicator.
- Communicate that the assistant is working.

Responding:
- A brief visual state transition while a response is being generated.
- Return to idle when complete.

Error:
- Use a clear error state, not a dramatic animation.

### Performance requirements

- Prefer CSS and lightweight SVG effects.
- Do not create a costly particle simulation.
- Avoid large animation libraries unless justified.
- Respect reduced-motion preferences.
- Avoid animations that interfere with text or controls.
- Keep the animation secondary to the actual workflow.

The assistant should feel alive without becoming distracting.

---

## PHASE 7 — WELCOME SCREEN

Use the heading:

"What can I help you with?"

Supporting text:

"Chat, analyze documents, research the web, and write code in one workspace."

Show four interactive feature cards.

### General Chat
Icon: MessageCircle or MessagesSquare.

Purpose: Ask questions, brainstorm, and explore ideas.

### Analyze Documents
Icon: FileText or FileSearch.

Purpose: Upload supported files, summarize information, and ask questions about documents.

### Web Research
Icon: Globe or Search.

Purpose: Discover information using existing research and search capabilities.

### Write & Debug Code
Icon: Code2 or Terminal.

Purpose: Generate, explain, and review code.

Each card must have:
- A consistent layout.
- A recognizable icon.
- A concise title.
- A short description.
- A restrained accent color.
- A clear hover and focus state.
- A functional click action.

Use a responsive grid that changes naturally from desktop to tablet and mobile.

Do not use a different saturated gradient for every card.

Below the cards, show a compact capabilities strip only if it adds real value. Suitable items include multiple AI models, file upload, web research, and privacy information.

Do not make unsupported security or capability claims.

---

## PHASE 8 — GENERAL CHAT

Implement a clean, readable conversation experience.

Requirements:
- Clear user and assistant message hierarchy.
- Markdown rendering.
- Lists and tables.
- Syntax-highlighted code blocks.
- Copy response and copy-code controls.
- Regenerate and feedback controls where supported.
- Streaming responses.
- Stop-generation control where supported.
- Loading indicators.
- Error and retry states.
- Conversation persistence.
- Appropriate scrolling behavior.

Use readable content widths and consistent vertical spacing.

Do not make every message a large colored bubble.

Preserve the existing message format, API integration, streaming behavior, and cancellation logic.

---

## PHASE 9 — MESSAGE COMPOSER

Create a refined, bottom-aligned composer with a white surface in light mode and an appropriately elevated surface in dark mode.

Include:
- Multiline auto-growing input.
- Attachment control.
- Existing file-upload integration.
- Web-search toggle when supported.
- Selected-model indicator.
- Send button.
- Attached-file chips.
- Upload progress.
- File validation feedback.
- Disabled and loading states.
- Stop-generation control where supported.
- Accessible tooltips and focus indicators.

Preserve existing keyboard behavior.

The composer must not cover messages or overflow on mobile.

Prevent duplicate submissions and preserve input where a request fails.

Every visible control must perform its intended action.

---

## PHASE 10 — MODEL SELECTION

Use a compact model selector in the top navigation and/or composer, according to the existing application design.

Show:
- Available model names.
- Provider metadata when available.
- Verified capabilities.
- Selected model.
- Search or filtering when useful.
- Loading and error states.

Use the existing model configuration and API flow.

Preserve automatic fallback.

Do not fabricate models, capabilities, availability, latency, or pricing.

Avoid permanently displaying a large model panel that consumes the main workspace.

---

## PHASE 11 — DOCUMENTS WORKSPACE

Implement a professional document-intelligence experience.

Include:
- Upload and drag-and-drop.
- Supported file types and limits from the backend.
- Existing document list.
- Filenames and sizes.
- Upload progress.
- Processing and indexing states.
- Error and retry states.
- Existing document retrieval and question-answering workflow.

Use a helpful empty state for users with no documents.

Once documents exist, prioritize document management and analysis over decorative upload graphics.

Do not replace the existing document-processing pipeline.

Do not claim successful indexing before the backend confirms it.

---

## PHASE 12 — RESEARCH WORKSPACE

Integrate the actual existing research or web-search workflow.

Include:
- Research input.
- Loading and progress states.
- Readable results.
- Source titles and domains when returned.
- Clickable source links.
- Citations when available.
- Empty, error, and retry states.

Do not fabricate sources or simulate real search results.

Keep source information distinguishable from generated summaries.

Preserve current backend contracts.

---

## PHASE 13 — CODE WORKSPACE

Create a clean developer-oriented interface.

Include:
- Readable Markdown.
- Syntax highlighting.
- Language labels when available.
- Copy-code actions.
- Horizontal scrolling for long lines.
- Clear separation between explanations and code.
- Existing model integration.

Add code execution or preview only when it is genuinely supported by the application.

Reuse existing dependencies where possible.

---

## PHASE 14 — ICONS AND INTERACTION DESIGN

Use Lucide icons if compatible with the existing project. Otherwise, reuse the established icon system.

Suggested icons:
- Bot, Sparkles.
- MessageCircle, MessagesSquare.
- Plus, Search, House.
- FileText, Files, FileSearch, Upload.
- Globe, ExternalLink.
- Code2, Terminal, Braces.
- Paperclip, ArrowUp, Square.
- Copy, RotateCcw.
- ThumbsUp, ThumbsDown.
- Ellipsis, Pin, Trash2, Pencil.
- Settings2, PanelLeftClose.
- Sun, Moon, Monitor.
- CheckCircle2, AlertCircle, LoaderCircle.

Maintain consistent icon sizing, stroke widths, and alignment.

Provide tooltips for icon-only controls.

Implement real interactions for:
- Navigation.
- Theme switching.
- Model selection.
- Conversation search.
- Context menus.
- File selection and removal.
- Composer submission.
- Streaming controls.
- Dialogs and confirmations.

Use clear success, error, and loading feedback.

Avoid dead buttons and decorative controls that appear interactive but do nothing.

---

## PHASE 15 — FRONTEND LIBRARIES AND ARCHITECTURE

First inspect the current technology stack.

Choose libraries based on compatibility and actual needs rather than adding every popular frontend tool.

Potential options:

- Bootstrap 5 for responsive grid and utilities if appropriate.
- Lucide for consistent icons.
- CSS variables for the theme system.
- The existing Markdown renderer.
- A suitable syntax-highlighting library.
- An existing accessible component library for dialogs and dropdowns.
- Lightweight CSS animation or SVG for the AI visualization.

Do not mix Bootstrap, Tailwind, and multiple component libraries without a clear reason.

If the project already uses React, Vue, or another framework, follow the established architecture.

If the project uses plain HTML, CSS, and JavaScript, retain that approach unless a migration is genuinely justified.

Use reusable components, modular styling, and separation of presentation from API logic.

Potential components include:
- AppShell.
- Sidebar.
- Navigation.
- ConversationHistory.
- TopNavigation.
- ThemeSwitcher.
- ModelSelector.
- WelcomeScreen.
- FeatureCard.
- AIVisualization.
- ChatConversation.
- ChatMessage.
- MessageComposer.
- AttachmentList.
- DocumentsWorkspace.
- ResearchWorkspace.
- CodeWorkspace.
- LoadingState.
- ErrorState.
- ConfirmationDialog.

Adapt these responsibilities to the actual project rather than forcing unnecessary abstractions.

---

## PHASE 16 — RESPONSIVE DESIGN

Desktop:
- Soft-color sidebar.
- Compact top bar.
- Spacious central workspace.
- No right sidebar.
- Bottom-aligned composer.

Tablet:
- Collapsible navigation.
- Responsive feature cards.
- Adaptable model selector.
- Appropriate workspace width.

Mobile:
- Accessible navigation drawer.
- Compact header.
- Single-column content.
- Touch-friendly buttons.
- Responsive composer.
- Scrollable code blocks.
- Appropriate dialogs and dropdowns.

Prevent horizontal overflow and clipped controls.

Test representative viewport widths and long conversation content.

---

## PHASE 17 — ACCESSIBILITY, SECURITY, AND PERFORMANCE

Implement:
- Semantic HTML.
- Keyboard navigation.
- Visible focus indicators.
- Accessible names for icon-only controls.
- Sufficient color contrast.
- Reduced-motion support.
- Safe Markdown rendering.
- Existing authorization checks.
- Appropriate error handling.
- Duplicate-submission prevention.
- Cleanup of subscriptions and event listeners.
- Responsive behavior at browser zoom.
- No unnecessary collection of user content.

Preserve backend validation and do not rely on frontend checks alone for security.

Optimize animation and rendering performance. Avoid unnecessary rerenders, oversized assets, and unjustified dependencies.

---

## PHASE 18 — IMPLEMENTATION AND VERIFICATION

Work in stages.

### Stage 1: Audit
Inspect the project and report the current architecture, existing functionality, and implementation plan.

### Stage 2: Design foundation
Create the theme tokens, typography, spacing, reusable controls, icon system, and light/dark theme behavior.

### Stage 3: Main shell
Implement the sidebar, top navigation, central workspace, responsive layout, and theme switcher.

### Stage 4: Core experience
Implement the animated AI visualization, welcome screen, feature cards, model selector, and composer.

### Stage 5: Integration
Connect General, Documents, Research, and Code to the existing backend and state management.

### Stage 6: Polish
Implement loading states, error handling, responsive behavior, accessibility, and subtle animations.

### Stage 7: Verification
Run the existing application and execute available:
- Build commands.
- Lint checks.
- Type checks.
- Unit tests.
- Integration tests.

Test theme switching, navigation, model selection, sending messages, streaming, file upload, research, and responsive behavior where possible.

Fix regressions introduced by the redesign.

Do not claim a test passed unless it was actually executed.

Do not stop after producing a plan or an isolated visual prototype. Implement the actual application changes.

---

## DEFINITION OF DONE

The redesign is complete when:

1. The application feels like a coherent, premium SaaS product.
2. The sidebar has a soft, professional color scheme.
3. There is no permanent right sidebar.
4. Light and dark themes have readable contrast.
5. The theme preference persists appropriately.
6. The AI visualization is animated and performance-conscious.
7. General, Documents, Research, and Code remain functional.
8. Existing backend contracts and authentication are preserved.
9. Model selection, fallback, and streaming continue to work.
10. The composer and file-upload workflow remain functional.
11. All visible controls have real actions.
12. Responsive layouts work without horizontal overflow.
13. Accessibility and error states have been considered.
14. The project builds successfully, or pre-existing failures are documented.
15. No mock data replaces working production functionality.

## FINAL REPORT

After implementation, provide:
- Summary of design and UX improvements.
- Files created and modified.
- Components and design tokens introduced.
- Theme implementation details.
- Libraries added and justification.
- Backend functionality preserved.
- Build, lint, type-check, and test results.
- Remaining issues.

Final directive: Think like a Figma product designer, establish a consistent design system, and implement it faithfully in the existing codebase. Build a functional product experience, not a static mockup.