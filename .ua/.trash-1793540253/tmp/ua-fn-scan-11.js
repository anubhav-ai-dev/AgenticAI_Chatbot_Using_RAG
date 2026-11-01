const fs = require("fs");
const path = require("path");
const root = "F:/Personal Projects/AgenticAI_Chatbot_Using_RAG";
const files = [
  "frontend/js/api.js",
  "frontend/js/app.js",
  "frontend/js/chat.js",
  "frontend/js/components/ai-viz.js",
  "frontend/js/components/code-preview.js",
  "frontend/js/components/dialog.js",
  "frontend/js/components/documents.js",
  "frontend/js/components/menu.js",
  "frontend/js/components/model-select.js",
  "frontend/js/components/motion.js",
  "frontend/js/components/notifications.js",
  "frontend/js/components/prefs.js",
];

function skipString(text, i) {
  const q = text[i];
  i++;
  while (i < text.length) {
    if (text[i] === "\\") { i += 2; continue; }
    if (text[i] === q) return i + 1;
    i++;
  }
  return i;
}

function extract(text) {
  const funcs = [];
  const re = /(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(/g;
  let m;
  while ((m = re.exec(text))) {
    const name = m[1];
    let i = m.index + m[0].length - 1; // at '('
    let depth = 0;
    let endParams = -1;
    for (; i < text.length; i++) {
      const ch = text[i];
      if (ch === "'" || ch === '"' || ch === "`") { i = skipString(text, i) - 1; continue; }
      if (ch === "(") depth++;
      else if (ch === ")") {
        depth--;
        if (depth === 0) { endParams = i; break; }
      }
    }
    if (endParams < 0) continue;
    let j = endParams + 1;
    while (j < text.length && text[j] !== "{" && text[j] !== ";" && text[j] !== "\n") j++;
    if (text[j] !== "{") continue;
    let b = 0;
    let end = -1;
    for (let k = j; k < text.length; k++) {
      const ch = text[k];
      if (ch === "'" || ch === '"' || ch === "`") { k = skipString(text, k) - 1; continue; }
      if (ch === "{") b++;
      else if (ch === "}") {
        b--;
        if (b === 0) { end = k; break; }
      }
    }
    if (end < 0) continue;
    const startLine = text.slice(0, m.index).split("\n").length;
    const endLine = text.slice(0, end).split("\n").length;
    const indent = (text.slice(text.lastIndexOf("\n", m.index) + 1, m.index).match(/^\s*/) || [""])[0].length;
    funcs.push({ name, start: startLine, end: endLine, span: endLine - startLine + 1, indent });
    re.lastIndex = endParams + 1;
  }
  return funcs;
}

const out = {};
for (const f of files) {
  const text = fs.readFileSync(path.join(root, f), "utf8");
  const funcs = extract(text);
  // last top-level-ish return of the IIFE: find "return {" nearest the end before "})();"
  const closeAt = text.lastIndexOf("})();");
  const retAt = text.lastIndexOf("return {", closeAt);
  let exported = [];
  if (retAt > 0) {
    const slice = text.slice(retAt, closeAt);
    const body = slice.match(/return\s*\{([\s\S]*)\}\s*$/);
    if (body) {
      exported = [...body[1].matchAll(/([A-Za-z_$][\w$]*)\s*(?:,|:)/g)].map(x => x[1])
        .filter(n => !["true","false","null"].includes(n));
    }
  }
  out[f] = {
    exported,
    funcs: funcs.filter(fn => fn.span >= 10 || exported.includes(fn.name)).map(fn => ({...fn, exported: exported.includes(fn.name)}))
  };
}
fs.writeFileSync(path.join(root, ".ua/tmp/ua-fn-list-11.json"), JSON.stringify(out, null, 2));
let n = 0;
for (const f of files) {
  console.log("\n## " + f + " export=" + out[f].exported.join(","));
  for (const fn of out[f].funcs) {
    n++;
    console.log(`  ${fn.exported ? "E" : " "} ${fn.name} ${fn.start}-${fn.end} (${fn.span}) i${fn.indent}`);
  }
}
console.log("\nTOTAL", n);
