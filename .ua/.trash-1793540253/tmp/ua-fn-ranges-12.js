const fs = require("fs");
const path = require("path");
const root = "F:/Personal Projects/AgenticAI_Chatbot_Using_RAG";
const files = [
  "frontend/js/components/radiogroup.js",
  "frontend/js/components/settings.js",
  "frontend/js/components/workspaces.js",
  "frontend/js/icons.js",
  "frontend/js/intro.js",
  "frontend/js/theme.js",
  "frontend/js/ui.js",
  "frontend/js/utils.js",
];

function extract(src) {
  const lines = src.split(/\n/);
  const fns = [];
  const re = /^  (async )?function ([A-Za-z0-9_]+)\s*\(/;
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(re);
    if (!m) continue;
    let depth = 0;
    let started = false;
    let end = i;
    for (let j = i; j < lines.length; j++) {
      for (const ch of lines[j]) {
        if (ch === "{") {
          depth++;
          started = true;
        } else if (ch === "}") {
          depth--;
          if (started && depth === 0) {
            end = j;
            break;
          }
        }
      }
      if (started && depth === 0) break;
    }
    fns.push({ name: m[2], start: i + 1, end: end + 1, lines: end - i + 1 });
  }
  const ret = src.match(/return \{([\s\S]*?)\};\s*\}\)\(\);/);
  let pubs = [];
  if (ret) {
    pubs = ret[1]
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => {
        const mm = s.match(/^([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)/);
        if (mm) return { exportName: mm[1], fn: mm[2] };
        const n = s.match(/^([A-Za-z0-9_]+)/);
        return n ? { exportName: n[1], fn: n[1] } : null;
      })
      .filter(Boolean);
  }
  return { fns, pubs };
}

for (const f of files) {
  const src = fs.readFileSync(path.join(root, f), "utf8");
  const { fns, pubs } = extract(src);
  const pubSet = new Set(pubs.map((p) => p.fn));
  console.log(
    "\n## " +
      f +
      " pubs=" +
      pubs.map((p) => (p.exportName === p.fn ? p.fn : p.exportName + "=>" + p.fn)).join(",")
  );
  for (const fn of fns) {
    const keep = pubSet.has(fn.name) || fn.lines >= 10;
    console.log(
      (keep ? "KEEP" : "skip") +
        " " +
        fn.name +
        " " +
        fn.start +
        "-" +
        fn.end +
        " (" +
        fn.lines +
        ")" +
        (pubSet.has(fn.name) ? " EXP" : "")
    );
  }
}
