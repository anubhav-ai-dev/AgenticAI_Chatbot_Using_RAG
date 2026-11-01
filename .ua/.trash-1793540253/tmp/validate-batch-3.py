import json
from pathlib import Path

root = Path(r"F:\Personal Projects\AgenticAI_Chatbot_Using_RAG\.ua")
graph = json.loads((root / "intermediate" / "batch-3.json").read_text(encoding="utf-8"))
prompt = json.loads((root / "tmp" / "batch-prompt-3.json").read_text(encoding="utf-8"))

ids = {n["id"] for n in graph["nodes"]}
neighbor_files = set()
neighbor_symbols = set()  # (path, symbol)
for src, neighbors in prompt["neighborMap"].items():
    neighbor_files.add(src)
    for n in neighbors:
        neighbor_files.add(n["path"])
        for s in n["symbols"]:
            neighbor_symbols.add((n["path"], s))

import_files = set()
for src, targets in prompt["batchImportData"].items():
    import_files.add(src)
    import_files.update(targets)

weights = {
    "contains": 1.0,
    "imports": 0.7,
    "calls": 0.8,
    "inherits": 0.9,
    "implements": 0.9,
    "exports": 0.8,
    "depends_on": 0.6,
    "tested_by": 0.5,
    "configures": 0.6,
    "documents": 0.5,
    "deploys": 0.7,
    "migrates": 0.7,
    "triggers": 0.6,
    "defines_schema": 0.8,
    "serves": 0.7,
    "provisions": 0.7,
    "routes": 0.6,
    "related": 0.5,
}

bad = []
for i, e in enumerate(graph["edges"]):
    if e["source"] == e["target"]:
        bad.append(f"self {i}")
    if e["direction"] != "forward":
        bad.append(f"dir {i}")
    if e["type"] not in weights or e["weight"] != weights[e["type"]]:
        bad.append(f"weight {i} {e['type']} {e['weight']}")
    for end in (e["source"], e["target"]):
        if end in ids:
            continue
        if end.startswith("file:"):
            path = end[len("file:"):]
            if path in neighbor_files or path in import_files:
                continue
            bad.append(f"file-miss {end}")
            continue
        if end.startswith("function:") or end.startswith("class:"):
            kind, rest = end.split(":", 1)
            # path may contain colons? no, windows not in ids. path has no colon.
            path, _, symbol = rest.rpartition(":")
            if (path, symbol) in neighbor_symbols:
                continue
            bad.append(f"symbol-miss {end}")
            continue
        bad.append(f"unknown {end}")

# import count
imp = [e for e in graph["edges"] if e["type"] == "imports"]
expected = sum(len(v) for v in prompt["batchImportData"].values())
print("import edges", len(imp), "expected", expected)
missing_imp = []
for src, targets in prompt["batchImportData"].items():
    got = {e["target"] for e in imp if e["source"] == f"file:{src}"}
    for t in targets:
        if f"file:{t}" not in got:
            missing_imp.append((src, t))
print("missing imports", missing_imp)
print("bad edges", len(bad))
for b in bad[:40]:
    print(" ", b)

# duplicate edges
seen = set()
dups = 0
for e in graph["edges"]:
    key = (e["source"], e["target"], e["type"])
    if key in seen:
        dups += 1
        print("dup", key)
    seen.add(key)
print("dups", dups)
print("file size", (root / "intermediate" / "batch-3.json").stat().st_size)
