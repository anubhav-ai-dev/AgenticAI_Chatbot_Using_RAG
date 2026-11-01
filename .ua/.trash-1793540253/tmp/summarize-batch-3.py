import json

p = r"F:\Personal Projects\AgenticAI_Chatbot_Using_RAG\.ua\tmp\ua-file-extract-results-3.json"
with open(p, encoding="utf-8") as f:
    data = json.load(f)

for r in data["results"]:
    print("=" * 80)
    print(r["path"], "lines", r["totalLines"], "nonempty", r["nonEmptyLines"])
    print("METRICS", r.get("metrics"))
    print("--- FUNCTIONS ---")
    for fn in r.get("functions") or []:
        span = fn["endLine"] - fn["startLine"] + 1
        print(f"  {fn['name']} {fn['startLine']}-{fn['endLine']} ({span}) params={fn.get('params')}")
    print("--- CLASSES ---")
    for c in r.get("classes") or []:
        span = c["endLine"] - c["startLine"] + 1
        print(f"  {c['name']} {c['startLine']}-{c['endLine']} ({span}) methods={c.get('methods')} props={c.get('properties')}")
    print("--- EXPORTS ---")
    for e in r.get("exports") or []:
        print(f"  {e['name']} line {e['line']} default={e.get('isDefault')}")
    print("--- CALLS (project-looking) ---")
    seen = set()
    for cg in r.get("callGraph") or []:
        callee = cg["callee"]
        # skip builtins-ish
        if callee.split(".")[0] in {
            "isinstance", "getattr", "int", "str", "len", "list", "dict", "set",
            "print", "range", "enumerate", "zip", "any", "all", "min", "max",
            "sorted", "sum", "float", "bool", "type", "super", "hasattr",
            "asyncio", "json", "os", "time", "logging", "pytest", "re",
        }:
            continue
        key = (cg["caller"], callee)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {cg['caller']} -> {callee} @ {cg['lineNumber']}")
