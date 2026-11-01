import json
p = r"F:\Personal Projects\AgenticAI_Chatbot_Using_RAG\.ua\tmp\ua-file-extract-results-6.json"
with open(p, encoding="utf-8") as f:
    data = json.load(f)
print("files", data["filesAnalyzed"], "skipped", data["filesSkipped"])
for r in data["results"]:
    print("=" * 80)
    print(r["path"], "lines", r["totalLines"], "nonempty", r["nonEmptyLines"])
    print("metrics", r.get("metrics"))
    print("FUNCTIONS:")
    for fn in r.get("functions") or []:
        span = fn["endLine"] - fn["startLine"] + 1
        print(f"  {fn['name']} L{fn['startLine']}-{fn['endLine']} ({span}) params={fn.get('params')}")
    print("CLASSES:")
    for c in r.get("classes") or []:
        span = c["endLine"] - c["startLine"] + 1
        print(f"  {c['name']} L{c['startLine']}-{c['endLine']} ({span}) methods={c.get('methods')} props={c.get('properties')}")
    print("EXPORTS:")
    for e in r.get("exports") or []:
        print(f"  {e}")
