import json
p = r"F:\Personal Projects\AgenticAI_Chatbot_Using_RAG\.ua\tmp\ua-file-extract-results-6.json"
with open(p, encoding="utf-8") as f:
    data = json.load(f)
for r in data["results"]:
    if r["path"] != "tests/test_frontend_guards.py":
        continue
    print("FUNCTIONS", len(r["functions"]))
    for fn in r["functions"]:
        span = fn["endLine"] - fn["startLine"] + 1
        print(f"{fn['name']}|{fn['startLine']}|{fn['endLine']}|{span}")
    print("EXPORTS", len(r.get("exports") or []))
    names = {e["name"] for e in r.get("exports") or []}
    fnames = {fn["name"] for fn in r["functions"]}
    print("exported not in functions", names - fnames)
    print("functions not exported", fnames - names)
