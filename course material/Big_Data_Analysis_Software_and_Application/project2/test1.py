import json

path = "./data/datamind_12k.json"

with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)

if isinstance(obj, dict):
    print("Top-level dict keys:", list(obj.keys())[:20])
    for k, v in obj.items():
        if isinstance(v, list) and len(v) > 0:
            print(f"\nFirst item under key={k}:")
            print(json.dumps(v[0], ensure_ascii=False, indent=2)[:5000])
            break
elif isinstance(obj, list):
    print("Top-level is list, len =", len(obj))
    print(json.dumps(obj[0], ensure_ascii=False, indent=2)[:5000])
else:
    print(type(obj))