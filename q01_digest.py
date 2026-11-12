import json, os, sys, hashlib, re
R = json.load(open("papers/index.json")); seen = {}; out = []
for r in R:
    t = open(os.path.join("papers/txt", r["txt"]), encoding="utf8").read()
    h = hashlib.md5(t[:20000].encode()).hexdigest()
    if h in seen: r["dup_of"] = seen[h]; continue
    seen[h] = r["txt"]; out.append(r)
json.dump(out, open("papers/unique.json", "w"))
a, b = int(sys.argv[1]), int(sys.argv[2]); A = int(sys.argv[3]) if len(sys.argv) > 3 else 1300; Cn = int(sys.argv[4]) if len(sys.argv) > 4 else 900
print(len(R), "docs,", len(out), "unique")
for i, r in enumerate(out[a:b], a):
    t = open(os.path.join("papers/txt", r["txt"]), encoding="utf8").read()
    k = t.find("Abstract"); k = k if 0 <= k < 4000 else t.find("ABSTRACT"); k = max(k, 0) if k < 4000 else 0
    head = (t[:250] + " ... " + t[k:k + A]) if k > 300 else t[:A + 250]
    c = r["concl_at"]; concl = t[c:c + Cn] if c > 0 else t[int(len(t) * 0.80):int(len(t) * 0.80) + Cn]
    print(f"\n=====[{i}] {r['tag']} | {r['file']} | {r['pages']}p\n" + re.sub(r"\s+", " ", head) + "\n--CONCL-- " + re.sub(r"\s+", " ", concl))
