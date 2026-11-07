import json, re, sys

# Chinese terms (substring match is fine)
CN = ["防障碍","防撞","防顶","硬停","障碍物","撞到","顶住","顶到","堵转","推不动",
      "退让","让步","防卡","撞上","压住","顶死"]
# English terms with word boundaries
EN = [r"\bstall(s|ed|ing)?\b", r"\bSTALL(s|_DEG|_BACKOFF)?\b", r"stall[-_](deg|backoff)",
      r"\bbackoff\b", r"\bback_off\b", r"\bback-off\b", r"防.*障"]

FILES = [
    r"C:\Users\jiang\.claude\projects\g--sim2real\3327b698-ccbe-45c0-8cac-792363d15f96.jsonl",
    r"C:\Users\jiang\.claude\projects\g--sim2real\774b2bbc-b8ab-454d-94b6-af0d3a061ed6.jsonl",
]
OUT = r"G:\sim2real\so101-real-harness\_scratch_out.txt"

en_re = re.compile("|".join(EN))

def role_of(rec):
    r = rec.get("type") or rec.get("role") or ""
    msg = rec.get("message")
    if isinstance(msg, dict):
        r = msg.get("role", r)
    return r

def strings(obj, out):
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            strings(v, out)

def find_hits(blob):
    hits = []
    for k in CN:
        if k in blob:
            hits.append(k)
    for m in en_re.finditer(blob):
        hits.append(m.group(0))
    return sorted(set(hits))

def main():
    with open(OUT, "w", encoding="utf-8") as o:
        for path in FILES:
            o.write("=" * 80 + "\nFILE: " + path + "\n" + "=" * 80 + "\n")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    role = role_of(rec)
                    ss = []
                    strings(rec, ss)
                    blob = "\n".join(ss)
                    hits = find_hits(blob)
                    if not hits:
                        continue
                    o.write("\n--- LINE %d role=%s hits=%s\n" % (lineno, role, ",".join(hits)))
                    shown = set()
                    for k in hits:
                        idx = blob.find(k)
                        if idx < 0:
                            m = en_re.search(blob)
                            idx = m.start() if m else -1
                        if idx < 0:
                            continue
                        s = max(0, idx - 300)
                        e = min(len(blob), idx + 300)
                        frag = blob[s:e].replace("\n", "\\n")
                        key = frag[:50]
                        if key in shown:
                            continue
                        shown.add(key)
                        o.write("  [%s] ...%s...\n" % (k, frag))
                        if len(shown) >= 4:
                            break
    sys.stderr.write("done\n")

if __name__ == "__main__":
    main()
