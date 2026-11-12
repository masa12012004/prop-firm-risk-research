"""Extract full text of every paper in the three literature folders; write a per-paper digest
(title page + abstract, and the conclusion section) for reading."""
import os, re, sys, json, zipfile
import fitz
Q = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = {"FX": "03_forex_and_propfirm", "IDX": "05_IndexLiterature", "GOLD": "06_literature_gold_silver"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")
rows = []
for tag, d in SRC.items():
    p = os.path.join(Q, "arxiv_papers", d)
    for f in sorted(os.listdir(p)):
        fp = os.path.join(p, f)
        if f.lower().endswith(".pdf"):
            try:
                doc = fitz.open(fp); txt = "\n".join(pg.get_text() for pg in doc); n = len(doc)
            except Exception as e:
                txt, n = f"EXTRACT FAILED {e!r}", 0
        elif f.lower().endswith(".docx"):
            x = zipfile.ZipFile(fp).read("word/document.xml").decode("utf8", "ignore")
            txt = re.sub(r"<[^>]+>", "", x.replace("</w:p>", "\n")); n = 0
        else:
            continue
        txt = re.sub(r"[ \t]+", " ", txt); txt = re.sub(r"\n\s*\n+", "\n", txt)
        name = f"{tag}__{re.sub(r'[^A-Za-z0-9._-]+', '_', os.path.splitext(f)[0])[:80]}.txt"
        open(os.path.join(OUT, "txt", name), "w", encoding="utf8").write(txt)
        m = [x.start() for x in re.finditer(r"\n\s*(\d+\.?\s*)?(Conclusions?|CONCLUSIONS?|Concluding [Rr]emarks|Discussion and [Cc]onclusion|Summary and [Cc]onclusions?)", txt)]
        rows.append({"tag": tag, "file": f, "txt": name, "pages": n, "chars": len(txt), "concl_at": m[-1] if m else -1})
json.dump(rows, open(os.path.join(OUT, "index.json"), "w"), indent=0)
print(len(rows), "documents;", sum(r["pages"] for r in rows), "pages;", sum(r["chars"] for r in rows) // 1000, "k chars")
print("no text:", [r["file"] for r in rows if r["chars"] < 3000])
