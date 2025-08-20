#!/usr/bin/env python3
# save as extract_outputs.py
import nbformat, base64, json, sys, pathlib

MIME_TEXT  = ["text/plain", "text/markdown", "text/html"]
MIME_BIN   = ["image/png", "image/jpeg", "image/gif", "application/pdf"]
EXT = {"image/png":".png","image/jpeg":".jpg","image/gif":".gif","application/pdf":".pdf"}

def as_str(x):
    return "".join(x) if isinstance(x, list) else (x or "")

def write_text(path, s):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")

def write_bytes(path, b64):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))

def extract(nb_path, outdir="notebook_outputs"):
    nb = nbformat.read(nb_path, as_version=4)
    outdir = pathlib.Path(outdir)
    log = []

    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        for j, out in enumerate(cell.get("outputs", [])):
            kind = out.get("output_type")
            tag = f"cell{i:03d}_out{j:02d}_{kind}"

            if kind == "stream":
                txt = as_str(out.get("text"))
                who = out.get("name", "stdout")
                log.append(f"### {tag} ({who})\n{txt}\n")

            elif kind in ("execute_result", "display_data"):
                data = out.get("data", {})
                # try binary first (images, pdf)
                saved = False
                for mime in MIME_BIN:
                    if mime in data:
                        f = outdir/"assets"/f"{tag}{EXT[mime]}"
                        write_bytes(f, data[mime])
                        log.append(f"### {tag}\n[saved {mime} ➜ assets/{f.name}]\n")
                        saved = True
                        break
                if saved:
                    continue
                # then text-ish formats
                for mime in MIME_TEXT:
                    if mime in data:
                        content = as_str(data[mime])
                        ext = "md" if mime=="text/markdown" else ("html" if mime=="text/html" else "txt")
                        f = outdir/"assets"/f"{tag}.{ext}"
                        write_text(f, content)
                        log.append(f"### {tag}\n{content}\n")
                        saved = True
                        break
                if not saved and data:
                    f = outdir/"assets"/f"{tag}.json"
                    write_text(f, json.dumps(data, ensure_ascii=False, indent=2))
                    log.append(f"### {tag}\n[saved mimebundle ➜ assets/{f.name}]\n")

            elif kind == "error":
                txt = f'{out.get("ename","Error")}: {out.get("evalue","")}\n' + "\n".join(out.get("traceback", []))
                log.append(f"### {tag}\n{txt}\n")

    write_text(outdir/"ALL_OUTPUTS.md", "\n".join(log))

if __name__ == "__main__":
    ipynb = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "notebook_outputs"
    extract(ipynb, outdir)
    print(f"Wrote outputs to {outdir}/ALL_OUTPUTS.md and {outdir}/assets/")
