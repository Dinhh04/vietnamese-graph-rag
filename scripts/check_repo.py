"""Kiểm soát toàn repo trong 1 lệnh:
- syntax-check mọi file .py
- validate JSON mọi notebook (+ đếm cell)
- liệt kê file + số dòng + vai trò
- nếu có artifacts/ thì kiểm tra nhanh

    python scripts/check_repo.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROLES = {
    "config.py": "config YAML + env (secret qua env)",
    "data.py": "load dữ liệu + parse nhãn + keyword/aspect + gazetteer",
    "embeddings.py": "PhoBERT encoder + maxsim (torch lazy)",
    "index.py": "DocumentIndex: vector+meta, lưu/nạp có version",
    "kg.py": "Knowledge Graph build/save/load + query (networkx lazy)",
    "retrieval.py": "HybridRetriever: bi-encoder -> MaxSim -> graph boost",
    "generate.py": "OpenAI generate + prompt grounding (openai lazy)",
    "pipeline.py": "GraphRAGPipeline orchestrate + observability",
    "observability.py": "log query latency/token/cost + feedback -> JSONL",
    "build_index.py": "CLI build & persist index + KG",
    "evaluate.py": "CLI P@k/MRR + regression gate",
    "import_artifacts.py": "CLI kiểm tra artifacts xuất từ notebook",
    "api.py": "FastAPI: /health /query /feedback",
    "ui.py": "Gradio UI + feedback 👍/👎",
    "test_core.py": "unit test no-GPU",
    "check_repo.py": "script kiểm soát repo (file này)",
    "build_notebooks.py": "sinh 2 notebook Kaggle",
}


def _check_py(globs) -> int:
    errs = 0
    print("# Python modules")
    files = []
    for g in globs:
        files += sorted(ROOT.glob(g))
    for f in files:
        txt = f.read_text(encoding="utf-8")
        n = txt.count("\n") + 1
        status = "OK"
        clean = "\n".join("" if ln.lstrip().startswith("!") else ln for ln in txt.splitlines())
        try:
            ast.parse(clean)
        except SyntaxError as e:
            status = f"ERR L{e.lineno}"
            errs += 1
        rel = f.relative_to(ROOT).as_posix()
        print(f"  [{status:>8}] {rel:32} {n:4}L  {ROLES.get(f.name, '')}")
    return errs


def _check_notebooks() -> int:
    errs = 0
    print("\n# Notebooks")
    for f in sorted(ROOT.glob("notebooks/*.ipynb")):
        try:
            nb = json.loads(f.read_text(encoding="utf-8"))
            cells = len(nb["cells"])
            code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
            # syntax-check code cells (bỏ dòng %/!)
            cerr = 0
            for c in nb["cells"]:
                if c["cell_type"] != "code":
                    continue
                src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                clean = "\n".join("" if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines())
                try:
                    ast.parse(clean)
                except SyntaxError:
                    cerr += 1
            status = "OK" if cerr == 0 else f"{cerr} CELL ERR"
            errs += cerr
            print(f"  [{status:>10}] {f.name:42} {cells} cells ({code} code)")
        except Exception as e:
            errs += 1
            print(f"  [ BAD JSON] {f.name}: {e}")
    return errs


def _check_artifacts():
    d = ROOT / "artifacts"
    print("\n# Artifacts")
    if not (d / "manifest.json").exists():
        print("  (chưa có artifacts/ — chạy notebook §7 hoặc `make index`)")
        return
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    files = [p.name for p in d.glob("*")]
    print(f"  manifest: {man}")
    print(f"  files   : {sorted(files)}")


def main():
    errs = _check_py(["src/**/*.py", "app/*.py", "tests/*.py", "scripts/*.py"])
    errs += _check_notebooks()
    _check_artifacts()
    print(f"\n{'❌ FAILED' if errs else '✅ ALL OK'}: {errs} error(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
