#!/usr/bin/env python3
"""In-session fix: bench_cuda_batch.rs tok scope + refresh deploy hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOTS = [
    Path("/teamspace/studios/this_studio/forge-current"),
    Path("/teamspace/studios/this_studio/forge"),
]

OLD = """    drop(cuda0);

    let params = GenerateParams {
        max_tokens: max_new,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: Some(tok.eos_token_id),
    };"""

NEW = """    drop(cuda0);

    let tok0 = ForgeTokenizer::from_dir(&dir, 151643, 151645).expect("tok");
    let params = GenerateParams {
        max_tokens: max_new,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: Some(tok0.eos_token_id),
    };"""


def main() -> None:
    root = next(r for r in ROOTS if (r / "runtime/examples/bench_cuda_batch.rs").is_file())
    rel = "runtime/examples/bench_cuda_batch.rs"
    path = root / rel
    text = path.read_text(encoding="utf-8")
    if "tok0.eos_token_id" in text:
        print("ALREADY_FIXED")
    elif OLD not in text:
        raise SystemExit("PATTERN_MISSING")
    else:
        path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8", newline="\n")
        print("FILE_PATCHED")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    man_path = root / ".forge-deploy-manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["files"][rel] = digest
    lines = "".join(f"{k}\t{man['files'][k]}\n" for k in sorted(man["files"]))
    man["content_hash"] = hashlib.sha256(lines.encode()).hexdigest()
    man_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(f"ROOT={root}")
    print(f"file_hash={digest}")
    print(f"content_hash={man['content_hash']}")
    print("PATCH_OK")


if __name__ == "__main__":
    main()
