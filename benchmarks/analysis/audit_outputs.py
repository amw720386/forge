"""Audit identical HTTP workloads; retain differences instead of hiding them."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--baseline", required=True)
p.add_argument("--candidate", required=True)
p.add_argument("--out", default="output-audit.json")
args = p.parse_args()
a = json.loads((args.root / args.baseline).read_text())
z = json.loads((args.root / args.candidate).read_text())
baseline = {(r["concurrency"], r["workload"]): r for r in a["results"]}
rows = []
for cell in z["results"]:
    ref = baseline[(cell["concurrency"], cell["workload"])]["trials"][0]["requests"]
    mismatches = []
    count = 0
    for trial, t in enumerate(cell["trials"]):
        for request, q in enumerate(t["requests"]):
            count += 1
            expected = ref[request]["text"]
            if q["text"] != expected:
                prefix = next(
                    (i for i, (x, y) in enumerate(zip(expected, q["text"])) if x != y),
                    min(len(expected), len(q["text"])),
                )
                mismatches.append(
                    dict(
                        trial=trial,
                        request=request,
                        common_prefix_characters=prefix,
                        baseline=expected,
                        candidate=q["text"],
                    )
                )
    rows.append(
        dict(
            workload=cell["workload"],
            concurrency=cell["concurrency"],
            requests=count,
            exact_text_matches=count - len(mismatches),
            differences=mismatches,
        )
    )
(args.root / args.out).write_text(json.dumps(rows, indent=2))
print(
    [
        (r["workload"], r["concurrency"], r["exact_text_matches"], r["requests"])
        for r in rows
    ]
)
