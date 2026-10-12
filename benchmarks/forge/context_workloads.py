"""Context sweep with token accounting for the server's plain-text chat wrapper."""

import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_http
from tokenizers import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--trials", type=int, default=3)
p.add_argument("--concurrency", default="1,8")
args = p.parse_args()
tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
settings = SimpleNamespace(
    runtime="forge", url="http://127.0.0.1:8080", max_new=8, arrival_gap_ms=0
)
rows = []
for length in [128, 512, 1024, 2048, 4096]:
    text = " test" * length
    body_ids = tok.encode(text, add_special_tokens=True).ids
    assert len(body_ids) == length, (length, len(body_ids))
    # Match server.formatPrompt; the wire prompt remains the same plain text.
    formatted = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        + "<|im_start|>user\n"
        + text
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    ids = tok.encode(formatted, add_special_tokens=True).ids
    label = "context" + str(length)
    bench_http.PROMPTS[label] = text
    for b in map(int, args.concurrency.split(",")):
        bench_http.run(settings, b, label)
        trials = [bench_http.run(settings, b, label) for _ in range(args.trials)]
        rows.append(
            dict(
                requested_body_tokens=length,
                actual_input_tokens=len(ids),
                concurrency=b,
                trials=trials,
            )
        )
        Path(args.out).write_text(
            json.dumps(dict(output_limit=8, eos=True, results=rows), indent=2)
        )
        print("CONTEXT", length, b, flush=True)
