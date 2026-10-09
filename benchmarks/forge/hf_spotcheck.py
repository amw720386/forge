#!/usr/bin/env python3
"""Optional Hugging Face Transformers greedy spot-check vs Forge JSON tokens.

Skips cleanly if transformers/torch are unavailable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--prompt",
        default="<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
    )
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        print(f"HF deps unavailable: {e}", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    inputs = tok(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    gen = out[0][inputs["input_ids"].shape[1] :].tolist()
    text = tok.decode(gen, skip_special_tokens=True)
    payload = {
        "runtime": "huggingface_transformers",
        "dtype": "fp32",
        "device": device,
        "prompt": args.prompt,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "tokens": gen,
        "text": text,
        "max_new": args.max_new,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
