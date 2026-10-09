#!/usr/bin/env python3
"""Hugging Face Transformers FP32 reference for Qwen2.5-0.5B-Instruct.

Prints first-token greedy id and a short greedy continuation for comparison
with Forge (`cargo run --release --example generate_qwen`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"

PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main() -> None:
    max_new = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    inputs = tok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1]
        first = int(torch.argmax(logits).item())
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new_ids = gen[0, inputs["input_ids"].shape[1] :].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    result = {
        "prompt_n_tokens": int(inputs["input_ids"].shape[1]),
        "first_greedy_id": first,
        "generated_ids": new_ids,
        "text": text,
    }
    print(json.dumps(result, indent=2))
    out_path = ROOT / "tests" / "reference" / "phase3_hf_out.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
