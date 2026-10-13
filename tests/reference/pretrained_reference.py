"""Sequential HF reference export; never runs a Forge model alongside PyTorch.
Usage: python tests/reference/pretrained_reference.py MODEL_DIR OUTPUT_DIR
"""

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if os.environ.get("FORGE_REFERENCE_TEXT_ONLY") == "1":
    # Optional incompatible torchvision installs can prevent text-only exports.
    # This text-only utility does not use vision; do not alter model operations.
    import transformers.utils.import_utils as hf_imports

    hf_imports._torchvision_available = False
from safetensors import safe_open

model_dir, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.float32, attn_implementation="eager"
).eval()
config = json.loads((model_dir / "config.json").read_text())
with safe_open(model_dir / "model.safetensors", framework="pt") as f:
    manifest = {
        name: {
            "shape": list(f.get_slice(name).get_shape()),
            "dtype": f.get_slice(name).get_dtype(),
        }
        for name in f.keys()
    }
(out / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2))
cases = []
prompts = [
    "Hello!",
    "What is the capital of France?",
    "Explain why the sky is blue in one sentence.",
    "A story about a robot exploring a garden. " * 20,
]
for i, prompt in enumerate(prompts):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tok.encode(text, add_special_tokens=False)
    if len(tokens) > 160:
        tokens = tokens[:160]
    ids = torch.tensor([tokens])
    hooks = []

    def hook(index):
        def save(module, args, result):
            t = result[0] if isinstance(result, tuple) else result
            (out / f"{i}_layer_{index}.bin").write_bytes(
                t.detach().float().contiguous().numpy().tobytes()
            )

        return save

    for j, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(hook(j)))
    with torch.no_grad():
        result = model(ids, use_cache=False)
    for h in hooks:
        h.remove()
    (out / f"{i}_logits.bin").write_bytes(
        result.logits[0, -1].float().contiguous().numpy().tobytes()
    )
    # Raw argmax, exactly Forge's policy; HF generation_config may add penalties.
    greedy = []
    with torch.no_grad():
        decoded = model(ids, use_cache=True)
        for step in range(12):
            next_token = int(decoded.logits[0, -1].argmax())
            greedy.append(next_token)
            if next_token == config["eos_token_id"]:
                break
            decoded = model(
                torch.tensor([[next_token]]),
                past_key_values=decoded.past_key_values,
                use_cache=True,
            )
    cases.append(
        {
            "tokens": tokens,
            "greedy": greedy,
            "text": tok.decode(greedy),
            "prompt": prompt,
        }
    )
    print(i, len(tokens), greedy, tok.decode(greedy), flush=True)
(out / "cases.json").write_text(
    json.dumps(
        {
            "model": str(model_dir),
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "cases": cases,
        },
        indent=2,
    )
)
