"""Verify cached GGUF precision and tensor samples against the HF checkpoint."""
import hashlib
import json
import pathlib
import sys
import struct
import numpy as np
from gguf import GGUFReader

model, gguf, output = map(pathlib.Path, sys.argv[1:])
reader = GGUFReader(str(gguf))
tensors = {t.name:t for t in reader.tensors}
checks = []
with (model / "model.safetensors").open("rb") as source:
    header_len = struct.unpack("<Q",source.read(8))[0]
    header = json.loads(source.read(header_len))
    for hf_name, gguf_name in [("model.embed_tokens.weight", "token_embd.weight"), ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"), ("model.layers.23.mlp.down_proj.weight", "blk.23.ffn_down.weight")]:
        desc = header[hf_name]
        dtype = desc["dtype"]
        width = desc["shape"][1]
        element_bytes = 4 if dtype == "F32" else 2
        rows = []
        for row in range(4):
            source_row = row
            source.seek(8+header_len+desc["data_offsets"][0]+source_row*width*element_bytes)
            raw=source.read(32*element_bytes)
            if dtype=="BF16":
                values=(np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16).view(np.float32)
            elif dtype in ("F16","F32"):
                values=np.frombuffer(raw,dtype="<f2" if dtype=="F16" else "<f4")
            else:
                raise RuntimeError("unsupported checkpoint dtype: "+dtype)
            rows.append(values.astype(np.float16))
        expected = np.stack(rows)
        actual = tensors[gguf_name].data[:4,:32]
        error = float(np.max(np.abs(actual.astype(np.float32)-expected.astype(np.float32))))
        checks.append(dict(hf=hf_name,gguf=gguf_name,max_abs_error=error,exact_sample_match=error == 0))
revision = (model / ".cache/huggingface/download/config.json.metadata").read_text().splitlines()[0]
result = dict(hf_revision=revision,architecture=reader.fields["general.architecture"].contents(),file_type=reader.fields["general.file_type"].contents(),tensor_types=sorted(set(int(t.tensor_type) for t in reader.tensors)),samples=checks)
result["verification_scope"] = "Four rows by 32 columns of three tensors; sample agreement does not prove complete checkpoint identity. Conversion transforms must be audited before treating mismatches as revision differences."
result["all_samples_match"] = all(c["exact_sample_match"] for c in checks)
for p in [model/"config.json",model/"model.safetensors",gguf]:
    with p.open("rb") as f:
        digest=hashlib.sha256()
        for chunk in iter(lambda:f.read(1024*1024),b""):digest.update(chunk)
    result[p.name+"_sha256"]=digest.hexdigest()
output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
if not result["all_samples_match"]:
    sys.exit(1)
