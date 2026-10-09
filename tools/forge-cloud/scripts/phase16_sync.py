"""Prepare a small, non-secret incremental validation patch and source audit."""
import base64, hashlib, json, tarfile, io
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=['runtime/src/model/tokenizer.rs','runtime/src/model/mod.rs','runtime/src/ffi/abi.rs',
       'runtime/src/bin/forge_bench.rs','runtime/src/bin/forge_int4_diag.rs',
       'runtime/examples/phase16_parity.rs','runtime/examples/phase16_checkpoint.rs',
       'runtime/tests/phase16_architecture.rs','runtime/tests/cuda_phase16.rs',
       'tests/reference/phase16_reference.py','tests/reference/phase2_reference.py',
       'scripts/phase16_validate.sh','scripts/phase16_second_model.sh']
audit={}
for directory in ['runtime/src','runtime/tests','runtime/examples','kernels/src','kernels/cuda/src','api/internal/scheduler','api/cmd/forge-server']:
    for p in (root/directory).rglob('*'):
        if p.is_file() and p.suffix in ['.rs','.cpp','.cu','.h','.hpp','.go','.c']:
            audit[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
out=root/'docs'
out.mkdir(parents=True,exist_ok=True)
(out/'_phase16_source_identity.json').write_text(json.dumps(audit,indent=2))
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
    for name in files:tar.add(root/name,arcname=name)
    tar.add(out/'_phase16_source_identity.json',arcname='phase16-source-identity.json')
(root/'docs/_phase16_patch.base64').write_text(base64.b64encode(buf.getvalue()).decode())
print('patch_bytes',len(buf.getvalue()))
