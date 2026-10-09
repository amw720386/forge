"""Record the exact retained sources after incremental GPU development."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
paths = ['runtime/src/compute/cuda.rs','kernels/cuda/include/forge_cuda.h',
         'kernels/cuda/src/forge_cuda_stub.c','runtime/build.rs','runtime/src/lib.rs','runtime/src/scheduling.rs',
         'runtime/src/cuda_model/batcher.rs','runtime/src/cuda_model/transformer.rs',
         'runtime/tests/cuda_phase15.rs','api/internal/scheduler/scheduler.go',
         'kernels/cuda/src/attention.cu','kernels/cuda/src/error.hpp',
         'kernels/cuda/src/linear.cu','kernels/cuda/src/elem.cu',
         'kernels/cuda/bench/bench_phase15_attention.cu',
         'benchmarks/forge/bench_http.py','benchmarks/forge/phase15_context.py',
         'scripts/phase15_http.sh','scripts/phase15_context.sh',
         'scripts/phase15_correctness.sh','scripts/phase15_accepted.sh',
         'scripts/phase15_scheduling.sh','scripts/phase15_refs.sh']
result = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
print(json.dumps(result,indent=2))
