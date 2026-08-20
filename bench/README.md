# Benchmark and correctness tools

Run the portable release gate from the repository root:

```bash
./verify.sh --full
```

The full gate uses only Python's standard library. For the optional C8
throughput proof:

```bash
python3 -m venv .bench-venv
.bench-venv/bin/pip install -r requirements-bench.txt
BASE_URL=http://127.0.0.1:18079 MODEL=qwen3.8-27b-nvfp4 \
  .bench-venv/bin/python bench/c8_proof.py 1,2,4,6,8 3
```

All endpoints, model names, and output paths can be overridden with
`BASE_URL`, `MODEL`, and `OUTPUT`.
