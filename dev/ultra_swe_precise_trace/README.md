# Ultra SWE precise-trace validation

This wrapper reproduces the `ultra-v3` SWE Teacher recipe for one training
step while pinning the production-only precise tracing revisions:

- 64 training nodes and 64 vLLM generation nodes on GB200
- 32 prompts x 16 generations (global batch size 512)
- 196,608-token maximum sequence length
- the released `swe.jsonl` blend and matching ARM64 per-instance SIF images
- Gym instrumentation rebased onto the exact Gym commit pinned by `ultra-v3`
- `youngeunkwon0405/nv-OpenHands@e806c4b4510639ca7deefb4e8254fc6c61773075`

Launch from the repository root:

```bash
dev/ultra_swe_precise_trace/launch.sh
```

Use `DRY_RUN=1` to print the resolved command without submitting. The wrapper
disables checkpoint writes because this is a one-step instrumentation
validation, while retaining the released model parallelism, batch shape, MTP
inference, context length, dataset, and sandbox configuration.
