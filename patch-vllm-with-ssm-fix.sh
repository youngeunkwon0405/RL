#!/bin/bash

set -eou pipefail

ORIG_UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT

#for uv_env in $ORIG_UV_PROJECT_ENVIRONMENT venvs/nemo_rl.models.generation.vllm.VllmGenerationWorker; do
for uv_env in venvs/nemo_rl.models.generation.vllm.VllmGenerationWorker; do
#for uv_env in $ORIG_UV_PROJECT_ENVIRONMENT; do

if [[ $uv_env == $ORIG_UV_PROJECT_ENVIRONMENT ]]; then
    # Install vllm if not already installed
    echo "Installing vllm..."
    uv sync --extra vllm
fi

export UV_PROJECT_ENVIRONMENT=$uv_env

# Find vllm installation path
VLLM_PATH=$(VLLM_LOGGING_LEVEL=ERROR uv run python -c "import vllm; print(vllm.__file__)")
VLLM_DIR=$(dirname "$VLLM_PATH")

echo "Found vllm at: $VLLM_DIR"

# Create patch file
cat > /tmp/vllm_ssm_fix.patch << "EOF"
diff --git a/vllm/model_executor/layers/mamba/ops/ssd_combined.py b/vllm/model_executor/layers/mamba/ops/ssd_combined.py
index b121275e9..5443f474d 100644
--- a/vllm/model_executor/layers/mamba/ops/ssd_combined.py
+++ b/vllm/model_executor/layers/mamba/ops/ssd_combined.py
@@ -112,7 +112,7 @@ def _mamba_chunk_scan_combined_fwd(x,
         if initial_states is not None else None,
         seq_idx=seq_idx,
         chunk_size=chunk_size,
-        out_dtype=C.dtype,
+        out_dtype=torch.float32,
         is_cont_batched=cu_seqlens is not None)
     states, final_states = (rearrange(t, "... (p n) -> ... p n", n=dstate)
                             for t in [states, final_states])
diff --git a/vllm/model_executor/models/mamba_cache.py b/vllm/model_executor/models/mamba_cache.py
index 49ba974c6..5126e9527 100644
--- a/vllm/model_executor/models/mamba_cache.py
+++ b/vllm/model_executor/models/mamba_cache.py
@@ -42,7 +42,7 @@ class MambaCacheManager(ConstantSizeCache):
                                  device="cuda")
         temporal_state = torch.empty(size=(num_mamba_layers, max_batch_size) +
                                      temporal_state_shape,
-                                     dtype=dtype,
+                                     dtype=torch.float32,
                                      device="cuda")

         self._mamba_cache = (conv_state, temporal_state)
EOF

# Apply patch
echo "Applying patch..."
echo "VLLM_DIR: $VLLM_DIR"
echo "Parent dir: $VLLM_DIR/.."
ls -l "$VLLM_DIR/.."
ls -l "$VLLM_DIR/../vllm/model_executor/layers/mamba/ops/"
cd "$VLLM_DIR/.." && patch -p1 < "/tmp/vllm_ssm_fix.patch"

echo "Patch applied successfully!"

done
