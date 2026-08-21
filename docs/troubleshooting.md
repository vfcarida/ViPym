# ViPym Troubleshooting & FAQ

This guide provides diagnostic steps and solutions for common operational and runtime issues encountered during LLM compression, serving, and benchmark evaluation.

---

## 1. GPU CUDA Out of Memory (OOM)

### Symptom
`torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X.XX GiB...`

### Diagnostic & Solutions
1. **Reduce Maximum Context Window**:
   In your recipe or configuration, lower `serving.max_model_len` (e.g., from `4096` to `2048` or `1024` for benchmark tasks).
   ```yaml
   serving:
     max_model_len: 2048
     gpu_memory_utilization: 0.85
   ```
2. **Enable KV-Cache Quantization**:
   Switch from FP16 KV-cache to FP8:
   ```yaml
   serving:
     kv_cache_dtype: "fp8"
   ```
3. **Use Memory Cleanup Utilities**:
   ViPym automatically triggers `safe_cuda_memory_cleanup()` between DAG stages. If running standalone scripts, call `vipym.utils.resilience.safe_cuda_memory_cleanup()`.

---

## 2. Docker / gVisor Sandbox Permissions

### Symptom
`DockerException: Error while fetching server API version` or `gVisor runtime 'runsc' not found`.

### Diagnostic & Solutions
1. **Verify Docker Daemon**: Ensure Docker Desktop or `dockerd` is running with `docker ps`.
2. **Degraded Subprocess Mode (CI / Local Dev)**:
   If running on a machine without Docker, enable non-isolated subprocess mode by setting both flags:
   ```bash
   export VIPYM_ALLOW_UNSAFE=1
   ```
   And in YAML:
   ```yaml
   evaluation:
     allow_unsafe_execution: true
     isolate_with_gvisor: false
   ```

---

## 3. Hugging Face Hub Rate Limiting

### Symptom
`HTTP 429 Too Many Requests` or `Warning: You are sending unauthenticated requests to the HF Hub`.

### Diagnostic & Solutions
Set your Hugging Face authentication token:
```bash
export HF_TOKEN="hf_your_access_token_here"
```
ViPym automatically routes authenticated requests through `huggingface_hub` and retries transient failures via `retry_with_backoff`.

---

## 4. Severe Pass@1 Loss Post-Quantization

### Symptom
A compressed model scores 0.00 or has $\ge 20\%$ drop on HumanEval/MBPP.

### Diagnostic & Solutions
1. **Missing Calibration Data**:
   Ensure you pass at least 128–256 high-quality code samples to AWQ/GPTQ via `CalibrationDatasetManager`.
2. **Salient Outlier Channels**:
   If using 4-bit quantization, pair it with **SpinQuant / QuaRot** orthogonal rotations to disperse activation outliers across channels:
   ```yaml
   compression_pipeline:
     - stage_id: spin_rotation
       method: transform_spinquant
     - stage_id: awq_quant
       method: quantize_awq
       parameters:
         bits: 4
   ```
