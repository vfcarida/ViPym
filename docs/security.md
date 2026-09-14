# Security Architecture & Sandbox Isolation

Evaluating LLM-generated code presents severe security risks. ViPym treats all generated code as untrusted and potentially malicious.

## Defense-in-Depth Security Controls

1. **AST Syntax Analysis:** Pre-execution syntax validation prevents invalid scripts from hitting execution runtimes.
2. **User-Space Kernel Virtualization:** Untrusted code runs inside **gVisor (`runsc`)** micro-sandboxes, preventing host kernel exploits.
3. **Network Isolation:** `--network=none` disables all sockets, preventing data exfiltration or botnet communication.
4. **Read-Only Root Filesystem:** Prevents filesystem tampering or persistence.
5. **Sanitized Environment:** Environment variables passed to execution containers are sanitized; AWS credentials, tokens, and SSH keys are never exposed.
6. **Resource Quotas:** Enforces hard limits: 2 CPU cores, 2GB RAM, 100 max PIDs, and 15s timeout.

---

## Turnkey Evaluation Sandbox (`Dockerfile.evaluator`)

ViPym provides a dedicated, pre-hardened evaluation container image in `docker/Dockerfile.evaluator`:

```bash
# Build evaluation sandbox container
docker build -t vipym-evaluator:latest -f docker/Dockerfile.evaluator .

# Run benchmark evaluation under complete network isolation
docker run --rm \
  --network=none \
  --pids-limit=100 \
  --memory=2g \
  --cpus=2 \
  -v $(pwd)/eval_scratch:/workspace/eval_scratch \
  vipym-evaluator:latest evaluate --suite humaneval --model HuggingFaceTB/SmolLM-135M

# Ultra-hardened mode with gVisor user-space kernel
docker run --rm \
  --runtime=runsc \
  --network=none \
  --memory=2g \
  vipym-evaluator:latest evaluate --suite swebench --model HuggingFaceTB/SmolLM-135M
```
