# Dev / Test GPU on RunPod

This project does not require an on-prem GPU. Local inference development and CI integration tests run against ephemeral RunPod pods, provisioned and torn down on demand.

## Setup

1. Create a RunPod account: <https://www.runpod.io/>
2. Generate an API key: **Settings → API Keys → Create**.
3. Copy `.env.example` to `.env` (gitignored) and fill in `RUNPOD_API_KEY`.
4. (Optional) Add `HUGGING_FACE_HUB_TOKEN` if you plan to use a gated model.

## Daily workflow

```bash
# Provision a 4090 + Qwen2.5-Coder-7B (defaults — cheapest viable combo)
python scripts/dev_gpu.py up

# ... do your work, pointing MODELMELD_VLLM_ENDPOINT at the printed URL ...

python scripts/dev_gpu.py status   # check it's still alive
python scripts/dev_gpu.py down     # terminate when finished
```

The script writes `.dev_gpu_state.json` at the repo root with the pod ID. This file is gitignored and is what `down` reads to find the pod to terminate. **Always run `down` when finished** — a forgotten H100 burns ~$72/day.

## Choosing GPU and model

| GPU | $/hr (approx) | Reasonable models | Use case |
|---|---|---|---|
| RTX 4090 (24GB) | $0.30–0.50 | Qwen2.5-Coder-7B (INT4/INT8), Llama-3.1-8B | Most dev work, early benchmarks |
| A100 80GB | $1.30–2.00 | Qwen2.5-Coder-32B, DeepSeek-Coder-V2 | Mid-quality, routing benchmarks |
| H100 80GB | $2.50–3.50 | Qwen2.5-Coder-32B, DeepSeek-Coder-V2.5 | High-quality, parity benchmarks vs cloud |

Override the defaults with flags:

```bash
python scripts/dev_gpu.py up \
  --gpu "NVIDIA A100 80GB PCIe" \
  --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
  --disk 80
```

Available GPU type IDs change over time; check RunPod's console if `up` fails with "GPU type not available."

## Budget guidance

| Activity | Typical duration | Cost (4090) | Cost (H100) |
|---|---|---|---|
| Quick smoke test of an adapter change | 30 min | $0.25 | $1.50 |
| Half-day dev session | 4 h | $2.00 | $12.00 |
| Per-tool benchmark run | 1–2 h | $1.00 | $5.00 |
| Overnight forgotten pod (DON'T) | 12 h | $6.00 | $36.00 |

Nightly CI workflow target: ≤$5/run on a 4090. Verified by checking RunPod billing weekly.

## CI integration

`.github/workflows/nightly-gpu.yml` runs at 06:00 UTC daily:

1. Provisions a pod via `dev_gpu.py up`.
2. The script writes `MODELMELD_VLLM_ENDPOINT` to `$GITHUB_ENV` so subsequent steps see it.
3. Runs `pytest --gpu` (collects tests marked `@pytest.mark.requires_gpu`).
4. Tears down the pod in an `if: always()` step.

The workflow needs a repository secret `RUNPOD_API_KEY` to be set under **Settings → Secrets → Actions**.

PR and main CI workflows do **not** run GPU tests — they would be too slow and too expensive to gate every PR on.

## Troubleshooting

**`RUNPOD_API_KEY is not set`** — populate `.env` and either `source` it before running or use `direnv` / `dotenv-cli`.

**`up` hangs on dots for >10 min** — most often a slow model download on the pod's first run. Check `python scripts/dev_gpu.py status` in another shell; if `status: RUNNING` but endpoint is unreachable, model download is in progress. If `status: STARTING` for >5 min, GPU pool is contested — retry later or pick a different `--gpu`.

**`down` reports 404** — pod was already terminated externally (e.g., from the RunPod web console). The script clears state automatically; safe to ignore.

**Stale `.dev_gpu_state.json` with no live pod** — delete the file: `rm .dev_gpu_state.json`.

**Endpoint returns 502 / connection refused** — vLLM is still warming up after the pod went RUNNING. Wait 30 s and retry. If persistent, check the pod's logs in the RunPod console.

## Switching providers

The script is hardcoded to RunPod for simplicity. If you need to switch (Modal, Lambda Labs, Vast.ai, or an on-prem GPU), the integration is small:

- Adapt `_client()`, `cmd_up`, `cmd_down`, `cmd_status` to the new provider's API.
- Keep the CLI surface (`up`, `down`, `status`, env vars) identical so nothing else changes.
- For a local GPU: skip provisioning entirely; just set `MODELMELD_VLLM_ENDPOINT=http://localhost:8000/v1` after starting vLLM yourself.

The pytest gating (`--gpu` flag, `requires_gpu` marker) is provider-agnostic and works either way.
