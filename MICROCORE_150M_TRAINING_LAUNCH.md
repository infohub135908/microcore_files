# MICROCORE 150M — Colab Launch Guide

**Experiment:** MICROCORE_150M, decoder-only LM trained from random initialization on the 3B-token FineWeb-Edu base selection prepared in Revision 9.1.1.

**Everything below was actually executed and verified** unless marked as a Colab-only step. The local smoke run passed (4 fresh steps + checkpoint-reload continuation, finite loss/grads); the test suite passed 55/55. **The full training run has NOT been started.** This document is the launch procedure, not a report of a completed experiment.

---

## Hardware and budget

- **Target: NVIDIA T4 (Colab free tier), 14.5 GiB usable VRAM, 15 GB Drive quota.**
- Effective batch: 8 × 30 × 2048 = **491,520 tokens/step**; 6,103 steps ≈ 3.0B tokens, one deterministic pass, no repetition.
- Memory plan: FP16 autocast + GradScaler + gradient checkpointing (on by default for GPU). If VRAM is exceeded, reduce `--microbatch` first and raise `--accum` to compensate — **do not change the architecture**. The effective batch must stay 491,520 tokens.
- **T4 limitation (documented, not hidden):** the T4 (SM75) has no FlashAttention kernels; the trainer uses PyTorch SDPA's memory-efficient/math backend. Expect roughly 25–40k tokens/s, i.e. **~20–35 hours** for the full 3B-token run. The free-tier runtime disconnects every ~12 hours — the checkpoint/resume path exists exactly for this and was tested.

## Storage layout

| Path | Contents | Size |
|---|---|---|
| `/content/drive/MyDrive/microcore_data/prepared/` | selection, manifests, hashes (already on Drive from 9.1.1) | ~610 MB |
| `/content/stream/` (`--stream-dir`) | packed uint32 token stream | **~11.2 GiB** |
| `/content/drive/MyDrive/microcore_data/train_150m/` | checkpoints (2 × 1.7 GB), JSONL log, run manifest | ~3.5 GB |

The stream is too large for the 15 GB Drive quota alongside checkpoints, so it is built on Colab's local disk (`/content`, ephemeral). If the runtime disconnects, re-run `--materialize-only` (it re-verifies every document hash and token count against the selection — the stream is byte-identical by construction, which is what makes checkpoint resume across stream rebuilds valid).

## Setup (once per runtime)

```python
from google.colab import drive
drive.mount('/content/drive')

# upload the revision files, then:
%pip -q install tiktoken==0.14.0 pyarrow requests numpy
%pip -q install torch   # Colab has torch preinstalled; version pin not required
%cd /content
# tokenizer + config sanity check (any invocation runs it first and aborts on mismatch):
!python3 -c "import MICROCORE_150M_TRAIN as T; T.validate_tokenizer(); print('tokenizer OK')"
```

The six files to upload to the Colab working directory: `MICROCORE_150M_TRAIN.py`, `MICROCORE_150M_CONFIG.json`, `microcore_paths.py`, and the updated `MICROCORE_REVISION_9_1_1_DATA_PREPARATION.py`, `MICROCORE_REVISION_9_1_1_COLAB_VALIDATION.py`. (The GitHub repo is read-only for this work; these are downloadable deliverables — commit them yourself if you want them versioned.)

## Pre-launch validation (already passed once — re-run on Colab before the real run)

```python
%run MICROCORE_150M_TRAIN_TESTS.py        # 55/55 expected; abort the launch on any FAIL
%run MICROCORE_150M_TRAIN.py --materialize-only --stream-dir /content/stream
%run MICROCORE_150M_TRAIN.py --smoke 4 --stream-dir /content/stream
```

The smoke test on Colab exercises the CUDA/FP16 path (the local sandbox could only run the CPU path): 4 steps, checkpoint save, reload, continuation, and GPU memory reporting. It is a **smoke test, not a completed experiment**.

## Full run — the launch command

```python
%run MICROCORE_150M_TRAIN.py --stream-dir /content/stream
```

This is the complete command for the first training experiment. It runs materialization if needed, then trains 6,103 steps (~3.0B tokens), checkpointing every 500 steps to Drive and logging every step to `train_log.jsonl`. Progress: `!tail -1 /content/drive/MyDrive/microcore_data/train_150m/train_log.jsonl`.

## After a disconnect

```python
%run MICROCORE_150M_TRAIN.py --resume --stream-dir /content/stream
```

If `/content/stream` was lost, re-run `--materialize-only` first (same command as above; it verifies every document hash again). `--resume` validates the checkpoint's param count, vocab/EOS, and data-manifest hash; a corrupt latest checkpoint falls back to the previous known-good.

## Outputs

- Model checkpoints: `microcore_data/train_150m/ckpt_last.pt`, `ckpt_prev.pt`
- Training log (JSONL): `microcore_data/train_150m/train_log.jsonl` — fields: ts, model, param_count, step, tokens_consumed, loss, lr, grad_norm, steps_per_s, tokens_per_s, gpu_mem_gib, checkpoint, event
- Run manifest: `microcore_data/train_150m/run_manifest.json`
- Nothing is uploaded to GitHub by these scripts.

## Known limitations

- T4 only: no FlashAttention, ~20–35 h wall clock, free-tier disconnects every ~12 h → use `--resume`.
- The stream lives on ephemeral `/content` (11.2 GiB does not fit the Drive quota next to checkpoints).
- CPU fallback exists for smoke tests only; a CPU full run is not practical (the sandbox OOM-killed even microbatch 8 there).
