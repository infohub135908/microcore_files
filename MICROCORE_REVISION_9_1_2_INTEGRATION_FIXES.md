# MICROCORE Revision 9.1.2 — Integration Fixes & Trainer Implementation

**Status:** complete. Test suite executed: **55/55 checks passed** (`MICROCORE_150M_TRAIN_TESTS.py`, output in `rev911/train_tests.log`). Smoke test executed and passed (`rev911/smoke.log`). **Training has NOT started** — no full run has been launched. All claims below come from actual executions in this session.

---

## 1. Drive path mismatch (fixed)

**Problem (carried from 9.1.1):** the preparation script wrote to `/content/drive/MyDrive/microcore_data`, while the Colab validation script looked in `/content/drive/MyDrive/microcore`. On Drive, the validator found nothing.

**Fix:** new shared module `microcore_paths.py` defines a single `DATA_ROOT` (`/content/drive/MyDrive/microcore_data` on Colab, `./microcore_data` locally). `MICROCORE_REVISION_9_1_1_DATA_PREPARATION.py`, `MICROCORE_REVISION_9_1_1_COLAB_VALIDATION.py`, and `MICROCORE_150M_TRAIN.py` all import it. Verified by test section 1 (5 checks: prep root == validator root == trainer root, manifest exists where the validator looks).

## 2. Base-data loading interface (fixed)

The 9.1.1 trainer proposal loaded `base_selection.jsonl` **directly** — that file contains only metadata (shard/row/doc_id/doc_hash/tokens), so it could never feed the model, and nothing verified the byte content.

**Fix:** `MICROCORE_150M_TRAIN.py --materialize-only` performs real reconstruction:

1. Reads the selection and groups documents by shard.
2. Re-downloads each pinned FineWeb-Edu shard (SHA `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`) and re-reads the exact rows.
3. For every document: recomputes **SHA-256 and the cl100k token count from the raw text** and compares both against the selection record. Any mismatch aborts (fail-closed). A changed document, a wrong token count, and a duplicate `doc_hash` are all rejected (test sections 6–7).
4. Writes a packed `uint32` token stream (`enc(text) + EOS` per document, selection order = training order) plus a manifest with file size, SHA-256, and counts. `verify_token_stream()` re-checks all of it.

## 3. Tokenizer correction (fixed)

**Problem:** the 9.1.1 proposal said the model vocabulary was 100,256 ("ordinary mergeable tokens") and pinned the version with `assert TIKTOKEN_VERSION == "0.14.0" or True` — an assertion that can never fail.

**Fix (verified by computation against the real encoding):**

- cl100k_base has **100,256 ordinary mergeable tokens, but valid token ids run to 100,277** — the model embedding and output head must have **100,278 rows**. The trainer now sizes vocab from the encoding itself and refuses anything else.
- **EOS = id 100257** (cl100k's `<|endoftext|>`; its dict key is an invisible Unicode string, not the empty string — the code looks the id up by value, never by that key).
- The version check is a real abort: the wrong tiktoken version kills preparation, validation, and training (tested against a fake 0.99.0).
- Token ids are `uint32` in the stream: cl100k ids exceed 65,535, so `uint16` cannot represent the vocabulary (test section 3 proves it).

## 4. Parameter count correction (fixed)

**Problem:** the 9.1.1 proposal derived 162,004,896 parameters — outside the ±0.5% band (that would be a 162M model, not a 150M model).

**Fix:** the actual instantiated architecture is **150,580,452 parameters (+0.387%)**:

- 73,403,496 — tied embedding (100,278 × 732, counted once)
- 77,176,224 — 12 layers × 6,431,352 (4·732² attention + 2·732·2928 MLP + 2·732 RMSNorm)
- 732 — final norm

12 layers, d_model 732, 6 heads (head_dim 122 — must be even for RoPE), MLP 2928, RoPE θ=10000, RMSNorm, GELU, no biases, tied embeddings, causal SDPA attention. `build_model()` prints the exact count and aborts outside ±0.5% **before allocating the optimizer** (test section 8). Head-dim 61 (odd) was rejected during development — RoPE requires an even head dimension.

## 5. Training loop (implemented)

`Trainer` in `MICROCORE_150M_TRAIN.py`:

- **Microbatch vs effective batch:** effective batch = microbatch × grad_accum × seq_len. Default 8 × 30 × 2048 = **491,520 tokens/step**. The architecture was NOT changed to fit memory — microbatching and accumulation are adjustable; on the 7.8 GB CPU sandbox the smoke run used microbatch 1 + gradient checkpointing (the 150M model + Adam states alone exceed RAM there; on the 14.5 GiB T4 the defaults apply).
- **Precision:** FP16 autocast + `GradScaler` on CUDA; CPU fallback for smoke tests only. `steps_per_s`, `tokens_per_s`, `gpu_mem_gib` are logged per step.
- **Token accounting:** `tokens_consumed` is incremented inside `train_step` by the actual tokens fed — the smoke test proves the counter advances correctly across a checkpoint reload (4 steps = 8,192 tokens; after resume, step 5 at 10,240 tokens). No examples are repeated: one deterministic pass over the stream (epochs=1); if the budget is exhausted the run stops.
- **Optimizer:** AdamW, lr 3e-4 peak, cosine decay to 3e-5, 2% linear warmup (122 steps), weight decay 0.1, betas (0.9, 0.95), grad clip 1.0, seed 1337.

## 6. Checkpointing and resume (implemented)

- Atomic writes: temp file + rename; `ckpt_last.pt` and `ckpt_prev.pt` (previous known-good) in `microcore_data/train_150m/`, interval 500 steps.
- Checkpoints carry model, optimizer, scaler, step, `tokens_consumed`, `seq_cursor`, RNG states, config, param count, and the **data-manifest SHA-256**. Resume validates parameter count, vocab/EOS, and manifest hash; a corrupt latest checkpoint falls back to the previous known-good (tested by deliberately corrupting a checkpoint); a checkpoint from different data is rejected.
- The smoke test executed a real round-trip: 4 steps → save → fresh process → reload → continued training at step 5 with correct counters.

## 7. Materializer resume (fixed during testing)

Test section 7 found a real defect: a mid-shard interruption left a partial tail that the naive resume (truncate to the raw byte counter) duplicated. **Fix:** the resume point is now derived from the **last completed shard boundary** (`shards_done` ∩ `shard_offsets`), the stream and doc-bounds files are truncated to that boundary, and the interrupted shard is rewritten exactly. Verified: interrupted → resumed → byte-exact stream (196 stream tokens, 5 documents).

## 8. Bugs found and fixed during development (all via real execution)

| Bug | Symptom | Fix |
|---|---|---|
| uint16 stream | token id 76,637 overflowed | stream dtype uint32 |
| odd head_dim 61 | RoPE reshape failure | n_head 6, head_dim 122 |
| `torch.bfloat32` typo | AttributeError | corrected |
| autocast as context manager | double-call error | factory returning a fresh context per use |
| CPU OOM kill | silent SIGKILL on 150M + Adam | microbatch 1 + gradient checkpointing for CPU smoke; T4 defaults unchanged |
| token counter not advancing on resume | counter frozen after reload | accounting moved into `train_step` |
| mid-shard resume duplication | 2 extra tokens after interruption | shard-boundary resume (section 7) |
| disk exhaustion | smoke checkpoints filled the 20 GB sandbox | test fixtures cleaned; checkpoints re-created on demand |

## 9. Quarantine note (agent stage)

The contamination quarantines from 9.1.1 (SWE-bench Verified vs SWE-rebench-V2, Terminal-Bench vs Nemotron-Pivot, 0 overlaps) apply to **later agent-stage training**, not to this 150M base run, whose data is FineWeb-Edu only.

## 10. Files changed / added in this revision

| File | Change |
|---|---|
| `MICROCORE_150M_TRAIN.py` | **new** — trainer, materializer, checkpointing, smoke test, CLI |
| `MICROCORE_150M_CONFIG.json` | **new** — full experiment config (exact param count 150,580,452, 6,103 steps) |
| `MICROCORE_150M_TRAIN_TESTS.py` | **new** — 55 production-function checks across 15 sections |
| `microcore_paths.py` | **new** — shared storage root |
| `MICROCORE_REVISION_9_1_1_DATA_PREPARATION.py` | import-safe, shared root, strict tokenizer abort, argparse/lock under `__main__` |
| `MICROCORE_REVISION_9_1_1_COLAB_VALIDATION.py` | shared root (fixes the Drive path mismatch) |

Per the standing constraints, nothing was committed to the GitHub repo; all files are downloadable deliverables. No datasets were added or re-discovered.
