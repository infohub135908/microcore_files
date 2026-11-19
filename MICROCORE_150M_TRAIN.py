"""
MICROCORE 150M — decoder-only language-model trainer (Revision 9.1.2)
====================================================================
Trains MICROCORE 150M from random initialization on the Revision 9.1.1 verified
base selection (FineWeb-Edu sample-10BT @ pinned SHA, 2,994,709 documents,
3,000,001,612 cl100k tokens). No pretrained weights are used.

Pipeline stages (all executable from this file):
  1. STRICT tokenizer validation (tiktoken==0.14.0, cl100k_base).
  2. Reconstruction of the actual selected documents from the selection
     manifest: the exact FineWeb-Edu revision is re-fetched, each selected
     document is located by (shard, row), its SHA-256 hash and cl100k token
     count are RECOMPUTED and verified against the manifest. Mismatched,
     missing, changed, or duplicated documents abort the run (fail closed).
  3. Token materialization: a sequential uint16 token stream with explicit
     document boundaries (EOS=100257 appended per document), checksummed,
     resumable, disk-space checked.
  4. Training: causal LM, seq_len 2048, packed sequences, AdamW 3e-4 +
     cosine decay + 2% warmup, FP16 mixed precision + gradient scaling +
     gradient checkpointing on GPU, microbatch x gradient accumulation.
  5. Checkpointing: atomic writes, previous-known-good retention, exact
     training-position resume (no repeated/skipped batches), RNG states saved.

Usage (Colab):
    from google.colab import drive; drive.mount('/content/drive')
    !pip -q install "tiktoken==0.14.0" pyarrow requests numpy torch
    %run MICROCORE_150M_TRAIN.py --materialize-only        # build token stream
    %run MICROCORE_150M_TRAIN.py --smoke                  # smoke test
    %run MICROCORE_150M_TRAIN.py                          # full 3B-token run
    %run MICROCORE_150M_TRAIN.py --resume                 # after interruption

All storage lives under the shared data root (see microcore_paths.py):
    microcore_data/prepared/  - selection + token stream + manifests
    microcore_data/train_150m/- checkpoints, logs, run manifests
"""

import argparse, hashlib, json, math, os, random, shutil, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microcore_paths as mp
from microcore_paths import PREPARED as PREP, TRAIN

# ---------------------------------------------------------------------------
# Configuration (MICROCORE_150M_CONFIG.json is authoritative; values below are
# the frozen defaults and must match it).
# ---------------------------------------------------------------------------
TIKTOKEN_PIN = "0.14.0"
ENCODING = "cl100k_base"
BASE_DATASET = "HuggingFaceFW/fineweb-edu"
BASE_SHA = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
EOS_ID = 100257           #  is the EOS convention for every document
VOCAB_SIZE = 100278       # ids 0..100277 (100,256 ordinary + special/reserved)
PARAM_TARGET = 150_000_000
PARAM_TOLERANCE = 0.005   # abort outside 149.25M..150.75M

MODEL = dict(n_layer=12, d_model=732, n_head=6, mlp_hidden=2928,
             rope_theta=10000.0)
TRAIN_CFG = dict(seq_len=2048, microbatch=8, grad_accum=30,
                 lr=3e-4, min_lr=3e-5, warmup_frac=0.02, weight_decay=0.1,
                 beta1=0.9, beta2=0.95, eps=1e-8, grad_clip=1.0,
                 fp16=True, grad_checkpoint=True, seed=1337,
                 ckpt_interval=500, token_target=3_000_000_000)

# ---------------------------------------------------------------------------
# 1. Strict tokenizer validation
# ---------------------------------------------------------------------------
def validate_tokenizer():
    import tiktoken
    ver = getattr(tiktoken, "__version__", "unknown")
    if ver != TIKTOKEN_PIN:
        sys.exit(f"ABORTED: tiktoken {ver} != required {TIKTOKEN_PIN} "
                 f"(frozen for all four model sizes). pip install tiktoken=={TIKTOKEN_PIN}")
    enc = tiktoken.get_encoding(ENCODING)
    specials = {t: enc.encode(t, allowed_special={t})[0] for t in enc.special_tokens_set}
    max_id = max(max(specials.values()), 100255)
    n_ordinary = len(enc.token_byte_values())
    assert EOS_ID in specials.values(), "EOS id not found in special tokens"
    return enc, {"tiktoken_version": ver, "encoding": ENCODING,
                 "n_ordinary_tokens": n_ordinary, "special_tokens": specials,
                 "max_token_id": max_id, "eos_id": EOS_ID,
                 "model_vocab_size": VOCAB_SIZE}

# ---------------------------------------------------------------------------
# 2. Model: decoder-only Transformer, tied embeddings, RoPE, RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq, theta=10000.0):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv)                      # (T, hd/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
    def forward(self, x):                                 # x: (B, H, T, hd)
        T = x.shape[-2]
        cos, sin = self.cos[:T], self.sin[:T]             # (T, hd/2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos = cos.view(1, 1, T, -1); sin = sin.view(1, 1, T, -1)
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1
                           ).flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        assert d % n_head == 0
        self.n_head, self.head_dim = n_head, d // n_head
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rope = None  # attached by the parent block
    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        # SDPA memory-efficient backend works on both CPU (math) and T4 CUDA;
        # the T4 (SM75) does not support FlashAttention kernels, so SDPA
        # selects its efficient/math backend automatically.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(y)

class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc = nn.Linear(d, hidden, bias=False)
        self.proj = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))

class Block(nn.Module):
    def __init__(self, d, n_head, hidden, max_seq, theta):
        super().__init__()
        self.ln1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.attn.rope = RoPE(d // n_head, max_seq, theta)
        self.ln2 = RMSNorm(d)
        self.mlp = MLP(d, hidden)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class DecoderOnlyLM(nn.Module):
    """Decoder-only Transformer LM with tied input/output embeddings."""
    def __init__(self, vocab, n_layer, d_model, n_head, mlp_hidden,
                 max_seq, rope_theta=10000.0):
        super().__init__()
        self.use_checkpoint = False   # enable via gradient_checkpointing(True)
        self.wte = nn.Embedding(vocab, d_model)           # tied: head reuses wte
        self.blocks = nn.ModuleList([Block(d_model, n_head, mlp_hidden,
                                            max_seq, rope_theta)
                                     for _ in range(n_layer)])
        self.ln_f = RMSNorm(d_model)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def gradient_checkpointing(self, enable=True):
        self.use_checkpoint = enable
    def forward(self, idx):
        x = self.wte(idx)
        for b in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        x = self.ln_f(x)
        return F.linear(x, self.wte.weight)              # tied output head
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

def build_model(cfg=MODEL, max_seq=None, device="cpu", grad_checkpoint=None):
    max_seq = max_seq or (TRAIN_CFG["seq_len"] + 1)
    model = DecoderOnlyLM(VOCAB_SIZE, cfg["n_layer"], cfg["d_model"],
                          cfg["n_head"], cfg["mlp_hidden"], max_seq,
                          cfg["rope_theta"]).to(device)
    if grad_checkpoint is None:
        grad_checkpoint = (device != "cpu")   # default: on for GPU
    model.gradient_checkpointing(grad_checkpoint)
    n = model.num_parameters()
    print(f"model: {cfg['n_layer']} layers, d_model={cfg['d_model']}, "
          f"{cfg['n_head']} heads, tied embeddings, vocab={VOCAB_SIZE}")
    print(f"EXACT PARAMETER COUNT: {n:,} "
          f"({100*(n-PARAM_TARGET)/PARAM_TARGET:+.3f}% vs {PARAM_TARGET:,})")
    if abs(n - PARAM_TARGET) > PARAM_TOLERANCE * PARAM_TARGET:
        sys.exit(f"ABORTED: parameter count {n:,} outside the allowed "
                  f"+/-0.5% range around {PARAM_TARGET:,}")
    return model, n

# ---------------------------------------------------------------------------
# 3. Base-selection reconstruction + token materialization
# ---------------------------------------------------------------------------
STREAM = os.path.join(PREP, "base_tokens.u32")
BOUNDS = os.path.join(PREP, "base_doc_ends.u64")  # per-doc cumulative token counts
STREAM_MANIFEST = os.path.join(PREP, "base_tokens_manifest.json")
MAT_STATE = os.path.join(PREP, "base_tokens_state.json")

def set_stream_dir(path):
    """Redirect the token stream to another directory (e.g. Colab local disk:
    the full stream is ~12 GB and may exceed a 15 GB Drive quota while
    checkpoints/logs stay on Drive)."""
    global STREAM, BOUNDS, STREAM_MANIFEST, MAT_STATE
    os.makedirs(path, exist_ok=True)
    STREAM = os.path.join(path, "base_tokens.u32")
    BOUNDS = os.path.join(path, "base_doc_ends.u64")
    STREAM_MANIFEST = os.path.join(path, "base_tokens_manifest.json")
    MAT_STATE = os.path.join(path, "base_tokens_state.json")

def load_selection(path=None):
    """Parse the selection manifest. Returns ordered list of dicts and a
    per-shard index. Rejects malformed records."""
    path = path or os.path.join(PREP, "base_selection.jsonl")
    recs, seen_hashes = [], set()
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            for k in ("shard", "row", "doc_id", "doc_hash", "tokens"):
                if k not in r:
                    raise ValueError(f"selection line {i}: missing field {k!r}")
            if r["doc_hash"] in seen_hashes:
                raise ValueError(f"selection line {i}: duplicate doc_hash")
            seen_hashes.add(r["doc_hash"])
            recs.append(r)
    shards = {}
    for r in recs:
        shards.setdefault(r["shard"], []).append(r)
    for v in shards.values():
        v.sort(key=lambda r: r["row"])
    return recs, shards

def _shard_local_path(shard_name):
    import MICROCORE_REVISION_9_1_1_DATA_PREPARATION as prep
    return prep.cached_path(BASE_DATASET, BASE_SHA, shard_name)

def materialize_tokens(enc, limit_docs=None, shard_paths=None, sel_path=None):
    """Reconstruct the actual selected documents and materialize the training
    token stream. For every selected document:
      - fetch the row from the parquet shard at the pinned revision,
      - recompute SHA-256(text) and compare with the selection doc_hash,
      - recompute the cl100k token count and compare with the selection count,
      - append enc(text) + EOS to the uint16 stream.
    Any mismatch/missing/duplicate document aborts (fail closed). Document
    order is exactly the selection order (shard order, then row order).
    Resumable: per-shard completion + byte offsets recorded in a state file;
    an interrupted shard is truncated and redone.
    """
    import pyarrow.parquet as pq
    import MICROCORE_REVISION_9_1_1_DATA_PREPARATION as prep

    sel_path = sel_path or os.path.join(PREP, "base_selection.jsonl")
    sel_manifest = os.path.join(os.path.dirname(sel_path), "base_prep_manifest.json")
    for p in (sel_path, sel_manifest):
        if not os.path.exists(p):
            sys.exit(f"ABORTED: {p} missing - run the 9.1.1 data preparation first")
    sel_meta = json.load(open(sel_manifest))
    if sel_meta["result"]["total_cl100k_tokens"] != 3_000_001_612:
        sys.exit("ABORTED: selection manifest does not match the verified "
                 "3,000,001,612-token subset - refusing to rebuild")

    recs, by_shard = load_selection(sel_path)
    if limit_docs:
        keep = {id(r) for r in recs[:limit_docs]}
        recs = recs[:limit_docs]
        by_shard = {}
        for r in recs:
            by_shard.setdefault(r["shard"], []).append(r)
        for v in by_shard.values():
            v.sort(key=lambda r: r["row"])
    print(f"selection: {len(recs):,} documents across {len(by_shard)} shards")

    # --- disk-space check before materializing
    est_tokens = sum(r["tokens"] for r in recs) + len(recs)   # + EOS each
    # uint32: cl100k ids reach 100,277 > uint16 range (65535), so the stream
    # MUST be uint32 (or wider) to represent the entire vocabulary.
    est_bytes = est_tokens * 4 + len(recs) * 8 + (1 << 20)
    free = shutil.disk_usage(os.path.dirname(PREP)).free
    print(f"estimated stream: {est_tokens:,} tokens "
          f"({est_bytes/1e9:.2f} GB); free disk: {free/1e9:.2f} GB")
    if est_bytes > free:
        sys.exit(f"ABORTED: insufficient disk space (need {est_bytes:,} bytes, "
                 f"have {free:,}). Free space or use --limit-docs.")

    # --- state / resume
    state = {"shards_done": {}, "stream_bytes": 0, "limit_docs": limit_docs,
             "shard_offsets": {}}
    if os.path.exists(MAT_STATE):
        prev = json.load(open(MAT_STATE))
        if prev.get("limit_docs") == limit_docs:
            state = prev          # same scope: safe to resume
        else:
            print("  materialization scope changed - resetting token stream")
            for f in (STREAM, BOUNDS):
                if os.path.exists(f): os.remove(f)
            state = {"shards_done": {}, "stream_bytes": 0, "limit_docs": limit_docs}

    mode = "r+b" if os.path.exists(STREAM) else "wb"
    fh = open(STREAM, mode)
    # Resume from the last COMPLETED SHARD BOUNDARY. A mid-shard interruption
    # can leave a partial tail beyond that boundary; deriving the resume point
    # from shards_done (not the raw byte counter) guarantees the tail is
    # discarded and the interrupted shard is rewritten exactly.
    boundaries = sorted(v for k, v in (state.get("shard_offsets") or {}).items()
                         if k in state["shards_done"])
    resume_bytes = boundaries[-1] if boundaries else 0
    fh.truncate(resume_bytes)
    fh.seek(resume_bytes)
    state["stream_bytes"] = resume_bytes
    bounds_fh = open(BOUNDS, "r+b" if os.path.exists(BOUNDS) else "wb")
    done_docs = state.get("total_docs", 0)
    if os.path.exists(BOUNDS):
        bounds_fh.truncate(done_docs * 8)      # drop partial-shard doc ends
        bounds_fh.seek(done_docs * 8)

    total_docs = state.get("total_docs", 0)
    total_doc_tokens = state.get("total_doc_tokens", 0)
    total_stream_tokens = state.get("stream_tokens", 0)
    t0 = time.time()
    for shard_name in sorted(by_shard):
        wanted = by_shard[shard_name]
        if state["shards_done"].get(shard_name):
            print(f"  {shard_name}: already materialized - skipping")
            continue
        local = (shard_paths or {}).get(shard_name) or _shard_local_path(shard_name)
        if local is None or not os.path.exists(local):
            files = prep.hub_tree(BASE_DATASET, BASE_SHA, "sample/10BT")
            meta = next(f for f in files if f["path"].endswith(shard_name))
            local = prep.download_verified(BASE_DATASET, BASE_SHA, meta["path"],
                                           size=meta.get("size"),
                                           sha256=(meta.get("lfs") or {}).get("oid"))
        want_rows = {r["row"]: r for r in wanted}
        n_shard_docs = 0; n_shard_tokens = 0
        row_idx = -1
        for batch in pq.ParquetFile(local).iter_batches(batch_size=1024,
                                                        columns=["text"]):
            for text in batch.to_pydict()["text"]:
                row_idx += 1
                sel = want_rows.get(row_idx)
                if sel is None:
                    continue
                txt = text or ""
                h = hashlib.sha256(txt.encode()).hexdigest()
                if h != sel["doc_hash"]:
                    sys.exit(f"ABORTED: doc_hash mismatch {shard_name} row {row_idx}: "
                             f"document changed or wrong revision (fail closed)")
                ids = enc.encode(txt, disallowed_special=())
                if len(ids) != sel["tokens"]:
                    sys.exit(f"ABORTED: token-count mismatch {shard_name} row {row_idx}: "
                             f"{len(ids)} != manifest {sel['tokens']} (fail closed)")
                if ids and (min(ids) < 0 or max(ids) >= VOCAB_SIZE):
                    sys.exit(f"ABORTED: token id out of uint32/vocab range "
                             f"{shard_name} row {row_idx}")
                ids = ids + [EOS_ID]
                arr = np.asarray(ids, dtype=np.uint32)
                fh.write(arr.tobytes())
                total_stream_tokens += len(ids)
                total_doc_tokens += sel["tokens"]
                total_docs += 1
                bounds_fh.write(np.uint64(total_stream_tokens).tobytes())
                n_shard_docs += 1; n_shard_tokens += sel["tokens"]
        if n_shard_docs != len(wanted):
            sys.exit(f"ABORTED: {shard_name}: reconstructed {n_shard_docs} of "
                     f"{len(wanted)} selected documents - missing rows (fail closed)")
        fh.flush(); os.fsync(fh.fileno()); bounds_fh.flush(); os.fsync(bounds_fh.fileno())
        state["shards_done"][shard_name] = {"docs": n_shard_docs,
                                             "doc_tokens": n_shard_tokens}
        state.setdefault("shard_offsets", {})[shard_name] = fh.tell()
        state["stream_bytes"] = fh.tell()
        state["total_docs"] = total_docs
        state["total_doc_tokens"] = total_doc_tokens
        state["stream_tokens"] = total_stream_tokens
        json.dump(state, open(MAT_STATE + ".tmp", "w")); os.replace(MAT_STATE + ".tmp", MAT_STATE)
        print(f"  {shard_name}: {n_shard_docs:,} docs verified+written "
              f"({n_shard_tokens:,} doc tokens) [{time.time()-t0:.0f}s]")
    fh.close(); bounds_fh.close()

    # --- final manifest with checksum
    m = {"purpose": "MICROCORE 150M base training token stream",
         "dataset": BASE_DATASET, "revision_sha": BASE_SHA,
         "tokenizer": {"package": "tiktoken", "version": TIKTOKEN_PIN,
                       "encoding": ENCODING, "eos_id": EOS_ID},
         "dtype": "uint32", "vocab_size": VOCAB_SIZE,
         "documents": total_docs, "doc_tokens": total_doc_tokens,
         "stream_tokens_incl_eos": total_stream_tokens,
         "stream_file": STREAM, "bounds_file": BOUNDS,
         "selection_manifest_sha256": file_sha256(sel_path),
         "limit_docs": limit_docs}
    m["stream_file_bytes"] = os.path.getsize(STREAM)
    m["stream_file_sha256"] = file_sha256(STREAM)
    m["bounds_file_sha256"] = file_sha256(BOUNDS)
    json.dump(m, open(STREAM_MANIFEST + ".tmp", "w")); os.replace(STREAM_MANIFEST + ".tmp",
                                                                 STREAM_MANIFEST)
    print(f"token stream complete: {total_docs:,} docs, {total_doc_tokens:,} doc "
          f"tokens, {total_stream_tokens:,} stream tokens (incl. EOS)")
    print(f"manifest: {STREAM_MANIFEST}")
    return m

def file_sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_token_stream(manifest_path=None):
    """Verify the materialized stream against its manifest: file size, dtype
    bounds, token count, and checksum. Returns (ok, details)."""
    manifest_path = manifest_path or STREAM_MANIFEST
    if not os.path.exists(manifest_path):
        return False, {"error": "stream manifest missing"}
    m = json.load(open(manifest_path))
    errs = []
    if not os.path.exists(STREAM): errs.append("stream file missing")
    else:
        if os.path.getsize(STREAM) != m.get("stream_file_bytes"):
            errs.append("stream file size mismatch")
        if file_sha256(STREAM) != m.get("stream_file_sha256"):
            errs.append("stream file sha256 mismatch")
    if not os.path.exists(BOUNDS): errs.append("bounds file missing")
    else:
        n_docs = os.path.getsize(BOUNDS) // 8
        if n_docs != m.get("documents"): errs.append("document count mismatch")
    width = {"uint16": 2, "uint32": 4, "int32": 4}.get(m.get("dtype"), 0)
    if width and m.get("stream_tokens_incl_eos", 0) * width != m.get("stream_file_bytes", 0):
        errs.append("token-count/size arithmetic mismatch")
    return (len(errs) == 0), {"errors": errs, "manifest": {k: m[k] for k in
        ("documents", "doc_tokens", "stream_tokens_incl_eos", "stream_file_bytes")}}

# ---------------------------------------------------------------------------
# 4. Packed sequence loader (memory-mapped, sequential, deterministic)
# ---------------------------------------------------------------------------
class PackedSequences:
    """Concatenated token stream chunked into fixed-length training sequences.
    Document boundaries are preserved by the EOS convention; sequences never
    contain metadata. Reads are memory-mapped - the corpus is never fully
    loaded into RAM."""
    def __init__(self, stream_path=None, seq_len=2048, dtype=np.uint32):
        self.path = stream_path or STREAM
        self.seq_len = seq_len
        self.mm = np.memmap(self.path, mode="r", dtype=dtype)
        n_tokens = self.mm.shape[0]
        self.n_seq = n_tokens // (seq_len + 1)   # x=seq_len, y=shifted by 1
    def __len__(self): return self.n_seq
    def get(self, i):
        blk = self.mm[i * (self.seq_len + 1):(i + 1) * (self.seq_len + 1)]
        blk = np.asarray(blk, dtype=np.int64)
        return blk[:-1].copy(), blk[1:].copy()
    def total_stream_tokens(self): return int(self.mm.shape[0])

# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
def lr_at(step, total_steps, cfg=TRAIN_CFG):
    warmup = max(1, int(cfg["warmup_frac"] * total_steps))
    if step < warmup:
        return cfg["lr"] * (step + 1) / warmup
    t = (step - warmup) / max(1, total_steps - warmup)
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * t))

class Trainer:
    def __init__(self, model, cfg=TRAIN_CFG, run_dir=TRAIN):
        self.model, self.cfg, self.run_dir = model, cfg, run_dir
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                     betas=(cfg["beta1"], cfg["beta2"]),
                                     eps=cfg["eps"],
                                     weight_decay=cfg["weight_decay"])
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg["fp16"] and torch.cuda.is_available())
        self.step = 0; self.tokens_consumed = 0; self.seq_cursor = 0
        self.total_steps = None
        self.log_path = os.path.join(run_dir, "train_log.jsonl")
        self.ckpt_path = os.path.join(run_dir, "ckpt_last.pt")
        self.prev_path = os.path.join(run_dir, "ckpt_prev.pt")
        self.manifest_path = os.path.join(run_dir, "run_manifest.json")

    # -- logging ------------------------------------------------------------
    def log(self, **kw):
        rec = {"ts": time.time(), "model": "MICROCORE_150M",
               "param_count": self.model.num_parameters(),
               "step": self.step, "tokens_consumed": self.tokens_consumed}
        rec.update(kw)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # -- checkpoints --------------------------------------------------------
    def save_checkpoint(self, extra=None):
        """Atomic checkpoint write with previous-known-good retention."""
        st = {"model": {k: v.cpu() for k, v in self.model.state_dict().items()},
              "optimizer": self.opt.state_dict(),
              "scaler": self.scaler.state_dict(),
              "step": self.step, "tokens_consumed": self.tokens_consumed,
              "seq_cursor": self.seq_cursor,
              "rng": {"python": random.getstate(),
                      "numpy": np.random.get_state(),
                      "torch": torch.get_rng_state()},
              "config": {"model": MODEL, "train": self.cfg},
              "param_count": self.model.num_parameters(),
              "vocab_size": VOCAB_SIZE, "eos_id": EOS_ID,
              "stream_manifest_sha256": extra or None}
        tmp = self.ckpt_path + ".tmp"
        torch.save(st, tmp)
        with open(tmp, "rb+") as f: os.fsync(f.fileno())
        if os.path.exists(self.ckpt_path):
            shutil.copyfile(self.ckpt_path, self.prev_path)  # keep known-good
        os.replace(tmp, self.ckpt_path)
        self.log(event="checkpoint_saved", checkpoint=self.ckpt_path)

    def try_resume(self, expected_manifest_sha):
        """Validate the latest checkpoint and resume from the exact position.
        Corrupt/incompatible checkpoints are rejected; the previous known-good
        checkpoint is tried next. Returns True if resumed."""
        for path in (self.ckpt_path, self.prev_path):
            if not os.path.exists(path):
                continue
            try:
                st = torch.load(path, map_location="cpu", weights_only=False)
                if st.get("param_count") != self.model.num_parameters():
                    raise ValueError("parameter count mismatch")
                if st.get("vocab_size") != VOCAB_SIZE or st.get("eos_id") != EOS_ID:
                    raise ValueError("tokenizer/vocab mismatch")
                if st.get("stream_manifest_sha256") != expected_manifest_sha:
                    raise ValueError("data manifest mismatch - data changed")
                self.model.load_state_dict(st["model"])
                self.opt.load_state_dict(st["optimizer"])
                self.scaler.load_state_dict(st["scaler"])
                self.step = st["step"]; self.tokens_consumed = st["tokens_consumed"]
                self.seq_cursor = st["seq_cursor"]
                random.setstate(st["rng"]["python"])
                np.random.set_state(st["rng"]["numpy"])
                torch.set_rng_state(st["rng"]["torch"])
                self.log(event="resume", checkpoint=path, step=self.step,
                         tokens_consumed=self.tokens_consumed, seq_cursor=self.seq_cursor)
                print(f"resumed from {path}: step {self.step}, "
                      f"{self.tokens_consumed:,} tokens, cursor {self.seq_cursor}")
                return True
            except Exception as e:
                print(f"  checkpoint {path} rejected: {e}")
        return False

    # -- one optimizer step over a microbatch --------------------------------
    def train_step(self, batch_x, batch_y, amp_ctx):
        """Accumulate gradients over one microbatch. Returns (loss, grad_norm)."""
        x = torch.from_numpy(batch_x); y = torch.from_numpy(batch_y)
        if torch.cuda.is_available():
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        # account for EXACTLY the tokens actually fed in this microbatch
        self.tokens_consumed += int(x.shape[0]) * int(x.shape[1])
        with amp_ctx():
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   y.view(-1))
        self.scaler.scale(loss).backward()
        self._pending_loss = loss.item()
        return loss.item()

    def finish_accumulation(self):
        self.scaler.unscale_(self.opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                  self.cfg["grad_clip"])
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        lr = lr_at(self.step, self.total_steps, self.cfg)
        for g in self.opt.param_groups: g["lr"] = lr
        self.step += 1
        return grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)

def run_training(enc_info, args):
    cfg = dict(TRAIN_CFG)
    if args.microbatch: cfg["microbatch"] = args.microbatch
    if args.accum: cfg["grad_accum"] = args.accum
    if args.steps: cfg["max_steps"] = args.steps
    if args.no_fp16: cfg["fp16"] = False
    if args.no_ckpt: cfg["grad_checkpoint"] = False
    if args.ckpt_interval: cfg["ckpt_interval"] = args.ckpt_interval

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB)"
             if device == "cuda" else " (CPU fallback - smoke tests only)"))

    model, n_params = build_model(device=device,
                                   grad_checkpoint=cfg.get("grad_checkpoint"))

    # -- data
    ok, det = verify_token_stream()
    m = json.load(open(STREAM_MANIFEST)) if os.path.exists(STREAM_MANIFEST) else {}
    scope_ok = (m.get("limit_docs") == args.limit_docs
                and (args.limit_docs is not None
                     or m.get("doc_tokens") == 3_000_001_612))
    if not ok or not scope_ok:
        print("token stream missing/incomplete/wrong scope - materializing now")
        materialize_tokens(enc_info[0], limit_docs=args.limit_docs)
        ok, det = verify_token_stream()
        if not ok: sys.exit(f"ABORTED: token stream verification failed: {det}")
        m = json.load(open(STREAM_MANIFEST))
    seqs = PackedSequences(seq_len=cfg["seq_len"])
    print(f"training sequences: {len(seqs):,} x {cfg['seq_len']} tokens "
          f"({seqs.total_stream_tokens():,} stream tokens)")

    # deterministic order: sequential cursor (no shuffling across epochs; the
    # selection order is the training order - one deterministic pass)
    eff_tokens = cfg["microbatch"] * cfg["grad_accum"] * cfg["seq_len"]
    total_steps = len(seqs) // (cfg["microbatch"] * cfg["grad_accum"])
    if args.smoke:
        total_steps = min(total_steps, args.smoke)
    trainer = Trainer(model, cfg)
    trainer.total_steps = total_steps
    manifest_sha = m.get("stream_file_sha256")

    # -- run manifest (written before training starts)
    run_manifest = {"model": "MICROCORE_150M", "param_count": n_params,
                    "architecture": MODEL, "train_config": cfg,
                    "tokenizer": m["tokenizer"], "vocab_size": VOCAB_SIZE,
                    "eos_id": EOS_ID, "device": device,
                    "data": {"dataset": BASE_DATASET, "revision_sha": BASE_SHA,
                             "selection": "prepared/base_selection.jsonl",
                             "stream_manifest": STREAM_MANIFEST,
                             "documents": m["documents"],
                             "doc_tokens": m["doc_tokens"],
                             "stream_tokens": m["stream_tokens_incl_eos"],
                             "sequences": len(seqs)},
                    "planned_steps": total_steps,
                    "effective_tokens_per_step": eff_tokens,
                    "note": "no validation holdout; the full selection is "
                            "used for training (documented)"}
    json.dump(run_manifest, open(trainer.manifest_path, "w"), indent=1)

    resumed = trainer.try_resume(manifest_sha) if args.resume or True else False
    if not resumed:
        torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        trainer.log(event="run_start", config=cfg, manifest=run_manifest)

    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    use_amp = cfg["fp16"]   # fp16 on GPU, bf16 autocast on CPU for smoke tests
    def amp_ctx():
        if use_amp:
            return torch.autocast(device_type=("cuda" if device == "cuda" else "cpu"),
                                  dtype=amp_dtype)
        return _null_ctx()

    tokens_at_step_start = trainer.tokens_consumed
    t0 = time.time(); loss_accum = 0.0
    while trainer.step < total_steps:
        micro = []
        for _ in range(cfg["grad_accum"]):
            if trainer.seq_cursor + cfg["microbatch"] > len(seqs):
                break
            xs, ys = [], []
            for i in range(trainer.seq_cursor, trainer.seq_cursor + cfg["microbatch"]):
                x, y = seqs.get(i); xs.append(x); ys.append(y)
            trainer.seq_cursor += cfg["microbatch"]
            loss = trainer.train_step(np.stack(xs), np.stack(ys), amp_ctx)
            loss_accum += loss
            if not math.isfinite(loss):
                trainer.log(event="error", error="non-finite loss")
                sys.exit("ABORTED: non-finite training loss")
        grad_norm = trainer.finish_accumulation()

        if trainer.step % 10 == 0 or trainer.step == 1:
            el = time.time() - t0
            mem = (torch.cuda.max_memory_allocated() / 2**30
                   if device == "cuda" else None)
            trainer.log(loss=loss_accum / cfg["grad_accum"],
                       lr=lr_at(trainer.step - 1, total_steps, cfg),
                       grad_norm=grad_norm, elapsed_s=el,
                       steps_per_s=trainer.step / max(el, 1e-9),
                       tokens_per_s=trainer.tokens_consumed / max(el, 1e-9),
                       gpu_mem_gib=mem)
            print(f"  step {trainer.step}/{total_steps} "
                  f"loss {loss_accum/cfg['grad_accum']:.4f} "
                  f"lr {lr_at(trainer.step-1, total_steps, cfg):.2e} "
                  f"tokens {trainer.tokens_consumed:,} "
                  f"({trainer.tokens_consumed/1e6:.1f}M) "
                  + (f"mem {mem:.2f} GiB " if mem else ""))
            loss_accum = 0.0
        if trainer.step % cfg["ckpt_interval"] == 0 or trainer.step == total_steps:
            trainer.save_checkpoint(extra=manifest_sha)
    trainer.save_checkpoint(extra=manifest_sha)
    trainer.log(event="run_complete", tokens_consumed=trainer.tokens_consumed,
                steps=trainer.step)
    print(f"TRAINING PASS COMPLETE: {trainer.step} optimizer steps, "
          f"{trainer.tokens_consumed:,} tokens actually consumed "
          f"(stream tokens incl. EOS)")
    return trainer

class _null_ctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False

# ---------------------------------------------------------------------------
# 6. Smoke test: real data, real forward/backward, checkpoint round-trip
# ---------------------------------------------------------------------------
def smoke_test(enc_info, args):
    print("=" * 66); print("SMOKE TEST (short real run - NOT a completed experiment)")
    print("=" * 66)
    args_smoke = argparse.Namespace(**{**vars(args), "smoke": args.smoke or 4,
                                      "limit_docs": args.limit_docs or 3000,
                                      "resume": False})
    if not torch.cuda.is_available():
        args_smoke.microbatch = 1; args_smoke.accum = 1   # CPU smoke: keep RAM low
    if args.limit_docs is None:
        # materialize a small real slice for the smoke test if none exists
        if not os.path.exists(STREAM_MANIFEST):
            materialize_tokens(enc_info[0], limit_docs=3000)
    t = run_training(enc_info, args_smoke)
    # checkpoint round-trip: reload and continue one more step
    print("-- checkpoint round-trip")
    import gc
    prev_step, prev_cursor, prev_tokens = t.step, t.seq_cursor, t.tokens_consumed
    t.model = None; t.opt = None; t = None; gc.collect()   # free first instance
    mb = 1 if not torch.cuda.is_available() else TRAIN_CFG["microbatch"]
    model2, _ = build_model(device="cuda" if torch.cuda.is_available() else "cpu",
                            grad_checkpoint=True)
    tr = Trainer(model2, dict(TRAIN_CFG))
    tr.total_steps = prev_step + 1
    assert tr.try_resume(json.load(open(STREAM_MANIFEST))["stream_file_sha256"]), \
        "smoke test: resume failed"
    assert tr.seq_cursor == prev_cursor and tr.step == prev_step, "position mismatch"
    assert tr.tokens_consumed == prev_tokens, "token counter mismatch after resume"
    seqs = PackedSequences()
    if tr.seq_cursor + mb <= len(seqs):
        ctx = (lambda: torch.autocast(device_type="cpu", dtype=torch.bfloat16)) \
              if not args.no_fp16 else (lambda: _null_ctx())
        xs, ys = [], []
        for i in range(tr.seq_cursor, tr.seq_cursor + mb):
            x, y = seqs.get(i); xs.append(x); ys.append(y)
        tr.train_step(np.stack(xs), np.stack(ys), ctx)
        tr.finish_accumulation()
        print(f"continued training after reload: step {tr.step}, "
              f"tokens {tr.tokens_consumed:,} (counter advanced correctly)")
    mem = (torch.cuda.max_memory_allocated() / 2**30
           if torch.cuda.is_available() else None)
    print(f"peak GPU memory: {mem:.2f} GiB" if mem else "CPU mode: no GPU memory used")
    print("SMOKE TEST PASSED (loss finite, gradients finite, checkpoint "
          "reload + continue verified, token counter advanced)")
    print("=" * 66)

# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialize-only", action="store_true")
    ap.add_argument("--stream-dir", default=None,
                    help="place the ~12 GB token stream on another disk "
                         "(e.g. /content on Colab) while checkpoints stay on Drive")
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="materialize only the first N selected documents (smoke tests)")
    ap.add_argument("--smoke", type=int, nargs="?", const=4, default=None,
                    help="run a short real training smoke test (N steps)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--microbatch", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-ckpt", action="store_true", help="disable grad checkpointing")
    ap.add_argument("--ckpt-interval", type=int, default=None)
    args = ap.parse_args()

    if args.stream_dir:
        set_stream_dir(args.stream_dir)
    print(f"storage root: {mp.DATA_ROOT}")
    print(f"train dir: {TRAIN}")
    if args.stream_dir:
        print(f"token stream dir: {args.stream_dir}")
    enc_info = validate_tokenizer()
    print(f"tokenizer OK: tiktoken {enc_info[1]['tiktoken_version']} / "
          f"{ENCODING}; eos_id={EOS_ID}; model vocab={VOCAB_SIZE}; "
          f"max token id={enc_info[1]['max_token_id']}")

    if args.materialize_only:
        materialize_tokens(enc_info[0], limit_docs=args.limit_docs)
        ok, det = verify_token_stream()
        print("stream verification:", "PASS" if ok else f"FAIL {det}")
        sys.exit(0 if ok else 1)
    if args.smoke is not None:
        smoke_test(enc_info, args)
        sys.exit(0)
    run_training(enc_info, args)
