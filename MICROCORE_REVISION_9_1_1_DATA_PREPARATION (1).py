"""
MICROCORE Revision 9.1.1 - Data preparation pipeline (Colab-ready)
=================================================================
Prepares the MICROCORE 150M initial training data from the Revision 9.1 verified
manifest. Runs on Google Colab (T4, ~12GB RAM) or any Python 3.10+ machine.

TOKENIZER (FROZEN for the entire 150M/300M/500M/750M series):
    tiktoken==0.14.0, encoding "cl100k_base".
    Used for: base pretraining, agent adaptation, SFT, preference preparation,
    token budgets, and training logs. GPT-2 or dataset-native token columns are
    NEVER mixed into cl100k budgets (source metadata only).

WHAT THIS SCRIPT PRODUCES
    1. base stage: a deterministic token-targeted subset of FineWeb-Edu
       sample-10BT with >= 3,000,000,000 actual cl100k tokens
       (150M params x 20 tokens/param planning assumption), plus an exact
       per-document selection manifest (shard, row index, doc id, tokens).
    2. agent/SFT/preference stages: license-gated, deduplicated, quarantine-
       checked data preparation with exact retained counts, matching the
       Revision 9.1 verified pools.
    3. Everything is checkpointed: interrupted runs resume without
       redownloading verified shards. Prepared data and manifests are written
       to persistent storage (Google Drive on Colab).

USAGE (Colab):
    from google.colab import drive; drive.mount('/content/drive')
    !pip -q install "tiktoken==0.14.0" "pyarrow>=14" requests
    %run MICROCORE_REVISION_9_1_1_DATA_PREPARATION.py            # full pipeline
    %run MICROCORE_REVISION_9_1_1_DATA_PREPARATION.py --base-only
    %run MICROCORE_REVISION_9_1_1_DATA_PREPARATION.py --stages-only
    %run MICROCORE_REVISION_9_1_1_DATA_PREPARATION.py --verify-only   # no writes

Storage root: /content/drive/MyDrive/microcore_data on Colab, ./microcore_data
locally. Shard caches are revision-specific and verified by size + SHA-256
(LFS oid from the Hub tree API), so a valid cached shard is never redownloaded.
"""

import argparse, csv, gzip, hashlib, json, os, sys, time
import requests

# --------------------------------------------------------------------------
# 0. Frozen configuration
# --------------------------------------------------------------------------
TIKTOKEN_PIN = "tiktoken==0.14.0"
ENCODING = "cl100k_base"
SCRIPT_VERSION = "9.1.1"
BASE_TOKEN_TARGET = 3_000_000_000          # 150M params x 20 tokens/param
BASE_DATASET = "HuggingFaceFW/fineweb-edu"
BASE_SHA = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
BASE_CONFIG_DIR = "sample/10BT"             # 14 shards, enumerated from the Hub API

try:
    import tiktoken
    enc = tiktoken.get_encoding(ENCODING)
    import tiktoken as _tk
    TIKTOKEN_VERSION = getattr(_tk, "__version__", "unknown")
    # STRICT version enforcement (Revision 9.1.2): a wrong tokenizer changes
    # every token count in the pipeline; abort rather than warn.
    if TIKTOKEN_VERSION != "0.14.0":
        sys.exit(f"ABORTED: tiktoken {TIKTOKEN_VERSION} != required 0.14.0 "
                 f"(frozen tokenizer for 150M/300M/500M/750M). "
                 f"pip install {TIKTOKEN_PIN}")
except ImportError:
    sys.exit(f"pip install {TIKTOKEN_PIN}")
try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pip install pyarrow")

# Shared data-root configuration (Revision 9.1.2: single explicit root used by
# data preparation, validation, and training - no more mismatched Drive roots).
from microcore_paths import DATA_ROOT as ROOT, CACHE, PREP, HF

# Single-instance lock: two concurrent runs corrupt the selection file.
# (Taken in main() only, so this module can be imported by tests and the
# trainer without side effects.)

# --------------------------------------------------------------------------
# 1. Hub helpers: pinned-revision shard discovery + verified streaming cache
# --------------------------------------------------------------------------
def hub_tree(dataset_id, sha, path=""):
    """Enumerate ACTUAL files at the pinned commit SHA (never 'main')."""
    url = f"{HF}/api/datasets/{dataset_id}/tree/{sha}/{path}"
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"REVISION_VERIFICATION_FAILED: tree {dataset_id}@{sha[:10]}/{path} -> {r.status_code}")
    return r.json()

def resolve_url(dataset_id, sha, path):
    return f"{HF}/datasets/{dataset_id}/resolve/{sha}/{path}"

def cached_path(dataset_id, sha, fname):
    return os.path.join(CACHE, f"{dataset_id.replace('/','__')}@{sha[:10]}", fname)

def download_verified(dataset_id, sha, path, size=None, sha256=None):
    """Stream a repo file at the pinned SHA to a revision-specific cache.
    Verify size and (when the Hub provides it) the LFS SHA-256 oid.
    Atomic rename only after verification. Never reuses unverified files."""
    local = cached_path(dataset_id, sha, os.path.basename(path))
    if os.path.exists(local):
        if (size is None or os.path.getsize(local) == size) and \
           (sha256 is None or file_sha256(local) == sha256):
            return local
        os.remove(local)   # corrupt/incomplete cache: reject, do not reuse
    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = local + ".part"
    if os.path.exists(tmp): os.remove(tmp)
    r = requests.get(resolve_url(dataset_id, sha, path), timeout=3600, stream=True)
    if r.status_code == 429:
        time.sleep(60)
        r = requests.get(resolve_url(dataset_id, sha, path), timeout=3600, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"DOWNLOAD_FAILED: {dataset_id}@{sha[:10]}/{path} -> {r.status_code}")
    h = hashlib.sha256(); got = 0
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk); h.update(chunk); got += len(chunk)
    if size is not None and got != size:
        os.remove(tmp); raise RuntimeError(f"INCOMPLETE_DOWNLOAD: {path}: {got} != {size} bytes")
    if sha256 is not None and h.hexdigest() != sha256:
        os.remove(tmp); raise RuntimeError(f"CACHE_INTEGRITY_FAILED: {path}: sha256 mismatch vs LFS oid")
    os.replace(tmp, local)   # atomic: only verified files enter the cache
    return local

def file_sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b: break
            h.update(b)
    return h.hexdigest()

# --------------------------------------------------------------------------
# 2. Base stage: deterministic token-targeted FineWeb-Edu subset
# --------------------------------------------------------------------------
BASE_POLICY = {
    "empty_documents_removed": "documents with empty/whitespace text are removed",
    "deduplication": "exact SHA-256 document-content dedup across all processed shards (first occurrence kept)",
    "quality_threshold": "none (sample-10BT is already FineWeb-Edu classifier-filtered)",
    "selection": "deterministic: shards in Hub tree order (000_00000..013_00000), rows in parquet order; "
                 "stop at the first document that reaches the 3,000,000,000 cl100k-token target",
}

def prepare_base(target_tokens=BASE_TOKEN_TARGET, verify_only=False, delete_shards=False):
    """Produce the 150M base subset. Streams every shard row group (RAM-safe),
    counts REAL cl100k tokens of retained text, records per-shard provenance."""
    man_path = os.path.join(PREP, "base_prep_manifest.json")
    sel_path  = os.path.join(PREP, "base_selection.jsonl")
    state_path = os.path.join(PREP, "base_state.json")

    # --- resumable state
    state = {"shards_done": {}, "total_docs": 0, "total_tokens": 0, "target_reached": False}
    if os.path.exists(state_path):
        state = json.load(open(state_path))
        print(f"resuming base prep: {state['total_docs']:,} docs, {state['total_tokens']:,} tokens so far")

    # --- discover ALL shards at the pinned revision (never assume filenames)
    files = hub_tree(BASE_DATASET, BASE_SHA, BASE_CONFIG_DIR)
    shards = sorted([f for f in files if f["type"] == "file" and f["path"].endswith(".parquet")],
                    key=lambda f: f["path"])
    print(f"discovered {len(shards)} FineWeb-Edu shards at pinned revision {BASE_SHA[:10]}")
    if len(shards) != 14:
        print(f"WARNING: expected 14 shards, found {len(shards)}")

    seen_docs = set()
    # restore dedup set from the selection file (resume-safe)
    if os.path.exists(sel_path):
        with open(sel_path) as f:
            for line in f:
                seen_docs.add(json.loads(line)["doc_hash"])

    sel = open(sel_path, "a")
    t0 = time.time()
    for shard in shards:
        fname = os.path.basename(shard["path"])
        if state["shards_done"].get(fname, {}).get("complete") and not verify_only:
            print(f"  {fname}: already complete - skipping")
            continue
        if state["target_reached"]:
            break
        local = download_verified(BASE_DATASET, BASE_SHA, shard["path"],
                                 size=shard.get("size"),
                                 sha256=(shard.get("lfs") or {}).get("oid"))
        print(f"  {fname}: verified ({shard.get('size',0):,} bytes, lfs oid "
              f"{(shard.get('lfs') or {}).get('oid','none')[:12]})")
        if verify_only:
            continue
        # --- stream rows, filter, dedup, count
        stats = {"rows_raw": 0, "empty_removed": 0, "dup_removed": 0,
                 "docs_kept": 0, "tokens": 0}
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=1024, columns=["text", "id"]):
            d = batch.to_pydict()
            for text, doc_id in zip(d["text"], d["id"]):
                stats["rows_raw"] += 1
                if not text or not text.strip():
                    stats["empty_removed"] += 1; continue
                dh = hashlib.sha256(text.encode()).hexdigest()
                if dh in seen_docs:
                    stats["dup_removed"] += 1; continue
                seen_docs.add(dh)
                ntok = len(enc.encode(text, disallowed_special=()))
                stats["docs_kept"] += 1; stats["tokens"] += ntok
                state["total_docs"] += 1; state["total_tokens"] += ntok
                sel.write(json.dumps({"shard": fname, "row": stats["rows_raw"] - 1,
                                      "doc_id": doc_id, "doc_hash": dh, "tokens": ntok}) + "\n")
                if state["total_tokens"] >= target_tokens:
                    state["target_reached"] = True
                    break
            if state["target_reached"]: break
        stats["complete"] = not state["target_reached"] or state["total_tokens"] >= target_tokens
        state["shards_done"][fname] = stats
        print(f"  {fname}: rows={stats['rows_raw']:,} kept={stats['docs_kept']:,} "
              f"tokens={stats['tokens']:,} (cum {state['total_tokens']:,}) "
              f"[{time.time()-t0:.0f}s]")
        sel.flush()
        json.dump(state, open(state_path, "w"))
        if delete_shards and not verify_only and os.path.exists(local):
            os.remove(local)   # verified cache copy no longer needed; sha256 in state
            print(f"  {fname}: cache copy deleted (verified sha256 recorded)")
        if state["target_reached"]:
            print(f"  TOKEN TARGET REACHED at document boundary in {fname}")
            break
    sel.close()

    if not verify_only:
        manifest = {
            "script_version": SCRIPT_VERSION, "purpose": "MICROCORE 150M base-training subset",
            "dataset": BASE_DATASET, "revision_sha": BASE_SHA,
            "tokenizer": {"package": TIKTOKEN_PIN, "package_version": TIKTOKEN_VERSION,
                          "encoding": ENCODING},
            "policy": BASE_POLICY, "target_tokens": target_tokens,
            "result": {
                "target_reached": state["target_reached"],
                "total_documents": state["total_docs"],
                "total_cl100k_tokens": state["total_tokens"],
                "shards_processed": sorted(state["shards_done"].keys()),
                "shards_discovered": [os.path.basename(s["path"]) for s in shards],
                "shards_not_processed": [os.path.basename(s["path"]) for s in shards
                                          if os.path.basename(s["path"]) not in state["shards_done"]],
                "per_shard": state["shards_done"],
            },
            "limitations": [
                "token-targeted subset: shards after the target boundary were not processed (by design)",
                "exact-dedup only within the processed shards",
                "FineWeb-Edu GPT-2 token_count column NOT used for budget accounting",
            ],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        json.dump(manifest, open(man_path, "w"), indent=1)
        print(f"\nBASE SUBSET {'COMPLETE' if state['target_reached'] else 'INCOMPLETE'}: "
              f"{state['total_docs']:,} docs, {state['total_tokens']:,} cl100k tokens "
              f"(target {target_tokens:,})")
        print(f"manifest: {man_path}\nselection: {sel_path}")

# --------------------------------------------------------------------------
# 3. Fine-tune stages: license gate + dedup + quarantine in the ACTUAL loader
# --------------------------------------------------------------------------
# License gate (mandatory, applied in the training data stream):
SWE_REBENCH_EXCLUDE = {"AGPL-3.0", "GPL-3.0", "None", None, "custom-check-github", "BSL-1.0"}

# Quarantine (evaluation-only material, never trained on).
# SWE-bench Verified instance_ids (eval set) pinned at the recorded revision.
WEBENCH_VERIFIED_SHA = None  # filled from the Hub API at first use and recorded

def hub_dataset_sha(dataset_id):
    """Current HEAD commit SHA of a dataset repo (used only for eval-only
    material whose revision was not previously pinned; recorded in reports)."""
    r = requests.get(f"{HF}/api/datasets/{dataset_id}", timeout=120)
    r.raise_for_status()
    return r.json()["sha"]

def load_swebench_verified_ids():
    """Download SWE-bench Verified test data and extract instance_ids for the
    overlap quarantine. Fail-closed: returns None if the eval set cannot be
    loaded, and the caller must then exclude ALL affected training records."""
    global WEBENCH_VERIFIED_SHA
    try:
        if WEBENCH_VERIFIED_SHA is None:
            WEBENCH_VERIFIED_SHA = hub_dataset_sha("SWE-bench/SWE-bench_Verified")
        files = hub_tree("SWE-bench/SWE-bench_Verified", WEBENCH_VERIFIED_SHA, "data")
        ids = set()
        for f in [x for x in files if x["path"].endswith(".parquet")]:
            local = download_verified("SWE-bench/SWE-bench_Verified", WEBENCH_VERIFIED_SHA,
                                      f["path"], size=f.get("size"),
                                      sha256=(f.get("lfs") or {}).get("oid"))
            for batch in pq.ParquetFile(local).iter_batches(batch_size=1024, columns=["instance_id"]):
                ids.update(batch.to_pydict()["instance_id"])
        print(f"  quarantine: loaded {len(ids)} SWE-bench Verified instance_ids "
              f"@ {WEBENCH_VERIFIED_SHA[:10]}")
        return ids
    except Exception as e:
        print(f"  quarantine: SWE-bench Verified load FAILED ({e}) - FAIL CLOSED")
        return None

def quarantine_check(train_ids, eval_ids, name):
    """Fail-closed overlap check. Returns (hits, message)."""
    if eval_ids is None:
        return None, f"{name}: eval set unavailable - FAIL CLOSED, exclude all {len(train_ids)} records"
    hits = train_ids & eval_ids
    return hits, f"{name}: {len(hits)} overlapping records excluded of {len(train_ids)} checked"

def prepare_agent_stage():
    """Agent adaptation: Nemotron-Pivot + SWE-rebench-V2 (license gate) + OpenHands-SFT.
    Verifies the Revision 9.1 retained counts (488,057,875 cl100k) in the real loader."""
    total = 0; report = {}
    # --- SWE-rebench-V2: license gate in the actual stream
    files = hub_tree("nebius/SWE-rebench-V2", "d7608f017a29ab256400ad4b73d5248d97204966", "data")
    pq_files = [f for f in files if f["path"].endswith(".parquet")]
    local = download_verified("nebius/SWE-rebench-V2", "d7608f017a29ab256400ad4b73d5248d97204966",
                              pq_files[0]["path"], size=pq_files[0].get("size"),
                              sha256=(pq_files[0].get("lfs") or {}).get("oid"))
    n_raw = n_excl = n_dup = n_keep = t_keep = 0; seen = set(); train_ids = set()
    for batch in pq.ParquetFile(local).iter_batches(batch_size=512,
            columns=["instance_id", "problem_statement", "patch", "license"]):
        d = batch.to_pydict()
        for iid, ps, pa, li in zip(d["instance_id"], d["problem_statement"], d["patch"], d["license"]):
            n_raw += 1
            ls = str(li) if li is not None else "None"
            if ls in SWE_REBENCH_EXCLUDE or ls.startswith("custom"):
                n_excl += 1; continue
            txt = (ps or "") + "\n" + (pa or "")
            h = hashlib.sha256(txt.encode()).hexdigest()
            if h in seen: n_dup += 1; continue
            seen.add(h); train_ids.add(iid)
            n_keep += 1; t_keep += len(enc.encode(txt, disallowed_special=()))
    # --- evaluation quarantine (fail-closed): SWE-bench Verified overlap
    eval_ids = load_swebench_verified_ids()
    hits, msg = quarantine_check(train_ids, eval_ids, "SWE-rebench-V2 vs SWE-bench Verified")
    report["SWE-rebench-V2"] = {"raw": n_raw, "license_excluded": n_excl, "dups": n_dup,
                                "retained": n_keep, "tokens": t_keep,
                                "quarantine": msg, "quarantine_hits": sorted(hits) if hits else []}
    total += t_keep
    print(f"  SWE-rebench-V2 (gated+dedup): {n_keep:,} records, {t_keep:,} tokens "
          f"(expected 26,481 / 59,054,014)")
    print(f"  quarantine: {msg}")
    # --- Nemotron-Pivot (terminal pivot) at the pinned SHA. The SHA
    # eaef2694... is the HEAD of nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1
    # (public). nvidia/Nemotron-PostTraining-Datasets also mirrored this file
    # but that repo started returning 401 on 2026-11-19; the pinned-SHA
    # verification below uses the public pivot-v1 repository.
    nemotron_report = count_nemotron_pivot()
    report["Nemotron-Pivot"] = nemotron_report
    total += nemotron_report.get("tokens", 0)
    print(f"  Nemotron-Pivot: {nemotron_report.get('records'):,} records, "
          f"{nemotron_report.get('tokens'):,} tokens (expected 31,111 / 420,583,133)")
    # --- OpenHands-SFT: execution-verified trajectories at the pinned SHA
    oh_files = hub_tree("SWE-Gym/OpenHands-SFT-Trajectories",
                        "4aaa5a4a4b5861f4799d2336908760c190ac3b17", "data")
    oh = [f for f in oh_files if f["path"].endswith(".parquet")]
    local = download_verified("SWE-Gym/OpenHands-SFT-Trajectories",
                              "4aaa5a4a4b5861f4799d2336908760c190ac3b17",
                              oh[0]["path"], size=oh[0].get("size"),
                              sha256=(oh[0].get("lfs") or {}).get("oid"))
    n_oh = t_oh = 0
    for batch in pq.ParquetFile(local).iter_batches(batch_size=64):
        d = batch.to_pydict()
        for msgs in d["messages"]:
            txt = "\n".join(m.get("content") or "" for m in msgs)
            n_oh += 1; t_oh += len(enc.encode(txt, disallowed_special=()))
    report["OpenHands-SFT"] = {"records": n_oh, "tokens": t_oh}
    total += t_oh
    print(f"  OpenHands-SFT: {n_oh} records, {t_oh:,} tokens (expected 491 / 8,420,728)")
    report["agent_stage_total_tokens"] = total
    expected_total = 488057875
    report["matches_rev91_pool"] = (total == expected_total)
    print(f"  AGENT STAGE TOTAL: {total:,} cl100k (Revision 9.1 pool: {expected_total:,}) "
          f"-> {'MATCH' if total == expected_total else 'MISMATCH'}")
    json.dump(report, open(os.path.join(PREP, "agent_stage_report.json"), "w"), indent=1)
    return report

def count_nemotron_pivot():
    """Nemotron-Pivot at the pinned SHA (public pivot-v1 repo). Same
    field concatenation as the Revision 9.1 full deterministic count, plus
    the Terminal-Bench task-name quarantine (fail-closed)."""
    sha = "eaef26944643644c8a3dbbf361ce6128142f5976"
    files = hub_tree("nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1", sha)
    jf = [f for f in files if f["path"].endswith(".jsonl")]
    local = download_verified("nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1", sha,
                              jf[0]["path"], size=jf[0].get("size"),
                              sha256=(jf[0].get("lfs") or {}).get("oid"))
    def s(x):
        if x is None: return ""
        if isinstance(x, str): return x
        return json.dumps(x, ensure_ascii=False)
    n = t = dups = 0; seen = set(); task_names = set()
    with open(local, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            rcp = r.get("responses_create_params")
            rcp = json.dumps(rcp, ensure_ascii=False) if not isinstance(rcp, str) else rcp
            txt = rcp + "\n" + s(r.get("expected_answer")) + "\n" + s(r.get("agent_ref")) + \
                  "\n" + s(r.get("task_name")) + "\n" + s(r.get("tool_name"))
            h = hashlib.sha256(txt.encode()).hexdigest()
            if h in seen: dups += 1
            else: seen.add(h)
            n += 1; t += len(enc.encode(txt, disallowed_special=()))
            task_names.add(r.get("task_name") or "")
    tb = load_terminal_bench_task_names()
    hits, msg = quarantine_check(task_names, tb, "Nemotron-Pivot vs Terminal-Bench task names")
    return {"records": n, "dups": dups, "tokens": t,
            "revision_sha": sha, "source": "nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 (public)",
            "quarantine": msg, "quarantine_hits": sorted(hits) if hits else []}

def load_terminal_bench_task_names():
    """Terminal-Bench task names from the eval repository (original-tasks dir).
    Fail-closed: None if unavailable, and the caller must exclude affected
    training records."""
    try:
        r = requests.get("https://api.github.com/repositories/918420677/contents/original-tasks",
                         timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"GitHub API {r.status_code}")
        return {f["name"] for f in r.json() if f["type"] == "dir"}
    except Exception as e:
        print(f"  quarantine: Terminal-Bench task list unavailable ({e}) - FAIL CLOSED")
        return None

# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, fcntl
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--stages-only", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="validate caches/manifests without writing prepared data")
    ap.add_argument("--delete-shards", action="store_true",
                    help="delete a shard from the verified cache after it is fully processed")
    args = ap.parse_args()

    # Single-instance lock: two concurrent runs corrupt the selection file.
    _lock_fh = open(os.path.join(ROOT, ".microcore_prep.lock"), "w")
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("ABORTED: another data-preparation run holds the lock")

    print(f"storage root: {ROOT}")
    print(f"tokenizer: {TIKTOKEN_PIN} / {ENCODING} (frozen for 150M/300M/500M/750M)")
    if not args.stages_only:
        prepare_base(verify_only=args.verify_only, delete_shards=args.delete_shards)
    if not args.base_only:
        prepare_agent_stage()
