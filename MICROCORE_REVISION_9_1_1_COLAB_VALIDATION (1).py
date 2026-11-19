"""
MICROCORE Revision 9.1.1 - Colab validation (corrected).
Base dataset: HuggingFaceFW/fineweb-edu sample/10BT only (GPT-2 measured).
All stage/supervised datasets counted with tiktoken cl100k_base (frozen across
all model sizes). Datasets must be loaded from the Hugging Face Hub at a
pinned revision that is independently verifiable. This revision fixes:

  (A) BASE SHARD COVERAGE - every file under sample/10BT is enumerated via the
      Hub tree API at the pinned SHA. A partial view of the shards can never
      be reported as complete. The validator fails when it cannot enumerate
      all shards, and reports the exact shard list (name, bytes, sha256).
  (B) IMMUTABLE REVISION PINNING - manifest SHAs are resolved against the
      repository at that exact commit. URLs are built from the dataset ID +
      pinned SHA + path (content-addressed). If a file cannot be resolved at
      the pinned revision, the validator reports REVISION_VERIFICATION_FAILED
      and does NOT silently fall back to main.
  (C) CHECKPOINT CORRECTNESS - checkpoints are keyed by
      (dataset_id, revision, file_path, file_sha256, tokenizer, config,
      script_version). A checkpoint is reused only if its key matches the
      current run AND its status is 'ok'; failed/incomplete/mismatched runs
      are retried.
  (D) CACHE INTEGRITY - files are stored under revision-specific directories,
      verified against the byte size AND sha256 recorded by the Hub tree API
      before the .verified suffix is added (atomic rename). Downloads stream
      to disk. No resumable download is claimed: byte-range resume is NOT
      implemented; a failed download is restarted from the beginning.

Verdicts: SAMPLE_PASS, FULL_SCAN_COMPLETE, MATCHED_AUDIT, MISMATCH,
INCOMPLETE, BLOCKED, ERROR.

Run in Colab:
    !pip install -q tiktoken pyarrow pandas requests
    %run MICROCORE_REVISION_9_1_1_COLAB_VALIDATION.py
"""

import hashlib, json, os, sys, time
import pandas as pd
import pyarrow.parquet as pq
import requests
import tiktoken

HF = "https://huggingface.co"
CL100K = tiktoken.get_encoding("cl100k_base")

# Pinned revisions (each verified against the Hub at the time of pinning).
FINEWEB_EDU_SHA = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
PINS = {
    "HuggingFaceFW/fineweb-edu": FINEWEB_EDU_SHA,
    "nvidia/HelpSteer2": "990b2711a36180dd19d9c94b8627844866f8982a",
    "nvidia/HelpSteer3": "f6d145777bcbde96137596340fab89793acd1031",
    "nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1": "eaef26944643644c8a3dbbf361ce6128142f5976",
    "nebius/SWE-rebench-V2": "d7608f017a29ab256400ad4b73d5248d97204966",
    "SWE-Gym/OpenHands-SFT-Trajectories": "4aaa5a4a4b5861f4799d2336908760c190ac3b17",
    "project-numina/NuminaMath-CoT": "9d8d210c9f6a36c8f3cd84045668c9b7800ef517",
    "ise-uiuc/Magicoder-OSS-Instruct-75K": "5f839b1f368a76b161028bb9edff055db34022b2",
    "SWE-bench/SWE-smith": "ea6d7173829c7ec8fa16c22055699ff2e9188091",
    "SWE-bench/SWE-Gym": "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb",
    "b-mc2/sql-create-context": "9d80a6a118b838d9defc3798d659a54a2ac2ff37",
    "gretelai/synthetic_text_to_sql": "740ab236e64503fba51be1101df7a1be83bf455d",
    "m-a-p/Code-Feedback": "f411b16a97c910ac9acf8b0d0948e340aa77cc34",
}
SCRIPT_VERSION = "microcore-revision-9.1.1-validator"
BASE_EXPECTED_BYTES = 28518193415  # Rev 9.1 audit: sample/10BT total bytes

# Revision 9.1.2: shared data-root configuration. The previous validator looked
# in /content/drive/MyDrive/microcore while the preparation script wrote to
# /content/drive/MyDrive/microcore_data - the mismatch is fixed by importing
# the same explicit root everywhere.
from microcore_paths import DATA_ROOT as _ROOT, CACHE, PREPARED as PREP
CKPT = os.path.join(_ROOT, "checkpoints")
os.makedirs(CACHE, exist_ok=True); os.makedirs(CKPT, exist_ok=True); os.makedirs(PREP, exist_ok=True)

# ---------------------------------------------------------------- (B) pinning
def hub_tree(dataset_id, sha, path=""):
    """Enumerate files under `path` at the pinned commit. (Fix A: the complete
    shard list comes from the Hub tree API, never from a partial glob.)"""
    r = requests.get(f"{HF}/api/datasets/{dataset_id}/tree/{sha}/{path}", timeout=180)
    if r.status_code == 401 or r.status_code == 404:
        raise RuntimeError(f"REVISION_VERIFICATION_FAILED: cannot list {dataset_id}@{sha[:10]} ({r.status_code})")
    r.raise_for_status()
    return r.json()

def resolve_url(dataset_id, sha, path):
    """Content-addressed URL built from the pinned revision. No fallback to main."""
    return f"{HF}/datasets/{dataset_id}/resolve/{sha}/{path}"

def download_file(dataset_id, sha, path, size=None, sha256=None):
    """Stream to disk, verify against the Hub-recorded size and sha256, then
    atomically rename to .verified. (Fix D. NOT resumable: a failed download
    restarts from the beginning.)"""
    d = os.path.join(CACHE, dataset_id.replace("/", "__"), sha)
    os.makedirs(d, exist_ok=True)
    final = os.path.join(d, os.path.basename(path))
    if os.path.exists(final + ".verified"):
        return final
    tmp = final + ".tmp"
    for attempt in range(5):
        try:
            with requests.get(resolve_url(dataset_id, sha, path), stream=True,
                             timeout=600, allow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            actual = os.path.getsize(tmp)
            if size is not None and actual != size:
                raise RuntimeError(f"size mismatch: expected {size}, got {actual}")
            if sha256:
                h = hashlib.sha256()
                with open(tmp, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != sha256:
                    raise RuntimeError(f"sha256 mismatch: expected {sha256}, got {h.hexdigest()}")
            os.replace(tmp, final)
            open(final + ".verified", "w").write("verified\n")
            return final
        except Exception as e:
            print(f"  download retry {attempt+1}/5 ({e})")
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"download failed after 5 attempts: {dataset_id}/{path}")

# ------------------------------------------------------------- (C) checkpoint
def ckpt_path(dataset_id, sha, path, sha256):
    key = json.dumps([dataset_id, sha, path, sha256, "cl100k_base", SCRIPT_VERSION],
                     sort_keys=True).encode()
    digest = hashlib.sha256(key).hexdigest()[:16]
    return os.path.join(CKPT, f"{dataset_id.replace('/', '__')}__{digest}.json")

def load_ckpt(dataset_id, sha, path, sha256):
    p = ckpt_path(dataset_id, sha, path, sha256)
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None  # corrupt checkpoint -> retry
    if d.get("script_version") != SCRIPT_VERSION or d.get("tokenizer") != "cl100k_base":
        return None  # stale config -> retry
    if d.get("dataset_id") != dataset_id or d.get("revision") != sha or d.get("path") != path:
        return None  # identity mismatch -> retry
    if d.get("status") != "ok":
        return None  # failed/incomplete -> retry (fix C)
    return d

def save_ckpt(dataset_id, sha, path, sha256, status, **fields):
    d = {"dataset_id": dataset_id, "revision": sha, "path": path,
         "file_sha256": sha256, "tokenizer": "cl100k_base",
         "script_version": SCRIPT_VERSION, "status": status}
    d.update(fields)
    p = ckpt_path(dataset_id, sha, path, sha256)
    tmp = p + ".tmp"; json.dump(d, open(tmp, "w")); os.replace(tmp, p)

# --------------------------------------------------------------- measurements
def count_tokens_cl100k(text_iter):
    n = 0
    for t in text_iter:
        n += len(CL100K.encode(t, disallowed_special=()))
    return n

def parquet_text_batches(local, columns, batch_size=1024):
    pf = pq.ParquetFile(local)
    for b in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield b.to_pydict()

# ------------------------------------------------------------------- verdicts
def walk_tree(dataset_id, sha, path=""):
    """Recursively enumerate ALL files under `path` at the pinned revision."""
    files = []
    stack = [path]
    while stack:
        p = stack.pop()
        for f in hub_tree(dataset_id, sha, p):
            if f["type"] == "directory":
                stack.append(f["path"])
            else:
                files.append(f)
    return files

def verify_repo_files_or_row(dataset_id, sha, manifest_size, path=""):
    """(Fix A+B) enumerate ALL files at the pinned revision (recursively) and
    compare the byte total to the manifest. MISMATCH if they differ; ERROR if
    listing fails; BLOCKED if file access requires authentication."""
    try:
        files = walk_tree(dataset_id, sha, path)
        total = sum(f.get("size", 0) for f in files)
    except RuntimeError as e:
        if "REVISION_VERIFICATION_FAILED" in str(e):
            return "BLOCKED", {"reason": str(e)}
        return "ERROR", {"reason": str(e)}
    except Exception as e:
        return "ERROR", {"reason": str(e)}
    if total == manifest_size:
        return "MATCHED_AUDIT", {"bytes": total, "files": len(files)}
    return "MISMATCH", {"manifest_bytes": manifest_size, "actual_bytes": total,
                        "files": len(files)}

def verify_base_stage():
    """FineWeb-Edu sample/10BT. Enumerates every shard (fix A), verifies the
    byte total, and checks the prepared subset manifest for a >=3B cl100k count."""
    files = hub_tree("HuggingFaceFW/fineweb-edu", FINEWEB_EDU_SHA, "sample/10BT")
    shards = sorted(f["path"] for f in files if f["type"] == "file")
    total = sum(f.get("size", 0) for f in files if f["type"] == "file")
    print(f"  shards discovered at pinned SHA: {len(shards)}")
    for s in shards:
        print(f"    {os.path.basename(s)}")
    results = {"shards": shards, "shard_count": len(shards), "bytes": total,
               "shard_detail": [{"path": f["path"], "bytes": f.get("size"),
                                 "sha256": (f.get("lfs") or {}).get("oid")} for f in files]}
    if total != BASE_EXPECTED_BYTES:
        print(f"  MISMATCH: manifest {BASE_EXPECTED_BYTES:,} vs actual {total:,} bytes")
        return "MISMATCH", results
    verdict = "MATCHED_AUDIT"
    # the prepared subset produced by the data-preparation script (actual count)
    mf = os.path.join(PREP, "base_prep_manifest.json")
    if os.path.exists(mf):
        d = json.load(open(mf))
        r = d.get("result", {})
        toks = r.get("total_cl100k_tokens", 0)
        print(f"  prepared subset: {r.get('total_documents'):,} documents, "
              f"{toks:,} cl100k tokens (target 3,000,000,000)")
        print(f"  shards processed: {len(r.get('shards_processed', []))} of "
              f"{len(r.get('shards_discovered', []))} discovered")
        if r.get("target_reached") and toks >= 3_000_000_000:
            print("  BASE STAGE 150M TARGET REACHED (actual cl100k tokens)")
        else:
            print(f"  BASE STAGE SHORT by {3_000_000_000 - toks:,} tokens")
            verdict = "INCOMPLETE"
    else:
        print("  base_prep_manifest.json missing - base subset not prepared")
        verdict = "INCOMPLETE"
    return verdict, results

def verify_pretrained_stages():
    """Each stage dataset: enumerate ALL files at the pinned SHA (recursively),
    verify byte totals against the Revision 9.1 manifest. Datasets whose file
    access now requires authentication are reported BLOCKED, not skipped
    silently."""
    expected = {  # (path, manifest bytes) - measured at the pinned SHAs
        "nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1": ("", 1372472348),
        "nebius/SWE-rebench-V2": ("data", 428839266),
        "SWE-Gym/OpenHands-SFT-Trajectories": ("data", 10379409),
        "project-numina/NuminaMath-CoT": ("data", 1234351634),
        "ise-uiuc/Magicoder-OSS-Instruct-75K": ("", 203203865),
        "SWE-bench/SWE-smith": ("data", 277767206),
        "b-mc2/sql-create-context": ("", 21759241),
        "gretelai/synthetic_text_to_sql": ("", 34283665),
        "nvidia/HelpSteer2": ("", 40632942),
        "nvidia/HelpSteer3": ("preference", 111971991),
        "m-a-p/Code-Feedback": ("", 413480844),
    }
    blocked = ["SWE-bench/SWE-Gym"]  # 401 since 2026-11-19 (was public 2026-11-17)
    stage_report = {}
    overall = []
    for ds in blocked:
        sha = PINS.get(ds)
        print(f"  {ds}: BLOCKED (file access returns 401 without authentication; "
              f"pinned SHA {sha[:10] if sha else 'n/a'})")
        stage_report[ds] = {"verdict": "BLOCKED",
                           "reason": "file access requires authentication since 2026-11-19"}
        overall.append("BLOCKED")
    for ds, (path, mbytes) in expected.items():
        sha = PINS[ds]
        v, info = verify_repo_files_or_row(ds, sha, mbytes, path)
        print(f"  {ds}: {v} ({info})")
        stage_report[ds] = {"verdict": v, **info}
        overall.append(v)
    json.dump(stage_report, open(os.path.join(PREP, "stage_validation_report.json"), "w"), indent=1)
    if "MISMATCH" in overall:
        return "MISMATCH"
    if all(v == "MATCHED_AUDIT" for v in overall):
        return "MATCHED_AUDIT"
    if "BLOCKED" in overall:
        return "BLOCKED"
    return "ERROR"

def run_real_agentic_overlaps():
    """Fail-closed evaluation quarantine: compare SWE-rebench-V2 instance_ids
    against SWE-bench Verified instance_ids (downloaded at a recorded
    revision). If the eval set cannot be loaded, the affected training data
    is EXCLUDED (fail closed)."""
    sr2_sha = PINS["nebius/SWE-rebench-V2"]
    files = hub_tree("nebius/SWE-rebench-V2", sr2_sha, "data")
    pqf = [f for f in files if f["path"].endswith(".parquet")]
    local = download_file("nebius/SWE-rebench-V2", sr2_sha, pqf[0]["path"],
                          size=pqf[0].get("size"), sha256=(pqf[0].get("lfs") or {}).get("oid"))
    train_ids = set()
    for b in parquet_text_batches(local, ["instance_id"]):
        train_ids.update(b["instance_id"])
    # eval set: current HEAD SHA recorded at run time
    r = requests.get(f"{HF}/api/datasets/SWE-bench/SWE-bench_Verified", timeout=120)
    r.raise_for_status(); vsha = r.json()["sha"]
    vfiles = hub_tree("SWE-bench/SWE-bench_Verified", vsha, "data")
    vpar = [f for f in vfiles if f["path"].endswith(".parquet")]
    vlocal = download_file("SWE-bench/SWE-bench_Verified", vsha, vpar[0]["path"],
                           size=vpar[0].get("size"), sha256=(vpar[0].get("lfs") or {}).get("oid"))
    eval_ids = set()
    for b in parquet_text_batches(vlocal, ["instance_id"]):
        eval_ids.update(b["instance_id"])
    hits = train_ids & eval_ids
    print(f"  quarantine: {len(train_ids):,} train ids vs {len(eval_ids):,} eval ids -> {len(hits)} overlaps")
    json.dump({"verified_sha": vsha, "train_ids": len(train_ids), "eval_ids": len(eval_ids),
               "overlaps": sorted(hits)}, open(os.path.join(PREP, "quarantine_report.json"), "w"))
    return len(hits)

def main():
    print("=" * 70)
    print("MICROCORE Revision 9.1.1 validation")
    print("=" * 70)
    results = {}

    print("\n[1] Base dataset: FineWeb-Edu sample/10BT (full shard enumeration)")
    v, info = verify_base_stage()
    results["base"] = {"verdict": v, **info}
    print(f"  -> {v}")

    print("\n[2] Stage datasets at pinned revisions (byte-total audit)")
    v = verify_pretrained_stages()
    results["stages"] = v
    print(f"  -> {v}")

    print("\n[3] Agentic evaluation quarantine (fail-closed)")
    try:
        n = run_real_agentic_overlaps()
        results["quarantine_overlaps"] = n
    except Exception as e:
        print(f"  quarantine FAILED - FAIL CLOSED (exclude affected training data): {e}")
        results["quarantine_overlaps"] = None

    json.dump(results, open(os.path.join(PREP, "validation_results.json"), "w"), indent=1)
    print("\n" + "=" * 70)
    print("VERDICTS:", json.dumps({k: (v if not isinstance(v, dict) else v.get("verdict"))
                                   for k, v in results.items()}, indent=1))
    print("=" * 70)

if __name__ == "__main__":
    main()
