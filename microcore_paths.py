"""
MICROCORE shared data-root configuration (Revision 9.1.2).

One explicit storage root used consistently by:
  - data preparation (MICROCORE_REVISION_9_1_1_DATA_PREPARATION.py)
  - dataset validation (MICROCORE_REVISION_9_1_1_COLAB_VALIDATION.py)
  - base-data reconstruction, token materialization, and training
    (MICROCORE_150M_TRAIN.py)
  - trainer checkpoints, training logs, model checkpoints, resume state

Colab root:   /content/drive/MyDrive/microcore_data
Local root:   ./microcore_data  (relative to this file's directory)

Revision 9.1.1 bug fixed here: the validator used
/content/drive/MyDrive/microcore while the preparation script wrote to
/content/drive/MyDrive/microcore_data. All components now import this module.
"""

import os

def _resolve_root():
    drive = "/content/drive/MyDrive/microcore_data"
    if os.path.isdir("/content/drive/MyDrive"):   # Colab with Drive mounted
        return drive
    # predictable local fallback: next to this file
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "microcore_data")

DATA_ROOT = _resolve_root()
CACHE      = os.path.join(DATA_ROOT, "cache")      # revision-specific shard caches
PREPARED   = os.path.join(DATA_ROOT, "prepared")    # selection manifests, token streams
PREP       = PREPARED                              # short alias used by scripts
TRAIN      = os.path.join(DATA_ROOT, "train_150m") # checkpoints, logs, run manifests

for _d in (DATA_ROOT, CACHE, PREPARED, TRAIN):
    os.makedirs(_d, exist_ok=True)

HF = "https://huggingface.co"
