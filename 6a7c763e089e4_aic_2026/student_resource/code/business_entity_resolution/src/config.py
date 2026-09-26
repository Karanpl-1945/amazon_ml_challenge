"""Central paths and constants for the entity-resolution pipeline.

Override locations with environment variables if the data lives elsewhere:
  ER_DATA_DIR  -> folder that contains train/ and test/  (default: <student_resource>/dataset)
  ER_WORK_DIR  -> folder for generated artifacts         (default: <student_resource>/work)
"""
import os
import zlib
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]            # code/business_entity_resolution
RESOURCE_DIR = PKG_DIR.parents[1]                        # student_resource

DATA_DIR = Path(os.environ.get("ER_DATA_DIR", RESOURCE_DIR / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", RESOURCE_DIR / "work"))
DICT_DIR = WORK_DIR / "dicts"
CLEAN_DIR = WORK_DIR / "clean"
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", RESOURCE_DIR / "output"))

TRAIN_FILES = {
    "s1": DATA_DIR / "train" / "train_source1.tsv",
    "s2": DATA_DIR / "train" / "train_source2.tsv",
    "s3": DATA_DIR / "train" / "train_source3.tsv",
}
TRAIN_GT = DATA_DIR / "train" / "train_ground_truth.tsv"
TEST_FILES = {
    "s1": DATA_DIR / "test" / "test_source1.tsv",
    "s2": DATA_DIR / "test" / "test_source2.tsv",
    "s3": DATA_DIR / "test" / "test_source3.tsv",
}

SEED = 42
VAL_FRACTION_MOD = 5          # 1 in 5 Source-1 entities -> validation fold


def is_val_entity(s1_id: str) -> bool:
    """Deterministic 80/20 split of Source-1 entities (stable across runs and machines).
    Dictionaries are mined on the train fold only so validation scores stay honest."""
    return zlib.crc32(s1_id.encode("utf-8")) % VAL_FRACTION_MOD == 0


for _d in (WORK_DIR, DICT_DIR, CLEAN_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def is_fit_entity(s1_id: str) -> bool:
    """Deterministic 20% sample of Source-1 entities used to build matcher training pairs."""
    return zlib.crc32(("fit:" + s1_id).encode("utf-8")) % 5 == 0
