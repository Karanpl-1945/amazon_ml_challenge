"""Fast, memory-lean TSV loading with pyarrow."""
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

STR_COLS = ["entity_id", "business_name", "business_address", "country",
            "source1_entity_id", "matched_entity_ids"]


def read_tsv_table(path) -> pa.Table:
    """Read a challenge TSV (tab separated, no quoting) as an Arrow table; empty fields stay ""."""
    return pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in STR_COLS},
                                             strings_can_be_null=False),
    )


def load_tsv(path) -> pd.DataFrame:
    """Arrow-backed DataFrame (~1x file size in RAM)."""
    return read_tsv_table(path).to_pandas(types_mapper=pd.ArrowDtype)


def load_gt_pairs(path):
    """Ground truth -> list of (s1_id, matched_id) tuples plus the list of singleton s1 ids."""
    gt = read_tsv_table(path).to_pydict()
    pairs, singletons = [], []
    for s1, ms in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        ids = [m for m in ms.split(",") if m]
        if not ids:
            singletons.append(s1)
        pairs.extend((s1, m) for m in ids)
    return pairs, singletons
