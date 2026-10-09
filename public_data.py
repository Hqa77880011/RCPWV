import json
from pathlib import Path

import numpy as np

from evaluate import project_spec

PROFILES = ("focus", "defocus")


def load_public(prefix: Path) -> tuple[list, np.ndarray, np.ndarray]:

    records = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as archive:
        packed = archive["masks_packed"]
        references = archive["aerial_grid"]
    if packed.shape != (len(records), 2048, 256):
        raise ValueError("Public masks_packed must have shape N,2048,256")
    if references.shape != (len(records), 2, 128, 128):
        raise ValueError("Public aerial_grid must have shape N,2,128,128")
    if not np.isfinite(references).all():
        raise ValueError("Public reference grids contain non-finite values")
    for record in records:
        if not record.get("source_group"):
            raise ValueError("Public records must declare source_group")
        specs = record["all_specs"]
        if not specs:
            raise ValueError(f"No fixed input-only specs for {record['id']}")
        for spec in specs:
            project_spec(spec, 256)
        for profile in PROFILES:
            if len(record[f"all_truth_{profile}"]) != len(specs):
                raise ValueError("Stored public truth must align with all_specs")
    return records, np.unpackbits(packed, axis=-1), references
