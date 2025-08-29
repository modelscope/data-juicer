from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.example_ir import ExampleIR, make_asset
from utils.parse_class import literal_eval_with_self

# Field aliases mapping for data normalization
FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "text": ("text",),
    "images": ("images",),
    "videos": ("videos", "video"),
    "audios": ("audios", "audio"),
    "answer": ("answer",),
}

# Reverse mapping from alias to canonical field name
ALIAS_TO_CANON: Dict[str, str] = {alias: canon for canon, aliases in FIELD_ALIASES.items() for alias in aliases}


def parse_literal_or_none(raw: str, attr_map: Dict[str, str]) -> Any:
    """Parse string literal using attribute mapping, return None on failure."""
    if not raw:
        return None
    try:
        return literal_eval_with_self(raw, attr_map)
    except Exception:
        return None


def _as_list_str(val: Any) -> List[str]:
    """Convert value to list of strings, handling str/list/tuple inputs."""
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple)) and all(isinstance(x, str) for x in val):
        return list(val)
    return []


def _normalize_sample_dict(item: Dict[str, Any], md_dir: Path) -> Dict[str, Any]:
    """
    Normalize raw sample dictionary to standardized format.

    Maps known fields to canonical names, converts media paths to assets,
    and stores unrecognized fields in 'meta' section.
    """
    sample: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}

    for k, v in item.items():
        canon = ALIAS_TO_CANON.get(k)
        if canon is None:
            # Unknown field -> store in meta
            meta[k] = v
            continue
        if canon in ("text", "answer"):
            if isinstance(v, str):
                sample[canon] = v
            else:
                # Convert to string as fallback
                sample[canon] = str(v)
        elif canon in ("images", "videos", "audios"):
            # Handle media fields - convert paths to assets
            kind = canon[:-1] if canon.endswith("s") else canon  # image/video/audio
            paths = _as_list_str(v)
            if paths:
                sample[canon] = [make_asset(p, md_dir, kind=kind) for p in paths]
            else:
                # Invalid media value -> store in meta
                meta[k] = v
        else:
            meta[k] = v

    if meta:
        sample["meta"] = meta
    return sample


def _to_samples(raw: Any, md_dir: Path) -> List[Dict[str, Any]]:
    """Convert raw data to normalized sample list."""
    samples: List[Dict[str, Any]] = []
    if isinstance(raw, list) and raw and all(isinstance(x, dict) for x in raw):
        # Handle list of dictionaries
        for item in raw:
            norm = _normalize_sample_dict(item, md_dir)
            samples.append(norm if norm else {"meta": item})
        return samples
    if isinstance(raw, dict):
        # Handle single dictionary
        norm = _normalize_sample_dict(raw, md_dir)
        return [norm if norm else {"meta": raw}]
    if isinstance(raw, list) and raw:
        return [{"list": str(raw)}]
    # Fallback: convert to string
    return [{"text": str(raw)}]


def samples_to_ds_tgt(obj: Any) -> Tuple[List[Any], List[Any]]:
    """
    Split samples with 'target' field into dataset and target lists.

    Extracts paired data where each item has exactly 2 keys: one data key and 'target'.
    """
    ds_obj = []
    tgt_obj = []
    if isinstance(obj, list) and all(isinstance(it, dict) for it in obj) and all("target" in it for it in obj):
        for it in obj:
            if len(it) != 2:
                continue
            other_key = (it.keys() - {"target"}).pop()
            ds_obj.append({other_key: it[other_key]})
            tgt_obj.append({other_key: it["target"]})
        return ds_obj, tgt_obj
    return None, None


def route(
    vals: Dict[str, str],
    attr_map: Dict[str, str],
    md_dir: Path,
    method: str,
) -> Optional[ExampleIR]:
    """
    Route parsed values to ExampleIR structure.

    Processes 'ds', 'tgt', and 'samples' fields from vals dictionary,
    normalizes them to sample format, and creates ExampleIR instance.
    Returns None if insufficient data is provided.
    """
    # Parse raw string values using attribute mapping
    ds_raw = vals.get("ds") or ""
    tgt_raw = vals.get("tgt") or ""
    samples_raw = vals.get("samples") or ""
    ds_obj = parse_literal_or_none(ds_raw, attr_map)
    tgt_obj = parse_literal_or_none(tgt_raw, attr_map)
    samples_obj = parse_literal_or_none(samples_raw, attr_map)

    # Validate input data availability
    if (ds_obj is None or tgt_obj is None) and samples_obj is None:
        return None
    elif (ds_obj is None or tgt_obj is None) and samples_obj is not None:
        # Extract ds/tgt from samples if direct ds/tgt not available
        ds_obj, tgt_obj = samples_to_ds_tgt(samples_obj)

    if ds_obj is None or tgt_obj is None:
        return None
    # Convert to normalized sample format
    ds_samples = _to_samples(ds_obj, md_dir)
    tgt_samples = _to_samples(tgt_obj, md_dir)

    def non_empty(s: Dict[str, Any]) -> bool:
        """Check if sample has content beyond meta field."""
        return any(k for k in s.keys() if k != "meta")

    # Filter out empty samples (keep if has content or meta)
    ds_samples = [s for s in ds_samples if non_empty(s) or s.get("meta")]
    tgt_samples = [s for s in tgt_samples if non_empty(s) or s.get("meta")]

    if not ds_samples and not tgt_samples:
        return None

    # Create ExampleIR instance with processed data
    return ExampleIR(
        method=method,
        op_code=vals.get("op_code") or "",
        input={"samples": ds_samples},
        output={"samples": tgt_samples},
    )
