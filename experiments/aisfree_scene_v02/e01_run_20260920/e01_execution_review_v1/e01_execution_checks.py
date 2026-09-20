#!/usr/bin/env python3
"""Read-only E01 output checks and tested integration helpers.

Does NOT modify repository scripts, train models, read labels from target files,
or repair an already-truncated identifier. Helpers must be integrated explicitly.
Only self-tests use synthetic labels. CLI reads identifier metadata only.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence
import numpy as np


def strings_exact(values: Sequence[str], *, unique: bool = False) -> np.ndarray:
    """Create inferred-width Unicode, never pickle/object dtype or hardcoded U64."""
    original = [str(v) for v in values]
    if any(not x for x in original):
        raise ValueError("Empty identifiers are not allowed")
    result = np.asarray(original, dtype=str)
    if result.tolist() != original:
        raise ValueError("Identifier round-trip failed")
    if unique and len(set(original)) != len(original):
        raise ValueError("Duplicate identifiers")
    return result


def strict_index(values: Sequence[str]) -> dict[str, int]:
    vals = strings_exact(values, unique=True)
    return {s: i for i, s in enumerate(vals.tolist())}


def product_to_port(manifest: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for port, parts in manifest["ports"].items():
        for role in ("adapt", "eval"):
            for prod in parts[role]:
                if prod in result:
                    raise ValueError(f"Repeated product in manifest: {prod}")
                result[prod] = port
    return result


def calibration_products(manifest: dict, fold_name: str) -> dict[str, list[str]]:
    """Use product registry, not a sample_id-keyed metadata dict."""
    fold = manifest["folds"][fold_name]
    mapping = product_to_port(manifest)
    expected_ports = set(fold["source_calibration_ports"])
    output = {p: [] for p in sorted(expected_ports)}
    for prod in fold["source_calibration_adapt_products"]:
        if prod not in mapping or mapping[prod] not in expected_ports:
            raise ValueError(f"Bad calibration assignment: {prod}")
        output[mapping[prod]].append(prod)
    if any(not products for products in output.values()):
        raise ValueError("Calibration port has no adapt products")
    return output


def port_class_probabilities(ports: Sequence[str], labels: Sequence[int]) -> np.ndarray:
    """P(port)=1/P; P(class|port)=1/Cp; P(instance|port,class)=1/Npc.

    Inputs must be source-fit, strict-fine, image-eligible supervised rows only.
    A stochastic draw has these EXPECTED proportions, not exact counts per epoch.
    """
    p = [str(x) for x in ports]
    y = list(labels)
    if not p or len(p) != len(y):
        raise ValueError("ports and labels must be nonempty and equal length")
    groups = Counter(zip(p, y))
    class_sets: dict[str, set] = defaultdict(set)
    for pi, yi in zip(p, y):
        class_sets[pi].add(yi)
    out = np.asarray([1 / (len(class_sets) * len(class_sets[pi]) * groups[(pi, yi)])
                      for pi, yi in zip(p, y)], dtype=np.float64)
    if not np.isclose(out.sum(), 1):
        raise ValueError("Sampling probabilities do not sum to 1")
    return out


def balanced_epoch_draw(ports, labels, seed: int, epoch: int, count: int | None = None):
    """Preserve draw order for the optimizer; locality prefetch must scatter back."""
    probabilities = port_class_probabilities(ports, labels)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    n = len(probabilities) if count is None else int(count)
    if n < 1:
        raise ValueError("count must be positive")
    return rng.choice(len(probabilities), size=n, replace=True, p=probabilities)


def classification_metrics(y_true, y_pred) -> dict:
    """Evaluator-only; BA averages recalls over truth-present classes."""
    y, p = np.asarray(y_true), np.asarray(y_pred)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("Need nonempty equal 1D label arrays")
    if y.dtype.kind == 'f' and not np.isfinite(y).all():
        raise ValueError("Nonfinite truth")
    per_class = []
    for cls in np.unique(y):
        keep = y == cls
        per_class.append({"class": str(cls), "n": int(keep.sum()),
                          "recall": float(np.mean(p[keep] == y[keep]))})
    return {"n": len(y), "accuracy": float(np.mean(y == p)),
            "balanced_accuracy": float(np.mean([r["recall"] for r in per_class])),
            "per_class": per_class}


def equal_port_ba(y_true, y_pred, ports) -> float:
    y, p, port = np.asarray(y_true), np.asarray(y_pred), np.asarray(ports)
    if y.ndim != 1 or p.shape != y.shape or port.shape != y.shape or len(y) == 0:
        raise ValueError("Need nonempty aligned truth, prediction, port arrays")
    return float(np.mean([classification_metrics(y[port == pi], p[port == pi])["balanced_accuracy"]
                          for pi in np.unique(port)]))


def augment_pair_reflect(chip: np.ndarray, seed: int, epoch: int, sample_index: int,
                         draw_index: int = 0) -> np.ndarray:
    """Same 180-degree rotation and reflect-padded translation for both channels.

    Caller must pass epoch/draw explicitly; persistent workers do not infer them.
    Use on an image-eligible chip. A validity mask needs the same spatial transform.
    """
    x = np.asarray(chip)
    if x.ndim != 3 or x.shape[0] != 2 or min(x.shape[1:]) <= 4:
        raise ValueError("Expected two-channel chip")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch),
                                                       int(sample_index), int(draw_index)]))
    if rng.random() < 0.5:
        x = x[:, ::-1, ::-1]
    dy, dx = (int(t) for t in rng.integers(-4, 5, size=2))
    h, w = x.shape[1:]
    pad = np.pad(x, ((0, 0), (4, 4), (4, 4)), mode="reflect")
    return np.ascontiguousarray(pad[:, 4-dy:4-dy+h, 4-dx:4-dx+w])


def centered_window(dataset, x: float, y: float, size: int = 128):
    """World x/y must already be in dataset CRS. Read mask/grid checks are separate."""
    from rasterio.windows import Window
    if size < 2 or size % 2:
        raise ValueError("size must be an even positive integer")
    row, col = dataset.index(x, y)
    return Window(col - size // 2, row - size // 2, size, size)


def audit_features(features: Path, relations: Path, manifest: dict, fold: str) -> dict:
    """Read only NPZ identifier members, not labels or full z/m arrays."""
    report = {"status": "pass", "errors": [], "warnings": [], "labels_read": False,
              "scope": "Identifier and manifest audit only; not crop/data/metric certification"}
    error = report["errors"].append
    mapping = product_to_port(manifest)
    with np.load(features, allow_pickle=False) as ff:
        required = ("sample_id", "product_id", "port")
        if not set(required).issubset(ff.files):
            return {**report, "status": "fail", "errors": ["Missing feature identifier arrays"]}
        arrays = {k: ff[k] for k in required}
        ids, prods, ports = (arrays[k].astype(str) for k in required)
        report["array_dtypes"] = {k: str(v.dtype) for k, v in arrays.items()}
        if not all(v.ndim == 1 for v in arrays.values()) or len(prods) != len(ids) or len(ports) != len(ids):
            return {**report, "status": "fail", "errors": ["Identifier arrays misaligned"]}
        report["feature_rows"] = len(ids)
        report["unique_sample_ids"] = len(set(ids.tolist()))
        report["max_sample_id_length"] = max(map(len, ids), default=0)
        report["min_max_product_id_length"] = [min(map(len, prods), default=0),
                                                max(map(len, prods), default=0)]
        if report["unique_sample_ids"] != len(ids):
            error("Duplicate feature sample_id; do not build dict joins")
        if len(ids) == 0:
            error("Empty features")
        absent = set(prods.tolist()) - set(mapping)
        if absent:
            error(f"{len(absent)} feature product IDs not in frozen manifest")
        wrong_port = sum(mapping.get(str(prod)) != str(port) for prod, port in zip(prods, ports))
        if wrong_port:
            error(f"{wrong_port} product/port mismatches")
        no_suffix = sum('|' not in str(sid) for sid in ids)
        if no_suffix:
            error(f"{no_suffix} sample IDs have no object suffix")
        prefix_bad = sum(str(sid).split('|', 1)[0] != str(prod) for sid, prod in zip(ids, prods))
        if prefix_bad:
            error(f"{prefix_bad} sample/product mismatches")
        report["image_eligibility_fields_present"] = [k for k in
              ("image_eligible", "valid_fraction_pair", "io_status") if k in ff.files]
        if not report["image_eligibility_fields_present"]:
            report["warnings"].append("No image-eligibility member; supply exact external ID-aligned I/O registry")
    with np.load(relations, allow_pickle=False) as rr:
        rid = rr['sample_id'].astype(str)
        if len(set(rid.tolist())) != len(rid):
            error("Duplicate relation sample IDs")
        miss = set(ids.tolist()) - set(rid.tolist())
        report["feature_ids_missing_in_relations"] = len(miss)
        if miss:
            error(f"{len(miss)} feature IDs not in relation index")
    f = manifest['folds'][fold]
    role_map = {"source_fit": "source_fit_products", "source_meta_query": "source_meta_query_products",
                "source_calibration": "source_calibration_products", "target_adapt": "target_adapt_products",
                "target_eval": "target_eval_products"}
    report['role_rows'] = {role: int(np.isin(prods, f[key]).sum()) for role, key in role_map.items()}
    for role, n in report['role_rows'].items():
        if n == 0:
            error(f"No rows found for {role}; check product IDs and joins")
    cal = calibration_products(manifest, fold)
    report['calibration_adapt_products_by_port'] = {p: len(v) for p, v in cal.items()}
    report['calibration_feature_rows_by_port'] = {p: int(np.isin(prods, v).sum()) for p, v in cal.items()}
    if any(n == 0 for n in report['calibration_feature_rows_by_port'].values()):
        error("At least one calibration port has zero feature rows")
    report['status'] = 'fail' if report['errors'] else 'pass'
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--relations', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--fold', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Refusing to overwrite an existing report')
    m = json.loads(a.manifest.read_text(encoding='utf-8'))
    report = audit_features(a.features, a.relations, m, a.fold)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x', encoding='utf-8') as h:
        json.dump(report, h, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['status'] == 'pass' else 2)

if __name__ == '__main__':
    main()
