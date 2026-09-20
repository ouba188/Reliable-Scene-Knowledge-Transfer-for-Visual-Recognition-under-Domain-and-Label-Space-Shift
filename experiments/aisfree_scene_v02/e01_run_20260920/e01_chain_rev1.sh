#!/bin/bash
# E01 fold chain REV1 — frozen revised implementation (2026-09-21).
# usage: bash e01_chain_rev1.sh Rotterdam
set -Eeuo pipefail
TARGET="${1:?usage: e01_chain_rev1.sh TARGET}"
PY=/root/miniconda3/bin/python
IN=/root/autodl-tmp/knowledge_841_20260920/e01_inputs
PKG=/root/autodl-tmp/knowledge_841_20260920/e01_first_batch
RUN=/root/autodl-tmp/e01_runs/${TARGET}_rev1

cd "$PKG"
echo "== $TARGET REV1 开始 ($(date '+%H:%M:%S')) =="
echo "code: chips=$(sha256sum "$IN/e01_chips_rev1.py" | cut -c1-12) features=$(sha256sum "$IN/e01_features_rev1.py" | cut -c1-12) glue=$(sha256sum "$IN/e01_glue_rev1.py" | cut -c1-12) core=$(sha256sum tools/e01_core.py | cut -c1-12)"

if [ ! -f "$RUN/features.done" ]; then
  echo "-- encoder 训练 + 特征导出 --"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY "$IN/e01_features_rev1.py" --fold "$TARGET" --epochs 30 --workers 4 \
    > /root/e01_train_${TARGET}_rev1.log 2>&1
  tail -4 /root/e01_train_${TARGET}_rev1.log
fi
[ -f "$RUN/features.done" ] && echo "features.done: $(cat "$RUN/features.done")"

echo "== 1) fit_groups =="
rm -rf "$RUN/groups"; $PY tools/fit_groups.py --features "$RUN/features.npz" --manifest "$PKG/split_manifest.json" \
  --target "$TARGET" --out "$RUN/groups" 2>&1 | tail -3

echo "== 2) fit_cached_heads (H=8) =="
rm -rf "$RUN/heads"; CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY tools/fit_cached_heads.py --features "$RUN/features.npz" \
  --labels "$RUN/source_fit_labels.csv" --manifest "$PKG/split_manifest.json" --config "$PKG/config.json" \
  --target "$TARGET" --out "$RUN/heads" --device cuda 2>&1 | tail -3

echo "== 3) fit_cached_relations (J=3) =="
rm -rf "$RUN/relations"; CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY tools/fit_cached_relations.py --features "$RUN/features.npz" \
  --relations /root/autodl-tmp/e01_runs/relations/relations.npz --group-model "$RUN/groups/group_model.npz" \
  --heads "$RUN/heads" --labels "$RUN/source_fit_labels.csv" --manifest "$PKG/split_manifest.json" \
  --config "$PKG/config.json" --target "$TARGET" --out "$RUN/relations" --device cuda 2>&1 | tail -3

echo "== 4) glue prep =="
$PY "$IN/e01_glue_rev1.py" prep --fold "$TARGET" --run "$RUN" 2>&1 | tail -20

echo "== 5) calibrate =="
$PY tools/calibrate_tolerance.py --input "$RUN/calibration_moments.npz" --out "$RUN/calibration.json" 2>&1 | tail -14

echo "== 6) filter =="
$PY tools/run_moment_filter.py --input "$RUN/filter_inputs.npz" --config "$PKG/config.json" \
  --calibration "$RUN/calibration.json" --out "$RUN/adapt_filter" 2>&1 | tail -3

echo "== 7) apply_retained =="
$PY tools/apply_retained.py --probabilities "$RUN/eval_probabilities.npy" \
  --retention "$RUN/adapt_filter/retention.json" --out "$RUN/eval_predictions.npz" 2>&1 | tail -3

echo "== 8) metrics =="
$PY "$IN/e01_glue_rev1.py" metrics --fold "$TARGET" --run "$RUN" 2>&1 | tail -60

echo "== $TARGET REV1 完成 ($(date '+%H:%M:%S')) =="
