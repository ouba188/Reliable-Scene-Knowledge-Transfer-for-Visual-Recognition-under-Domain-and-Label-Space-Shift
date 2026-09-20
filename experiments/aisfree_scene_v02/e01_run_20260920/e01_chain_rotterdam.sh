#!/bin/bash
# E01 Rotterdam fold: run the package pipeline end to end after encoder training finishes.
set -u
PY=/root/miniconda3/bin/python
IN=/root/autodl-tmp/knowledge_841_20260920/e01_inputs
PKG=/root/autodl-tmp/knowledge_841_20260920/e01_first_batch
RUN=/root/autodl-tmp/e01_runs/Rotterdam
LOG=/root/e01_chain_rotterdam.log
TARGET=Rotterdam

cd "$PKG" || exit 1
echo "== 等待特征导出完成 ($(date '+%H:%M:%S')) =="
for i in $(seq 1 240); do
  if [ -f "$RUN/features.npz" ] && [ -f "$RUN/source_fit_labels.csv" ]; then echo "特征就绪"; break; fi
  sleep 30
done
[ -f "$RUN/features.npz" ] || { echo "特征未生成，退出"; exit 2; }

echo "== 1) fit_groups =="
$PY tools/fit_groups.py --features "$RUN/features.npz" --manifest "$PKG/split_manifest.json" \
  --target $TARGET --out "$RUN/groups" 2>&1 | tail -5

echo "== 2) fit_cached_heads (H=8) =="
CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY tools/fit_cached_heads.py --features "$RUN/features.npz" \
  --labels "$RUN/source_fit_labels.csv" --manifest "$PKG/split_manifest.json" --config "$PKG/config.json" \
  --target $TARGET --out "$RUN/heads" --device cuda 2>&1 | tail -4

echo "== 3) fit_cached_relations (J=3) =="
CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY tools/fit_cached_relations.py --features "$RUN/features.npz" \
  --relations /root/autodl-tmp/e01_runs/relations/relations.npz --group-model "$RUN/groups/group_model.npz" \
  --heads "$RUN/heads" --labels "$RUN/source_fit_labels.csv" --manifest "$PKG/split_manifest.json" \
  --config "$PKG/config.json" --target $TARGET --out "$RUN/relations" --device cuda 2>&1 | tail -4

echo "== 4) glue prep =="
$PY "$IN/e01_glue.py" prep --fold $TARGET --run "$RUN" 2>&1 | tail -14

echo "== 5) calibrate =="
$PY tools/calibrate_tolerance.py --input "$RUN/calibration_moments.npz" --out "$RUN/calibration.json" 2>&1 | tail -12

echo "== 6) filter =="
$PY tools/run_moment_filter.py --input "$RUN/filter_inputs.npz" --config "$PKG/config.json" \
  --calibration "$RUN/calibration.json" --out "$RUN/adapt_filter" 2>&1 | tail -4

echo "== 7) apply_retained =="
$PY tools/apply_retained.py --probabilities "$RUN/eval_probabilities.npy" \
  --retention "$RUN/adapt_filter/retention.json" --out "$RUN/eval_predictions.npz" 2>&1 | tail -4

echo "== 8) metrics =="
$PY "$IN/e01_glue.py" metrics --fold $TARGET --run "$RUN" 2>&1 | tail -40

echo "== 完成 ($(date '+%H:%M:%S')) =="
