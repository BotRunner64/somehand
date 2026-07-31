# Quantitative Experiment

This workflow runs the frozen cross-hand comparison and the distance/frame ablations without a viewer or hardware controller. Raw recordings and generated results stay under `recordings/` and are not committed.

## 1. Preflight

Install development dependencies, download MJCF assets, and validate the complete matrix:

```bash
pip install -e ".[dev]"
python scripts/setup/download_assets.py --only mjcf
python scripts/experiments/preflight_quantitative.py
```

The v1 manifest covers 19 right-hand configs. Every hand must resolve four distance constraints, one thumb-frame constraint, five common base-to-tip evaluation directions, five fingertips, and fingertip collision geometry. The planned matrix contains 19 Full runs plus `no_distance` and `no_frame` for `linkerhand_l20_right`.

## 2. Use an existing PICO recording

An exploratory pass does not require a new fixed-protocol recording. The included
available-data manifest uses `recordings/pico_right.pkl`, six representative hands,
and the two L20 ablations:

```bash
python scripts/experiments/run_quantitative.py \
  --manifest configs/experiments/quantitative_existing_pico.yaml
python scripts/experiments/summarize_available_quantitative.py
```

Because this recording has no action labels, the exploratory summary uses every
saved frame for Direction Error, Normalized Semantic Keypoint Error, site-distance
tracking error, and thumb-frame error. Site-distance tracking error is the absolute
difference between each robot thumb-to-finger site distance and the corresponding
human fingertip distance. It is continuous, threshold-free, and independent of
collision-mesh thickness. Generated CSV, JSON, and figure names end in `available`
and record the all-frame scope explicitly.

## 3. Record the fixed PICO sequence

Connect PICO Bridge, keep the right hand active, and run:

```bash
python scripts/experiments/record_quantitative_pico.py
```

The terminal cues seven actions: fist, four pairwise pinches, tripod pinch, and thumb opposition. Each action is repeated five times. The default 80 Hz protocol records 12,600 frames over 157.5 seconds, excluding the countdown.

The recorder writes:

- `recordings/quantitative/pico_right_v1.pkl`
- `recordings/quantitative/pico_right_v1.annotations.json`

Any missing sampled frame aborts the recording without saving formal artifacts. Existing files are not overwritten unless `--force` is explicitly supplied.

Validate gesture completion and sampling continuity before running any robot output:

```bash
python scripts/experiments/validate_quantitative_recording.py
```

## 4. Run the formal matrix

Run all 21 jobs:

```bash
python scripts/experiments/run_quantitative.py
```

Useful scoped commands:

```bash
python scripts/experiments/run_quantitative.py --dry-run
python scripts/experiments/run_quantitative.py \
  --hand linkerhand_l20_right \
  --method full
```

For an unannotated smoke check only:

```bash
python scripts/experiments/run_quantitative.py \
  --recording recordings/samples/pico_right_short.pkl \
  --allow-unannotated \
  --hand linkerhand_l20_right \
  --method full \
  --max-frames 20 \
  --output-dir /tmp/somehand-quantitative-smoke
```

Formal results are stored as a compressed NumPy array file plus JSON metadata for each `hand × method`. Metadata includes input/config hashes, the resolved config, semantic mappings, model coupling counts, environment versions, and SLSQP status.

## 5. Produce formal tables and figures

After every formal run completes:

```bash
python scripts/experiments/summarize_quantitative.py
python scripts/experiments/render_ablation_snapshots.py
```

The summary directory contains the complete and six-hand tables, the distance-ablation table/curve, the frame-ablation table/bar chart, and Full/w-o-frame renders from the same opposition input frame.

## Frozen metric choices

- Primary Direction Error uses five common finger-base-to-tip rays, independent of each hand's objective topology. Per-objective-segment errors are also saved for supplementary diagnosis.
- Normalized Semantic Keypoint Error uses five fingertips, wrist/world origins, and wrist-to-middle-base palm scales.
- Site Distance Tracking MAE is the mean absolute difference between robot fingertip-site distance and the corresponding human fingertip distance. Formal distance-ablation results use annotated pinch hold frames; the unlabeled exploratory pass uses every saved frame.
- Collision-surface distances remain available as frame-level diagnostics but are excluded from cross-hand scores, rankings, and ablation claims.
- Mean Solve Time measures only the call to SciPy SLSQP after a throwaway warm-up.
- Thumb-Frame Error uses the Full semantic frame mapping even for `no_frame`.

Do not change the recording, annotation, thresholds, mappings, or manifest after inspecting formal robot outputs. Optimizer failures and invalid episodes must be reported rather than discarded.
