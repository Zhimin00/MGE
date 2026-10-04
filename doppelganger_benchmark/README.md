# Doppelganger Camera-Pose Benchmark

This release contains 43 fixed mixed-input subsets from eight outdoor Heinly scenes.
Each subset contains exactly 20 input images in model input order:

- indices 0-14: 15 normal reference views;
- indices 15-19: 5 visually similar Doppelganger views.

The evaluation scripts, manifests, poses, intrinsics, and calibration files are stored on
the GitHub `evaluation` branch. The 860 benchmark images are hosted in the
[MGE Hugging Face repository](https://huggingface.co/shaozhimin/MGE) to keep large binary
assets out of Git history.

## Download Images

From the root of an `evaluation` branch checkout, install the Hugging Face CLI and download
the images into their expected relative paths:

```bash
pip install -U "huggingface_hub[cli]"
hf download shaozhimin/MGE \
  --include "doppelganger_benchmark/subsets/*/*/images/*.jpg" \
  --local-dir .
```

Verify that all images were downloaded:

```bash
find doppelganger_benchmark/subsets -path '*/images/*.jpg' | wc -l
# Expected: 860
```

## Layout

```text
subsets/<scene>/<subset_id>/
  images/                    # 20 model inputs
  ordered_images.csv         # authoritative input order and role
  gt_c2w.npy                 # pseudo-GT camera-to-world poses, (20, 4, 4)
  gt_c2w.csv                 # human-readable copy of the poses
  intrinsics.npy             # camera intrinsics, (20, 3, 3)
  camera_calibration.json    # COLMAP camera model and distortion
  subset.json                # compact subset metadata
```

`benchmark.json` lists all 43 subsets. Image paths in `ordered_images.csv` are relative to
each subset directory and are restored by the Hugging Face download command above.

## Run Pi3

The benchmark does not duplicate the Pi3 repository or model weights. Point the inference
script at a Pi3/SWAPi3 checkout and a checkpoint:

```bash
python eval_scripts/run_pi3_inference.py \
  --pi3-repo /path/to/SWAPi3 \
  --checkpoint /path/to/checkpoint.pth \
  --output-dir predictions/pi3
```

For every subset this writes:

```text
predictions/pi3/<scene>/<subset_id>/pred_c2w.npy
```

The predicted matrices must follow `ordered_images.csv` and use the camera-to-world convention.

## Evaluate

```bash
python eval_scripts/evaluate_camera_poses.py \
  --predictions-root predictions/pi3 \
  --output-dir results/pi3
```

The evaluator reports relative camera-pose AUC, relative rotation accuracy (RRA), and
relative translation-direction accuracy (RTA) at 5, 10, 15, and 30 degrees for:

- `all`: all 190 image pairs;
- `reference`: 105 reference-reference pairs;
- `cross`: 75 reference-Doppelganger pairs;
- `doppelganger`: 10 Doppelganger-Doppelganger pairs.

Overall metrics are the direct arithmetic mean of the 43 subset metrics. No trajectory scale
or ATE is used.
