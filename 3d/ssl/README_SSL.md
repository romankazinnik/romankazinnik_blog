# ssl_scene_probe.py

Self-supervised representation test + semantic classifier for colored/labeled point clouds.

**Pipeline:** (1) SSL pretext — a masked point autoencoder reconstructs masked per-point features (standardized RGB and/or a 10-d invariant geometry descriptor: normal/direction verticality, planarity, linearity, curvature at 2 scales) and denoises coordinates, with random z-rotation augmentation, producing a per-point embedding; (2) a classifier ("model") is trained on the labeled points using those embeddings and predicts the unlabeled ones; (3) instances are recovered per class via DBSCAN. The SSL embedding is compared against a raw-geometry baseline so you can see whether pretraining actually bought anything.

Deps: `torch numpy scipy scikit-learn plyfile pye57` (`open3d`/`laspy` only for those formats).

---

## Inputs & labels

- `--data PATH` — the point cloud (`.ply/.e57/.npy/.las/.pcd`). Labels come from a `semantic_id` vertex field if present, else are decoded from vertex colors (distinct color = class, `0,0,0` = unlabeled).
- `--color_ply PATH` — separate cloud (same points) carrying the **real scan colors** used as SSL features. Aligned to `--data` by nearest neighbor (prints max match distance; ~0 = same points).
- Color source for SSL = `--color_ply` if given, else `--data`'s colors when they aren't the labels, else geometry-only.

| flag | default | description |
|---|---|---|
| `--data` | — | input cloud; `semantic_id` field or vertex colors carry labels |
| `--color_ply` | — | same-points cloud supplying scan RGB for SSL features |
| `--sem_field` | `semantic_id` | per-vertex integer label field name |
| `--labels_from_colors` | `False` | force color-decoded labels even when `semantic_id` exists |
| `--color_min_count` | `50` | drop colors with fewer points than this (anti-alias noise) |
| `--max_color_classes` | `50` | hard-error if `--data` has more distinct colors (looks like a scan, not labels) |
| `--names_json` | — | `{"semantic_names":{"1":"wall"}}` (id→name) or `{"color_names":{"0,180,0":"wall"}}` (rgb→name) |
| `--palette_ply` | — | derive output colors per `semantic_id` from a type-colored cloud (id-label mode only) |

## Classifier ("model") & post-processing

| flag | default | description |
|---|---|---|
| `--head` | `mlp` | `mlp` (128-unit hidden layer) or `linear` (logistic regression) |
| `--eval_frac` | `0.2` | per-class fraction held out for eval (≥1 pt if class size ≥2); eval = all non-fit labeled points |
| `--probe_max_train` | `50000` | cap on points fit; drawn **balanced** per class (sklearn is CPU-only) |
| `--probe_pca` | `64` | PCA-reduce SSL embeddings before the classifier (0 = off) |
| `--conf_thresh` | `0.5` | predictions below this are left unlabeled (black in `pred_new_points`) |
| `--smooth_k` | `12` | KNN majority smoothing of predictions (0 = off); removes speckle |
| `--dbscan_eps` | `0.3` | DBSCAN neighborhood radius, **meters** (tune to point spacing) |
| `--dbscan_min` | `30` | DBSCAN min samples per object |
| `--dbscan_cap` | `150000` | per-class DBSCAN cap (subsample + NN-propagate above this) |

## SSL training

| flag | default | description |
|---|---|---|
| `--epochs` | `40` | pretext epochs |
| `--dim / --depth / --heads` | `256 / 4 / 4` | transformer width / layers / attention heads |
| `--K / --n_blocks` | `2048 / 512` | points per training block / blocks per epoch |
| `--mask_ratio` | `0.5` | fraction of per-point features masked for reconstruction |
| `--xyz_noise / --w_xyz` | `0.05 / 1.0` | coord-denoising noise std / loss weight |
| `--lr / --batch` | `0.001 / 16` | AdamW LR / block batch size |
| `--max_minutes` | `25.0` | wall-clock training budget (early exit) |

## Runtime & caching

| flag | default | description |
|---|---|---|
| `--voxel` | `0.0` | voxel downsample size (m); `0` = keep all points (E57 self-downsamples at 0.04) |
| `--max_points` | `4000000` | random cap after voxel |
| `--device` | `cpu` | e.g. `cuda:1` |
| `--out` | `ssl_out` | output dir; cache lives in `<out>/cache/` |
| `--run_classifier` | — | label a new unlabeled cloud using `DIR/cache/{model.pt, classifier.pkl}` from a previous run |
| `--extra_data` | — | comma-separated **unlabeled** clouds that join SSL pretraining (no labels needed) — the key transfer lever: include the target cloud here when training |
| `--block_scale` | `2.0` | fixed metric scale (m) for SSL block coords; density/scale-stable across clouds |
| `--force_retrain` | `False` | ignore cached `model.pt` / `features.npz` |
| `--seed` | `0` | RNG seed |

**Cache (`<out>/cache/`):** `model.pt` gates SSL training; `features.npz` gates dense embedding + descriptor. Independent — a present `model.pt` with a missing `features.npz` reloads the model and only re-embeds. `classifier.pkl` (SSL classifier + fitted PCA + palette + names) is saved after every training run and is what `--run_classifier` consumes. Path-trust; a change in feature width auto-invalidates. `features.npz` ≈ `N × dim × 4` bytes (~1.4 GB at 1.3M pts, dim 256).

## Labeling a new cloud (`--run_classifier`)

`--run_classifier DIR` labels a **new, unlabeled** cloud using a previous run's artifacts (`DIR/cache/model.pt` + `classifier.pkl`). No training happens: the frozen source encoder embeds the new cloud, the source-fitted PCA projects it, and the saved classifier predicts every point. The new cloud's color availability must match the training run (rgb+geom classifier needs `--color_ply` or colored `--data`; geometry-only needs neither) — mismatch is a hard error. Embeddings are cached under the new `--out`. Outputs: `pred_types.ply` (source palette/names; abstentions below `--conf_thresh` unlabeled) and `pred_confidence.ply`. **For best transfer, add the target cloud to `--extra_data` during training** (SSL is unsupervised — it can pretrain on the unlabeled target), and keep both clouds at the same `--voxel` (a spacing-mismatch warning fires at inference otherwise). Transfer is further helped by design: all classifier position features are invariant (height above floor / below ceiling in meters — no absolute xy/z), the descriptor is rotation/translation-invariant, RGB is per-cloud standardized, and SSL is z-rotation-augmented. Still expect some drop vs. the training cloud on dissimilar sites.

## Outputs (`<out>/`)

`subsampled.ply` (input at working resolution) · `pred_types.ply` (all points, known + predicted, by class) · `pred_new_points.ply` (only newly-classified points; low-conf → black) · `pred_confidence.ply` (green=confident) · `pred_instances.ply` (DBSCAN objects) · `pred_train_points.ply` / `pred_eval_points.ply` (fit / held-out labeled, colored by prediction) · `mistake_train_points.ply` / `mistake_eval_points.ply` (green=correct, red=wrong). Console reports per-epoch losses, per-class IoU (SSL vs raw), confidence buckets, and object counts.

---

## Examples

**1. Labels + scan color in separate files (typical scan-to-BIM).** `--data` has `semantic_id`; real color in a second cloud.
```bash
python ssl_scene_probe.py --device cuda:1 --epochs 10 \
  --data working-cloud-instance.ply \
  --color_ply working-cloud-orig.ply \
  --names_json labels.json
```

**2. Class-colored labels, no `semantic_id`.** `--data` is color-coded per class (`0,0,0` = unlabeled); scan color separate; names bound to colors. Output colors match the input scheme automatically.
```bash
python ssl_scene_probe.py --device cuda:1 --epochs 10 \
  --data classes_colored.ply \
  --color_ply working-cloud-orig.ply \
  --names_json color_names.json      # {"color_names":{"0,180,0":"wall",...}}
```

**3. Geometry-only (no scan color available).** SSL learns the geometry descriptor only; still classifies from `semantic_id`.
```bash
python ssl_scene_probe.py --device cuda:1 --epochs 10 \
  --data working-cloud-instance.ply --names_json labels.json
```

**4. Fast iteration on the downstream (reuse cached SSL).** After a full run, tweak classifier/DBSCAN/viz without retraining or re-embedding.
```bash
python ssl_scene_probe.py --device cuda:1 \
  --data working-cloud-instance.ply --color_ply working-cloud-orig.ply \
  --out ssl_out --head linear --smooth_k 24 --dbscan_eps 0.15 --conf_thresh 0.7
```

**5. Raw E57, no labels — SSL sanity check.** Trains, then writes PCA/k-means clouds to eyeball whether the embedding separates structure.
```bash
python ssl_scene_probe.py --device cuda:1 --epochs 15 \
  --data 808_Brannan.e57 --voxel 0.04 --out ssl_raw
```

**6. Label a brand-new cloud with a trained classifier.** Train once on building A (example 1 or 2), then label building B's unlabeled scan — no retraining.
```bash
python ssl_scene_probe.py --device cuda:1 \
  --data buildingB.ply --color_ply buildingB-orig.ply \
  --run_classifier ssl_out_buildingA --out ssl_out_buildingB
```
