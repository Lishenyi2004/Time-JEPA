# Data preparation

Obtain raw data from its original provider and accept the applicable access terms. No processed dataset mirror is provided. Place datasets at the following paths, or edit their YAML paths:

| Domain/task | Original source | Default root | Preparation |
|---|---|---|---|
| autonomous driving | [CoVLA](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset) | `data/CoVLA-Dataset` | Videos, `metadata.jsonl` and per-video state/caption files |
| robot manipulation | [AgiBot World](https://huggingface.co/agibot-world) | `data/agibot` | The paper's task/session subset, episode videos, LeRobot Parquet state and metadata |
| human motion | [Ego-Exo4D access instructions](https://docs.ego-exo4d-data.org/getting-started/) | `data/EgoExo4D_minimal` | Exocentric videos, `takes.json` and 3D body-pose annotations |
| GRASP | [GRASP](https://github.com/i-machine-think/grasp) | `external_baselines/grasp/data/videos/level2` | Official Level-2 paired videos |
| Physion | [Physion evaluator](https://github.com/neuroailab/physion_evaluator) | `data/physion_v1_5` | Compact Physion v1.5 videos and labels |

Split JSONs use dataset IDs and window keys. AgiBot splits are included; CoVLA and Ego-Exo4D splits and per-sample manifests must be generated locally. Keep encoder training, probe training, validation and testing separate. For human motion, seen exocentric views are 01/02/04 and held-out view 03 supplies condition OOD. Probe labels are supervised even for variables excluded from encoder auxiliary supervision.

## Generate restricted-domain metadata locally

Run from the repository root after obtaining the complete source data required by the paper protocol:

```bash
# Preview the fixed recipes without data access or writes.
python scripts/prepare_metadata.py covla --dry-run
python scripts/prepare_metadata.py egoexo --dry-run

# Generate splits and the full per-video/per-take manifest on your own machine.
python scripts/prepare_metadata.py covla --dataset-root /path/to/CoVLA-Dataset
python scripts/prepare_metadata.py egoexo --dataset-root /path/to/EgoExo4D_minimal
```

`metadata-protocol.json` stores the fixed generation arguments and expected split counts/checksums without sample IDs or annotation content. CoVLA keeps the fixed capture-date/rainy-vote protocol. Ego-Exo4D explicitly uses seed 239, the `exo_view_ood` protocol, seen views 01/02/04 and held-out view 03; the raw split script's defaults use a different view protocol, so use the wrapper above.

The generator checks the resulting split files against the paper's expected hashes. Missing source videos, a different data snapshot or preprocessing can produce different splits; a mismatch is reported as failure. Existing split files are never overwritten. Use `--verify-only` to check existing splits or `--output-dir` for a new destination. Point the selected YAMLs at the same external data root used for generation.

Preparation entry points are `time-jepa/make_covla_weather_ood_splits.py`, `time-jepa/make_agibot_splits.py`, `time-jepa/make_egoexo_splits.py`, `time-jepa/fit_agibot_waist_pca.py` and `time-jepa/transcode_agibot_h264.py`; use each script's `--help` to inspect input layouts and options.

`time-jepa/cache/` contains the selected AgiBot normalization statistics and frozen train-only waist PCA; AgiBot kinematic-probe statistics are also included. They are separately licensed under CC BY-NC-SA 4.0; see [DATA-LICENSES.md](../DATA-LICENSES.md).

CoVLA/Ego-Exo4D normalization and kinematic statistics are omitted. The training datasets and kinematic-bank runners compute missing caches from their configured training splits using the original algorithms. The shared probe computes its own training-target statistics. Do not fit normalization on validation/test data. These locally generated files are ignored by Git and excluded from packaging; `metadata-policy.json` identifies their exact paths and generating entrypoints. Initial runs may take longer while statistics are computed.

## Resources deliberately omitted

The G2 URDF and frozen GT-gate/calibration JSONs are **not included**, as requested. Their YAML paths are retained as expected resource locations. Obtain the official URDF separately before generating robot kinematic gates:

- Repository: `https://github.com/AgibotTech/genie_sim`
- Revision: `09ed792f5c8d07d9a8d963662dc9eab409c621a3`
- File: `source/geniesim_benchmark/src/geniesim_benchmark/utils/IK-SDK/G2_NO_GRIPPER.urdf`
- Expected SHA-256: `76d056b159af5ffc1aa9a7d7b931f6221f6cff5818e10fe4af4b7552162f0aef`
- Expected local location: `physics_consistency/agibot/assets/G2_NO_GRIPPER.urdf`

Generate gates/calibration before reconstruction using `physics_consistency.covla.covla_gt_consistency_gate`, `physics_consistency.agibot_reach_rate.gt_gate`, `physics_consistency.egoexo.gt_kinematic_gate` and `physics_consistency.egoexo_knee.gt_kinematic_gate`. The law-supervision preprocessing gate in `time-jepa/cache/` is also omitted; generate it with `python covla_law_rjob/gate.py --config configs/train_time_jepa/weather_ood/covla_vitg384_64f_time_jepa_conv1x1_final.yaml --output time-jepa/cache/covla_law_train6305_gate_video_pts_v2.json --workers 8`.

All dataset access and redistribution terms remain with the original dataset sources; this package contains the selected processing implementation and experiment metadata.
