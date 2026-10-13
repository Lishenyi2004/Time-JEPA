# Paper reproduction index

For exact per-YAML training/probe/encoder associations, use [experiments.md](experiments.md), [models.md](models.md) and the [CSV index](experiment-index.csv). This guide describes the overall execution order.

Run from the repository root. Configurations retain the original module layout and exact scientific hyperparameters, with machine-local roots made relative and adapted encoder filenames aliased to `last`. No training or model selection is performed by this staging package.

| Paper scope | Configurations / code |
|---|---|
| Main single-domain comparisons | `configs/train_time_jepa/weather_ood`, the two selected AgiBot n15000 training YAMLs, and the two selected EgoExo exo124/view03 training YAMLs |
| Current-state probes | `covla_probe_rjob/configs/covla/weather_ood`, `agibot_wholebody_v2`, `egoexo/exo124_view3ood/whole` |
| Multi-domain comparisons | The selected `configs/train_time_jepa/covla_agibot_egoexo` pair and `covla_probe_rjob/configs/covla_agibot_egoexo/mix_exposure_agibot15000` |
| Appendix F matched human-motion sequence | Selected `configs/train_time_jepa/covla_egoexo` and `covla_probe_rjob/configs/covla_egoexo` |
| Five-horizon future state | `covla_probe_rjob/configs/future_state_1s`, plus `future_state_single_domain` |
| Appendix G eight-horizon extension | `covla_probe_rjob/configs/future_state_1p6s_8step` |
| Kinematic reconstruction / Appendices E/F/K | `physics_consistency/{covla,agibot_reach_rate,egoexo,egoexo_knee}/configs` |
| GRASP | `grasp_eval_rjob/configs` |
| Physion ordered/shuffled and leave-one-scenario-out | `physion_ocp_rjob/configs/main` |
| Additional backbones / iVideoGPT adaptation | `world_model_probe_rjob/configs` |
| Appendix I head aggregation | Six selected `configs/train_time_jepa/egoexo_head_sweep` YAMLs and corresponding probes |
| Appendix I probe capacity / breadth | `ablation_a3`, selected `ablation_a4` probes, and the reduced 1D/5D training YAMLs |
| Appendix J sensitivity | Cross-take A4 readout, 0.8 s offset and Gaussian-noise readout configurations |
| Appendix K alternative supervision | `external_baselines/psg_jepa/configs/*n15000*` and `covla_law_rjob/probes/configs` |

The materialized eight-horizon YAMLs apply the original launcher overrides (`future_horizon_seconds=1.6`, eight horizons and a distinct output directory) to the five-horizon base. They are separate fitted probes, not a reinterpretation of five-horizon results.

## Execution order

1. Install dependencies; prepare source dependencies only for the backbones being evaluated.
2. Obtain the data and official pretrained snapshots separately. Use `scripts/prepare_metadata.py` to regenerate and verify CoVLA/Ego-Exo4D splits and their full manifests locally; their missing statistics are computed from training data by the configured runners. Retain the separately licensed AgiBot split/statistics/PCA metadata. See `data.md`.
3. Train the video-fine-tuned control and Time-JEPA encoders with their selected YAMLs. Preserve per-domain quotas and split checks.
4. Select the paper-equivalent saved encoders and prepare `last` aliases as described in `checkpoints.md`.
5. Fit current-state and future-state probes on probe training data, select fitted probes on validation loss, then evaluate ID/OOD using the same training normalization.
6. Generate required GT gates/calibration before fitting independent kinematic probe banks. Five probe seeds are 0 through 4. The URDF/gate resources must be supplied/generated separately.
7. Recompute per-view, per-variable and per-horizon aggregates from the resulting JSONs.

GRASP command sequence is `python -m grasp_eval_rjob.audit_data`, `python -m grasp_eval_rjob.freeze_protocol`, `python -m grasp_eval_rjob.run_eval`, `python -m grasp_eval_rjob.score`, then `python -m grasp_eval_rjob.compare_models`; inspect each command's `--help`. Run against the included model YAMLs with a generated data audit. The reported aggregate gives equal weight to the eight properties.

Physion first extracts frozen features with `python -m physion_ocp_rjob.extract_features`, then fits/evaluates readouts with `python -m physion_ocp_rjob.run_probe`. Preserve chronological/shuffled test ordering and the distinction between all-scenario and leave-one-scenario-out protocols.

For estimated persistence in the eight-horizon extension, use the existing probe runner's `--eval-only --probe-checkpoint <eight-horizon best_probe.pt> --estimated-persistence-checkpoint <matched current-state best_probe.pt>` arguments with the corresponding eight-horizon YAML. Keep future-valid and paired current/future-valid comparisons separate.

Result summarizers are retained in `scripts/summarize_future_state_ood.py`, `scripts/summarize_single_domain_future.py`, `paper/scripts/aggregate_ivideogpt_current_state.py` and the selected kinematic aggregation modules. They require newly produced result files, which are not included.

## Excluded experiment material

SSv2, Diving48, IntPhys2 evaluation configurations, PEZ/Kubric experiments, KITTI protocols, old CoVLA splits, AgiBot v1/left-arm development variants, EgoExo bestexo/view04 protocols, A1, clip-level A5, sensor-density experiments, historical smoke/pilot/debug YAMLs, cluster launch scripts, raw results, paper drafting material and cleanup scripts are excluded. The public [quickstart](quickstart.md) is a separate bounded integration check, excluded from the formal experiment index. `intphys2_eval_rjob/model.py` remains solely because the GRASP evaluator imports this predictive-model implementation.

Every selected file and its source checksum are recorded in `release-manifest.json`. Every changed encoder reference is recorded in `checkpoint-aliases.json`.
