# Real-data integration acceptance

The staged repository was exercised on 2026-10-13 using the existing jepa environment and one NVIDIA H200 GPU. Data and the paper-selected CoVLA Time-JEPA checkpoint were referenced through local symbolic links; neither dataset nor encoder weights were copied. All runtime output is under the ignored `outputs/acceptance/` directory.

## Verified pipeline

- Real CoVLA videos and synchronized state labels load using the fixed paper splits. Split counts are 1,349 probe-train, 449 probe-validation, 900 ID-test and 521 rainy OOD videos. These split lists now require local generation in the public package.
- Five sampled videos decode to `[3,64,384,384]`; state targets and validity masks have shape `[32,3]`. Frame sampling uses 5 FPS with a 12.6-second first-to-last-frame span.
- The `last.pth.tar` link loads the paper-selected `target_encoder` with **zero missing or unexpected parameter keys**.
- A frozen-encoder Conv1x1 probe completes two optimization steps, one validation pass and validation checkpoint selection.
- The saved fitted probe completes one ID-test and one rainy OOD-test example. Both report 32 valid tubelets per target, and loss/MAE/RMSE/nRMSE/Pearson/CCC/R2 are finite.
- The final evaluation task succeeds. The retained 68 regression tests pass after the sample-limit fix, and release-file validation also passes.

These tiny-sample results establish execution of the integration path. They are not a reproduction of paper scores. The probe was trained on only two videos for one epoch, with batch size one and BF16 autocast; formal evaluation requires the full selected configuration.

## Issue found and corrected

`max_test_videos` originally restricted the ID split but omitted the additional CoVLA rainy OOD split. This would silently turn a small smoke test into a 521-video evaluation. The release runner now applies the same limit to both CoVLA test splits, and its CLI help documents that contract. Formal configurations use zero (all videos), so their sampling and reported protocols are unchanged. The original experimental runner was not edited.

The first GPU task was stopped after its training and validation finished. The second task reused that `best_probe.pt` and ran the corrected limited ID/OOD evaluation, avoiding redundant training.

## Run the published quickstart

Use [quickstart.md](quickstart.md) and the included `configs/quickstart/covla_probe_smoke.yaml` to repeat the same bounded setup with your own external CoVLA data and compatible encoder checkpoint. First generate and verify CoVLA metadata locally with `scripts/prepare_metadata.py`. The public entrypoint supports a resource-free command preview, resource-path checks, training plus ID/OOD evaluation, and evaluation of a fitted smoke probe without retraining.

The earlier acceptance logs and fitted probe remain local ignored runtime files. They are not required by the published quickstart and are excluded from the public release package. The original real-data acceptance exercised the probe runner; the new wrapper and clean package are checked separately without launching another GPU task.
