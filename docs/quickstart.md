# Quick integration check

This entrypoint uses the same bounded CoVLA setup as the completed integration acceptance: two training videos, one validation video, one ID test video and one rainy OOD video, with one epoch, batch size one and BF16. It verifies encoder loading, frozen-probe training and ID/OOD evaluation. Its scores are not paper results. The quickstart YAML is separate from the formal experiment index.

## Preview without data or weights

Install the README dependencies and run from the repository root:

```bash
python scripts/validate_release.py
python scripts/prepare_metadata.py covla --dry-run
python scripts/quickstart.py --dry-run
```

The preview needs only PyYAML. It prints the selected configuration and command without loading a model, creating output files or launching GPU work.

## Supply external inputs

Supply CoVLA videos, metadata and state files described in [data.md](data.md), plus a compatible V-JEPA 2.1 ViT-g/384 encoder checkpoint. Generate the paper splits locally first, then check your existing resource paths:

```bash
python scripts/prepare_metadata.py covla --dataset-root /path/to/CoVLA-Dataset

python scripts/quickstart.py --check-only \
  --covla-root /path/to/CoVLA-Dataset \
  --encoder-checkpoint /path/to/last.pth.tar
```

The default locations are in [the quickstart YAML](../configs/quickstart/covla_probe_smoke.yaml). The input check verifies resource paths and locally generated split JSONs; it does not load the checkpoint, inspect every video or verify CUDA. The runner validates compatibility during execution. Data and weights are never downloaded or copied by the quickstart.

## Run on an allocated GPU

Use a GPU with enough memory for the frozen ViT-g/384 encoder at 64 frames; the previous integration run used one H200. GPU allocation is managed by your own cluster or machine.

```bash
python scripts/quickstart.py \
  --covla-root /path/to/CoVLA-Dataset \
  --encoder-checkpoint /path/to/last.pth.tar
```

Outputs are written under `outputs/quickstart/covla_probe_smoke/`: `history.json`, `best_probe.pt` and `test_metrics.json`. The last JSON contains both `test` and `ood_test`. The entrypoint reports success after confirming those outputs. Runtime outputs are ignored and never included in the release package.

An existing nonempty output directory is refused. To run again, use a new `--output-dir`. Paths supplied on the command line are resolved relative to the repository root unless absolute. To evaluate the fitted smoke probe again without retraining:

```bash
python scripts/quickstart.py --eval-only \
  --covla-root /path/to/CoVLA-Dataset \
  --encoder-checkpoint /path/to/last.pth.tar
```

This reads the quickstart's `best_probe.pt` and writes new metrics under its `eval/` subdirectory. For a different fitted probe, supply `--probe-checkpoint`; for a different base output directory, supply `--output-dir` consistently in both commands.
