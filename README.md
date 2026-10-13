# Time-JEPA

Code and configurations for **Time-JEPA: Grounding Video World Models with Physical Time Series**.
Time-JEPA fine-tunes a video encoder with temporally aligned physical-state supervision. Evaluation uses video-only input.

The manuscript is submitted to ICLR 2027 and is being prepared as an arXiv preprint; it has not been accepted. **arXiv: pending submission.**

Authors: Shenyi Li, Ziyang Zhang, Wen Wu, Ming Liu, Chao Zhang.
Code: [Lishenyi2004/Time-JEPA](https://github.com/Lishenyi2004/Time-JEPA). The repository citation is in [CITATION.cff](CITATION.cff); a paper citation will be added when the preprint is available. See [publication information](docs/publication.md).

## Installation

Use Python 3.11. `requirements.txt` records the package versions exported from the existing `jepa` environment, including PyTorch 2.9.0/CUDA 12.8. `environment-jepa.json` records the Conda package inventory for provenance; it is not a Conda environment YAML.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Install FFmpeg separately for the timestamp-aligned decoding paths. Run commands from this repository root.
Third-party sources, video datasets, pretrained weights and adapted weights are not included. See [dependency preparation](docs/dependencies.md) and [data preparation](docs/data.md).

## Quick integration check

```bash
# Preview without datasets, weights or a GPU.
python scripts/quickstart.py --dry-run

# After obtaining CoVLA under its official terms, generate the fixed splits locally.
python scripts/prepare_metadata.py covla --dataset-root /path/to/CoVLA-Dataset

# On an allocated GPU, using externally supplied resources.
python scripts/quickstart.py \
  --covla-root /path/to/CoVLA-Dataset \
  --encoder-checkpoint /path/to/last.pth.tar
```

This runs one epoch on two training videos, one validation video and one video each for ID/rainy OOD evaluation. See [quickstart.md](docs/quickstart.md) for input checks, output files and evaluation without retraining. Quickstart scores are not paper results.

## Training and probing

Prepare the data, locally generated CoVLA/Ego-Exo4D splits and the official `vjepa2_1_vitg_384.pt` checkpoint first. See [data.md](docs/data.md) for the fixed recipes. Edit the data/checkpoint/output paths in the selected YAML if needed; published configurations use repository-relative paths.

```bash
# Example: CoVLA Time-JEPA fine-tuning. The reported main training setup uses 8 GPUs.
torchrun --standalone --nproc_per_node=8 covla_jepa_rjob/train_from_yaml.py \
  --fname configs/train_time_jepa/weather_ood/covla_vitg384_64f_time_jepa_conv1x1_final.yaml

# Frozen current-state probe.
python covla_probe_rjob/train_covla_probe.py \
  --config covla_probe_rjob/configs/covla/weather_ood/readout/time/covla_readout_conv.yaml

# Frozen five-horizon future-state probe.
python covla_probe_rjob/train_covla_probe.py \
  --config covla_probe_rjob/configs/future_state_1s/mix_e19/covla/readout.yaml
```

The probe configurations refer to adapted encoder aliases named `last.pth.tar` or `last.pt`. Official pretrained filenames are preserved. Adapted iVideoGPT uses `last/transformer/model.safetensors` with its accompanying config and adaptation manifest. **These are release aliases for the paper-selected checkpoints; they do not redefine checkpoint selection as the final training epoch.** See [checkpoint conventions](docs/checkpoints.md).

## Repository guide

| Location | Purpose |
|---|---|
| `app/`, `src/` | V-JEPA 2.1 training, Time-JEPA targets, losses and data handling |
| `configs/train_time_jepa/` | Selected single-domain, multi-domain and reported ablation training configurations |
| `covla_probe_rjob/` | Shared current/future-state probe implementation and configurations |
| `physics_consistency/` | Independent probes, GT-gate generation and kinematic reconstruction |
| `grasp_eval_rjob/`, `physion_ocp_rjob/` | External physical evaluations |
| `world_model_probe_rjob/` | InternVideo-Next, VideoMAE V2 and iVideoGPT adapters |
| `external_baselines/psg_jepa/` | PSG-JEPA adaptation and probing |
| `time-jepa/splits/`, `time-jepa/cache/` | Included AgiBot metadata and destinations for locally generated CoVLA/Ego-Exo4D metadata |
| `scripts/` | Data-independent validation, dependency preparation and result summaries |

Original module names are retained so the selected implementation keeps its import relationships. Cluster submission scripts and historical experiment logs are excluded.
Use [the formal experiment index](docs/experiments.md) for exact training/probe/encoder associations and [the encoder/resource catalog](docs/models.md) for required inputs. The same mappings are available as [CSV](docs/experiment-index.csv) and [JSON](docs/experiment-index.json). See [the reproduction guide](docs/reproduction.md) for execution order.

## Validation

```bash
python scripts/validate_release.py
python -m pip install -r requirements-dev.txt
python -m pytest tests covla_probe_rjob/tests
```

No data or weights are required for the static validation and retained synthetic protocol tests. Full training/evaluation additionally requires the resources listed in the documentation.

A real CoVLA/H200 integration run has also passed; see [the acceptance record](docs/acceptance.md).

## Clean release package

```bash
python scripts/validate_release.py
python scripts/package_release.py
```

This builds a manifest-selected directory and archive under `releases/`, excluding data, weights, local acceptance outputs, symlinks and Git history. See [release.md](docs/release.md).

## Provenance

`release-manifest.json` records source files, selection reasons and checksums. `third-party.json` records the inspected upstream revisions. The included license files retain upstream V-JEPA notices. Confirmed authors/repository and the pending preprint link are recorded in `publication.json`; the release workflow is described in [publication.md](docs/publication.md).

Source-data access and metadata redistribution are separate from the code license. Retained AgiBot metadata uses CC BY-NC-SA 4.0; CoVLA/Ego-Exo4D split lists, per-sample manifests and statistics are generated locally. See [DATA-LICENSES.md](DATA-LICENSES.md) and [data preparation](docs/data.md).
