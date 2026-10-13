# Data and metadata notices

The code license does not relicense source datasets, annotations or processed data. No videos, per-frame state/pose labels or processed dataset mirror is distributed in this repository. Obtain source data directly from its provider and accept the applicable agreement. This release does not publish a Hugging Face dataset.

## CoVLA

The [official CoVLA terms](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset) require prior written permission to redistribute the dataset or derivative works. CoVLA split lists, per-video weather audit, normalization caches and kinematic statistics are excluded from the public package. Generate them locally using authorized source data. The package contains processing code, fixed protocol arguments, split counts and file checksums, rather than the sample lists or source annotations.

## Ego-Exo4D

Access to videos and annotations requires the [official license agreement](https://docs.ego-exo4d-data.org/getting-started/). The [published model agreement](https://ego4d-data.org/pdfs/Ego-Exo4D-Model-License.pdf) restricts providing the database, in whole or in part, to third parties. Follow the agreement actually accepted for your source copy. Split lists, the per-take manifest, normalization caches and kinematic statistics are excluded here and must be generated locally.

## Retained AgiBot metadata

The following derived AgiBot metadata is distributed under **CC BY-NC-SA 4.0**, separately from the code license:

- `time-jepa/splits/agibot_*.json`: the selected episode/window split keys and protocol manifest.
- `time-jepa/cache/agibot_*.json`: selected normalization statistics and the fitted train-only waist PCA.
- `physics_consistency/agibot_reach_rate/results/cache/*.json`: selected kinematic-probe statistics.

Source attribution: AgiBot World Team / AGIBOT. The [official AgiBot World 2026 distribution](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) and [AgiBot World Beta distribution](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) declare CC BY-NC-SA 4.0. Use the source version required by the selected configuration; the links do not substitute a different source snapshot. Cite the corresponding original dataset when using these resources.

Time-JEPA processing changes comprise selected episode/window partitions, train-only normalization, task-balanced waist PCA fitting and kinematic target statistics. These files are derived metadata, not original video or state records, and the source provider does not endorse this project.

The [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/) requires attribution, a license link, an indication of changes, noncommercial use and the same license for distributed adaptations. Preserve this notice with redistributed metadata.

## Other evaluations

GRASP and Physion data are external inputs. Obtain them from the original sources listed in [data.md](docs/data.md); their original dataset terms continue to apply. Source and protocol revisions are recorded in `third-party.json`.

The explicit local-only file list is in `metadata-policy.json`. Data access links and notices were reviewed on 2026-10-13. Split preparation instructions are in [data.md](docs/data.md).
