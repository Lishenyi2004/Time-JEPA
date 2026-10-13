# Dependencies and upstream sources

`requirements.txt` is the exact pip package/version export of the existing `jepa` environment. It is intentionally not replaced with an inferred minimal dependency set. Python is 3.11; PyTorch/torchvision use CUDA 12.8 wheels. Install FFmpeg as a system executable for timestamp decoding.

Some additional-backbone and PSG experiments originally used separate environments. Their extra dependencies are documented separately and do not change the `jepa` snapshot. iVideoGPT needs a compatible `diffusers` installation; the inspected setup used `diffusers==0.39.0`. PSG training imports `lightning`, `omegaconf` and `stable_pretraining`, together with the pinned PSG-JEPA source. The observed extra versions are recorded in `requirements-ivideogpt.txt` and `requirements-psg.txt`. Install the relevant file in a separate environment with the matching pinned source; retain any additional upstream installation requirements.

No third-party source tree has been copied. The local adapters expect these source directories:

```bash
# Download selected source dependencies only. This does not fetch data or weights.
python scripts/setup_third_party.py VideoMAEv2 iVideoGPT PSG-JEPA
```

The script clones into the locations recorded in `third-party.json`, checks out the exact inspected revisions, and refuses to use an existing checkout at a different revision. It has not been executed as part of this preparation.

InternVideo-Next loads its custom modeling Python files from the full pinned Hugging Face checkpoint snapshot, so a standalone InternVideo checkout is not required by this adapter. Preserve snapshot metadata and configuration files alongside the weights. iVideoGPT likewise needs the full official tokenizer/transformer snapshot, not an isolated weight file. Snapshot model IDs/revisions are recorded in `third-party.json`; obtain weights separately.

GRASP and Physion evaluation implementations here run independently of their upstream repositories. Upstream repository/revision records are included as protocol provenance. Third-party installation and notices remain with the corresponding upstream projects.
