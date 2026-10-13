# Preparation validation

Validation ran in the staged Time-JEPA directory with the existing jepa Python 3.11 interpreter. The test runner was supplied read-only from an already installed pytest in another environment; jepa itself was not modified. For a new installation, use requirements-dev.txt.

- Retained scientific/protocol test suite: **68 passed**, with one upstream timm deprecation warning.
- Python compilation, local package imports, configuration inheritance and release-file checksums: passed.
- All 196 published YAMLs parse (195 formal experiment configurations plus one public quickstart); 118 adapted-encoder aliases match their configurations.
- Original-to-release YAML comparison: no changes to non-path scientific values. The six generated eight-horizon YAMLs materialize the documented launcher overrides.
- Ten command-line entry points load and display help: main training, shared probes, GRASP evaluation/scoring, Physion extraction/probing, InternVideo/VideoMAE probing, frozen iVideoGPT probing, robot reach-rate probing and human knee reconstruction.
- Formal AdamW initialization with small synthetic modules: passed. The unused upstream RAdamW branch references an absent source file; the release rejects that unsupported option explicitly while preserving the selected AdamW path.
- Checksums of all inspected original source/configuration/metadata files remained unchanged.
- Public quickstart: matches the accepted small-sample YAML except the output directory; runner argument parsing and external-path overrides pass. Command preview works from outside the repository, and resource checks report missing data/weights in the clean package without creating outputs.
- Clean preprint package: 439 files exactly match the manifest plus the manifest itself, with no symlinks, raw data, weights, runtime outputs or Git history. CoVLA/Ego-Exo4D split/manifests/statistics are excluded by the metadata policy; retained AgiBot metadata carries a separate license notice. Packaged validation passes; rebuilding unchanged contents produces the same archive checksum. The packager rejects published symlinks, checksum mismatches, excluded runtime paths and local-only metadata.
- Metadata preparation: both fixed recipe previews work without data or writes. Verification passes for all 12 original CoVLA/Ego-Exo4D split lists using the ignored local copies; missing splits fail with explicit guidance. Full split regeneration and new GPU execution were not performed during this publication update.
- Publication metadata: the five confirmed authors appear in the supplied order in `publication.json` and `CITATION.cff`; the GitHub URL is set, the arXiv URL remains unassigned, and the manuscript is recorded as under review rather than accepted.

The original preparation checks did not run real weights. A subsequent small real-data GPU run now verifies encoder loading, probe training/validation and ID/OOD evaluation; see `acceptance.md`. Full paper-scale training/evaluation, optional-backbone dependency installation and third-party source download have not been performed. Existing raw data and encoder weights remain externally linked; newly generated probe weights and metrics are local ignored acceptance outputs. URDF and frozen GT-gate calibration resources remain external.
