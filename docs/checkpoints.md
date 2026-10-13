# Checkpoint conventions

No weights were copied into this directory. Official pretrained names remain unchanged; adapted encoder filenames are release aliases:

- V-JEPA: `e17.pth.tar`, `e19.pth.tar`, etc. become `last.pth.tar` in their original run directory.
- PSG-JEPA: the exported `encoder_final.pt` becomes `last.pt`. Do not substitute the Lightning trainer state `last.ckpt`; it is a different format.
- iVideoGPT: `checkpoints/epoch_019/transformer/model.safetensors` becomes `checkpoints/last/transformer/model.safetensors`. Keep `transformer/config.json` and the parent `adaptation_manifest.json` together.
- Fitted probe files such as `best_probe.pt` retain their names.

The aliases identify the **same saved weights selected for the paper**, including loss-selected single-domain checkpoints and fixed endpoints where specified. Renaming does not select a newer epoch or switch online/EMA encoder keys. `checkpoint-aliases.json` retains every configuration's original selection.

The V-JEPA trainer still writes `latest.pth.tar` and epoch snapshots. After training, select the appropriate snapshot and expose it under the configured `last.pth.tar` alias. The iVideoGPT trainer similarly retains epoch directories; create the `last` directory alias after selection. This preparation stage does not change training algorithms or automatic selection behavior.

iVideoGPT validates `adaptation_manifest.json`, including mode, epoch and base snapshot. When relocating externally supplied weights, update path metadata to the new base snapshot location while retaining the scientific fields and weights unchanged.

GRASP YAMLs retain checkpoint checksums. A renamed copy of the same weights has the same checksum; do not replace these checkpoints with a different epoch under the same alias.
