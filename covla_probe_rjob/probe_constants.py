"""Constants and architecture/target tables for the CoVLA probe."""

import probe_paths  # noqa: F401  (ensures REPO_ROOT is on sys.path before app/src imports)

from app.vjepa_2_1.models.vision_transformer import (
    vit_giant_xformers_rope as vit_giant_xformers_rope_21,
)
from app.vjepa_2_1.models.vision_transformer import vit_large_rope as vit_large_rope_21
from src.models.vision_transformer import (
    vit_giant_xformers_rope as vit_giant_xformers_rope_legacy,
)
from src.models.vision_transformer import vit_large_rope as vit_large_rope_legacy


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

ARCH_SPECS = {
    "vitg_384": {
        "builder": vit_giant_xformers_rope_21,
        "legacy_builder": vit_giant_xformers_rope_legacy,
        "img_size": 384,
        "num_frames": 64,
        "embed_dim": 1408,
        "num_heads": 22,
    },
    "vitl_256": {
        "builder": vit_large_rope_21,
        "legacy_builder": vit_large_rope_legacy,
        "img_size": 256,
        "num_frames": 16,
        "embed_dim": 1024,
        "num_heads": 16,
    },
}

TARGET_ALIASES = {
    "yaw_rate": "angular_velocities_calib[2]",
    "omega_z": "angular_velocities_calib[2]",
    "x_1s": "trajectory_x_1s",
    "traj_x_1s": "trajectory_x_1s",
}

# CoVLA trajectory points are sampled at roughly 20 Hz. trajectory[20][0]
# is therefore the longitudinal displacement about 1.0 s after the current
# state frame.
DERIVED_COVLA_TARGETS = {
    "trajectory_x_1s": {
        "source": "trajectory",
        "point_index": 20,
        "axis_index": 0,
        "horizon_seconds": 1.0,
        "display_name": "trajectory_x_1s",
    },
}

KITTI_TARGET_ALIASES = {
    "vEgo": "vf",
    "aEgo": "af",
    "yaw_rate": "wz",
    "omega_z": "wz",
}

KITTI_OXTS_FIELDS = (
    "lat",
    "lon",
    "alt",
    "roll",
    "pitch",
    "yaw",
    "vn",
    "ve",
    "vf",
    "vl",
    "vu",
    "ax",
    "ay",
    "az",
    "af",
    "al",
    "au",
    "wx",
    "wy",
    "wz",
    "wf",
    "wl",
    "wu",
    "posacc",
    "velacc",
    "navstat",
    "numsats",
    "posmode",
    "velmode",
    "orimode",
)
