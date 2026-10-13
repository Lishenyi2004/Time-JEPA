"""Dependency-light URDF forward kinematics and train-only frame calibration."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ARM_CHAINS = {
    "left": tuple(f"idx2{index}_arm_l_joint{index}" for index in range(1, 8)),
    "right": tuple(f"idx6{index}_arm_r_joint{index}" for index in range(1, 8)),
}
END_JOINTS = {"left": "arm_l_end_joint", "right": "arm_r_end_joint"}


def _parse_vector(text: str | None, width: int = 3) -> np.ndarray:
    if text is None:
        return np.zeros(width, dtype=np.float64)
    values = np.fromstring(str(text), sep=" ", dtype=np.float64)
    if values.shape != (width,):
        raise ValueError(f"Expected {width} numbers, got {text!r}")
    return values


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def homogeneous(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def axis_angle_batch(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("Revolute joint axis must be non-zero")
    axis = axis / norm
    x, y, z = axis
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    skew_sq = skew @ skew
    angles = np.asarray(angles, dtype=np.float64).reshape(-1)
    sine = np.sin(angles)[:, None, None]
    cosine = np.cos(angles)[:, None, None]
    eye = np.eye(3, dtype=np.float64)[None, :, :]
    return eye + sine * skew[None, :, :] + (1.0 - cosine) * skew_sq[None, :, :]


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray | None
    lower: float | None
    upper: float | None


class G2ForwardKinematics:
    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        root = ET.parse(self.urdf_path).getroot()
        parsed = {}
        for element in root.findall("joint"):
            name = element.attrib["name"]
            origin_element = element.find("origin")
            xyz = _parse_vector(origin_element.attrib.get("xyz") if origin_element is not None else None)
            rpy = _parse_vector(origin_element.attrib.get("rpy") if origin_element is not None else None)
            origin = homogeneous(rpy_matrix(rpy), xyz)
            axis_element = element.find("axis")
            axis = None if axis_element is None else _parse_vector(axis_element.attrib.get("xyz"))
            limit_element = element.find("limit")
            lower = None if limit_element is None or "lower" not in limit_element.attrib else float(limit_element.attrib["lower"])
            upper = None if limit_element is None or "upper" not in limit_element.attrib else float(limit_element.attrib["upper"])
            parsed[name] = JointSpec(
                name=name,
                joint_type=element.attrib.get("type", "fixed"),
                origin=origin,
                axis=axis,
                lower=lower,
                upper=upper,
            )
        required = set(sum((list(value) for value in ARM_CHAINS.values()), [])) | set(END_JOINTS.values())
        missing = sorted(required - set(parsed))
        if missing:
            raise ValueError(f"URDF is missing required G2 arm joints: {missing}")
        self.joints = parsed

    def joint_limits(self, arm: str) -> np.ndarray:
        self._check_arm(arm)
        return np.asarray(
            [[self.joints[name].lower, self.joints[name].upper] for name in ARM_CHAINS[arm]],
            dtype=np.float64,
        )

    def forward(self, q: np.ndarray, arm: str, include_end: bool = True) -> np.ndarray:
        self._check_arm(arm)
        q = np.asarray(q, dtype=np.float64)
        original_shape = q.shape[:-1]
        if q.ndim < 1 or q.shape[-1] != 7:
            raise ValueError(f"Expected q[...,7], got {q.shape}")
        flat = q.reshape(-1, 7)
        batch = flat.shape[0]
        transform = np.broadcast_to(np.eye(4, dtype=np.float64), (batch, 4, 4)).copy()
        for index, name in enumerate(ARM_CHAINS[arm]):
            joint = self.joints[name]
            if joint.joint_type not in {"revolute", "continuous"} or joint.axis is None:
                raise ValueError(f"Expected revolute G2 arm joint, got {joint}")
            rotation = axis_angle_batch(joint.axis, flat[:, index])
            motion = np.broadcast_to(np.eye(4, dtype=np.float64), (batch, 4, 4)).copy()
            motion[:, :3, :3] = rotation
            transform = transform @ joint.origin[None, :, :] @ motion
        if include_end:
            transform = transform @ self.joints[END_JOINTS[arm]].origin[None, :, :]
        return transform[:, :3, 3].reshape(*original_shape, 3)

    @staticmethod
    def _check_arm(arm: str) -> None:
        if arm not in ARM_CHAINS:
            raise ValueError(f"arm must be one of {tuple(ARM_CHAINS)}, got {arm!r}")


def fit_rigid_alignment(source: np.ndarray, target: np.ndarray) -> dict:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    source, target = source[finite], target[finite]
    if len(source) < 3:
        raise ValueError("At least three finite points are required for rigid alignment")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise RuntimeError("Rigid alignment produced an improper rotation")
    return {
        "rotation": rotation,
        "translation": translation,
        "fit_count": int(len(source)),
    }


def apply_rigid_alignment(points: np.ndarray, calibration: dict) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(calibration["rotation"], dtype=np.float64)
    translation = np.asarray(calibration["translation"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Invalid rigid calibration shapes")
    return np.einsum("ij,...j->...i", rotation, points) + translation


def position_error_summary(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(prediction), axis=1) & np.all(np.isfinite(target), axis=1)
    prediction, target = prediction[finite], target[finite]
    if not len(prediction):
        return {"count": 0}
    delta = prediction - target
    distance = np.linalg.norm(delta, axis=1)
    denominator = np.sum((target - target.mean(axis=0)) ** 2)
    r2 = float(1.0 - np.sum(delta**2) / denominator) if denominator > 0 else float("nan")
    return {
        "count": int(len(distance)),
        "coordinate_rmse_m": float(np.sqrt(np.mean(delta**2))),
        "mean_euclidean_m": float(np.mean(distance)),
        "rmse_euclidean_m": float(np.sqrt(np.mean(distance**2))),
        "p50_euclidean_m": float(np.quantile(distance, 0.50)),
        "p90_euclidean_m": float(np.quantile(distance, 0.90)),
        "p95_euclidean_m": float(np.quantile(distance, 0.95)),
        "p99_euclidean_m": float(np.quantile(distance, 0.99)),
        "global_r2": r2,
    }


def save_calibration(calibration: dict, path: str | Path) -> None:
    payload = {
        arm: {
            "rotation": np.asarray(value["rotation"], dtype=np.float64).tolist(),
            "translation": np.asarray(value["translation"], dtype=np.float64).tolist(),
            "fit_count": int(value.get("fit_count", 0)),
        }
        for arm, value in calibration.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_calibration(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    result = {}
    for arm in ("left", "right"):
        if arm not in payload:
            raise ValueError(f"Calibration is missing arm {arm!r}")
        result[arm] = {
            "rotation": np.asarray(payload[arm]["rotation"], dtype=np.float64),
            "translation": np.asarray(payload[arm]["translation"], dtype=np.float64),
            "fit_count": int(payload[arm].get("fit_count", 0)),
        }
    return result
