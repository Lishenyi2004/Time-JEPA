from __future__ import annotations


# This order is the one released by the Meta JEPA intuitive-physics evaluator.
FAMILIES: tuple[str, ...] = (
    "Collision",
    "Continuity",
    "Gravity",
    "GravityContinuity",
    "GravityInertia",
    "GravityInertia2",
    "GravitySupport",
    "Inertia",
    "Inertia2",
    "ObjectPermanence",
    "ObjectPermanence2",
    "ObjectPermanence3",
    "SolidityContinuity",
    "SolidityContinuity2",
    "Unchangeableness",
    "Unchangeableness2",
)


# Exact multi-label mapping from facebookresearch/jepa-intuitive-physics,
# figures.ipynb (commit 302a2e821d908b4a8667e0c0e48fe733e43c9156).
# GRASP's own paper calls these eight concepts.  The later JEPA paper's Table
# S2 says "10", but its public aggregation code and per-property results use
# precisely the eight non-empty categories below.  In particular,
# Unchangeableness is reported as Color Constancy.
FAMILY_TO_PROPERTIES: dict[str, tuple[str, ...]] = {
    "Collision": ("Collision",),
    "Continuity": ("Continuity",),
    "Gravity": ("Gravity",),
    "GravityContinuity": ("Gravity", "Continuity"),
    "GravityInertia": ("Gravity", "Inertia"),
    "GravityInertia2": ("Gravity", "Inertia"),
    "GravitySupport": ("Gravity", "Support"),
    "Inertia": ("Inertia",),
    "Inertia2": ("Inertia",),
    "ObjectPermanence": ("Object Permanence",),
    "ObjectPermanence2": ("Object Permanence",),
    "ObjectPermanence3": ("Object Permanence",),
    "SolidityContinuity": ("Solidity", "Continuity"),
    "SolidityContinuity2": ("Solidity", "Continuity"),
    "Unchangeableness": ("Color Constancy",),
    "Unchangeableness2": ("Color Constancy",),
}


PROPERTIES: tuple[str, ...] = (
    "Object Permanence",
    "Continuity",
    "Color Constancy",
    "Support",
    "Inertia",
    "Gravity",
    "Solidity",
    "Collision",
)


EXPECTED_PAIRS_PER_FAMILY = 128
EXPECTED_TOTAL_PAIRS = len(FAMILIES) * EXPECTED_PAIRS_PER_FAMILY
EXPECTED_TOTAL_VIDEOS = EXPECTED_TOTAL_PAIRS * 2
EXPECTED_ARCHIVE_BYTES = 4_980_560_348


def validate_constants() -> None:
    if set(FAMILY_TO_PROPERTIES) != set(FAMILIES):
        raise RuntimeError("Family/property mapping does not cover the family list")
    mapped = {value for values in FAMILY_TO_PROPERTIES.values() for value in values}
    if mapped != set(PROPERTIES):
        raise RuntimeError(
            f"Property list and family mapping disagree: {mapped} != {set(PROPERTIES)}"
        )


validate_constants()
