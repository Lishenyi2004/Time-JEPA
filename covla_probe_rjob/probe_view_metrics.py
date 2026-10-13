"""View/target aggregation helpers for probe reports."""

from numbers import Real


AGIBOT_FISHEYE_VIEW_SIDES = {
    "observation.images.head_left_fisheye": "left",
    "observation.images.head_right_fisheye": "right",
}

_DIMENSIONLESS_REGRESSION_METRICS = (
    "nrmse",
    "pearson",
    "ccc",
    "r2",
    "skill_train_mean",
)


def _macro_average_values(values):
    if all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
        return sum(float(value) for value in values) / len(values)
    if all(isinstance(value, list) for value in values):
        lengths = {len(value) for value in values}
        if len(lengths) != 1:
            return None
        averaged = [
            _macro_average_values([value[index] for value in values])
            for index in range(next(iter(lengths)))
        ]
        return averaged if all(value is not None for value in averaged) else None
    if all(isinstance(value, dict) for value in values):
        common_keys = set.intersection(*(set(value) for value in values))
        averaged = {}
        for key in sorted(common_keys):
            result = _macro_average_values([value[key] for value in values])
            if result is not None:
                averaged[key] = result
        return averaged
    return None


def macro_average_view_metrics(metrics_by_view):
    """Recursively macro-average numeric metrics across camera views."""
    if not metrics_by_view:
        return {}
    metric_names = set.intersection(
        *(set(metrics) for metrics in metrics_by_view.values())
    )
    macro = {}
    for metric_name in sorted(metric_names):
        result = _macro_average_values(
            [metrics[metric_name] for metrics in metrics_by_view.values()]
        )
        if result is not None:
            macro[metric_name] = result
    return macro


def _bilateral_target_indices(target_fields):
    indices = {"left": [], "right": []}
    for index, target_field in enumerate(target_fields):
        matching_sides = [
            side for side in indices if str(target_field).endswith(f"_{side}")
        ]
        if len(matching_sides) != 1:
            raise ValueError(
                "AgiBot same-side Shot OOD requires every target field to end in "
                f"exactly one of '_left'/'_right'; got {target_field!r}"
            )
        indices[matching_sides[0]].append(index)
    if not indices["left"] or len(indices["left"]) != len(indices["right"]):
        raise ValueError(
            "AgiBot same-side Shot OOD requires non-empty, balanced left/right "
            f"targets; got left={len(indices['left'])}, right={len(indices['right'])}"
        )
    return indices


def validate_agibot_same_side_ood(target_fields, ood_view_keys):
    """Validate the visibility-aligned AgiBot Shot OOD protocol before a run."""
    target_indices = _bilateral_target_indices(target_fields)
    actual_views = set(ood_view_keys)
    expected_views = set(AGIBOT_FISHEYE_VIEW_SIDES)
    if actual_views != expected_views:
        raise ValueError(
            "AgiBot same-side Shot OOD requires exactly head_left_fisheye and "
            "head_right_fisheye; got " + ", ".join(sorted(actual_views))
        )
    return target_indices


def _target_vector_metrics(view_metrics, target_count):
    vector_metrics = {}
    for metric_name, values in view_metrics.items():
        if not isinstance(values, (list, tuple)) or len(values) != target_count:
            continue
        if not all(
            isinstance(value, Real) and not isinstance(value, bool)
            for value in values
        ):
            continue
        vector_metrics[metric_name] = [float(value) for value in values]
    return vector_metrics


def _build_selected_target_report(
    metrics_by_view,
    target_fields,
    target_names,
    target_indices_by_view,
    target_side_by_view,
):
    views = {}
    macro_values = {
        metric_name: [] for metric_name in _DIMENSIONLESS_REGRESSION_METRICS
    }
    target_count = len(target_fields)
    for view_key in AGIBOT_FISHEYE_VIEW_SIDES:
        indices = list(target_indices_by_view[view_key])
        vector_metrics = _target_vector_metrics(metrics_by_view[view_key], target_count)
        selected_metrics = {
            metric_name: [values[index] for index in indices]
            for metric_name, values in vector_metrics.items()
        }
        for metric_name in macro_values:
            macro_values[metric_name].extend(selected_metrics.get(metric_name, []))
        views[view_key] = {
            "camera_side": AGIBOT_FISHEYE_VIEW_SIDES[view_key],
            "target_side": target_side_by_view[view_key],
            "target_indices": indices,
            "target_fields": [target_fields[index] for index in indices],
            "targets": [target_names[index] for index in indices],
            "metrics": selected_metrics,
        }
    macro = {
        metric_name: sum(values) / len(values)
        for metric_name, values in macro_values.items()
        if values
    }
    return {"views": views, "macro": macro}


def build_agibot_shot_ood_report(metrics_by_view, target_fields, target_names=None):
    """Build primary same-side and retained hard-OOD summaries for bilateral Shot.

    The left fisheye is paired with left-arm targets and the right fisheye with
    right-arm targets. Full-8D and cross-side aggregates remain available as
    diagnostics, so changing the primary reporting target does not erase the
    original, harder evidence.
    """
    target_fields = list(target_fields)
    target_names = list(target_fields if target_names is None else target_names)
    if len(target_names) != len(target_fields):
        raise ValueError(
            "target_names and target_fields must have the same length for AgiBot "
            "same-side Shot OOD reporting"
        )
    target_indices = validate_agibot_same_side_ood(
        target_fields,
        metrics_by_view.keys(),
    )
    all_indices = list(range(len(target_fields)))
    same_side_indices = {
        view_key: target_indices[view_side]
        for view_key, view_side in AGIBOT_FISHEYE_VIEW_SIDES.items()
    }
    cross_side_indices = {
        view_key: target_indices["right" if view_side == "left" else "left"]
        for view_key, view_side in AGIBOT_FISHEYE_VIEW_SIDES.items()
    }
    full_indices = {view_key: all_indices for view_key in AGIBOT_FISHEYE_VIEW_SIDES}
    same_side_targets = dict(AGIBOT_FISHEYE_VIEW_SIDES)
    cross_side_targets = {
        view_key: "right" if view_side == "left" else "left"
        for view_key, view_side in AGIBOT_FISHEYE_VIEW_SIDES.items()
    }
    full_targets = {view_key: "both" for view_key in AGIBOT_FISHEYE_VIEW_SIDES}
    return {
        "protocol": "agibot_shot_same_side_v1",
        "scope": "current-state bilateral ordinary Shot OOD only",
        "primary": {
            "name": "same_side",
            "description": (
                "left fisheye -> four left-arm targets; right fisheye -> "
                "four right-arm targets"
            ),
            **_build_selected_target_report(
                metrics_by_view,
                target_fields,
                target_names,
                same_side_indices,
                same_side_targets,
            ),
        },
        "diagnostics": {
            "full_8d_hard": {
                "description": "both fisheye views evaluated on all eight bilateral targets",
                **_build_selected_target_report(
                    metrics_by_view,
                    target_fields,
                    target_names,
                    full_indices,
                    full_targets,
                ),
            },
            "cross_side_hard": {
                "description": (
                    "left fisheye -> right-arm targets; right fisheye -> left-arm targets"
                ),
                **_build_selected_target_report(
                    metrics_by_view,
                    target_fields,
                    target_names,
                    cross_side_indices,
                    cross_side_targets,
                ),
            },
        },
    }
