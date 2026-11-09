"""Shared conventions for the Bernoulli posterior and saved numerical results."""

from __future__ import annotations

from typing import Any, Mapping


INFERENCE_MODEL = {
    "target_distribution": "bernoulli_posterior",
    "rho_used_as_inference_prior": True,
    "fixed_total_mines": False,
}
OVERLAP_CONVENTION = {
    "definition": "matching_fraction",
    "variable_set": "unobserved_cells",
    "empty_value": 1.0,
}
RUN_SCHEMA_VERSION = 4


def validate_model_options(data: Mapping[str, Any]) -> None:
    """Reject explicitly requested alternatives instead of silently ignoring them.

    Omitted options use the current model, so existing input configurations
    without model metadata can still be used for a fresh calculation.
    """

    for section, expected in (
        ("inference", INFERENCE_MODEL), ("overlap", OVERLAP_CONVENTION)
    ):
        options = data.get(section, {})
        if not isinstance(options, Mapping):
            raise ValueError(f"{section} must be an object")
        for key, value in options.items():
            if key not in expected or value != expected[key]:
                raise ValueError(f"unsupported {section} option: {key}={value!r}")


def validate_run_model(record: Mapping[str, Any]) -> None:
    """Prevent reuse or analysis of results with a different inference/overlap."""

    overlap = record.get("overlap", {})
    if (
        record.get("schema_version") != RUN_SCHEMA_VERSION
        or record.get("inference") != INFERENCE_MODEL
        or not isinstance(overlap, Mapping)
        or any(overlap.get(key) != value for key, value in OVERLAP_CONVENTION.items())
    ):
        raise ValueError(
            "saved run uses an incompatible CSP or overlap convention; "
            "recompute with the current model in a new output directory"
        )
