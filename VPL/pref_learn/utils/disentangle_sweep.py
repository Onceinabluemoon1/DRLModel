"""Disentanglement sweep metadata: model_type -> swept hyperparameter."""

from pathlib import Path

# model_type -> (flag name, plot label)
MODEL_SWEEP_PARAMS: dict[str, tuple[str, str]] = {
    "VAE": ("kl_weight", "β_KL"),
    "TCVAE": ("tc_weight", "β_TC"),
    "QLAE": ("weight_decay", "λ_wd"),
}

SWEEP_SUBRUNS_DIRNAME = "subruns"
SWEEP_NON_SUBRUN_DIRNAMES = frozenset(
    {
        SWEEP_SUBRUNS_DIRNAME,
        "evals_sweep",
        "evals_latentgrid",
        "evals_rewardmap",
        "evals_individual",
        "evals_subsweep",
    }
)


def enrich_variant_sweep_metadata(variant: dict) -> dict:
    """Write ``sweep_param_{name,value,label}`` into ``variant`` (train time)."""
    entry = MODEL_SWEEP_PARAMS.get(str(variant.get("model_type", "")))
    if entry is None:
        return variant
    name, label = entry
    variant["sweep_param_name"] = name
    variant["sweep_param_value"] = float(variant.get(name, 0.0))
    variant["sweep_param_label"] = label
    return variant


def sweep_from_variant(variant: dict) -> tuple[str, float, str]:
    """Return ``(param_name, param_value, param_label)`` for plots / sorting."""
    if "sweep_param_name" in variant:
        name = str(variant["sweep_param_name"])
        return (
            name,
            float(variant.get("sweep_param_value", variant.get(name, 0.0))),
            str(variant.get("sweep_param_label", name)),
        )
    entry = MODEL_SWEEP_PARAMS.get(str(variant.get("model_type", "")))
    if entry is None:
        name = "kl_weight"
        return name, float(variant.get(name, 0.0)), name
    name, label = entry
    return name, float(variant.get(name, 0.0)), label


def sweep_subruns_dir(*, sweep_root: Path) -> Path:
    """``<sweep_root>/subruns/`` — one training checkpoint dir per array task."""
    return sweep_root.expanduser().resolve() / SWEEP_SUBRUNS_DIRNAME


def is_sweep_subrun_dir(path: Path) -> bool:
    return path.is_dir() and (path / "variant.json").is_file()


def discover_sweep_subruns(sweep_root: Path) -> list[Path]:
    """List subrun dirs under ``subruns/``; fall back to legacy flat layout."""
    root = sweep_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Sweep root is not a directory: {root}")

    nested = sweep_subruns_dir(sweep_root=root)
    if nested.is_dir():
        return sorted(p for p in nested.iterdir() if is_sweep_subrun_dir(p))

    return sorted(
        p
        for p in root.iterdir()
        if is_sweep_subrun_dir(p) and p.name not in SWEEP_NON_SUBRUN_DIRNAMES
    )
