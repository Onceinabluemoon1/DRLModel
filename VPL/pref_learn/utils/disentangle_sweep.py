"""Disentanglement sweep metadata: model_type -> swept hyperparameter."""

# model_type -> (flag name, plot label)
MODEL_SWEEP_PARAMS: dict[str, tuple[str, str]] = {
    "VAE": ("kl_weight", "β_KL"),
    "VAEClassifier": ("kl_weight", "β_KL"),
    "TCVAE": ("tc_weight", "β_TC"),
}


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
