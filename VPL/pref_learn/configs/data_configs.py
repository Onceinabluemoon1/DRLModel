"""Named pref-learning datasets for VAE training sweeps."""

TRAIN_VAE_DATASETS = {
    "sort_2goals_T=1": {
        "env": "sort-easy-2goals-v0",
        "pref_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-2goals-v0/_vpl_official/pref_datasets_20260518_143219_Q=5000_S=16_T=1/queries_relabelled.pkl",
        "pref_infomec_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-2goals-v0/_vpl_official/pref_datasets_20260523_214225_Q=5000_S=16_T=1/queries_relabelled.pkl",
    },
    "sort_2goals_T=10": {
        "env": "sort-easy-2goals-v0",
        "pref_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-2goals-v0/_vpl_official/pref_datasets_20260518_143030_Q=5000_S=16_T=10/queries_relabelled.pkl",
        "pref_infomec_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-2goals-v0/_vpl_official/pref_datasets_20260523_214500_Q=5000_S=16_T=10/queries_relabelled.pkl",
    },
    "sort_4goals_T=1": {
        "env": "sort-easy-4goals-v0",
        "pref_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-4goals-v0/20260517_212722_ntrajs=5000/pref_datasets_20260518_143204_Q=5000_S=16_T=1/queries_relabelled.pkl",
        "pref_infomec_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-4goals-v0/20260517_212722_ntrajs=5000/pref_datasets_20260523_214211_Q=5000_S=16_T=1/queries_relabelled.pkl",
    },
    "sort_4goals_T=10": {
        "env": "sort-easy-4goals-v0",
        "pref_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-4goals-v0/20260517_212722_ntrajs=5000/pref_datasets_20260518_142917_Q=5000_S=16_T=10/queries_relabelled.pkl",
        "pref_infomec_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-4goals-v0/20260517_212722_ntrajs=5000/pref_datasets_20260523_214433_Q=5000_S=16_T=10/queries_relabelled.pkl",
    },
}


def get_train_vae_dataset(*, pref_dataset: str) -> dict:
    if pref_dataset not in TRAIN_VAE_DATASETS:
        known = ", ".join(sorted(TRAIN_VAE_DATASETS))
        raise ValueError(f"Unknown pref_dataset {pref_dataset!r}. Known: {known}")
    return TRAIN_VAE_DATASETS[pref_dataset]


def pref_dataset_name_from_variant(*, variant: dict) -> str:
    pref_dataset = str(variant.get("pref_dataset", "")).strip()
    if not pref_dataset:
        known = ", ".join(sorted(TRAIN_VAE_DATASETS))
        raise ValueError(
            "variant.json missing pref_dataset; cannot resolve dataset paths from "
            f"data_configs. Known keys: {known}"
        )
    return pref_dataset


def pref_dataset_path_for_variant(*, variant: dict) -> str:
    pref_dataset = pref_dataset_name_from_variant(variant=variant)
    return get_train_vae_dataset(pref_dataset=pref_dataset)["pref_dataset_path"]


def infomec_dataset_path_for_variant(*, variant: dict) -> str:
    pref_dataset = pref_dataset_name_from_variant(variant=variant)
    return get_train_vae_dataset(pref_dataset=pref_dataset)["pref_infomec_dataset_path"]
