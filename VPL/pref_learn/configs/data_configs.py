"""Named pref-learning datasets for VAE training sweeps."""

TRAIN_VAE_DATASETS = {
    # * sort 2 goals task
    # updated with obs=15 to include 2 zone xy
    "sort_2goals_obs=15_T=1_Q=20000": {
        "env": "sort-easy-2goals-v0",
        "pref_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-2goals-v0/20260627_005939_ntrajs=5000_obs=15/pref_datasets_20260627_022915_Q=20000_S=16_T=1/queries_relabelled.pkl",
        "pref_infomec_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-2goals-v0/20260627_005939_ntrajs=5000_obs=15/pref_datasets_20260627_022915_Q=20000_S=16_T=1/queries_relabelled.pkl",
    },
    "sort_2goals_obs=15_T=25_Q=20000": {
        "env": "sort-easy-2goals-v0",
        "pref_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-2goals-v0/20260627_005939_ntrajs=5000_obs=15/pref_datasets_20260627_051751_Q=20000_S=16_T=25/queries_relabelled.pkl",
        "pref_infomec_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-2goals-v0/20260627_005939_ntrajs=5000_obs=15/pref_datasets_20260627_051751_Q=20000_S=16_T=25/queries_relabelled.pkl",
    },
    # * sort 4 goals task
    # updated with obs=15 to include 4 zone xy
    "sort_4goals_obs=15_T=1_Q=20000": {
        "env": "sort-easy-4goals-v0",
        "pref_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-4goals-v0/20260627_005919_ntrajs=5000_obs=15/pref_datasets_20260627_022822_Q=20000_S=16_T=1/queries_relabelled.pkl",
        "pref_infomec_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-4goals-v0/20260627_005919_ntrajs=5000_obs=15/pref_datasets_20260627_022822_Q=20000_S=16_T=1/queries_relabelled.pkl",
    },
    "sort_4goals_obs=15_T=25_Q=20000": {
        "env": "sort-easy-4goals-v0",
        "pref_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-4goals-v0/20260627_005919_ntrajs=5000_obs=15/pref_datasets_20260627_051733_Q=20000_S=16_T=25/queries_relabelled.pkl",
        "pref_infomec_dataset_path": "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-4goals-v0/20260627_005919_ntrajs=5000_obs=15/pref_datasets_20260627_051733_Q=20000_S=16_T=25/queries_relabelled.pkl",
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
