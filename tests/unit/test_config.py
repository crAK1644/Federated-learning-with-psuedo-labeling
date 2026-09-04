from pathlib import Path

import pytest
from pydantic import ValidationError

from ssfl.config import (
    DawidSkeneAbstentionMode,
    DawidSkeneConfusionModel,
    ExperimentConfig,
    LabelRepresentation,
    VotingMode,
    experiment_config_from_run_config,
    load_experiment_config,
    load_yaml,
    parse_run_config_string,
)
from ssfl.protocols.dawid_skene import DawidSkeneSettings

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
ALL_PROFILES = ["paper", "paper_batch100", "smoke", "robustness", "deployment", "debug"]


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_all_profiles_validate(profile: str) -> None:
    cfg = load_experiment_config(CONFIGS_DIR / f"{profile}.yaml")
    assert cfg.profile == profile


def test_unknown_key_rejected() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    base["totally_unknown_field"] = 1
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base)


def test_dawid_skene_abstention_mode_defaults_to_missing_and_parses_explicit() -> None:
    base = load_yaml(CONFIGS_DIR / "experiment1_s3_ds_only.yaml")
    default = ExperimentConfig.model_validate(base)
    assert default.dawid_skene_abstention_mode == DawidSkeneAbstentionMode.missing
    base["dawid_skene_abstention_mode"] = "explicit"
    explicit = ExperimentConfig.model_validate(base)
    assert explicit.dawid_skene_abstention_mode == DawidSkeneAbstentionMode.explicit
    assert DawidSkeneSettings.from_config(explicit).explicit_abstention


def test_dawid_skene_one_coin_configuration_is_wired_into_settings() -> None:
    base = load_yaml(CONFIGS_DIR / "experiment1_s3_ds_only.yaml")
    base.update(
        dawid_skene_confusion_model="one_coin",
        dawid_skene_one_coin_min_accuracy=0.9,
        dawid_skene_one_coin_pseudocount=2.0,
        dawid_skene_temporal_window=2,
        dawid_skene_temporal_decay=0.8,
    )
    config = ExperimentConfig.model_validate(base)
    settings = DawidSkeneSettings.from_config(config)

    assert config.dawid_skene_confusion_model == DawidSkeneConfusionModel.one_coin
    assert settings.confusion_model == "one_coin"
    assert settings.one_coin_min_accuracy == 0.9
    assert settings.one_coin_pseudocount == 2.0
    assert settings.temporal_window == 2
    assert settings.temporal_decay == 0.8


def test_dawid_skene_one_coin_rejects_class_conditional_abstention() -> None:
    base = load_yaml(CONFIGS_DIR / "experiment1_s3_ds_only.yaml")
    base.update(
        dawid_skene_confusion_model="one_coin",
        dawid_skene_abstention_mode="explicit",
    )

    with pytest.raises(ValidationError, match="requires.*missing"):
        ExperimentConfig.model_validate(base)


def test_temporal_dawid_skene_requires_one_coin_parameterization() -> None:
    base = load_yaml(CONFIGS_DIR / "experiment1_s3_ds_only.yaml")
    base["dawid_skene_temporal_window"] = 2

    with pytest.raises(ValidationError, match="requires.*one_coin"):
        ExperimentConfig.model_validate(base)


@pytest.mark.parametrize(
    "label_repr,voting",
    [
        (LabelRepresentation.hard, VotingMode.disabled),
        (LabelRepresentation.soft, VotingMode.enabled),
    ],
)
def test_invalid_label_voting_combination_rejected(label_repr, voting) -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    base["ssfl_label_representation"] = label_repr.value
    base["ssfl_voting_mode"] = voting.value
    if label_repr == LabelRepresentation.soft:
        base["ssfl_soft_label_round_decimals"] = 4
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base)


def test_soft_label_requires_valid_rounding() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    base["ssfl_label_representation"] = "soft"
    base["ssfl_voting_mode"] = "disabled"
    base["ssfl_soft_label_round_decimals"] = 3  # not in {2,4,6,8}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base)


def test_bad_scenario_rejected() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    base["scenario"] = 4
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base)


def test_run_config_string_overrides_profile() -> None:
    overrides = parse_run_config_string("algorithm=fl scenario=2 num-server-rounds=5 seed=99")
    cfg = load_experiment_config(CONFIGS_DIR / "smoke.yaml", overrides=overrides)
    assert cfg.algorithm.value == "fl"
    assert cfg.scenario.value == 2
    assert cfg.num_server_rounds == 5
    assert cfg.seed == 99


def test_empty_resume_default_is_treated_as_none() -> None:
    cfg = experiment_config_from_run_config(
        {
            "profile": "smoke",
            "algorithm": "ssfl",
            "scenario": 1,
            "device": "cpu",
            "resume-from": "",
        }
    )
    assert cfg.resume_from is None
    assert cfg.data_path == CONFIGS_DIR.parent / "artifacts/data"
    assert cfg.output_path == CONFIGS_DIR.parent / "artifacts/runs"


def test_num_clients_matches_scenario() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    for scenario, expected in [(1, 27), (2, 89), (3, 89)]:
        base["scenario"] = scenario
        cfg = ExperimentConfig.model_validate(base)
        assert cfg.num_clients() == expected


def test_config_hash_stable_and_excludes_paths() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    cfg_a = ExperimentConfig.model_validate(base)
    base2 = dict(base)
    base2["output_path"] = "/some/other/path"
    cfg_b = ExperimentConfig.model_validate(base2)
    assert cfg_a.config_hash() == cfg_b.config_hash()


def test_paper_text_batch_overrides_effective_batch_size() -> None:
    cfg = load_experiment_config(CONFIGS_DIR / "paper.yaml")
    assert cfg.effective_batch_size == 80
    cfg100 = load_experiment_config(CONFIGS_DIR / "paper_batch100.yaml")
    assert cfg100.effective_batch_size == 100


def test_backbone_restricted_to_ssfl() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    base["algorithm"] = "fl"
    base["backbone"] = "mlp"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base)
