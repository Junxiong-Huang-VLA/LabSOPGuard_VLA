from __future__ import annotations

from key_action_indexer.pipeline import _fast_locate_expected_experiment_count
from key_action_indexer.schemas import DetectionConfig


def test_expected_experiment_count_ignores_env_by_default(monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_EXPECTED_EXPERIMENT_COUNT", "6")
    monkeypatch.delenv("KEY_ACTION_ALLOW_EXPECTED_EXPERIMENT_COUNT_ENV", raising=False)

    assert _fast_locate_expected_experiment_count(DetectionConfig()) is None


def test_expected_experiment_count_uses_explicit_config(monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_EXPECTED_EXPERIMENT_COUNT", "6")
    config = DetectionConfig(expected_experiment_count=4)

    assert _fast_locate_expected_experiment_count(config) == 4


def test_expected_experiment_count_env_requires_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_EXPECTED_EXPERIMENT_COUNT", "6")
    monkeypatch.setenv("KEY_ACTION_ALLOW_EXPECTED_EXPERIMENT_COUNT_ENV", "1")

    assert _fast_locate_expected_experiment_count(DetectionConfig()) == 6
