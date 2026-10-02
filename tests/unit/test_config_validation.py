from __future__ import annotations

import pytest

from gridworld_rl.config import ExperimentConfig


@pytest.mark.parametrize("value", [None, 8, "8", {"size": 8}, True])
def test_hidden_sizes_rejects_non_sequences(value):
    with pytest.raises(ValueError, match=r"network\.hidden_sizes"):
        ExperimentConfig.from_dict({"network": {"hidden_sizes": value}})


@pytest.mark.parametrize("value", [None, [], "config"])
def test_from_dict_rejects_non_object_roots(value):
    with pytest.raises(ValueError, match="configuration root"):
        ExperimentConfig.from_dict(value)


def test_configuration_json_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="Invalid JSON in configuration file"):
        ExperimentConfig.from_json(path)
