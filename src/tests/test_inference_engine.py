from pathlib import Path

import yaml

from experiments.inference_engine.pipeline import load_config


def test_analysis_config_accepts_multiple_immutable_trace_roots(tmp_path: Path):
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "artifact_root": "artifacts",
                    "traces_dir": "legacy",
                },
                "conditions": {
                    "vllm": {"traces_dir": "one"},
                    "sglang": {"traces_dirs": ["two", "three"]},
                },
            }
        )
    )

    config = load_config(config_path)

    assert config["conditions"]["vllm"]["traces_dirs"] == [
        (tmp_path / "one").resolve()
    ]
    assert config["conditions"]["sglang"]["traces_dirs"] == [
        (tmp_path / "two").resolve(),
        (tmp_path / "three").resolve(),
    ]
