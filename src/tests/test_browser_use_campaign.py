import pytest

from browser_use_campaign import CampaignError, CampaignRunner


def _runner(**settings):
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.settings = settings
    return runner


def test_local_engine_requires_exactly_one_backend():
    with pytest.raises(CampaignError, match="exactly one"):
        CampaignRunner._local_engine({"agent_id": "model"})

    with pytest.raises(CampaignError, match="exactly one"):
        CampaignRunner._local_engine(
            {
                "agent_id": "model",
                "vllm": {},
                "sglang": {},
            }
        )


def test_local_engine_requires_mapping():
    with pytest.raises(CampaignError, match="must be a mapping"):
        CampaignRunner._local_engine(
            {"agent_id": "model", "sglang": [{"model": "example/model"}]}
        )


def test_vllm_command_remains_backward_compatible():
    runner = _runner(vllm_executable="/env/bin/vllm")

    command = runner._server_command(
        "vllm",
        {
            "model": "example/model",
            "port": 3031,
            "args": ["--tensor-parallel-size", "2"],
        },
    )

    assert command == [
        "/env/bin/vllm",
        "serve",
        "example/model",
        "--port",
        "3031",
        "--tensor-parallel-size",
        "2",
    ]


def test_sglang_command_uses_isolated_python():
    runner = _runner(sglang_executable="/envs/w_serve/bin/python")

    command = runner._server_command(
        "sglang",
        {
            "model": "example/model",
            "served_model_name": "example/model-alias",
            "port": 3031,
            "args": ["--tp-size", "1"],
        },
    )

    assert command == [
        "/envs/w_serve/bin/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        "example/model",
        "--port",
        "3031",
        "--served-model-name",
        "example/model-alias",
        "--tp-size",
        "1",
    ]
