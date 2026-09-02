from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


def test_umi_sparse_temporal_indices():
    config = DiffusionConfig(
        n_obs_steps=2,
        observation_stride_frames=3,
        horizon=16,
        n_action_steps=8,
        action_gap_frames=6,
        action_stride_frames=3,
        down_dims=(256, 512, 1024),
    )

    assert config.observation_delta_indices == [-3, 0]
    assert config.action_delta_indices == list(range(6, 52, 3))
    assert len(config.action_delta_indices) == 16
    assert config.action_delta_indices[-1] == 51


def test_legacy_diffusion_indices_are_unchanged():
    config = DiffusionConfig(n_obs_steps=2, horizon=16)
    assert config.observation_delta_indices == [-1, 0]
    assert config.action_delta_indices == list(range(-1, 15))
