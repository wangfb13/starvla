from pathlib import Path

import torch
from transformers import AutoConfig

from lerobot.policies.starvla.configuration_starvla import StarVLAConfig
from lerobot.policies.starvla.modeling_starvla import StarVLAPolicy
from lerobot.policies.starvla.processor_starvla import StarVLAPrepareTactileHistoryProcessorStep
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI, TactileEncoder


repo = Path("/public/home/eng_ywang/wangfubo/model_train/starvla")
for relative in (
    "src/lerobot/policies/starvla/configuration_starvla.py",
    "src/lerobot/policies/starvla/modeling_starvla.py",
    "src/lerobot/policies/starvla/processor_starvla.py",
    "src/starVLA/model/framework/VLM4A/QwenPI.py",
):
    path = repo / relative
    compile(path.read_text(encoding="utf-8"), str(path), "exec")

common = dict(
    framework_name="QwenPI",
    action_dim=20,
    state_dim=32,
    chunk_size=32,
    n_action_steps=16,
    action_gap_frames=6,
    use_relative_actions=True,
    relative_action_keyframes=["0:3", "10:13"],
    relative_exclude_joints=["gripper"],
)
no_state = StarVLAConfig(use_state_input=False, **common)
with_state = StarVLAConfig(
    use_state_input=True,
    state_history_horizon=2,
    relative_state_history_to_current=True,
    relative_scalar_state_history_indices=[9, 19],
    inter_arm_relative_state=True,
    state_full_dropout_prob=0.05,
    inter_arm_state_dropout_prob=0.05,
    state_available_index=31,
    **common,
)
no_state.validate_features()
with_state.validate_features()
assert no_state.action_delta_indices == list(range(6, 38))
assert no_state.to_official_config()["framework"]["action_model"]["state_dim"] == 0
assert with_state.to_official_config()["framework"]["action_model"]["state_dim"] == 32
assert with_state.get_observation_delta_indices("observation.state") == [-1, 0]

tactile_force_indices = [32, 33, 34, 38, 39, 40, 44, 45, 46, 50, 51, 52]
tactile = StarVLAConfig(
    use_state_input=True,
    use_tactile=True,
    tactile_history_len=30,
    tactile_force_indices=tactile_force_indices,
    state_dim=56,
    qpos_state_dim=32,
    **{key: value for key, value in common.items() if key != "state_dim"},
)
tactile.validate_features()
assert tactile.get_observation_delta_indices(OBS_STATE) == list(range(-29, 1))
official = tactile.to_official_config()["framework"]
assert official["use_tactile"] is True
assert official["tactile_history_len"] == 30
assert official["tactile_channels"] == 4
assert official["action_model"]["state_dim"] == 32

policy = StarVLAPolicy.__new__(StarVLAPolicy)
torch.nn.Module.__init__(policy)
policy.config = tactile
policy.eval()
assert policy._get_model_state({OBS_STATE: torch.zeros(2, 30, 56)}).shape == (2, 1, 32)

state = torch.zeros(1, 30, 56)
state[..., 32:35] = torch.tensor([3.0, 4.0, -2.0])
state[..., 38:41] = torch.tensor([0.0, 2.0, 1.0])
state[..., 44:47] = torch.tensor([6.0, 8.0, -3.0])
state[..., 50:53] = torch.tensor([0.0, 1.0, 4.0])
pad = torch.zeros(1, 30, dtype=torch.bool)
pad[:, :2] = True
step = StarVLAPrepareTactileHistoryProcessorStep(
    enabled=True, history_len=30, force_indices=tuple(tactile_force_indices)
)
transition = step(
    {
        TransitionKey.OBSERVATION: {
            OBS_STATE: state,
            f"{OBS_STATE}_is_pad": pad,
        }
    }
)
tactile_history = transition[TransitionKey.OBSERVATION]["observation.tactile_history"]
assert tactile_history.shape == (1, 30, 4)
assert torch.equal(tactile_history[:, :2], torch.zeros(1, 2, 4))
assert torch.allclose(tactile_history[0, -1], torch.tensor([3.0, 7.0, 7.0, 11.0]))

framework = Qwen_PI.__new__(Qwen_PI)
torch.nn.Module.__init__(framework)
framework.use_tactile = True
framework.tactile_encoder = TactileEncoder(30, 4, 128)
framework.tactile_proj = torch.nn.Linear(128, 16)
hidden = [torch.randn(2, 7, 16), torch.randn(2, 7, 16)]
mask = torch.ones(2, 7, dtype=torch.bool)
conditioned, conditioned_mask = framework._append_tactile_condition(
    hidden,
    mask,
    [{"tactile_history": tactile_history[0].numpy()} for _ in range(2)],
)
assert [tuple(value.shape) for value in conditioned] == [(2, 8, 16), (2, 8, 16)]
assert torch.allclose(conditioned[0][:, -1], conditioned[1][:, -1])
assert conditioned_mask.shape == (2, 8) and conditioned_mask[:, -1].all()

qwen_cfg = AutoConfig.from_pretrained(
    "/public/home/eng_ywang/wangfubo/pretrained_models/Qwen3.5-0.8B",
    local_files_only=True,
)
text_cfg = getattr(qwen_cfg, "text_config", qwen_cfg)
assert int(text_cfg.hidden_size) == 1024
assert int(text_cfg.num_hidden_layers) == 24
print(
    "VALID StarVLA-QwenPI: original configs unchanged; tactile=30x4, "
    "left-zero-pad, causal deltas=-29..0, Qwen=24x1024"
)
