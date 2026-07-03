# StarvlaGroot (QwenGR00T) policy

Qwen-VL backbone (prefix encoder) + GR00T flow-matching DiT action head, ported
from starVLA into the LeRobot policy framework. Registered policy type:
`starvla_groot`.

## Layout
- `configuration_starvla_groot.py` — `StarvlaGrootConfig(PreTrainedConfig)`
- `modeling_starvla_groot.py` — `StarvlaGrootPolicy(PreTrainedPolicy)`
- `processor_starvla_groot.py` — `make_starvla_groot_pre_post_processors`
- `qwen_vl_interface.py` — generic `AutoModelForImageTextToText` wrapper
- `action_head/` — vendored GR00T flow-matching head (`flow_matching_head.py`,
  `cross_attention_dit.py` [needs `diffusers`], `action_encoder.py`)

No YAML is required: the config is a draccus-registered dataclass, exactly like
`act` / `pi05`. Override any field on the command line with `--policy.<field>=...`.

## Minimal training command
```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=<your/dataset> \
  --policy.type=starvla_groot \
  --policy.base_vlm=./playground/pretrained_models/Qwen3.5-0.8B \
  --policy.chunk_size=8 \
  --policy.n_action_steps=8 \
  --policy.state_mode=joint \
  --policy.top_camera_keys='["observation.images.cam_top"]' \
  --policy.wrist_camera_keys='["observation.images.cam_right_wrist"]' \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/train/starvla_groot
```

Useful overrides:
- `--policy.action_model_type=DiT-B|DiT-L`
- `--policy.repeated_diffusion_steps=8` (noise samples per batch element)
- `--policy.num_inference_timesteps=4` (denoising Euler steps at inference)
- `--policy.use_state_input=true` (feed `observation.state` to the action head)
- `--policy.wrist_only=true` (single-view) / `--policy.state_mode=none`
- `--policy.train_expert_only=true` (freeze the VLM, train only the action head)
- `--policy.freeze_vision_encoder=true`
- `--policy.use_relative_actions=true` to train relative action targets with the
  native LeRobot action/state processor; use
  `--policy.relative_action_keyframes='["0:3","10:13"]'` for xyz+rot6d blocks.
- `--policy.canonicalize_bimanual_rot6d=true` to reuse the PI0.5 bimanual
  canonical representation: `left/right = xyz + rot6d(6) + gripper`.
- `action_valid_mask` or `observation.action_mask` is used as the StarVLA action
  loss mask when present in the batch.

Migration notes:
- `tactile_mode=none`, `tactile_mode=as_image`, and `tactile_mode=encode` are available.
- `tactile_mode=encode` requires `--policy.tactile_encoder_path` pointing to a tactile-MAE
  checkpoint or converted HF-style directory.
- Control vectors are passed through unchanged: `state_mode=joint` keeps the raw
  `observation.state`, and `action_mode=joint` trains/predicts the raw `action`.
  Any flange/TCP conversion or robot-space mapping should stay in the existing
  humanoid data/control pipeline.

## VLM / transformers compatibility
The action head is VLM-agnostic; the backbone is loaded by
`AutoModelForImageTextToText.from_pretrained(base_vlm)`, so `base_vlm` must be an
architecture your installed `transformers` recognizes.

- `Qwen/Qwen3-VL-*` (model_type `qwen3_vl`) works on transformers >= 4.57.0.
- `Qwen3.5-0.8B` (model_type `qwen3_5`) is **not** in any stable transformers
  release as of 4.57.x — its `config.json` reports `transformers_version
  4.57.0.dev0`. To use it, install a transformers build that ships `qwen3_5`
  (e.g. from source), or point `base_vlm` at a `qwen3_vl` checkpoint instead.
