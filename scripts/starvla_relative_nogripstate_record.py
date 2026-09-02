#!/usr/bin/env python3
"""Robot-side launcher adapter for the no-gripper-state relative StarVLA checkpoint.

The online Spark3 adapter normally marks a valid gripper observation as model input.
This checkpoint was trained with gripper state dimensions 9 and 19 masked, while its
gripper action dimensions remain valid and relative. Patch only the state mask so
online preprocessing exactly matches training.
"""

from __future__ import annotations

from lerobot.utils.constants import OBS_STATE_MASK


LEFT_GRIPPER_STATE_INDEX = 9
RIGHT_GRIPPER_STATE_INDEX = 19


def install_no_gripper_state_adapter() -> None:
    from lerobot.async_inference import helpers

    original_builder = helpers._build_spark3_online_pi05_state_and_masks

    def build_state_without_gripper_input(
        raw_observation,
        *,
        action_mask_key: str,
        state_mask_key: str,
    ):
        result = original_builder(
            raw_observation,
            action_mask_key=action_mask_key,
            state_mask_key=state_mask_key,
        )
        state_mask = result[state_mask_key].clone()
        if state_mask.shape[-1] < 20:
            raise RuntimeError(
                f"Expected canonical state mask with at least 20 dims, got {tuple(state_mask.shape)}"
            )
        state_mask[..., LEFT_GRIPPER_STATE_INDEX] = 0.0
        state_mask[..., RIGHT_GRIPPER_STATE_INDEX] = 0.0
        result[state_mask_key] = state_mask
        result[OBS_STATE_MASK] = state_mask
        # Deliberately leave action_mask_key unchanged: gripper actions are trained
        # and decoded as g_target - g_current.
        return result

    helpers._build_spark3_online_pi05_state_and_masks = build_state_without_gripper_input
    print(
        "[no-gripper-state] model state dims 9 and 19 are masked; "
        "gripper action dims remain enabled and relative",
        flush=True,
    )


def main() -> None:
    install_no_gripper_state_adapter()
    from lerobot.scripts.lerobot_record import main as record_main

    record_main()


if __name__ == "__main__":
    main()
