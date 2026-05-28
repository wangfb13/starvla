#!/usr/bin/env python

import torch

from lerobot.policies.pi05.processor_pi05 import Pi05ApplyMasksProcessorStep as Pi05ApplyMasksProcessorStep
from lerobot.policies.pi05_full.processor_pi05 import (
    Pi05ApplyMasksProcessorStep as Pi05FullApplyMasksProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_ACTION_MASK, OBS_STATE, OBS_STATE_MASK


def _transition_with_action_chunk():
    action = torch.ones(2, 3, 4)
    action_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    return {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.ones(2, 5),
            OBS_STATE_MASK: torch.ones(2, 5),
            OBS_ACTION_MASK: action_mask,
        },
        TransitionKey.ACTION: action,
    }


def _assert_action_chunk_masked(step):
    output = step(_transition_with_action_chunk())
    action = output[TransitionKey.ACTION]

    assert action.shape == (2, 3, 4)
    assert torch.allclose(action[0], torch.tensor([[1.0, 0.0, 1.0, 0.0]]).expand(3, -1))
    assert torch.allclose(action[1], torch.tensor([[0.0, 1.0, 1.0, 0.0]]).expand(3, -1))


def test_pi05_apply_masks_broadcasts_action_mask_over_action_chunk():
    _assert_action_chunk_masked(Pi05ApplyMasksProcessorStep())


def test_pi05_full_apply_masks_broadcasts_action_mask_over_action_chunk():
    _assert_action_chunk_masked(Pi05FullApplyMasksProcessorStep())
