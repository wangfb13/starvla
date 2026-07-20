#!/usr/bin/env python

"""LeRobot policy wrapper around the vendored official StarVLA frameworks."""

from collections import deque
import importlib
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_starvla import StarVLAConfig


_FRAMEWORK_MODULES = {
    "ABot_M0": "starVLA.model.framework.VLM4A.ABot_M0",
    "CosmosGR00T": "starVLA.model.framework.VLM4A.CosmosGR00T",
    "CosmoPredict2GR00T": "starVLA.model.framework.WM4A.CosmoPredict2GR00T",
    "CosmoPredict2OFT": "starVLA.model.framework.WM4A.CosmoPredict2OFT",
    "CosmoPredict2PI": "starVLA.model.framework.WM4A.CosmoPredict2PI",
    "Gemma4GR00T": "starVLA.model.framework.VLM4A.Gemma4GR00T",
    "Gemma4PI": "starVLA.model.framework.VLM4A.Gemma4PI",
    "InternVLA-M1": "starVLA.model.framework.VLM4A.M1",
    "LangForce": "starVLA.model.framework.VLM4A.LangForce",
    "MiniCPMGR00T": "starVLA.model.framework.VLM4A.MiniCPMGR00T",
    "MiniCPMPI": "starVLA.model.framework.VLM4A.MiniCPMPI",
    "PI0": "starVLA.model.framework.VLM4A.PI0",
    "Pi0": "starVLA.model.framework.VLM4A.PI0",
    "PI05": "starVLA.model.framework.VLM4A.PI05",
    "Pi05": "starVLA.model.framework.VLM4A.PI05",
    "QwenAdapter": "starVLA.model.framework.VLM4A.QwenAdapter",
    "QwenDiscreteDiffusion": "starVLA.model.framework.VLM4A.QwenDiscreteDiffusion",
    "QwenDual": "starVLA.model.framework.VLM4A.QwenDual",
    "QwenFast": "starVLA.model.framework.VLM4A.QwenFast",
    "QwenFM": "starVLA.model.framework.VLM4A.QwenPI",
    "QwenGR00T": "starVLA.model.framework.VLM4A.QwenGR00T",
    "QwenOFT": "starVLA.model.framework.VLM4A.QwenOFT",
    "QwenPI": "starVLA.model.framework.VLM4A.QwenPI",
    "QwenPI_v3": "starVLA.model.framework.VLM4A.QwenPI_v3",
    "WanGR00T": "starVLA.model.framework.WM4A.WanGR00T",
    "WanOFT": "starVLA.model.framework.WM4A.WanOFT",
    "WanPI": "starVLA.model.framework.WM4A.WanPI",
}


class StarVLAPolicy(PreTrainedPolicy):
    """Run an official StarVLA framework inside the LeRobot policy interface."""

    config_class = StarVLAConfig
    name = "starvla"

    def __init__(self, config: StarVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        try:
            from omegaconf import OmegaConf
            from starVLA.model.framework.share_tools import apply_config_compat
            from starVLA.model.tools import FRAMEWORK_REGISTRY
        except Exception as exc:
            raise ImportError(
                "StarVLA policy requires the vendored starVLA backend and its dependencies "
                "(for example omegaconf, numpy, Pillow, transformers)."
            ) from exc

        official_cfg = OmegaConf.create(config.to_official_config())
        apply_config_compat(official_cfg)
        self.starvla_config = official_cfg
        self.starvla_model = self._build_selected_framework(official_cfg, FRAMEWORK_REGISTRY)
        self.to(config.device)
        self.last_starvla_output: Any | None = None
        self.reset()

    def _build_selected_framework(self, official_cfg: Any, framework_registry: Any) -> torch.nn.Module:
        framework_name = str(self.config.framework_name)
        module_name = _FRAMEWORK_MODULES.get(framework_name)
        if module_name is None:
            available = ", ".join(sorted(_FRAMEWORK_MODULES))
            raise NotImplementedError(
                f"StarVLA framework '{framework_name}' is not mapped in the LeRobot wrapper. "
                f"Mapped official frameworks: {available}"
            )
        importlib.import_module(module_name)
        if framework_name not in framework_registry._registry:
            available = ", ".join(sorted(framework_registry._registry))
            raise NotImplementedError(
                f"StarVLA framework '{framework_name}' did not register after importing "
                f"{module_name}. Available registered frameworks: {available}"
            )
        return framework_registry[framework_name](official_cfg)

    def _image_keys(self) -> list[str]:
        keys = self.config.resolved_image_keys()
        if not keys:
            raise ValueError("StarVLA requires at least one VISUAL input feature or policy.image_keys.")
        return keys

    @staticmethod
    def _tensor_to_pil(image: Tensor) -> Image.Image:
        image = image.detach().to("cpu")
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor [C,H,W], got {tuple(image.shape)}")
        if image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        array = image.numpy()
        if array.dtype.kind == "f":
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255.0).round().astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        return Image.fromarray(array)

    def _batch_size(self, batch: dict[str, Any]) -> int:
        for key in self._image_keys():
            if key in batch:
                return int(batch[key].shape[0])
        if OBS_STATE in batch:
            return int(batch[OBS_STATE].shape[0])
        if ACTION in batch:
            return int(batch[ACTION].shape[0])
        return 1

    def _get_tasks(self, batch: dict[str, Any], bsize: int) -> list[str]:
        tasks = batch.get(self.config.task_key)
        if tasks is None:
            tasks = batch.get("task")
        if tasks is None:
            return [""] * bsize
        if isinstance(tasks, str):
            return [tasks] * bsize
        return [str(task) for task in list(tasks)]

    def _to_examples(self, batch: dict[str, Any], include_actions: bool) -> list[dict[str, Any]]:
        image_keys = self._image_keys()
        bsize = self._batch_size(batch)
        tasks = self._get_tasks(batch, bsize)

        examples: list[dict[str, Any]] = []
        for batch_idx in range(bsize):
            images = []
            for key in image_keys:
                if key not in batch:
                    continue
                images.append(self._tensor_to_pil(batch[key][batch_idx]))
            example: dict[str, Any] = {
                "image": images,
                "lang": tasks[batch_idx] if batch_idx < len(tasks) else "",
            }
            if self.config.use_state_input and OBS_STATE in batch:
                state = batch[OBS_STATE][batch_idx].detach().to("cpu").numpy()
                if state.ndim == 1:
                    state = state[None, :]
                example["state"] = state
            if include_actions:
                if ACTION not in batch:
                    raise ValueError("StarVLA training forward requires 'action' in the batch.")
                example["action"] = batch[ACTION][batch_idx].detach().to("cpu").numpy()
            examples.append(example)
        return examples

    @staticmethod
    def _loss_from_starvla_output(output: Any) -> tuple[Tensor, dict[str, Any]]:
        if not isinstance(output, dict):
            raise TypeError(f"StarVLA forward must return a dict of losses, got {type(output)}")

        tensor_items = {key: value for key, value in output.items() if isinstance(value, Tensor)}
        if "loss" in tensor_items:
            loss = tensor_items["loss"]
        elif "action_loss" in tensor_items:
            loss = tensor_items["action_loss"]
        elif tensor_items:
            loss = sum(tensor_items.values())
        else:
            raise ValueError("StarVLA forward returned no tensor loss values.")

        log_dict: dict[str, Any] = {}
        for key, value in output.items():
            if isinstance(value, Tensor):
                log_dict[key] = value.detach().mean().item()
            elif isinstance(value, (int, float, str, bool)) or value is None:
                log_dict[key] = value
        log_dict.setdefault("loss", loss.detach().mean().item())
        return loss, log_dict

    @staticmethod
    def _actions_from_starvla_output(output: Any) -> Any:
        if isinstance(output, dict):
            for key in ("normalized_actions", "actions", "action"):
                if key in output:
                    return output[key]
            raise KeyError(
                "StarVLA predict_action returned a dict without one of "
                "'normalized_actions', 'actions', or 'action'."
            )
        return output

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        examples = self._to_examples(batch, include_actions=True)
        output = self.starvla_model.forward(examples)
        self.last_starvla_output = output
        loss, log_dict = self._loss_from_starvla_output(output)
        if reduction == "none" and loss.ndim == 0:
            return loss[None], log_dict
        return loss.mean(), log_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        examples = self._to_examples(batch, include_actions=False)
        output = self.starvla_model.predict_action(examples, **kwargs)
        self.last_starvla_output = output
        actions = self._actions_from_starvla_output(output)
        if isinstance(actions, Tensor):
            action_tensor = actions.detach().to(device=self.config.device, dtype=torch.float32)
        else:
            action_tensor = torch.as_tensor(actions, device=self.config.device, dtype=torch.float32)
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)
        return action_tensor

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            skip_start_steps = max(0, int(self.config.action_execution_skip_start_steps))
            if skip_start_steps:
                actions = actions[:, skip_start_steps:]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()
