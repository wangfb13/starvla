#!/usr/bin/env python3
"""Fine-tune StarVLA with DD action conditioning and safely cropped recovery data.

Only the original flow-matching action loss is optimized: no contact classifier,
auxiliary contact loss, or classification warmup. Invoke with torchrun or python.
The original pretrained processors (including all action statistics) are loaded
without dataset-stat overrides. All generated files live under --output-dir.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, Subset

# Allow direct invocation from a repository checkout, independent of installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DD_PREFIX = "starvla_model.dd_contact."
DD_KEY = "observation.tactile_dd"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--info-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Our checkpoints/NNNNNN directory (or latest symlink).")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=6, help="Microbatch size per GPU.")
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--action-lr", type=float, default=3e-5)
    parser.add_argument("--dd-lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=100, help="LR ramp for action training, not a separate training stage.")
    parser.add_argument("--normal-fraction", type=float, default=0.7)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--smoke-test", action="store_true", help="Two real updates, bounded validation, full save/reload and DD-gradient checks.")
    args = parser.parse_args()
    if args.smoke_test:
        args.steps = 2
        args.accumulation_steps = 1
        args.eval_batches = min(args.eval_batches, 2)
        args.save_every = args.eval_every = 2
        args.log_every = 1
        args.lr_warmup_steps = 0
    if min(args.steps, args.batch_size, args.accumulation_steps, args.save_every,
           args.eval_every, args.eval_batches, args.log_every) < 1:
        parser.error("Steps, batches, accumulation and intervals must be positive.")
    if not 0 < args.normal_fraction < 1:
        parser.error("--normal-fraction must be strictly between 0 and 1.")
    if args.workers < 0 or args.lr_warmup_steps < 0:
        parser.error("Worker count and warmup must be nonnegative.")
    if args.smoke_test and args.resume:
        parser.error("Use a fresh output for --smoke-test; --resume is a separate check.")
    return args


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(temp, path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def processor_fingerprint(*pipelines):
    """Hash tensor values, names and dtypes, independent of device/container files."""
    digest = hashlib.sha256()
    names = []
    for pipeline_index, pipeline in enumerate(pipelines):
        for step_index, step in enumerate(pipeline.steps):
            for key, tensor in sorted(step.state_dict().items()):
                tensor = tensor.detach().cpu().contiguous()
                name = f"{pipeline_index}/{step_index}/{type(step).__name__}/{key}"
                names.append(name)
                digest.update(name.encode())
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    if not names:
        raise RuntimeError("Pretrained processors contain no statistics; refusing to silently retrain normalization.")
    return {"sha256": digest.hexdigest(), "tensor_count": len(names), "tensor_names": names}


def checkpoint_tensor_keys(directory):
    """Only keys physically serialized in the checkpoint, not metadata aliases."""
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        keys = set(json.loads(index.read_text())["weight_map"])
    else:
        from safetensors import safe_open
        with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as stream:
            keys = set(stream.keys())
    if not keys:
        raise RuntimeError(f"Checkpoint contains no serialized parameter keys: {directory}")
    return keys


def verify_loaded_keys(model, serialized_keys, missing, unexpected, *, allow_new_dd):
    """Enforce strict loading, accounting only for proven tied parameter aliases.

    Hugging Face saves one physical copy of tied weights. Its sharded loader
    can report an omitted alias as missing even though that exact Parameter
    was loaded through the saved alias. Never accept equal *values* alone:
    the missing alias must share the same storage view with a serialized,
    successfully loaded peer. This is also enforced on our saved checkpoints.
    """
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    missing = set(missing)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    loaded_peers = {key: parameter for key, parameter in parameters.items()
                    if key in serialized_keys and key not in missing}
    initialized, aliases, rejected = [], {}, []

    def same_parameter_storage(left, right):
        if left is right:
            return True
        if left.numel() == 0 or right.numel() == 0:
            return False
        return (left.device == right.device and left.dtype == right.dtype
                and left.shape == right.shape and left.stride() == right.stride()
                and left.storage_offset() == right.storage_offset()
                and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr())

    for key in sorted(missing):
        if allow_new_dd and key.startswith(DD_PREFIX):
            initialized.append(key)
            continue
        parameter = parameters.get(key)
        peers = [] if parameter is None else [name for name, value in loaded_peers.items()
                   if same_parameter_storage(parameter, value)]
        if not peers:
            rejected.append(key)
            continue
        peer = sorted(peers)[0]
        aliases[key] = {"serialized_peer": peer, "same_parameter_object": parameter is loaded_peers[peer],
                        "same_storage_view": True, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
    if rejected:
        raise RuntimeError(f"Uninitialized checkpoint keys without a verified loaded tied peer: {rejected}")
    return {"initialized_dd_keys": initialized, "verified_tied_aliases": aliases}


class WeightedRankSampler(Sampler):
    """One deterministic weighted global stream, partitioned across DDP ranks.

    Sampling probabilities retain the requested source mixture. Resume consumes
    the same stream after the completed global optimizer steps.
    """
    def __init__(self, weights, *, steps, completed_steps, batch_size, accumulation_steps,
                 rank, world_size, seed):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if not torch.isfinite(self.weights).all() or (self.weights < 0).any() or self.weights.sum() <= 0:
            raise ValueError("Invalid sampling weights.")
        per_step = batch_size * accumulation_steps * world_size
        self.total = steps * per_step
        self.skip = completed_steps * per_step
        self.rank, self.world_size, self.seed = rank, world_size, seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        stream = torch.multinomial(self.weights, self.total, replacement=True, generator=generator)
        yield from stream[self.skip + self.rank::self.world_size].tolist()

    def __len__(self):
        return (self.total - self.skip) // self.world_size


def optimizer_save(optimizer, directory, max_bytes=700_000_000):
    """Save Adam states in bounded safetensors shards, below cluster file limits."""
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = optimizer.state_dict()
    tensor_keys, scalars, shards = {}, {}, []
    pending, pending_bytes = {}, 0

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        name = f"optimizer-{len(shards):04d}.safetensors"
        save_file(pending, str(directory / name))
        shards.append(name)
        pending, pending_bytes = {}, 0

    for param_id, values in state["state"].items():
        for key, value in values.items():
            name = f"{param_id}/{key}"
            if not isinstance(value, torch.Tensor):
                scalars[name] = value
                continue
            value = value.detach().cpu().contiguous()
            size = value.numel() * value.element_size()
            if size > max_bytes:
                raise RuntimeError(f"Single optimizer tensor exceeds safe shard limit: {name}, {size} bytes")
            if pending_bytes + size > max_bytes:
                flush()
            tensor_keys[name] = [int(param_id), key]
            pending[name] = value
            pending_bytes += size
    flush()
    json_write(directory / "optimizer.json", {"param_groups": state["param_groups"],
               "tensor_keys": tensor_keys, "scalars": scalars, "shards": shards})


def optimizer_load(optimizer, directory):
    from safetensors.torch import load_file
    directory = Path(directory)
    index = json.loads((directory / "optimizer.json").read_text())
    states = {}
    for shard in index["shards"]:
        for name, value in load_file(str(directory / shard)).items():
            param_id, key = index["tensor_keys"][name]
            states.setdefault(int(param_id), {})[key] = value
    for name, value in index["scalars"].items():
        param_id, key = name.split("/", 1)
        states.setdefault(int(param_id), {})[key] = value
    optimizer.load_state_dict({"state": states, "param_groups": index["param_groups"]})


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def set_training(policy):
    policy.train()
    # Frozen Qwen remains deterministic; action and tactile/DD modules train.
    policy.starvla_model.qwen_vl_interface.eval()


def prepare_batch(raw, preprocessor, device, first=False):
    if "contact_label" in raw:
        raise RuntimeError("This trainer accepts no contact labels or classification supervision.")
    dd = raw.get(DD_KEY)
    if dd is None or dd.ndim != 5 or tuple(dd.shape[1:]) != (2, 3, 96, 128):
        raise ValueError(f"Expected DD [B,2,3,96,128], got {None if dd is None else tuple(dd.shape)}")
    if not torch.isfinite(dd).all():
        raise ValueError("Non-finite DD input.")
    if "action_is_pad" in raw and raw["action_is_pad"].any():
        raise RuntimeError("Unsafe action padding reached training; sampler boundaries are incorrect.")
    batch = preprocessor(raw)
    if DD_KEY not in batch:
        raise RuntimeError("Pretrained processor dropped the DD observation.")
    batch[DD_KEY] = batch[DD_KEY].to(device=device, dtype=torch.float32, non_blocking=True)
    if first and not torch.equal(dd.cpu(), batch[DD_KEY].detach().cpu()):
        raise RuntimeError("Pretrained processor altered physical DD values.")
    return batch


@torch.no_grad()
def validate(policy, preprocessor, dataset, args, device, step):
    """Bounded episode-heldout action-loss check; no test-set model selection."""
    state = capture_rng()
    try:
        random.seed(args.seed + 99)
        np.random.seed(args.seed + 99)
        torch.manual_seed(args.seed + 99)
        torch.cuda.manual_seed(args.seed + 99)
        if not len(dataset):
            return {"step": step, "samples": 0, "note": "No validation anchors."}
        generator = np.random.default_rng(args.seed + 77)
        sources = sorted({anchor["source"] for anchor in dataset.anchors})
        # Recovery first, so even a two-batch smoke validation checks recovery.
        sources.sort(key=lambda source: source != "recovery")
        candidates = {source: generator.permutation([i for i, anchor in enumerate(dataset.anchors)
                      if anchor["source"] == source]).tolist() for source in sources}
        by_source = {source: {"samples": 0, "action_loss_sum": 0.0} for source in sources}
        policy.eval()
        for batch_index in range(args.eval_batches):
            source = sources[batch_index % len(sources)]
            offset = (batch_index // len(sources)) * args.batch_size
            indices = candidates[source][offset:offset + args.batch_size]
            if not indices:
                continue
            raw = next(iter(DataLoader(Subset(dataset, indices), batch_size=len(indices), num_workers=0)))
            batch = prepare_batch(raw, preprocessor, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite held-out action loss.")
            by_source[source]["action_loss_sum"] += float(loss) * len(indices)
            by_source[source]["samples"] += len(indices)
        count = sum(item["samples"] for item in by_source.values())
        total = sum(item["action_loss_sum"] for item in by_source.values())
        summary = {source: {"samples": item["samples"],
                   "action_loss": item["action_loss_sum"] / item["samples"] if item["samples"] else None}
                   for source, item in by_source.items()}
        return {"step": step, "samples": count, "action_loss": total / count, "by_source": summary,
                "split": "validation; normal episodes are retention probes for the pretrained VLA"}
    finally:
        restore_rng(state)
        set_training(policy)


def save_checkpoint(policy, preprocessor, postprocessor, optimizer, scheduler, args,
                    step, fingerprint, rank, world_size, provenance):
    local_rng = capture_rng()
    rng_states = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(local_rng, rng_states, dst=0)
    else:
        rng_states = [local_rng]
    if rank == 0:
        if processor_fingerprint(preprocessor, postprocessor) != fingerprint:
            raise RuntimeError("Pretrained normalization statistics changed during training.")
        final = args.output_dir / "checkpoints" / f"{step:06d}"
        temporary = final.with_name(final.name + ".incomplete")
        if final.exists() or temporary.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint {final} or incomplete {temporary}")
        destination = temporary / "pretrained_model"
        destination.mkdir(parents=True)
        policy.save_pretrained(destination)
        preprocessor.save_pretrained(destination)
        postprocessor.save_pretrained(destination)
        optimizer_save(optimizer, temporary / "training_state")
        torch.save(rng_states, temporary / "training_state" / "rng.pt")
        json_write(temporary / "training_state" / "state.json", {
            "step": step, "scheduler": scheduler.state_dict(), "world_size": world_size,
            "args": vars(args), "normalization": fingerprint, "provenance": provenance})
        # Reload saved processors to prove normalization survived serialization.
        from lerobot.policies.factory import make_pre_post_processors
        saved_pre, saved_post = make_pre_post_processors(policy.config, str(destination))
        if processor_fingerprint(saved_pre, saved_post) != fingerprint:
            raise RuntimeError("Saved processor statistics differ from the source checkpoint.")
        force_steps = [s for s in saved_pre.steps if type(s).__name__ == "StarVLAPrepareTactileHistoryProcessorStep"]
        if len(force_steps) != 1 or not force_steps[0].enabled or tuple(force_steps[0].force_indices) != tuple(policy.config.tactile_force_indices):
            raise RuntimeError("Saved force-history processor lost its settings; get_config() must serialize them.")
        os.replace(temporary, final)
        link = args.output_dir / "checkpoints" / ".latest.tmp"
        link.symlink_to(final.name)
        os.replace(link, args.output_dir / "checkpoints" / "latest")
        print(json.dumps({"event": "checkpoint", "step": step, "path": str(final)}, ensure_ascii=False), flush=True)
    if world_size > 1:
        dist.barrier()


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("The full QwenPI training smoke test requires a CUDA GPU.")
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    if rank == 0:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
            raise FileExistsError(f"Use a new --output-dir or --resume: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    from huggingface_hub import load_torch_model
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.starvla.modeling_starvla import StarVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.dd_recovery import make_dd_recovery_datasets

    checkpoint = args.resume / "pretrained_model" if args.resume else args.source_checkpoint
    config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
    config.device = str(device)
    config.freeze_vlm = True
    config.gradient_checkpointing = False  # Frozen backbone needs no activation-gradient checkpointing.
    config.use_dd_contact = True
    config.optimizer_lr = args.action_lr
    config.contact_optimizer_lr = args.dd_lr
    config.vlm_optimizer_lr = 0.0
    config.push_to_hub = False
    # Preserve the trained action convention; wrong checkpoint fails early.
    if (config.action_gap_frames, config.chunk_size, config.n_action_steps,
        config.use_state_input, config.use_tactile) != (6, 32, 4, False, True):
        raise ValueError("This entry expects the audited gap6/chunk32/action4/no-state/tactile checkpoint.")
    if config.action_gap_dataset_overrides:
        raise ValueError("Per-source action gap overrides are unsupported by the audited recovery manifest.")
    config.drop_n_last_frames = 37
    policy = StarVLAPolicy(config)
    missing, unexpected = load_torch_model(policy, str(checkpoint), strict=False, map_location=str(device))
    loading_audit = verify_loaded_keys(policy, checkpoint_tensor_keys(checkpoint), missing, unexpected,
                                      allow_new_dd=args.resume is None)
    if rank == 0:
        print(json.dumps({"event": "checkpoint_loading_audit", **loading_audit}), flush=True)
    if not args.resume and not loading_audit["initialized_dd_keys"]:
        raise RuntimeError("Expected an original checkpoint without the DD branch; use --resume for DD checkpoints.")

    # Critical: do not pass dataset_stats or normalizer/relative-action overrides.
    force_history_override = {
        "enabled": config.use_tactile,
        "history_len": config.tactile_history_len,
        "force_indices": list(config.tactile_force_indices),
        "output_key": config.tactile_history_key,
    }
    # The original 060000 checkpoint saved this step as config={} because the
    # old class omitted get_config(). Restore its intended settings from the
    # saved policy config; action normalization/relative settings stay intact.
    preprocessor, postprocessor = make_pre_post_processors(
        config, str(checkpoint), preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "starvla_prepare_tactile_history": force_history_override,
        })
    fingerprint = processor_fingerprint(preprocessor, postprocessor)
    train, val, test = make_dd_recovery_datasets(
        config, args.manifest, info_path=args.info_path, video_backend=args.video_backend)
    if not len(train) or not len(val):
        raise RuntimeError("Both training and held-out validation anchors are required.")

    dd_params, action_params = [], []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(DD_PREFIX):
            dd_params.append(parameter)
        elif name.startswith(("starvla_model.action_model.", "starvla_model.tactile_encoder.",
                              "starvla_model.tactile_proj.")):
            action_params.append(parameter)
        else:
            raise RuntimeError(f"Unexpected unfrozen parameter: {name}")
    if not dd_params or not action_params:
        raise RuntimeError("DD and action parameter groups must both be trainable.")
    optimizer = torch.optim.AdamW([
        {"params": action_params, "lr": args.action_lr, "name": "action_and_original_force"},
        {"params": dd_params, "lr": args.dd_lr, "name": "dd_conditioning"},
    ], betas=tuple(config.optimizer_betas), eps=config.optimizer_eps,
        weight_decay=config.optimizer_weight_decay)

    def learning_rate_scale(step):
        if args.lr_warmup_steps and step < args.lr_warmup_steps:
            return (step + 1) / args.lr_warmup_steps
        progress = min(1.0, max(0.0, (step - args.lr_warmup_steps) /
                              max(1, args.steps - args.lr_warmup_steps)))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    start_step = 0
    resume_state = None
    if args.resume:
        resume_state = json.loads((args.resume / "training_state" / "state.json").read_text())
        if resume_state["world_size"] != world_size:
            raise ValueError("Exact resume requires the same world size.")
        if resume_state["normalization"] != fingerprint:
            raise RuntimeError("Resume normalization fingerprint mismatch.")
        for key in ("manifest", "seed", "batch_size", "accumulation_steps", "steps", "normal_fraction",
                    "action_lr", "dd_lr", "lr_warmup_steps"):
            if str(resume_state["args"][key]) != str(getattr(args, key)):
                raise ValueError(f"Exact resume requires unchanged {key}.")
        optimizer_load(optimizer, args.resume / "training_state")
        scheduler.load_state_dict(resume_state["scheduler"])
        start_step = resume_state["step"]
    if start_step >= args.steps:
        raise ValueError("No remaining optimizer steps; --steps must exceed completed steps.")

    provenance = {"source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_config_sha256": file_hash(args.source_checkpoint / "config.json"),
        "manifest_sha256": file_hash(args.manifest), "trainer_sha256": file_hash(__file__),
        "train_anchors": len(train), "validation_anchors": len(val), "test_anchors": len(test),
        "test_set_used": False, "missing_keys_initialized": loading_audit["initialized_dd_keys"],
        "verified_tied_aliases": loading_audit["verified_tied_aliases"],
        "loss": "original flow-matching action loss only", "no_contact_classifier": True,
        "restored_force_history_step": force_history_override,
        "normalization": fingerprint, "effective_batch_size": args.batch_size * args.accumulation_steps * world_size,
        "trainable_action_parameters": sum(p.numel() for p in action_params),
        "trainable_dd_parameters": sum(p.numel() for p in dd_params),
        "created_utc": datetime.now(timezone.utc).isoformat()}
    if resume_state and resume_state["provenance"]["manifest_sha256"] != provenance["manifest_sha256"]:
        raise RuntimeError("Resume manifest content changed.")
    if rank == 0:
        json_write(args.output_dir / ("resume_run.json" if args.resume else "run.json"),
                   {"args": vars(args), "provenance": provenance})
        print(json.dumps({"event": "ready", **provenance}, ensure_ascii=False), flush=True)

    sampler = WeightedRankSampler(train.sampling_weights(normal_fraction=args.normal_fraction),
        steps=args.steps, completed_steps=start_step, batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps, rank=rank, world_size=world_size, seed=args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                        generator=torch.Generator().manual_seed(args.seed + rank + 111))
    train_model = policy
    if world_size > 1:
        train_model = DistributedDataParallel(policy, device_ids=[local_rank], broadcast_buffers=False,
                                               find_unused_parameters=False)
    if resume_state:
        states = torch.load(args.resume / "training_state" / "rng.pt", weights_only=False)
        restore_rng(states[rank])
    iterator = iter(loader)
    set_training(policy)
    metrics_path = args.output_dir / "train_metrics.jsonl"
    maximum_dd_grad, maximum_encoder_grad = 0.0, 0.0
    started = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for microstep in range(args.accumulation_steps):
            raw = next(iterator)
            batch = prepare_batch(raw, preprocessor, device, first=(step == start_step + 1 and microstep == 0))
            sync = nullcontext() if world_size == 1 or microstep == args.accumulation_steps - 1 else train_model.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = train_model(batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite training action loss at step {step}.")
                    scaled = loss / args.accumulation_steps
                scaled.backward()
            accumulated_loss += float(loss.detach()) / args.accumulation_steps
        dd_grad = math.sqrt(sum(float(p.grad.detach().float().square().sum())
                               for p in dd_params if p.grad is not None))
        encoder_grad = math.sqrt(sum(float(p.grad.detach().float().square().sum())
            for name, p in policy.named_parameters() if name.startswith(DD_PREFIX + "finger_encoder.") and p.grad is not None))
        maximum_dd_grad = max(maximum_dd_grad, dd_grad)
        maximum_encoder_grad = max(maximum_encoder_grad, encoder_grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(action_params + dd_params, config.optimizer_grad_clip_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step}.")
        optimizer.step()
        scheduler.step()
        if rank == 0 and (step % args.log_every == 0 or step == start_step + 1):
            record = {"step": step, "action_loss": accumulated_loss, "dd_grad_norm": dd_grad,
                      "dd_encoder_grad_norm": encoder_grad, "grad_norm": float(grad_norm),
                      "action_lr": optimizer.param_groups[0]["lr"], "dd_lr": optimizer.param_groups[1]["lr"],
                      "elapsed_s": time.monotonic() - started,
                      "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({"event": "train", **record}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                record = validate(policy, preprocessor, val, args, device, step)
                with (args.output_dir / "validation_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps({"event": "validation", **record}), flush=True)
            if world_size > 1:
                dist.barrier()
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(policy, preprocessor, postprocessor, optimizer, scheduler, args,
                            step, fingerprint, rank, world_size, provenance)

    if args.smoke_test:
        if maximum_dd_grad <= 0 or maximum_encoder_grad <= 0:
            raise RuntimeError("Smoke test failed: action loss did not reach both DD projector and encoder.")
        if rank == 0:
            saved = args.output_dir / "checkpoints" / "latest" / "pretrained_model"
            saved_config = PreTrainedConfig.from_pretrained(str(saved), local_files_only=True)
            if not saved_config.use_dd_contact or saved_config.dd_contact_input_key != DD_KEY:
                raise RuntimeError("Serialized policy config lost DD input requirements.")
            # Reload into the same instance to avoid a second full VLM allocation.
            missing, unexpected = load_torch_model(policy, str(saved), strict=False, map_location=str(device))
            reload_audit = verify_loaded_keys(policy, checkpoint_tensor_keys(saved), missing, unexpected,
                                             allow_new_dd=False)
            optimizer_load(optimizer, args.output_dir / "checkpoints" / "latest" / "training_state")
            json_write(args.output_dir / "smoke_test_passed.json", {
                "two_real_forward_backward_updates": True, "full_policy_strict_reload": True,
                "optimizer_reload": True, "normalization_exact": True,
                "verified_tied_aliases_on_reload": reload_audit["verified_tied_aliases"],
                "dd_gradient_max": maximum_dd_grad, "dd_encoder_gradient_max": maximum_encoder_grad})
            print("SMOKE_TEST_PASSED", flush=True)
    if rank == 0:
        json_write(args.output_dir / "completed.json", {"step": args.steps,
                   "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.monotonic() - started})
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
