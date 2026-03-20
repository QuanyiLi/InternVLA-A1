#!/usr/bin/env python
"""
Standalone evaluation script for InternVLA-A1 on the WISER/ManiSkill environment.

Usage:
    python evaluation/wiser/inference.py \
        --ckpt_path <checkpoint_dir> \
        --cfg_name config_0 \
        --split both \
        --action_mode delta \
        --num_envs 12
"""
import json
import shutil
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import tyro

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import load_json
from lerobot.policies.InternVLA_A1_3B.modeling_internvla_a1 import QwenA1Config, QwenA1Policy
from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import Qwen3_VLProcessorTransformFn
from lerobot.transforms.core import (
    NormalizeTransformFn,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    RemapImageKeyTransformFn,
    compose,
)
from lerobot.utils.constants import OBS_IMAGES

# Patch sys.modules before importing vla_align modules that depend on official lerobot
from lerobot.utils.vla_compat import (
    patch_vla_align_lerobot,
    obs_state_key, image_1_key, wrist_image_key, task_key,
)
patch_vla_align_lerobot()

# vla_align imports for ManiSkill evaluation
from vla_align.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE_EVAL
from vla_align.utils.env import build_endless_env
from vla_align.utils.rollout import rollout
from vla_align.utils.helpers import batch_tensor_to_string


def resolve_ckpt_dir(ckpt_path: Union[str, Path]) -> Path:
    """Resolve checkpoint path to local directory."""
    local_dir = Path(ckpt_path).expanduser()
    if local_dir.exists():
        return local_dir.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=str(ckpt_path)))


def build_policy_and_transforms(ckpt_path, stats_key, resize_size, dtype):
    """Load policy and build input/output transforms."""
    ckpt_dir = resolve_ckpt_dir(ckpt_path)
    config = PreTrainedConfig.from_pretrained(ckpt_dir)
    if not isinstance(config, QwenA1Config):
        raise ValueError(f"Expected QwenA1Config, got {type(config)}")

    policy = QwenA1Policy.from_pretrained(config=config, pretrained_name_or_path=ckpt_dir)
    policy.cuda().to(dtype).eval()

    stats = load_json(ckpt_dir / "stats.json")
    # Handle nested stats (keyed by robot type) or flat structure
    if stats_key and stats_key in stats:
        stats = stats[stats_key]
    elif "observation.state" not in stats:
        # Auto-detect: stats are keyed by robot type, pick the first one
        robot_keys = list(stats.keys())
        if len(robot_keys) == 1:
            print(f"Auto-detected stats key: {robot_keys[0]}")
            stats = stats[robot_keys[0]]
        else:
            raise KeyError(
                f"stats.json has multiple robot-type keys {robot_keys} but no "
                f"--stats_key was provided. Please specify one explicitly."
            )

    stat_keys = ["min", "max", "mean", "std"]
    state_stat = {"observation.state": {k: np.asarray(stats["observation.state"][k]) for k in stat_keys}}
    action_stat = {"action": {k: np.asarray(stats["action"][k]) for k in stat_keys}}

    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=["action"],
        mode="mean_std",
        norm_stats=action_stat,
    )

    image_keys = [f"{OBS_IMAGES}.image{i}" for i in range(2)]  # only 2 cameras from env
    input_transforms = compose(
        [
            ResizeImagesWithPadFn(height=resize_size, width=resize_size),
            RemapImageKeyTransformFn(mapping={k: k for k in image_keys}),
            Qwen3_VLProcessorTransformFn(),
            NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
        ]
    )

    return policy, input_transforms, unnormalize_fn


@dataclass
class InferenceArgs:
    """Configuration for WISER/ManiSkill evaluation."""
    ckpt_path: Union[str, Path] = "outputs/train/latest/checkpoints/last-checkpoint/pretrained_model"
    stats_key: str = ""  # Robot type key for stats; empty = auto-detect
    cfg_name: str = "config_0"
    split: str = "both"  # "train", "test", or "both"
    num_envs: int = 12
    max_steps: int = MAX_EPISODE_STEP_WORKSPACE_EVAL
    resize_size: int = 224
    action_mode: str = "delta"  # delta | abs
    dtype: str = "bfloat16"  # float32 | bfloat16
    action_dim: int = 8
    n_action_steps: int = 20  # Replanning frequency
    log_level: str = "INFO"
    eval_rounds: int = 1
    result_dir: str = ""  # If set, save results to this directory
    start_subset: int = -1  # If >= 0, loop config_{start_subset..end_subset-1}
    end_subset: int = -1
    aggregate_only: bool = False  # If True, only aggregate existing results


def run_eval(args: InferenceArgs):
    """Run evaluation on WISER/ManiSkill configs.

    Supports three modes:
    1. aggregate_only: just aggregate existing results.
    2. start_subset/end_subset >= 0: loop over config_{start..end-1}.
    3. Otherwise: evaluate a single cfg_name.
    """
    # ── Aggregate-only mode ──────────────────────────────────────────────
    if args.aggregate_only:
        assert args.result_dir, "--result_dir is required for --aggregate_only"
        _aggregate_results(args.result_dir)
        return

    # ── Build config list ────────────────────────────────────────────────
    if args.start_subset >= 0 and args.end_subset >= 0:
        cfg_names = [f"config_{i}" for i in range(args.start_subset, args.end_subset)]
    else:
        cfg_names = [args.cfg_name]

    # ── Load policy once ─────────────────────────────────────────────────
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    policy, input_transforms, unnormalize_fn = build_policy_and_transforms(
        args.ckpt_path, args.stats_key, args.resize_size, dtype
    )

    action_dim = args.action_dim
    n_action_steps = args.n_action_steps

    # Action queues for replanning every n_action_steps
    _action_queues = None

    def _a1_policy_forward(obs):
        """Wraps InternVLA-A1 policy for the vla_align rollout interface.
        Replans every n_action_steps by caching action chunks in per-env queues.
        """
        nonlocal _action_queues
        start = time.perf_counter()
        num_envs = obs[obs_state_key].shape[0]
        if _action_queues is None:
            _action_queues = [deque() for _ in range(num_envs)]

        all_actions = torch.zeros((num_envs, action_dim), device=obs[obs_state_key].device)

        for env_idx in range(num_envs):
            # If we have cached actions, use them
            if len(_action_queues[env_idx]) > 0:
                all_actions[env_idx] = _action_queues[env_idx].popleft()
                continue

            # Otherwise, run inference to get a new action chunk
            state = obs[obs_state_key][env_idx][:action_dim].float()
            img1 = obs[image_1_key][env_idx].float() / 255.0
            img1 = img1.permute(2, 0, 1)
            wrist_img = obs[wrist_image_key][env_idx].float() / 255.0
            wrist_img = wrist_img.permute(2, 0, 1)
            task_str = batch_tensor_to_string(obs[task_key][env_idx:env_idx+1])[0]

            image0 = torch.stack([img1, img1], dim=0).to(dtype).cuda()
            image1 = torch.stack([wrist_img, wrist_img], dim=0).to(dtype).cuda()

            sample = {
                f"{OBS_IMAGES}.image0": image0,
                f"{OBS_IMAGES}.image1": image1,
                "observation.state": state.cuda(),
                "task": task_str,
            }

            sample = input_transforms(sample)

            inputs = {}
            for key in sample.keys():
                if key == "task":
                    inputs[key] = [sample[key]]
                elif isinstance(sample[key], torch.Tensor) and sample[key].dtype == torch.int64:
                    inputs[key] = sample[key][None].cuda()
                elif isinstance(sample[key], torch.Tensor):
                    inputs[key] = sample[key][None].cuda().to(dtype=dtype)
                else:
                    inputs[key] = sample[key]

            inputs.update({
                f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(),
                f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(),
                f"{OBS_IMAGES}.image2_mask": torch.tensor([False]).cuda(),
            })

            with torch.no_grad():
                action_pred, _ = policy.predict_action_chunk(inputs)

            # Get the full action chunk
            action_chunk = action_pred[0, :n_action_steps, :action_dim]
            action_chunk = unnormalize_fn({"action": action_chunk})["action"]

            if args.action_mode == "delta":
                init_state = obs[obs_state_key][env_idx][:action_dim].float().cuda()
                init_state[action_dim - 1] = 0.0
                action_chunk = action_chunk + init_state[None, :]

            # First action goes to output, rest go to queue
            all_actions[env_idx] = action_chunk[0].float()
            for t in range(1, action_chunk.shape[0]):
                _action_queues[env_idx].append(action_chunk[t].float().to(obs[obs_state_key].device))

        elapsed = time.perf_counter() - start
        return all_actions, all_actions, {"inference_time": elapsed}

    splits = ["train", "test"] if args.split == "both" else [args.split]

    # Set up result directory
    if args.result_dir:
        result_dir = Path(args.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = None

    # ── Evaluation loop ──────────────────────────────────────────────────
    for split in splits:
        for cfg_name in cfg_names:
            # Skip if already evaluated
            if result_dir:
                subset_dir = str(result_dir / f"{cfg_name}_{split}")
                metrics_file = os.path.join(subset_dir, "episode_metrics.json")
                if os.path.exists(metrics_file):
                    print(f"Skipping {cfg_name} ({split}) — already evaluated")
                    continue
                if os.path.exists(subset_dir):
                    shutil.rmtree(subset_dir)
                os.makedirs(subset_dir)
            else:
                subset_dir = None

            # Reset action queues between configs
            _action_queues = None
            scene_cfg = dict(
                robot_init_qpos_noise=0.0,
                cube_size_noise=0.0,
                cfg_name=cfg_name,
                mode=split,
            )
            env_cfg = get_env_cfg(
                num_env=args.num_envs,
                max_steps=args.max_steps,
                obs_mode="rgb+segmentation",
                scene_cfg_to_overwrite=scene_cfg,
            )
            envs = build_endless_env(env_cfg, record_video=False, data_record_dir=None)

            print(f"\n{'=' * 80}")
            print(f"Starting eval: {cfg_name} ({split})")
            print(f"{'=' * 80}")

            eval_start = time.perf_counter()
            with torch.no_grad():
                performance = rollout(
                    envs,
                    _a1_policy_forward,
                    round_to_collect=args.eval_rounds,
                    demo_saving_dir=subset_dir,
                    debug_mode=True if subset_dir else False,
                    indices_to_save=[],
                )
            elapsed = time.perf_counter() - eval_start

            print(f"\n{'=' * 80}")
            print(f"{cfg_name} ({split}) — {elapsed:.1f}s")
            print(f"{'=' * 80}")
            for key, v in performance.items():
                print(f"  {key}: {v}")

            envs.unwrapped.close()

    # Aggregate results into final_results_train.json / final_results_test.json
    if result_dir:
        _aggregate_results(str(result_dir))


def _aggregate_results(result_dir):
    """Compute final aggregated results for train and test splits."""
    from vla_align.utils.rollout import calculate_averages
    for split in ["train", "test"]:
        pattern = os.path.join(result_dir, f"*{split}*")
        final_results = calculate_averages(pattern)
        if final_results:
            out_path = os.path.join(result_dir, f"final_results_{split}.json")
            with open(out_path, "w") as f:
                json.dump(final_results, f, indent=2)
            print(f"Final results saved to {out_path}")
        else:
            print(f"No results found for split '{split}'")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    args = tyro.cli(InferenceArgs)
    run_eval(args)


if __name__ == "__main__":
    main()
