"""
Compatibility shim for vla_align.utils.lerobot.

When InternVLA-A1's patched `lerobot` package is installed, importing
`vla_align.utils.lerobot` fails because it tries to import modules from the
official lerobot (e.g. `lerobot.processor`, `lerobot.envs.configs`) that don't
exist in this fork.

This module re-defines the constants and patches `sys.modules` so that
downstream `vla_align` modules (like `rollout.py`) can import without error.

Usage (call once before importing vla_align.utils.rollout):
    from lerobot.utils.vla_compat import patch_vla_align_lerobot
    patch_vla_align_lerobot()
"""

import sys
import types

# These mirror vla_align/utils/lerobot.py lines 26-36
OBS_IMAGES_PREFIX = "observation.images"
obs_state_key = "observation.state"
reward_key = "next.reward"
action_key = "action"
image_1_key = f"{OBS_IMAGES_PREFIX}.image_1"
image_1_robot_state = f"{OBS_IMAGES_PREFIX}.image_1_robot_state"
image_1_segmentation_mask_key = f"{OBS_IMAGES_PREFIX}.image_1_segmentation_mask"
image_2_key = f"{OBS_IMAGES_PREFIX}.image_2"
wrist_image_key = f"{OBS_IMAGES_PREFIX}.wrist_image"
task_key = "task"


def patch_vla_align_lerobot():
    """Inject a fake `vla_align.utils.lerobot` into sys.modules.

    This allows `from vla_align.utils.rollout import rollout` to succeed
    even though the real vla_align.utils.lerobot depends on official lerobot.
    """
    if "vla_align.utils.lerobot" in sys.modules:
        return  # already loaded (or patched)

    fake = types.ModuleType("vla_align.utils.lerobot")
    fake.OBS_IMAGES_PREFIX = OBS_IMAGES_PREFIX
    fake.obs_image_prefix = OBS_IMAGES_PREFIX
    fake.obs_state_key = obs_state_key
    fake.reward_key = reward_key
    fake.action_key = action_key
    fake.image_1_key = image_1_key
    fake.image_1_robot_state = image_1_robot_state
    fake.image_1_segmentation_mask_key = image_1_segmentation_mask_key
    fake.image_2_key = image_2_key
    fake.wrist_image_key = wrist_image_key
    fake.task_key = task_key

    sys.modules["vla_align.utils.lerobot"] = fake
