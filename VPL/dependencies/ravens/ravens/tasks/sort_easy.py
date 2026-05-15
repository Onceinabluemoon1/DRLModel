# coding=utf-8
# Copyright 2023 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-block sort-easy: oracle targets one ``goal_pose``; optional ``zone_poses`` mark all contenders."""

import numpy as np
import pybullet as p

from ravens.tasks.task import Task
from ravens.utils import utils

BLOCK_META = dict(
    size=(0.05, 0.05, 0.05),
    color=utils.COLORS["blue"] + [1],
    urdf="sort/large_block.urdf",
)


class SortingEasy(Task):
    """Place one blue block into the goal selected by SortEnv.set_mode(...)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = 4
        self.no_rot = False
        self.goal_pose = None  # ((xyz), (quat)), set in jaxrl_m.envs.sort_easy.SortEnv
        self.zone_poses = None  # visual markers for goal zones
        self.debug = False
        self.initial_pose = None
        self.block_id = None  # only one block in this task

    def init_pose(self, poses):
        del poses

    def reset(self, env):
        super().reset(env)
        if self.goal_pose is None:
            raise ValueError("SortingEasy.goal_pose must be set before reset().")

        # * Fixed zone meshes: one URDF per element of ``zone_poses``, else only ``goal_pose``.
        zone_poses = self.zone_poses if self.zone_poses is not None else [self.goal_pose]
        for zp in zone_poses:
            env.add_object("zone/zone.urdf", zp, "fixed")

        # * add block objects

        avoid_xys = [np.asarray(zp[0][:2], dtype=np.float32) for zp in zone_poses]
        block_pose = self._sample_block_pose(env, BLOCK_META["size"], avoid_xys=avoid_xys)
        if self.debug:
            block_pose = self.initial_pose

        block_pose = (block_pose[0], (0, 0, 0, 1))
        block_id = env.add_object(BLOCK_META["urdf"], block_pose)
        p.changeVisualShape(block_id, -1, rgbaColor=BLOCK_META["color"])
        self.block_id = block_id

        # One goal step: 8-tuple consumed by Task._discrete_oracle (planning) and
        # Task.reward (scoring); see unpacking in ravens/tasks/task.py.
        # [0] objs — [(body_id, (z symm. rad., unused)), ...]
        # [1] matches — matrix |objs|×|targs|; 1 = allowed object→target pairing
        # [2] targs — list of target ((x,y,z), quaternion); here one zone
        # [3] replace — False: zero satisfied rows/cols so pairs are not planned again
        # [4] rotations — False would force placement orientation to upright identity
        # [5] metric — "pose" = is_match(xy, yaw); "zone" uses params instead
        # [6] params — only for metric=="zone"; unused here
        # [7] max_reward — episode progress chunk when this subgoal completes

        self.goals = [
            (
                [(block_id, (np.pi / 2, None))],
                np.ones((1, 1)),
                [self.goal_pose],
                False,
                True,
                "pose",
                None,
                1,
            ),
        ]

    def _sample_block_pose(
        self,
        env,
        block_size: tuple[float, float, float],
        avoid_xys: list[np.ndarray],
    ):
        """Reject starts whose block XY falls within ``min_sep`` of any zone center."""
        min_sep = 0.08
        for _ in range(100):
            block_pose = self.get_random_pose(env, block_size)
            block_xy = np.asarray(block_pose[0][:2], dtype=np.float32)
            if all(float(np.linalg.norm(block_xy - gxy)) > min_sep for gxy in avoid_xys):
                return block_pose
        raise RuntimeError("Failed to sample a block pose away from all zone XY centers.")

    def get_world_obs(self):
        obs = self._env._get_obs()
        return (
            obs["state"],
            obs["goal"],
            int(obs["object_grabbed"]),
            int(obs["suction_state"]),
        )

    def get_observation(self, env):
        obs = super().get_observation(env)
        obs["goal"] = np.array([0, 0, 0, 0])
        return obs


class SortingEasyFixed(SortingEasy):
    pass
