import os
from collections.abc import Callable, Sequence
from pathlib import Path

import gym
import numpy as np
import pybullet as p
import timeout_decorator
from d4rl.offline_env import OfflineEnv
from ravens import tasks
from ravens.environments.environment import Environment


def _multimodal_goal_layout(*, n_possible_goals: int) -> list:
    """
    Explicit poses in ``_multimodal_goal_layout``: 1 fixed stripe; 2 ±y stripes at x=0.5; 4 corners.
    """
    quat = (0, 0, 0, 1)
    if n_possible_goals == 1:
        return [
            ((0.5, 0.2, 0.0), quat),  # right goal (only one)
        ]
    if n_possible_goals == 2:
        return [
            ((0.5, 0.2, 0.0), quat),  # right goal
            ((0.5, -0.2, 0.0), quat),  # left goal
        ]
    if n_possible_goals == 4:
        return [
            ((0.32, -0.18, 0.0), quat),  # bottom left goal
            ((0.32, 0.18, 0.0), quat),  # bottom right
            ((0.68, -0.18, 0.0), quat),  # top left goal
            ((0.68, 0.18, 0.0), quat),  # top right goal
        ]
    raise ValueError(f"n_possible_goals must be 1, 2, or 4; got {n_possible_goals!r}")


class SortEnv(OfflineEnv, gym.Env):
    def __init__(
        self,
        mode=-1,
        disp=False,
        debug=False,
        ee=False,
        oracle=False,
        *,
        n_possible_goals: int = 4,
        dataset_path: os.PathLike[str] | str | None = None,
        **kwargs,
    ):
        del ee, oracle, kwargs  # unused
        super().__init__()

        vpl_root = Path(__file__).resolve().parents[2]
        assets_root = vpl_root / "dependencies/ravens/ravens/environments/assets"

        self.env = Environment(
            str(assets_root),
            disp=disp,
            shared_memory=False,
            hz=480,
            use_egl=not disp,
        )
        self.mode = mode
        self.is_multimodal = mode < 0

        dataset_fp = (
            Path(dataset_path).expanduser().resolve()
            if dataset_path is not None
            else Path(__file__).resolve().parent / "dataset/sort.hdf5"
        )
        self.dataset_path = str(dataset_fp)

        if mode == 0:
            self.task = tasks.names["sort-easy-fixed"](continuous=True)
            self.goal_poses = _multimodal_goal_layout(n_possible_goals=1)
            self.n_possible_goals = 1
        else:
            self.task = tasks.names["sort-easy"](continuous=True)
            self.goal_poses = _multimodal_goal_layout(n_possible_goals=n_possible_goals)
            self.n_possible_goals = int(n_possible_goals)

        # todo: get random goal?
        self.curr_goal_id = 0 if self.is_multimodal else int(mode)
        if self.curr_goal_id < 0 or self.curr_goal_id >= len(self.goal_poses):
            raise ValueError(
                f"Invalid initial goal mode {self.curr_goal_id}; "
                f"expected 0..{len(self.goal_poses) - 1}"
            )
        self.mode = self.curr_goal_id
        self.curr_goal_pose = self.goal_poses[self.curr_goal_id]
        self.task.goal_pose = self.curr_goal_pose
        self.task.zone_poses = list(self.goal_poses)
        self.task.mode = "train"
        self.env.set_task(self.task)

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(4,),  # 8,)
        )  # TODO: setup action space, self.env.action_space

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),  # 15,)
        )  # TODO: setup observation space self.env.observation_space

        self._max_episode_steps = 25  # kwargs.get("max_episode_steps", 300)
        self.relabel_offline_reward = True
        self.debug = debug

        # * Offline stats: prefer dataset keys if ``sort.hdf5`` exists (else oracle bounds below).
        if dataset_fp.exists():
            dataset = self.get_dataset()
            self.action_stats = {
                "min": dataset["action_min"],
                "max": dataset["action_max"],
            }
        else:
            # Needed for collecting trajectories before sort.hdf5 exists.
            # These are workspace/action bounds used to normalize oracle actions.
            self.action_stats = {
                "min": np.array([0.25, -0.5, 0.0, 0.0], dtype=np.float32),
                "max": np.array([0.75, 0.5, 0.35, 1.0], dtype=np.float32),
            }

            # note: these are read from h5 of VPL's official sort.hdf5
            # min: [-0.03602862 -0.5022791  -0.03594894  0.]
            # max: [0.74327385 0.49079713 0.39760613 1.]

    @property
    def target(self):
        return self.curr_goal_pose

    def goal_pose_for_mode(self, mode: int):
        """
        args:
            mode: int, index into ``goal_poses`` (0..len(goal_poses)-1 in multimodal)
        returns:
            goal_pose: tuple, ((x, y, z), (qx, qy, qz, qw))
        """
        mode = int(mode)
        if mode < 0 or mode >= len(self.goal_poses):
            raise ValueError(
                f"Invalid goal mode {mode}; expected 0..{len(self.goal_poses) - 1}"
            )
        return self.goal_poses[mode]

    def get_dataset(self, h5path=None, path=None):
        """
        Load a D4RL-style HDF5 dict (see ``d4rl.offline_env.OfflineEnv.get_dataset``).

        ``d4rl.qlearning_dataset(env, h5path=...)`` forwards ``h5path`` here; ``path`` is
        an alias. Default is ``self.dataset_path`` (bundled ``sort.hdf5`` unless overridden
        via env kw ``dataset_path=`` in :meth:`__init__`).
        """
        p = h5path or path or self.dataset_path
        return super().get_dataset(h5path=p)

    def get_obs(self):
        ee_pose = self.env.get_ee_pose()  # ((x, y, z), (qx, qy, qz, qw))
        ee_xyz = np.array(ee_pose[0])  # (x, y, z)

        block_id = getattr(self.task, "block_id", None)
        assert block_id is not None, "task did not expose a block_id after reset()."
        block_xyz, _ = p.getBasePositionAndOrientation(block_id)
        block_xyz = np.array(block_xyz)  # (x, y, z)

        grasp = np.array([self.env.ee.check_grasp()])  # (1,) bool

        obs = np.concatenate([ee_xyz, block_xyz, grasp]).flatten()
        return obs  # (7,), (ee x,y,z, obj x,y,z, grasp)

    def array2dict(self, action):
        """
        normalize the action to the range of [0, 1], then wrap it as dict

        args:
            action: (4,), (delta_x, delta_y, delta_z, suction)
        returns:
            dict: {
                "move_cmd": (pose, (0, 0, 0, 1)),
                "suction_cmd": suction,
            }

        """
        action = np.clip(action, -1, 1)
        action = (action + 1) / 2
        action = (
            action * (self.action_stats["max"] - self.action_stats["min"])
            + self.action_stats["min"]
        )
        delta_pos = action[:3]
        suction = action[3]
        # current_pos = np.array(self.env.get_ee_pose()[0])
        # pose = current_pos + delta_pos
        pose = delta_pos
        suction = int(suction > 0.2)

        return {
            "move_cmd": (pose, (0, 0, 0, 1)),
            "suction_cmd": suction,
        }

    def compute_reward(self, state, mode: int):
        """
        args:
            state: (A, T, O), where O = [ee_x, ee_y, ee_z, obj_x, obj_y, obj_z, grasp]
            mode: int, decides which goal pose to use
        returns:
            reward: (A, T)
        """
        goal_pose = self.goal_pose_for_mode(mode)  # ((x, y, z), (qx, qy, qz, qw))
        eef_xyz = state[:, :, 0:3]  # (A, T, 3)
        obj_xyz = state[:, :, 3:6]  # (A, T, 3)
        goal_xyz = np.array(goal_pose[0])  # (3,)

        eef_dist = np.linalg.norm(eef_xyz - obj_xyz, axis=-1)  # (A, T)
        goal_dist = np.linalg.norm(obj_xyz[:, :, :2] - goal_xyz[:2], axis=-1)  # (A, T)
        within = goal_dist < 0.05  # (A, T)
        grasped = state[:, :, -1]  # (A, T)

        # Piecewise dense shaping (scaled by ./100 below):
        #   within_goal & ~grasped: strong bonus at goal XY
        #   within_goal & grasped: smaller bonus still at goal
        #   ~within_goal & grasped: intermediate + decay with goal_dist
        #   ~within_goal & ~grasped: encourage EEF→object proximity
        reward = (
            within * (1 - grasped) * 100.0
            + within * grasped * 5.0
            + (1 - within) * grasped * (2 + np.exp(-goal_dist))
            + (1 - within) * (1 - grasped) * np.exp(-eef_dist)
        )
        return reward / 100.0

    def compute_sparse_reward(self, obs):
        """
        args:
            obs: (7,), where obs = [ee_x, ee_y, ee_z, obj_x, obj_y, obj_z, grasp]
        returns:
            reward: float
        """
        goal_pose = self.goal_pose_for_mode(self.mode)  # ((x, y, z), (qx, qy, qz, qw))
        goal_xyz = np.array(goal_pose[0])  # (3,)
        obj_xyz = obs[3:6]  # (3,)
        goal_dist = np.linalg.norm(obj_xyz[:2] - goal_xyz[:2])
        obj_z = obj_xyz[2]
        grasp = self.env.ee.check_grasp()
        return float(goal_dist < 0.05 and obj_z < 0.03 and not grasp)

    def reset(self):
        self.env.reset()
        self.elasped_steps = 0
        return self.get_obs()

    @timeout_decorator.timeout(2)
    def step(self, action):
        action_dict = self.array2dict(action)
        obs, reward, done, info = self.env.step(action_dict)
        obs = self.get_obs()

        sparse_reward = float(self.compute_sparse_reward(obs))
        dense_reward = float(self.compute_reward(obs[None, None], self.mode)[0, 0])
        reward = sparse_reward
        done = done or reward > 0.99

        info = {}
        goal_pos = self.curr_goal_pose[0][:2]
        dist = np.linalg.norm(goal_pos - obs[3:5])
        info["debug_string"] = f"{dist:.3f}, {action[-1]:.3f}, is grasped: {obs[-1]}"
        info["dense_reward"] = dense_reward
        info["sparse_reward"] = sparse_reward
        if done:
            info["distance_to_goal"] = dist
        self.elasped_steps += 1
        done = done or (self.elasped_steps >= 50)  # self.max_episode_steps
        # info["is_success"] = (reward > 0.99)
        return obs, reward, done, info

    @timeout_decorator.timeout(2)
    def step_oracle(self, action):
        obs, reward, env_done, info_in = self.env.step(action)
        obs = self.get_obs()
        goal_dist = np.linalg.norm(obs[3:5] - self.curr_goal_pose[0][:2])

        dense_reward = float(self.compute_reward(obs[None, None], self.mode)[0, 0])
        sparse_reward = float(self.compute_sparse_reward(obs))

        info = dict(info_in) if info_in is not None else {}
        info["dense_reward"] = dense_reward
        info["sparse_reward"] = sparse_reward
        info["distance_to_goal"] = float(goal_dist)
        self.elasped_steps += 1
        # done = (
        #     bool(env_done)
        #     or sparse_reward > 0.99
        #     or (self.elasped_steps >= self._max_episode_steps)
        # )
        done = self.elasped_steps >= self._max_episode_steps
        reward = sparse_reward
        return obs, reward, done, info

    ## Functions to handle multimodality
    def get_num_modes(self):
        if self.is_multimodal:
            return len(self.goal_poses)
        return 1

    def sample_mode(self) -> int:
        if self.is_multimodal:
            return np.random.randint(len(self.goal_poses))
        return self.mode

    def reset_mode(self):
        self.set_mode(self.sample_mode())

    def set_mode(self, mode: int):
        mode = int(mode)
        if mode < 0 or mode >= len(self.goal_poses):
            raise ValueError(
                f"Invalid goal mode {mode}; expected 0..{len(self.goal_poses) - 1}"
            )
        self.mode = mode
        self.curr_goal_id = mode
        self.curr_goal_pose = self.goal_poses[mode]
        self.task.goal_pose = self.curr_goal_pose

    def render(self, mode="rgb_array"):
        return self.env.render(mode)


class SortPreferenceTwoGoals:
    """Trajectory hypotheses over exactly two ``goal_poses``; inject reward from :class:`SortEnv`."""

    SOURCES_NAMES = ["pref_horizontal"]
    SOURCES_DIMS = [2]
    SOURCES_TYPES = ["discrete"]

    ANNOTATOR_SPECS = [
        {"name": "left", "pref_horizontal": 0},
        {"name": "right", "pref_horizontal": 1},
    ]

    def __init__(
        self,
        *,
        goal_poses: Sequence,
        env_reward_fn: Callable[[np.ndarray, int], np.ndarray],
    ):
        gp = list(goal_poses)
        if len(gp) != 2:
            raise ValueError(
                f"SortPreferenceTwoGoals requires exactly two goal poses; got len={len(gp)}"
            )
        self.goal_poses = gp
        self.env_reward_fn = env_reward_fn

    @property
    def n_sources(self) -> int:
        return len(self.SOURCES_NAMES)

    @property
    def n_pref_modes(self) -> int:
        return 2

    @property
    def source_specs(self) -> dict:
        return dict(
            n_sources=self.n_sources,
            source_names=self.SOURCES_NAMES,
            source_types=self.SOURCES_TYPES,
            source_dims=self.SOURCES_DIMS,
        )

    def sample_pref_mode(self) -> int:
        return int(np.random.randint(0, self.n_pref_modes))

    def pref_spec_for_mode(
        self, annotator_id: int, *, as_array: bool = False
    ) -> dict | np.ndarray:
        """
        ``as_array=False``: metadata dict from ``ANNOTATOR_SPECS``.

        ``as_array=True``: discrete source vector ``(1,)`` — ``0`` = left, ``1`` = right
        (from ``pref_horizontal``, aligned with ``annotator_id``).
        """
        annotator_id = int(annotator_id)
        specs = self.ANNOTATOR_SPECS
        assert 0 <= annotator_id < len(specs), (
            f"Invalid annotator_id {annotator_id}; expected 0..{len(specs) - 1}"
        )
        spec = specs[annotator_id]
        if as_array:
            code = [spec["pref_horizontal"]]
            return np.array(code, dtype=np.int32)  # (1,)
        return spec

    def compute_reward(self, state: np.ndarray, mode: int) -> np.ndarray:
        mode = int(mode)
        if mode not in list(range(self.n_pref_modes)):
            raise ValueError(
                f"SortPreferenceTwoGoals.compute_reward expects mode in {{0, 1}}; got {mode}"
            )
        return self.env_reward_fn(state, mode)


class SortPreferenceFourGoals:
    """
    Four trajectory hypotheses plus annotator preference anchors for relabeling.

    Inject ``goal_poses`` (length 4) and ``compute_reward`` from a live :class:`SortEnv`.

    Table view (``collect_trajs`` plot — rgb front camera: bottom ≈ larger world x, right = +y):
        * ``pref_vertical`` (world x, table bottom↔top): 0 = ignore; 1 = top (high x); 2 = bottom (low x).
        * ``pref_horizontal`` (world y, table left↔right): 0 = ignore; 1 = left (low y); 2 = right (high y).
    Uses 4-corner workspace span for anchors (covers 2-stripe layouts too).
    """

    SOURCES_NAMES = ["pref_vertical", "pref_horizontal"]
    SOURCES_DIMS = [3, 3]
    SOURCES_TYPES = ["discrete", "discrete"]

    ANNOTATOR_SPECS = [
        {"name": "left", "pref_vertical": 0, "pref_horizontal": 1},
        {"name": "right", "pref_vertical": 0, "pref_horizontal": 2},
        {"name": "top", "pref_vertical": 1, "pref_horizontal": 0},
        {"name": "bottom", "pref_vertical": 2, "pref_horizontal": 0},
        {"name": "top_left", "pref_vertical": 1, "pref_horizontal": 1},
        {"name": "top_right", "pref_vertical": 1, "pref_horizontal": 2},
        {"name": "bottom_left", "pref_vertical": 2, "pref_horizontal": 1},
        {"name": "bottom_right", "pref_vertical": 2, "pref_horizontal": 2},
    ]

    ANNOT_X_LIM = (0.32, 0.68)
    ANNOT_Y_LIM = (-0.18, 0.18)

    def __init__(
        self,
        *,
        goal_poses: Sequence,
        env_reward_fn: Callable[[np.ndarray, int], np.ndarray],
    ):
        gp = list(goal_poses)
        if len(gp) != 4:
            raise ValueError(
                f"SortPreferenceFourGoals requires exactly four goal poses; got len={len(gp)}"
            )
        self.goal_poses = gp
        self.env_reward_fn = env_reward_fn

    @staticmethod
    def _annot_pref_edge(
        *, pref: int, low: float, high: float, vertical_axis: bool
    ) -> float | None:
        """
        pref ``1``/``2`` → ``low``/``high`` on horizontal world ``y``;
        XOR swap for world ``x`` (``vertical_axis=True``: ``1`` = top/high ``x``, ``2`` = bottom/low ``x``).
        """
        if pref == 0:
            return None
        if pref not in (1, 2):
            raise ValueError(f"preference code must be 0, 1, or 2; got {pref!r}")
        use_high = (pref == 2) ^ vertical_axis
        return float(high if use_high else low)

    @property
    def n_sources(self) -> int:
        return len(self.SOURCES_NAMES)

    @property
    def n_pref_modes(self) -> int:
        return len(self.ANNOTATOR_SPECS)

    @property
    def source_specs(self) -> dict:
        return dict(
            n_sources=self.n_sources,
            source_names=self.SOURCES_NAMES,
            source_types=self.SOURCES_TYPES,
            source_dims=self.SOURCES_DIMS,
        )

    def sample_pref_mode(self) -> int:
        return int(np.random.randint(0, self.n_pref_modes))

    def pref_spec_for_mode(
        self, annotator_id: int, *, as_array: bool = False
    ) -> dict | np.ndarray:
        """
        ``as_array=False``: full annotator metadata dict.

        ``as_array=True``: ``(pref_vertical, pref_horizontal)`` with values in ``{0,1,2}``
        per the class docstring.
        """
        annotator_id = int(annotator_id)
        specs = self.ANNOTATOR_SPECS
        assert 0 <= annotator_id < len(specs), (
            f"Invalid annotator_id {annotator_id}; expected 0..{len(specs) - 1}"
        )
        spec = specs[annotator_id]
        if as_array:
            code = [spec["pref_vertical"], spec["pref_horizontal"]]
            return np.array(code, dtype=np.int32)  # (2,)
        return spec

    def compute_reward(self, state: np.ndarray, mode: int) -> np.ndarray:
        mode = int(mode)
        if mode not in list(range(self.n_pref_modes)):
            raise ValueError(
                f"SortPreferenceFourGoals.compute_reward expects mode in {{0..{self.n_pref_modes - 1}}}; got {mode}"
            )
        return self.compute_annotator_reward(state, mode)
        # return self.env_reward_fn(state, mode)

    def compute_annotator_reward(
        self, state: np.ndarray, annotator_id: int
    ) -> np.ndarray:
        """
        Preference-only dense surrogate for relabeling (not the task MDP reward).

        Mirrors :meth:`SortEnv.compute_reward` piecewise shaping; ``pref_dist`` replaces goal distance—
        Euclidean distance in world XY from the object to the annotator anchor (vertical and/or horizontal targets).

        args:
            state: (A, T, O), where O = [ee_x, ee_y, ee_z, obj_x, obj_y, obj_z, grasp]
            annotator_id: index into ``ANNOTATOR_SPECS``
        returns:
            reward: (A, T)
        """
        spec = self.pref_spec_for_mode(annotator_id=annotator_id, as_array=False)
        eef_xyz = state[:, :, 0:3]  # (A, T, 3)
        obj_xyz = state[:, :, 3:6]  # (A, T, 3)
        grasped = state[:, :, -1]  # (A, T)
        obj_x = obj_xyz[:, :, 0]
        obj_y = obj_xyz[:, :, 1]

        x_lo, x_hi = self.ANNOT_X_LIM
        y_lo, y_hi = self.ANNOT_Y_LIM

        t_vert = self._annot_pref_edge(
            pref=int(spec["pref_vertical"]), low=x_lo, high=x_hi, vertical_axis=True
        )
        t_horiz = self._annot_pref_edge(
            pref=int(spec["pref_horizontal"]), low=y_lo, high=y_hi, vertical_axis=False
        )

        sq_terms: list[np.ndarray] = []
        if t_vert is not None:
            sq_terms.append((obj_x - float(t_vert)) ** 2)
        if t_horiz is not None:
            sq_terms.append((obj_y - float(t_horiz)) ** 2)
        if not sq_terms:
            raise RuntimeError(
                f"Annotator {annotator_id} ({spec['name']}) has "
                f"pref_vertical=pref_horizontal=0 (no active axis)."
            )

        pref_dist = np.sqrt(sum(sq_terms))  # (A, T), meters on active axes only

        eef_dist = np.linalg.norm(eef_xyz - obj_xyz, axis=-1)  # (A, T)
        within = pref_dist < 0.05  # (A, T), same band as ``compute_reward`` goal_xy

        # Piecewise dense shaping (scaled by ./100 below)—same weights as ``compute_reward``:
        #   within_pref & ~grasped: strong bonus at preferred XY anchor
        #   within_pref & grasped: smaller bonus still there
        #   ~within_pref & grasped: intermediate + decay with pref_dist
        #   ~within_pref & ~grasped: encourage EEF→object proximity
        reward = (
            within * (1 - grasped) * 100.0
            + within * grasped * 5.0
            + (1 - within) * grasped * (2 + np.exp(-pref_dist))
            + (1 - within) * (1 - grasped) * np.exp(-eef_dist)
        )
        return reward / 100.0

    def compute_sparse_annotator_reward(
        self, obs: np.ndarray, annotator_id: int
    ) -> float:
        """
        Preference sparse surrogate for relabeling (not the task MDP reward).

        Uses observation grasp channel ``obs[-1]`` (released iff zero), unlike :meth:`SortEnv.compute_sparse_reward`
        which queries the simulator—so offline HDF5 segments stay consistent without a live ``env``.

        args:
            obs: (7,), where obs = [ee_x, ee_y, ee_z, obj_x, obj_y, obj_z, grasp]
            annotator_id: index into ``ANNOTATOR_SPECS``
        returns:
            reward: float, 1.0 iff preferred XY band, low object height, and released.
        """
        spec = self.pref_spec_for_mode(annotator_id=annotator_id, as_array=False)
        obj_xyz = obs[3:6]  # (3,)
        obj_x = float(obj_xyz[0])
        obj_y = float(obj_xyz[1])
        obj_z = float(obj_xyz[2])

        x_lo, x_hi = self.ANNOT_X_LIM
        y_lo, y_hi = self.ANNOT_Y_LIM

        t_vert = self._annot_pref_edge(
            pref=int(spec["pref_vertical"]), low=x_lo, high=x_hi, vertical_axis=True
        )
        t_horiz = self._annot_pref_edge(
            pref=int(spec["pref_horizontal"]), low=y_lo, high=y_hi, vertical_axis=False
        )

        sq_terms: list[float] = []
        if t_vert is not None:
            sq_terms.append((obj_x - float(t_vert)) ** 2)
        if t_horiz is not None:
            sq_terms.append((obj_y - float(t_horiz)) ** 2)
        if not sq_terms:
            raise RuntimeError(
                f"Annotator {annotator_id} ({spec['name']}) has "
                f"pref_vertical=pref_horizontal=0 (no active axis)."
            )

        pref_dist = float(np.sqrt(sum(sq_terms)))
        grasped = float(obs[-1]) != 0.0
        return float(pref_dist < 0.05 and obj_z < 0.03 and not grasped)


def annotator_from_env(
    env: SortEnv,
) -> SortPreferenceTwoGoals | SortPreferenceFourGoals:
    """Build the trajectory-hypothesis preference helper matching ``len(env.goal_poses)``."""
    n = len(env.goal_poses)
    if n == 2:
        return SortPreferenceTwoGoals(
            goal_poses=env.goal_poses,
            env_reward_fn=env.compute_reward,
        )
    if n == 4:
        return SortPreferenceFourGoals(
            goal_poses=env.goal_poses,
            env_reward_fn=env.compute_reward,
        )
    raise ValueError(
        f"sort_preference_from_sort_env expects SortEnv with 2 or 4 goal poses; got {n}"
    )
