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

        sort_dataset = Path(__file__).resolve().parent / "dataset/sort.hdf5"
        self.dataset_path = str(sort_dataset)

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
        if sort_dataset.exists():
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

    def get_dataset(self, path=None):
        path = path or self.dataset_path
        return super().get_dataset(h5path=path)

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
        done = (
            bool(env_done)
            or sparse_reward > 0.99
            or (self.elasped_steps >= self._max_episode_steps)
        )
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
