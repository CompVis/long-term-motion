# Adapted from https://github.com/pairlab/AMPLIFY
import copy

import numpy as np
from robosuite.wrappers import Wrapper

from .custom_venv import DummyVectorEnv, SubprocVectorEnv


class EnvStateWrapper(Wrapper):
    """
    Wrapper to capture full state information for reset including sim, controller, and gripper states
    """

    def __init__(self, env):
        super(EnvStateWrapper, self).__init__(env)

    def _get_controller_state(self):
        """
        Makes a deep copy of controller state, excluding MjSim reference to avoid pickling error
        """
        controller_state = self.env.robots[0].controller.__dict__
        # copy all attributes except sim
        controller_state_copy = {}
        for key, value in controller_state.items():
            if key != "sim":
                controller_state_copy[key] = copy.deepcopy(value)

        return controller_state_copy

    def _set_controller_state(self, controller_state):
        """
        Sets the controller state from a dictionary
        """
        self.env.robots[0].controller.__dict__.update(controller_state)

    def _get_gripper_state(self):
        return self.env.robots[0].gripper.current_action.copy()

    def _set_gripper_state(self, gripper_state):
        self.env.robots[0].gripper.current_action = gripper_state

    def get_env_state(self):
        return {
            "sim_state": self.env.get_sim_state(),
            "controller_state": self._get_controller_state(),
            "gripper_state": self._get_gripper_state(),
        }

    def set_env_state(self, env_state):
        self.env.set_state(env_state["sim_state"])
        self.env.sim.forward()
        self._set_controller_state(env_state["controller_state"])
        self._set_gripper_state(env_state["gripper_state"])


class FourDOFWrapper(Wrapper):
    """
    Wrapper to take 4 DOF action (no rotations) and convert to 7 DOF action.
    Action indices are 0,1,2,6; rotation indices are 3,4,5, which are set to zero
    """

    def __init__(self, env):
        super(FourDOFWrapper, self).__init__(env)

    def step(self, action):
        full_action = np.zeros(7)
        full_action[[0, 1, 2, 6]] = action
        return self.env.step(full_action)


class LiberoTaskEmbWrapper(Wrapper):
    """Wrapper to add task embeddings to the returned info"""

    def __init__(self, env, task_emb):
        super().__init__(env)
        self.task_emb = task_emb

    def reset(self):
        obs = self.env.reset()
        obs["task_emb"] = self.task_emb
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["task_emb"] = self.task_emb
        return obs, reward, done, info


class LiberoResetWrapper(Wrapper):
    """Wrap the complex state initialization process in LIBERO"""

    def __init__(self, env, init_states, init_state_no=0):
        super().__init__(env)
        self.init_states = init_states
        self.reset_times = init_state_no

    def reset(self):
        _ = self.env.reset()
        print(f"Resetting environment")
        obs = self.env.set_init_state(self.init_states[self.reset_times])

        # dummy actions all zeros for initial physics simulation
        dummy = np.zeros(7)
        dummy[-1] = -1.0  # set the last action to -1 to open the gripper
        for _ in range(5):
            obs, _, _, _ = self.env.step(dummy)

        self.reset_times += 1
        if self.reset_times == len(self.init_states):
            self.reset_times = 0
        return obs

    def seed(self, seed):
        self.env.seed(seed)


class LiberoObservationWrapper(Wrapper):
    """Wrapper to stack observations from multiple cameras, adds combined obs as 'image' key in dict"""

    def __init__(self, env, masks, cameras):
        super(LiberoObservationWrapper, self).__init__(env)
        self.masks = masks
        self.cameras = cameras

    def reset(self):
        obs = self.env.reset()
        obs_dict = self._stack_obs(obs)
        return obs_dict

    def regenerate_obs_from_state(self, mujoco_state):
        obs = self.env.regenerate_obs_from_state(mujoco_state)
        obs_dict = self._stack_obs(obs)
        return obs_dict

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_dict = self._stack_obs(obs)
        return obs_dict, reward, done, info

    def _stack_obs(self, obs):
        obs_dict = copy.deepcopy(obs)
        obs_dict["image"] = []
        obs_dict["depth"] = []
        obs_dict["segmentation"] = []
        for c in self.cameras:
            mod = obs[f"{c}_image"]
            obs_dict["image"].append(mod)

            if f"{c}_depth" in obs.keys():
                mod = obs[f"{c}_depth"]
                mod = mod.squeeze(-1)
                obs_dict["depth"].append(mod)

            if f"{c}_segmentation_instance" in obs.keys():
                mod = obs[f"{c}_segmentation_instance"]
                mod = mod.squeeze(-1)
                obs_dict["segmentation"].append(mod)

        # check for vectorized env
        if obs_dict["image"][0].ndim == 4:
            axis = 1
        else:
            axis = 0

        obs_dict["image"] = np.stack(obs_dict["image"], axis=axis)
        obs_dict["depth"] = np.stack(obs_dict["depth"], axis=axis) if len(obs_dict["depth"]) > 0 else None
        obs_dict["segmentation"] = (
            np.stack(obs_dict["segmentation"], axis=axis) if len(obs_dict["segmentation"]) > 0 else None
        )

        return obs_dict


class LiberoImageUpsideDownWrapper(Wrapper):
    """Wrapper to flip the image upside down. Not vectorized like the ATM one!"""

    def __init__(self, env):
        super(LiberoImageUpsideDownWrapper, self).__init__(env)

    def flip_image(self, obs):
        keys = [
            "agentview_image",
            "robot0_eye_in_hand_image",
            "agentview_depth",
            "robot0_eye_in_hand_depth",
            "agentview_segmentation_instance",
            "robot0_eye_in_hand_segmentation_instance",
        ]
        for k in keys:
            if k in obs.keys():
                obs[k] = obs[k][::-1, :, :]
        return obs

    def reset(self):
        obs = self.env.reset()
        return self.flip_image(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.flip_image(obs), reward, done, info

    def regenerate_obs_from_state(self, mujoco_state):
        obs = self.env.regenerate_obs_from_state(mujoco_state)
        return self.flip_image(obs)


class LiberoExpandObsWrapper(Wrapper):
    """Wrapper to add first dimension to single-view observations"""

    def __init__(self, env):
        super(LiberoExpandObsWrapper, self).__init__(env)

    def reset(self):
        obs = self.env.reset()
        obs["agentview_image"] = np.expand_dims(obs["agentview_image"], axis=0)
        obs["robot0_eye_in_hand_image"] = np.expand_dims(obs["robot0_eye_in_hand_image"], axis=0)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["agentview_image"] = np.expand_dims(obs["agentview_image"], axis=0)
        obs["robot0_eye_in_hand_image"] = np.expand_dims(obs["robot0_eye_in_hand_image"], axis=0)
        return obs, reward, done, info


class LiberoSuccessWrapper(Wrapper):
    """Wrapper to check for success in the environment"""

    def __init__(self, env):
        super(LiberoSuccessWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = self.env.check_success()
        return obs, reward, done, info


def merge_dict(dict_obj):
    merged_dict = {}
    for k in dict_obj[0].keys():
        merged_dict[k] = np.stack([d[k] for d in dict_obj], axis=0)
    return merged_dict


######################################
### Custom Vectorized environments ###
######################################


class StackDummyVectorEnv(DummyVectorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, id=None):
        obs = super().reset(id=id)
        return merge_dict(obs)

    def step(
        self,
        action: np.ndarray,
        id=None,
    ):
        obs, reward, done, info = super().step(action, id)
        return merge_dict(obs), reward, done, merge_dict(info)

    def regenerate_obs_from_state(self, mujoco_state):
        obs = super().regenerate_obs_from_state(mujoco_state)
        return merge_dict(obs)


class StackSubprocVectorEnv(SubprocVectorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        obs = super().reset()
        return merge_dict(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return merge_dict(obs), reward, done, merge_dict(info)

    def regenerate_obs_from_state(self, mujoco_state):
        obs = super().regenerate_obs_from_state(mujoco_state)
        return merge_dict(obs)
