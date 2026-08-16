"""Compatibility layer for COLA's legacy Gym MuJoCo environments.

COLA was written against Gym 0.15 and ``mujoco-py``.  ``mujoco-py`` 2.x
does not support Windows, while Gymnasium uses the maintained official
``mujoco`` bindings.  This adapter keeps the small part of the old
``MujocoEnv`` surface used by COLA without changing the environment rewards
or dynamics.
"""

from functools import wraps
from types import SimpleNamespace

import numpy as np
import mujoco
from gymnasium import spaces
from gymnasium.envs.mujoco import mujoco_env as _mujoco_env
from gymnasium.utils import seeding


class _LegacySim(object):
    """Expose the ``sim.data`` and ``sim.forward()`` calls used by COLA."""

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def forward(self):
        mujoco.mj_forward(self.model, self.data)


class MujocoEnv(_mujoco_env.MujocoEnv):
    """Accept the old constructor and expose the old ``sim.data`` alias."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init_subclass__(cls, **kwargs):
        """Convert each legacy subclass's 4-tuple step to Gymnasium's API."""
        super(MujocoEnv, cls).__init_subclass__(**kwargs)
        legacy_step = cls.__dict__.get("step")
        if legacy_step is None:
            return

        @wraps(legacy_step)
        def modern_step(self, action):
            result = legacy_step(self, action)
            if not isinstance(result, tuple):
                raise TypeError("MuJoCo environment step() must return a tuple")
            if len(result) == 5:
                return result
            if len(result) != 4:
                raise TypeError(
                    "MuJoCo environment step() must return 4 or 5 values"
                )
            observation, reward, done, info = result
            return observation, reward, bool(done), False, info

        cls.step = modern_step

    def __init__(self, model_path, frame_skip, observation_space=None, **kwargs):
        if observation_space is None:
            # The model must be initialized before COLA can compute its exact
            # observation shape.  Gymnasium only needs a temporary space here.
            observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float64,
            )

        super(MujocoEnv, self).__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            **kwargs
        )

        # Legacy COLA environments consistently access self.sim.data.  The
        # official bindings expose the same data directly as self.data.
        self.sim = _LegacySim(self.model, self.data)

        observation = np.asarray(self._get_obs())
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation.shape,
            dtype=observation.dtype,
        )

    def seed(self, seed=None):
        """Provide COLA's legacy seed hook while using Gymnasium internally."""
        self.np_random, actual_seed = seeding.np_random(seed)
        self.action_space.seed(actual_seed)
        return [actual_seed]


# Preserve the old import/call shape: ``mujoco_env.MujocoEnv``.
mujoco_env = SimpleNamespace(MujocoEnv=MujocoEnv)
