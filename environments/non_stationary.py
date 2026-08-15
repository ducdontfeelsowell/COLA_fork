"""Reusable non-stationarity support for Gym environments.

The classes in this module intentionally use only Gym and NumPy so that they
remain compatible with the dependencies pinned in ``environmentV2.yml``.
"""

from __future__ import absolute_import

import copy
import math
from collections.abc import Mapping

import gym
import numpy as np
from gym.utils import seeding


_NOT_PROVIDED = object()


def _rng_integer(rng, high):
    """Draw an integer from either NumPy's RandomState or Generator API."""
    if hasattr(rng, "integers"):
        return int(rng.integers(high))
    return int(rng.randint(high))


def _sample_distribution(distribution, rng):
    """Sample one value from a supported distribution-like object.

    Supported inputs are Gym spaces, SciPy frozen distributions, callables
    accepting a NumPy random generator, mappings of distributions, and
    constants. Gym spaces are reseeded from ``rng`` before each sample. This
    keeps samples reproducible without relying on global NumPy state.
    """
    if isinstance(distribution, Mapping):
        return {
            key: _sample_distribution(value, rng)
            for key, value in distribution.items()
        }

    if isinstance(distribution, gym.spaces.Space):
        distribution.seed(_rng_integer(rng, 2 ** 31 - 1))
        return distribution.sample()

    if hasattr(distribution, "rvs") and callable(distribution.rvs):
        return distribution.rvs(random_state=rng)

    if callable(distribution):
        return distribution(rng)

    return copy.deepcopy(distribution)


def _snapshot(value):
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return copy.deepcopy(value)


class NormalDistribution(object):
    """Seedable normal distribution usable by the wrapper and scheduler.

    The optional bounds clip samples rather than resampling them. Bounds are
    useful for keeping physical parameters positive and timestep intervals in
    a practical range. Leave them unset for an ordinary unbounded normal
    distribution.

    Args:
        mean: Mean of the distribution.
        standard_deviation: Strictly positive standard deviation.
        low: Optional inclusive lower clipping bound.
        high: Optional inclusive upper clipping bound.
        size: Optional output shape. With no size, scalar parameters produce a
            scalar sample and array parameters use their broadcast shape.
    """

    def __init__(
        self,
        mean,
        standard_deviation,
        low=None,
        high=None,
        size=None,
    ):
        mean_array = np.asarray(mean)
        deviation_array = np.asarray(standard_deviation)
        if not np.all(np.isfinite(mean_array)):
            raise ValueError("normal distribution mean must be finite")
        if not np.all(np.isfinite(deviation_array)):
            raise ValueError("normal distribution standard deviation must be finite")
        if np.any(deviation_array <= 0):
            raise ValueError("normal distribution standard deviation must be positive")

        if low is not None and high is not None:
            try:
                invalid_bounds = np.any(np.asarray(low) > np.asarray(high))
            except ValueError as error:
                raise ValueError("normal distribution bounds do not broadcast") from error
            if invalid_bounds:
                raise ValueError("normal distribution low must not exceed high")

        if isinstance(size, int):
            size = (size,)
        elif size is not None:
            size = tuple(size)
        if size is not None and any(dimension < 0 for dimension in size):
            raise ValueError("normal distribution size cannot be negative")

        self.mean = _snapshot(mean)
        self.standard_deviation = _snapshot(standard_deviation)
        self.low = _snapshot(low)
        self.high = _snapshot(high)
        self.size = size

    def __call__(self, rng):
        sample = rng.normal(
            loc=self.mean,
            scale=self.standard_deviation,
            size=self.size,
        )
        if self.low is not None:
            sample = np.maximum(sample, self.low)
        if self.high is not None:
            sample = np.minimum(sample, self.high)

        sample_array = np.asarray(sample)
        if sample_array.ndim == 0:
            return sample_array.item()
        return sample_array


class RandomizedScheduler(object):
    """Schedule recurring events using randomized inter-event intervals.

    Args:
        interval_distribution: Distribution-like object sampled for the number
            of steps until the next event. See :func:`_sample_distribution`.
        seed: Optional random seed.
        interval_offset: Value added to every sampled interval. This is handy
            with ``gym.spaces.Discrete``; for example, ``Discrete(401)`` with
            an offset of 100 samples intervals from 100 through 500.
        minimum_interval: Smallest accepted interval.
        start_step: Initial scheduler clock value.

    Fractional samples are rounded up to the next whole environment step.
    """

    def __init__(
        self,
        interval_distribution,
        seed=None,
        interval_offset=0,
        minimum_interval=1,
        start_step=0,
    ):
        if interval_distribution is None:
            raise ValueError("interval_distribution must be provided")
        if int(minimum_interval) < 1:
            raise ValueError("minimum_interval must be at least 1")

        self.interval_distribution = interval_distribution
        self.interval_offset = interval_offset
        self.minimum_interval = int(minimum_interval)
        self._start_step = int(start_step)
        self.np_random = None
        self.seed_value = None
        self.current_step = None
        self.next_update_step = None
        self.last_interval = None
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, actual_seed = seeding.np_random(seed)
        self.seed_value = int(actual_seed)
        self.reset(self._start_step)
        return [self.seed_value]

    def sample_interval(self):
        sample = _sample_distribution(
            self.interval_distribution, self.np_random
        )
        array = np.asarray(sample)
        if array.size != 1:
            raise ValueError(
                "interval_distribution must produce exactly one value; got "
                "shape {}".format(array.shape)
            )

        value = float(array.reshape(-1)[0]) + float(self.interval_offset)
        if not np.isfinite(value):
            raise ValueError("sampled interval must be finite; got {}".format(value))

        interval = int(math.ceil(value))
        if interval < self.minimum_interval:
            raise ValueError(
                "sampled interval {} is below minimum_interval {}".format(
                    interval, self.minimum_interval
                )
            )
        return interval

    def reset(self, start_step=0):
        self.current_step = int(start_step)
        self.last_interval = self.sample_interval()
        self.next_update_step = self.current_step + self.last_interval
        return self.next_update_step

    def tick(self):
        """Advance one step and report whether an event is due."""
        self.current_step += 1
        if self.current_step < self.next_update_step:
            return False

        self.last_interval = self.sample_interval()
        self.next_update_step = self.current_step + self.last_interval
        return True


class NonStationaryEnv(gym.Wrapper):
    """Make any Gym environment non-stationary at randomized time steps.

    The default updater changes one or more attributes on ``env.unwrapped``.
    A custom ``update_fn(env, degree)`` can instead implement arbitrary
    changes. Scheduled updates happen immediately before the action at the due
    (one-based) global step is passed to the underlying environment.

    Distribution arguments accept Gym spaces, SciPy frozen distributions,
    callables accepting a NumPy random generator, constants, and (for degrees)
    mappings from parameter paths to any of those objects.
    """

    _CHANGE_MODES = ("scale", "add", "multiply", "set")

    def __init__(
        self,
        env,
        parameter_paths=None,
        degree_distribution=None,
        interval_distribution=None,
        update_fn=None,
        change_mode="scale",
        seed=None,
        interval_offset=0,
        minimum_interval=1,
        reset_schedule_on_env_reset=False,
        add_update_info=True,
        forward_after_update=True,
    ):
        super(NonStationaryEnv, self).__init__(env)

        if degree_distribution is None:
            raise ValueError("degree_distribution must be provided")
        if interval_distribution is None:
            raise ValueError("interval_distribution must be provided")
        if change_mode not in self._CHANGE_MODES:
            raise ValueError(
                "change_mode must be one of {}; got {!r}".format(
                    self._CHANGE_MODES, change_mode
                )
            )

        if isinstance(parameter_paths, str):
            parameter_paths = (parameter_paths,)
        elif parameter_paths is None:
            parameter_paths = ()
        else:
            parameter_paths = tuple(parameter_paths)
        if any(not isinstance(path, str) or not path for path in parameter_paths):
            raise ValueError("parameter_paths must contain non-empty strings")
        if any("" in path.split(".") for path in parameter_paths):
            raise ValueError("parameter paths cannot contain empty components")
        if not parameter_paths and update_fn is None:
            raise ValueError("provide at least one parameter path or update_fn")

        self.parameter_paths = parameter_paths
        self.degree_distribution = degree_distribution
        self.update_fn = update_fn
        self.change_mode = change_mode
        self.reset_schedule_on_env_reset = bool(reset_schedule_on_env_reset)
        self.add_update_info = bool(add_update_info)
        self.forward_after_update = bool(forward_after_update)

        self._initial_values = {
            path: _snapshot(self._get_parameter(path))
            for path in self.parameter_paths
        }
        self._degree_rng = None
        self.seed_value = None
        self.update_count = 0
        self.last_update = None

        master_rng, actual_seed = seeding.np_random(seed)
        self.seed_value = int(actual_seed)
        scheduler_seed = _rng_integer(master_rng, 2 ** 31 - 1)
        degree_seed = _rng_integer(master_rng, 2 ** 31 - 1)
        self.scheduler = RandomizedScheduler(
            interval_distribution=interval_distribution,
            seed=scheduler_seed,
            interval_offset=interval_offset,
            minimum_interval=minimum_interval,
        )
        self._degree_rng, _ = seeding.np_random(degree_seed)
        if hasattr(self.env, "seed"):
            self.env.seed(self.seed_value)

    @property
    def elapsed_steps(self):
        return self.scheduler.current_step

    @property
    def next_update_step(self):
        return self.scheduler.next_update_step

    def seed(self, seed=None):
        master_rng, actual_seed = seeding.np_random(seed)
        self.seed_value = int(actual_seed)
        scheduler_seed = _rng_integer(master_rng, 2 ** 31 - 1)
        degree_seed = _rng_integer(master_rng, 2 ** 31 - 1)
        self.scheduler.seed(scheduler_seed)
        self._degree_rng, _ = seeding.np_random(degree_seed)
        self.update_count = 0
        self.last_update = None

        if hasattr(self.env, "seed"):
            self.env.seed(self.seed_value)
        return [self.seed_value]

    def reset_scheduler(self, start_step=0):
        """Restart only the update clock, without resetting the environment."""
        self.update_count = 0
        self.last_update = None
        return self.scheduler.reset(start_step)

    def reset(self, **kwargs):
        if self.reset_schedule_on_env_reset:
            self.reset_scheduler()
        return self.env.reset(**kwargs)

    def step(self, action):
        event = self.update() if self.scheduler.tick() else None
        result = self.env.step(action)

        if event is None or not self.add_update_info:
            return result
        if not isinstance(result, tuple) or len(result) not in (4, 5):
            raise TypeError(
                "wrapped env.step() must return a Gym 4-tuple or 5-tuple"
            )

        result_parts = list(result)
        info = {} if result_parts[-1] is None else dict(result_parts[-1])
        info["non_stationary_update"] = event
        result_parts[-1] = info
        return tuple(result_parts)

    def update(self, degree=_NOT_PROVIDED):
        """Apply one change immediately and return its metadata.

        Supplying ``degree`` bypasses degree sampling. Manual updates do not
        alter the next scheduled update time.
        """
        if degree is _NOT_PROVIDED:
            degree = _sample_distribution(
                self.degree_distribution, self._degree_rng
            )

        callback_result = None
        if self.update_fn is not None:
            callback_result = self.update_fn(self.env.unwrapped, degree)
        else:
            self._apply_default_update(degree)

        self._forward_simulation()
        self.update_count += 1
        event = {
            "step": int(self.scheduler.current_step),
            "count": int(self.update_count),
            "degree": _snapshot(degree),
        }
        if self.parameter_paths:
            event["parameters"] = {
                path: _snapshot(self._get_parameter(path))
                for path in self.parameter_paths
            }
        if callback_result is not None:
            event["result"] = _snapshot(callback_result)

        self.last_update = event
        return event

    def restore_parameters(self):
        """Restore path-based parameters to values captured at construction."""
        for path, value in self._initial_values.items():
            self._set_parameter(path, _snapshot(value))
        self._forward_simulation()

    def _forward_simulation(self):
        if not self.forward_after_update:
            return
        simulation = getattr(self.env.unwrapped, "sim", None)
        if simulation is not None and hasattr(simulation, "forward"):
            simulation.forward()

    def _apply_default_update(self, degree):
        if isinstance(degree, Mapping):
            missing = set(self.parameter_paths).difference(degree)
            if missing:
                raise KeyError(
                    "degree mapping is missing parameter(s): {}".format(
                        ", ".join(sorted(missing))
                    )
                )

        for path in self.parameter_paths:
            path_degree = degree[path] if isinstance(degree, Mapping) else degree
            initial = self._initial_values[path]
            current = self._get_parameter(path)

            if self.change_mode == "scale":
                new_value = np.asarray(initial) * path_degree
            elif self.change_mode == "add":
                new_value = np.asarray(initial) + path_degree
            elif self.change_mode == "multiply":
                new_value = np.asarray(current) * path_degree
            else:
                new_value = path_degree

            self._set_parameter(path, new_value)

    def _get_parameter(self, path):
        owner = self.env.unwrapped
        for component in path.split("."):
            if not hasattr(owner, component):
                raise AttributeError(
                    "{} has no parameter path {!r}".format(
                        type(self.env.unwrapped).__name__, path
                    )
                )
            owner = getattr(owner, component)
        return owner

    def _set_parameter(self, path, value):
        owner = self.env.unwrapped
        components = path.split(".")
        for component in components[:-1]:
            if not hasattr(owner, component):
                raise AttributeError(
                    "{} has no parameter path {!r}".format(
                        type(self.env.unwrapped).__name__, path
                    )
                )
            owner = getattr(owner, component)

        name = components[-1]
        if not hasattr(owner, name):
            raise AttributeError(
                "{} has no parameter path {!r}".format(
                    type(self.env.unwrapped).__name__, path
                )
            )

        current = getattr(owner, name)
        if isinstance(current, np.ndarray):
            try:
                current[...] = value
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "cannot assign sampled value to parameter {!r} with shape "
                    "{}: {}".format(path, current.shape, error)
                )
            return

        if isinstance(current, (int, float, np.number)):
            array = np.asarray(value)
            if array.size != 1:
                raise ValueError(
                    "scalar parameter {!r} requires one sampled value; got "
                    "shape {}".format(path, array.shape)
                )
            scalar = array.reshape(-1)[0]
            if isinstance(current, np.generic):
                value = current.dtype.type(scalar)
            else:
                value = type(current)(scalar)

        setattr(owner, name, value)


NonStationaryWrapper = NonStationaryEnv


__all__ = [
    "NormalDistribution",
    "NonStationaryEnv",
    "NonStationaryWrapper",
    "RandomizedScheduler",
]
