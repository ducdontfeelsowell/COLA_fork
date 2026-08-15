import argparse
import unittest

import gym
import numpy as np

from environments.non_stationary import (
    NormalDistribution,
    NonStationaryEnv,
    RandomizedScheduler,
    add_non_stationary_arguments,
    wrap_environment_from_args,
)


class SequenceDistribution(object):
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def __call__(self, rng):
        value = self.values[self.index % len(self.values)]
        self.index += 1
        return value


class DummySimulation(object):
    def __init__(self):
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class DummyModel(object):
    def __init__(self):
        self.body_mass = np.array([1.0, 2.0], dtype=np.float64)


class DummyEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.ones(1, dtype=np.float32),
            high=np.ones(1, dtype=np.float32),
            dtype=np.float32,
        )
        self.gain = 10.0
        self.vector = np.array([1.0, 2.0], dtype=np.float64)
        self.model = DummyModel()
        self.sim = DummySimulation()
        self.steps = 0
        self.seed_value = None

    def seed(self, seed=None):
        self.seed_value = seed
        return [seed]

    def reset(self, **kwargs):
        self.steps = 0
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        return (
            np.array([self.gain], dtype=np.float32),
            self.gain,
            False,
            {"underlying_step": self.steps},
        )


class RandomizedSchedulerTests(unittest.TestCase):
    def test_uses_sampled_intervals_and_offset(self):
        scheduler = RandomizedScheduler(
            SequenceDistribution([0, 2]), interval_offset=2, seed=7
        )
        self.assertEqual(scheduler.next_update_step, 2)
        self.assertFalse(scheduler.tick())
        self.assertTrue(scheduler.tick())
        self.assertEqual(scheduler.next_update_step, 6)

    def test_rejects_non_positive_intervals(self):
        with self.assertRaises(ValueError):
            RandomizedScheduler(0, seed=7)

    def test_normal_intervals_are_bounded_and_reproducible(self):
        def make_scheduler():
            return RandomizedScheduler(
                NormalDistribution(
                    mean=10,
                    standard_deviation=3,
                    low=2,
                    high=20,
                ),
                seed=29,
            )

        first = make_scheduler()
        second = make_scheduler()
        first_intervals = [first.last_interval]
        second_intervals = [second.last_interval]
        for _ in range(4):
            first.reset()
            second.reset()
            first_intervals.append(first.last_interval)
            second_intervals.append(second.last_interval)

        self.assertEqual(first_intervals, second_intervals)
        self.assertTrue(all(2 <= value <= 20 for value in first_intervals))


class NormalDistributionTests(unittest.TestCase):
    def test_samples_requested_shape_and_clips_bounds(self):
        distribution = NormalDistribution(
            mean=1.0,
            standard_deviation=0.25,
            low=0.8,
            high=1.2,
            size=1000,
        )
        samples = distribution(np.random.RandomState(31))

        self.assertEqual(samples.shape, (1000,))
        self.assertTrue(np.all(samples >= 0.8))
        self.assertTrue(np.all(samples <= 1.2))
        self.assertGreater(np.std(samples), 0.0)

    def test_rejects_non_positive_standard_deviation(self):
        with self.assertRaises(ValueError):
            NormalDistribution(mean=1.0, standard_deviation=0.0)


class NonStationaryCliTests(unittest.TestCase):
    def make_parser(self):
        parser = argparse.ArgumentParser()
        add_non_stationary_arguments(parser)
        return parser

    def test_stationary_environment_is_unchanged_without_flag(self):
        base_env = DummyEnv()
        args = self.make_parser().parse_args([])

        env = wrap_environment_from_args(base_env, args, seed=37)

        self.assertIs(env, base_env)

    def test_normal_cli_configuration_wraps_and_updates_environment(self):
        args = self.make_parser().parse_args(
            [
                "--non-stationary",
                "--ns-parameter", "gain",
                "--ns-degree-mean", "2.0",
                "--ns-degree-std", "0.1",
                "--ns-degree-low", "2.0",
                "--ns-degree-high", "2.0",
                "--ns-interval-mean", "1.0",
                "--ns-interval-std", "0.1",
                "--ns-interval-low", "1",
                "--ns-interval-high", "1",
            ]
        )

        env = wrap_environment_from_args(DummyEnv(), args, seed=41)
        _, reward, _, info = env.step(0)

        self.assertIsInstance(env, NonStationaryEnv)
        self.assertEqual(reward, 20.0)
        self.assertIn("non_stationary_update", info)

    def test_uniform_cli_configuration_uses_requested_interval_range(self):
        args = self.make_parser().parse_args(
            [
                "--non-stationary",
                "--ns-parameter", "gain",
                "--ns-degree-distribution", "uniform",
                "--ns-degree-low", "0.8",
                "--ns-degree-high", "1.2",
                "--ns-interval-distribution", "uniform",
                "--ns-interval-low", "2",
                "--ns-interval-high", "4",
            ]
        )

        env = wrap_environment_from_args(DummyEnv(), args, seed=43)

        self.assertIsInstance(env, NonStationaryEnv)
        self.assertTrue(2 <= env.next_update_step <= 4)

    def test_cli_configuration_rejects_invalid_interval_bounds(self):
        args = self.make_parser().parse_args(
            [
                "--non-stationary",
                "--ns-interval-low", "0",
            ]
        )

        with self.assertRaises(ValueError):
            wrap_environment_from_args(DummyEnv(), args, seed=47)


class NonStationaryEnvTests(unittest.TestCase):
    def test_updates_before_due_steps_and_scales_from_initial_value(self):
        wrapper = NonStationaryEnv(
            DummyEnv(),
            parameter_paths="gain",
            degree_distribution=SequenceDistribution([2.0, 3.0]),
            interval_distribution=SequenceDistribution([2, 3]),
            seed=3,
        )

        _, reward, _, info = wrapper.step(0)
        self.assertEqual(reward, 10.0)
        self.assertNotIn("non_stationary_update", info)

        _, reward, _, info = wrapper.step(0)
        self.assertEqual(reward, 20.0)
        self.assertEqual(info["non_stationary_update"]["step"], 2)
        self.assertEqual(wrapper.next_update_step, 5)

        wrapper.step(0)
        wrapper.step(0)
        _, reward, _, info = wrapper.step(0)
        self.assertEqual(reward, 30.0)
        self.assertEqual(info["non_stationary_update"]["count"], 2)

    def test_mapping_samples_independent_degrees_per_parameter(self):
        wrapper = NonStationaryEnv(
            DummyEnv(),
            parameter_paths=("gain", "vector"),
            degree_distribution={"gain": 0.5, "vector": 2.0},
            interval_distribution=1,
            change_mode="scale",
            seed=11,
        )

        _, reward, _, info = wrapper.step(0)
        self.assertEqual(reward, 5.0)
        np.testing.assert_allclose(wrapper.unwrapped.vector, [2.0, 4.0])
        np.testing.assert_allclose(
            info["non_stationary_update"]["parameters"]["vector"],
            [2.0, 4.0],
        )

        wrapper.restore_parameters()
        self.assertEqual(wrapper.unwrapped.gain, 10.0)
        np.testing.assert_allclose(wrapper.unwrapped.vector, [1.0, 2.0])

    def test_nested_mujoco_style_path_is_updated_and_forwarded(self):
        wrapper = NonStationaryEnv(
            DummyEnv(),
            parameter_paths="model.body_mass",
            degree_distribution=1.25,
            interval_distribution=1,
            seed=17,
        )

        wrapper.step(0)
        np.testing.assert_allclose(wrapper.unwrapped.model.body_mass, [1.25, 2.5])
        self.assertEqual(wrapper.unwrapped.sim.forward_calls, 1)

    def test_all_change_modes_have_explicit_semantics(self):
        expected = {
            "scale": 20.0,
            "add": 12.0,
            "multiply": 20.0,
            "set": 2.0,
        }
        for mode, expected_value in expected.items():
            with self.subTest(mode=mode):
                wrapper = NonStationaryEnv(
                    DummyEnv(),
                    parameter_paths="gain",
                    degree_distribution=2.0,
                    interval_distribution=10,
                    change_mode=mode,
                    seed=23,
                )
                wrapper.update()
                self.assertEqual(wrapper.unwrapped.gain, expected_value)

    def test_custom_updater_can_change_arbitrary_environment_state(self):
        def update_gain(env, degree):
            env.gain += float(degree)
            return {"gain": env.gain}

        wrapper = NonStationaryEnv(
            DummyEnv(),
            degree_distribution=1.5,
            interval_distribution=1,
            update_fn=update_gain,
            seed=5,
        )

        _, reward, _, info = wrapper.step(0)
        self.assertEqual(reward, 11.5)
        self.assertEqual(
            info["non_stationary_update"]["result"], {"gain": 11.5}
        )

    def test_gym_space_degree_sampling_is_reproducible(self):
        def make_wrapper():
            return NonStationaryEnv(
                DummyEnv(),
                parameter_paths="gain",
                degree_distribution=gym.spaces.Box(
                    low=np.array([0.5], dtype=np.float32),
                    high=np.array([1.5], dtype=np.float32),
                    dtype=np.float32,
                ),
                interval_distribution=1,
                seed=19,
            )

        first = make_wrapper()
        second = make_wrapper()
        first_degrees = [
            first.step(0)[3]["non_stationary_update"]["degree"]
            for _ in range(3)
        ]
        second_degrees = [
            second.step(0)[3]["non_stationary_update"]["degree"]
            for _ in range(3)
        ]

        np.testing.assert_allclose(first_degrees, second_degrees)

    def test_clock_spans_resets_by_default(self):
        wrapper = NonStationaryEnv(
            DummyEnv(),
            parameter_paths="gain",
            degree_distribution=2.0,
            interval_distribution=3,
            seed=13,
        )
        wrapper.step(0)
        wrapper.reset()
        wrapper.step(0)
        _, reward, _, _ = wrapper.step(0)
        self.assertEqual(wrapper.elapsed_steps, 3)
        self.assertEqual(reward, 20.0)


if __name__ == "__main__":
    unittest.main()
