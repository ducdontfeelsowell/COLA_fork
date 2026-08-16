import math
import unittest

import numpy as np
import torch

from base import MOMemory, QMemory
from multi_step import MOMultiStepMemory
from utils import (
    envelope_actor_loss_candidates,
    generate_simplex_grid,
    normalize_action,
    scale_action,
    select_min_q_vectors,
)


from model import Conflict_caculate, PCGrad


class SimplexGridTests(unittest.TestCase):
    def test_grid_contains_every_integer_composition(self):
        for dimension, step_size in ((3, 0.05), (4, 0.1), (5, 0.2)):
            with self.subTest(dimension=dimension):
                grid = generate_simplex_grid(dimension, step_size)
                units = round(1.0 / step_size)
                expected = math.comb(units + dimension - 1, dimension - 1)
                self.assertEqual(len(grid), expected)
                np.testing.assert_allclose(grid.sum(axis=1), 1.0)
                self.assertEqual(len(np.unique(grid, axis=0)), expected)

    def test_rejects_step_that_does_not_partition_one(self):
        with self.assertRaises(ValueError):
            generate_simplex_grid(3, 0.3)


class ActionTransformTests(unittest.TestCase):
    def test_round_trip_for_asymmetric_action_bounds(self):
        low = np.array([-2.0, 1.0], dtype=np.float32)
        high = np.array([4.0, 5.0], dtype=np.float32)
        normalized = np.array([-0.25, 0.75], dtype=np.float32)

        environment_action = scale_action(normalized, low, high)
        recovered = normalize_action(environment_action, low, high)

        np.testing.assert_allclose(recovered, normalized, atol=1e-6)
        np.testing.assert_allclose(scale_action([-1.0, 1.0], low, high), [-2.0, 5.0])


class TwinQSelectionTests(unittest.TestCase):
    def test_selects_a_complete_vector_independently_for_each_sample(self):
        q1 = torch.tensor([[1.0, 10.0], [9.0, 2.0]])
        q2 = torch.tensor([[3.0, 4.0], [1.0, 8.0]])
        preferences = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        selected = select_min_q_vectors(q1, q2, preferences)

        torch.testing.assert_close(selected, torch.tensor([[1.0, 10.0], [9.0, 2.0]]))

    def test_actor_loss_does_not_broadcast_across_batch(self):
        scalarized_q = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        entropy = torch.tensor([[0.1], [0.2], [0.3]])

        losses = envelope_actor_loss_candidates(scalarized_q, entropy, alpha=0.5)

        self.assertEqual(losses.shape, (3, 2))
        torch.testing.assert_close(
            losses,
            torch.tensor([[-1.05, -2.05], [-3.10, -4.10], [-5.15, -6.15]]),
        )


class ReplayMemoryTests(unittest.TestCase):
    def test_load_preserves_preferences(self):
        memory = MOMemory(4, (1,), 2, (1,), torch.device("cpu"))
        batch = (
            np.array([[1.0], [2.0]], dtype=np.float32),
            np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32),
            np.array([[0.1], [0.2]], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[2.0], [3.0]], dtype=np.float32),
            np.array([[0.0], [1.0]], dtype=np.float32),
        )

        memory.load(batch)

        loaded = memory.get()
        for actual, expected in zip(loaded, batch):
            np.testing.assert_allclose(actual, expected)

    def test_multistep_keeps_vector_rewards_and_flushes_episode_tail(self):
        memory = MOMultiStepMemory(
            8, (1,), 2, (1,), torch.device("cpu"), gamma=0.5, multi_step=3
        )
        memory.append(
            [0.0], [0.4, 0.6], [0.0], [1.0, 10.0], [1.0], False
        )
        memory.append(
            [1.0], [0.4, 0.6], [0.1], [2.0, 20.0], [2.0], True,
            episode_done=True,
        )

        states, preferences, _, rewards, next_states, dones = memory.get()
        self.assertEqual(len(memory), 2)
        np.testing.assert_allclose(states, [[0.0], [1.0]])
        np.testing.assert_allclose(preferences, [[0.4, 0.6], [0.4, 0.6]])
        np.testing.assert_allclose(rewards, [[2.0, 20.0], [2.0, 20.0]])
        np.testing.assert_allclose(next_states, [[2.0], [2.0]])
        np.testing.assert_allclose(dones, [[1.0], [1.0]])

    def test_q_memory_freezes_historical_critics(self):
        critic = torch.nn.Linear(2, 1)
        memory = QMemory(1)
        memory.append(critic)

        self.assertFalse(memory.sample()[0].training)
        self.assertTrue(all(not p.requires_grad for p in memory.sample()[0].parameters()))


class GradientLogicTests(unittest.TestCase):
    def test_pcgrad_sum_reduction_and_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        optimizer = PCGrad(torch.optim.SGD([parameter], lr=0.1), reduction="sum")

        optimizer.pc_backward([parameter[0], parameter[1]])
        optimizer.step()

        torch.testing.assert_close(parameter, torch.tensor([0.9, 0.9]))

    def test_stiffness_is_finite_for_zero_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        calculator = Conflict_caculate(torch.optim.SGD([parameter], lr=0.1))

        stiffness = calculator.get_stiffness(
            [torch.zeros(3), torch.tensor([1.0, 2.0, 3.0])]
        )

        self.assertTrue(torch.isfinite(stiffness))
        self.assertEqual(stiffness.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
