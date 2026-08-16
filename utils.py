from collections import deque
import itertools
import numpy as np
import torch


def to_batch(state, preference, action, reward, next_state, done, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    preference = torch.FloatTensor(preference).unsqueeze(0).to(device)
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    return state, preference, action, reward, next_state, done


def generate_simplex_grid(dimension, step_size):
    """Generate an exact, duplicate-free grid on a probability simplex.

    Building the grid from integer compositions avoids losing valid points to
    floating-point equality checks such as ``weights.sum() == 1``.
    """
    dimension = int(dimension)
    step_size = float(step_size)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")

    units = int(round(1.0 / step_size))
    if units < 1 or not np.isclose(units * step_size, 1.0, atol=1e-10):
        raise ValueError("step_size must divide 1 exactly")

    integer_points = (
        point for point in itertools.product(range(units + 1), repeat=dimension)
        if sum(point) == units
    )
    return np.asarray(list(integer_points), dtype=np.float64) / units


def select_min_q_vectors(q1, q2, preferences):
    """Select one complete twin-Q vector per sample by scalarized value."""
    if q1.shape != q2.shape:
        raise ValueError("q1 and q2 must have the same shape")
    if q1.ndim != 2 or preferences.shape != q1.shape:
        raise ValueError("q tensors and preferences must all have shape (batch, objectives)")

    choose_q1 = (q1 * preferences).sum(dim=-1) < (q2 * preferences).sum(dim=-1)
    return torch.where(choose_q1.unsqueeze(-1), q1, q2)


def envelope_actor_loss_candidates(scalarized_q, entropy, alpha):
    """Return one SAC actor-loss candidate per sample and envelope choice."""
    if scalarized_q.ndim != 2:
        raise ValueError("scalarized_q must have shape (batch, candidates)")
    if entropy.ndim == 2 and entropy.shape[-1] == 1:
        entropy = entropy.squeeze(-1)
    if entropy.ndim != 1 or entropy.shape[0] != scalarized_q.shape[0]:
        raise ValueError("entropy must have shape (batch,) or (batch, 1)")
    return -scalarized_q - alpha * entropy.unsqueeze(-1)


def scale_action(normalized_action, low, high):
    """Map an action from [-1, 1] to the environment's Box bounds."""
    normalized_action = np.asarray(normalized_action, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
        raise ValueError("action bounds must be finite")
    if np.any(high <= low):
        raise ValueError("every action upper bound must exceed its lower bound")
    normalized_action = np.clip(normalized_action, -1.0, 1.0)
    return low + 0.5 * (normalized_action + 1.0) * (high - low)


def normalize_action(action, low, high):
    """Map an environment Box action to the policy/replay [-1, 1] space."""
    action = np.asarray(action, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
        raise ValueError("action bounds must be finite")
    if np.any(high <= low):
        raise ValueError("every action upper bound must exceed its lower bound")
    normalized = 2.0 * (action - low) / (high - low) - 1.0
    return np.clip(normalized, -1.0, 1.0)


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)
