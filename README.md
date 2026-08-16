# (NeurIPS 2025) COLA: Towards Efficient Multi-Objective Reinforcement Learning with Conflict Objective Regularization in Latent Space

**:triangular_flag_on_post: SOTA Performance** This repository contains the official implementation of **COLA**, a *general-policy* Multi-Objective Reinforcement Learning (MORL) framework that learns in a shared latent space and mitigates optimization conflicts across preferences.

**Paper:** COLA: Towards Efficient Multi-Objective Reinforcement Learning with Conflict Objective Regularization in Latent Space (NeurIPS 2025). 

## ✨ Overview

COLA addresses two key challenges in general-policy MORL:
- **Objective-agnostic Latent Dynamics Model (OADM):** builds a shared latent space capturing environment dynamics via temporal consistency, enabling efficient knowledge sharing across diverse preferences.
- **Conflict Objective Regularization (COR):** regularizes value updates when optimization directions under different preferences conflict, stabilizing value approximation and improving policy learning.

We adopt **Envelope SAC** as the backbone (general-policy) algorithm and condition both value and policy on preferences to cover the entire preference space.


## ✅ Features

- **Objective-agnostic latent space (OADM):** compact state & state–action representations for efficient multi-objective optimization.
- **Conflict-aware value learning (COR):** reduces interference among preferences when their optimization directions conflict.
- **General policy conditioning:** learn a single policy \(\pi(a\mid s, \omega)\) that generalizes across preferences \(\omega\).
- **CPU-friendly:** the code supports CPU-only training for MuJoCo-based tasks.

## 🔧 Installation

We recommend using Conda.

```bash
git clone <your-repo-url>
cd COLA

# Option A: create the tested Windows/CPU environment
conda env create --solver libmamba -f environmentV2.yml
conda activate cola38

# Option B: create a fresh env (example)
conda create -n cola38 python=3.8 -y
conda activate cola38
# Install PyTorch 1.13.1 first, then:
pip install -r requirements.txt
```

## 🚀 Quick Start

Train COLA on a 2D multi-objective Ant task:

```bash
python main.py   --env_id "MO-Ant-v2"   --seed 1   --Use_Critic_Preference   --Use_Policy_Preference   --Policy_use_latent   --Policy_use_s   --Policy_use_w   --Critic_use_both   --Critic_use_s   --Critic_use_a   --latent_dim 50   --regular_alpha 0.001   --regular_bar 0.25
```

You can also use the pre-configured launcher in `run.sh` to reproduce experiments for all tasks.

## 🌍 Supported Environments

We follow the paper’s multi-objective MuJoCo tasks (2–5 objectives). Example task set:

- **2D:** `MO-HalfCheetah-v2`, `MO-New-HalfCheetah-v2`, `MO-Hopper-v2`,
  `MO-Walker2d-v2`, `MO-Ant-v2`, `MO-Swimmer-v2`, `MO-Humanoid-v2`
- **3D:** `MO-Hopper-v3`, `MO-Ant-v3`
- **4D:** `MO-Ant-v4`
- **5D:** `MO-HalfCheetah-v5`, `MO-Hopper-v5`, `MO-Ant-v5`,
  `MO-Humanoid-v5`

The authoritative environment IDs and their Python entry points are registered
in `environments/__init__.py`.

Each task runs for 500 steps per episode, with objectives including forward/axis speed, jump height, and energy efficiency; some 4D/5D variants add per-limb energy costs.

## Non-stationary environments

`NonStationaryEnv` can wrap any existing Gymnasium environment and change one or
more core parameters at randomized global time steps. It uses only Gymnasium and
NumPy already pinned in `environmentV2.yml`. For example, the following changes
all Ant body masses to between 80% and 120% of their original values every
100--500 steps:

The training entry point can apply the wrapper directly. There are no separate
wrapped environment IDs: `--env_id` selects one of the base IDs registered in
`environments/__init__.py`, and `--non-stationary` wraps that instance before
`SacAgent` is created.

```bash
python main.py \
  --env_id "MO-Ant-v2" \
  --seed 1 \
  --non-stationary \
  --ns-parameter model.body_mass \
  --ns-degree-distribution normal \
  --ns-degree-mean 1.0 \
  --ns-degree-std 0.1 \
  --ns-degree-low 0.8 \
  --ns-degree-high 1.2 \
  --ns-interval-distribution normal \
  --ns-interval-mean 300 \
  --ns-interval-std 75 \
  --ns-interval-low 100 \
  --ns-interval-high 500 \
  --ns-change-mode scale
```

Repeat `--ns-parameter` to alter several parameters with the same sampled
degree. Omitting `--non-stationary` preserves the original stationary training
path. The equivalent Python API is:

```python
import gymnasium as gym
import numpy as np

from environments import NonStationaryEnv

base_env = gym.make("MO-Ant-v2")
env = NonStationaryEnv(
    base_env,
    parameter_paths="model.body_mass",
    degree_distribution=gym.spaces.Box(
        low=np.array([0.8], dtype=np.float32),
        high=np.array([1.2], dtype=np.float32),
        dtype=np.float32,
    ),
    interval_distribution=gym.spaces.Discrete(401),
    interval_offset=100,  # Discrete(401) + 100 gives 100,...,500 steps.
    change_mode="scale",
    seed=1,
)
```

Updates occur immediately before the action at the scheduled step. The global
clock spans episode resets by default, and `env.next_update_step`,
`env.elapsed_steps`, and `env.last_update` expose its state. On an updated
transition, the same metadata is available as
`info["non_stationary_update"]`. Call `env.update(degree=...)` for a manual
update, `env.reset_scheduler()` to restart the clock, or
`env.restore_parameters()` to restore values captured when wrapping.

The built-in path updater supports shared MuJoCo fields such as
`model.body_mass`, `model.geom_friction`, `model.dof_damping`,
`model.actuator_gear`, and `model.opt.gravity`. A mapping gives each parameter
its own distribution:

```python
env = NonStationaryEnv(
    gym.make("MO-Hopper-v2"),
    parameter_paths=("model.body_mass", "model.geom_friction"),
    degree_distribution={
        "model.body_mass": gym.spaces.Box(0.9, 1.1, shape=(1,)),
        "model.geom_friction": gym.spaces.Box(0.7, 1.3, shape=(1,)),
    },
    interval_distribution=lambda rng: rng.randint(250, 751),
    seed=1,
)
```

Gym spaces, SciPy frozen distributions, callables that accept a NumPy random
generator, constants, and the included `NormalDistribution` are accepted as
distributions. For example, this uses bounded normal samples for both the mass
scaling factor and update interval:

```python
from environments import NonStationaryEnv, NormalDistribution

env = NonStationaryEnv(
    gym.make("MO-Ant-v2"),
    parameter_paths="model.body_mass",
    degree_distribution=NormalDistribution(
        mean=1.0,
        standard_deviation=0.1,
        low=0.8,
        high=1.2,
    ),
    interval_distribution=NormalDistribution(
        mean=300,
        standard_deviation=75,
        low=100,
        high=500,
    ),
    seed=1,
)
```

The scheduler rounds a fractional normal interval up to the next whole step.
The bounds are optional; without them, `NormalDistribution` is an ordinary
unbounded normal distribution. Bounds clip outlying samples, which helps avoid
negative physical scales or timestep intervals. SciPy's `stats.norm` and
`stats.truncnorm` frozen distributions are also accepted directly.

For a parameter that cannot be represented by an attribute path, pass an
`update_fn(env, degree)` callback instead:

```python
def change_one_body(env, degree):
    env.model.body_mass[1] *= float(degree)

env = NonStationaryEnv(
    gym.make("MO-Humanoid-v2"),
    update_fn=change_one_body,
    degree_distribution=lambda rng: rng.uniform(0.95, 1.05),
    interval_distribution=lambda rng: rng.randint(100, 501),
    seed=1,
)
```

## 📊 Training 

### Preference grids
During training/evaluation we use preference grids to cover the space and report:
- **HV (Hypervolume)** and **UT (Utility)** on the discovered policies.
- Typical preference step sizes (per paper):  
  - 2-objective: `0.005`  
  - 3-objective: `0.05`  
  - 4-objective: `0.1`
  - 5-objective: `0.2`
  
More details can be found in the Appendix of paper.

### Hardware
Experiments can be run on CPU; GPU is optional.

## 📁 Project Structure

```
COLA/
├── agent.py              # SAC-based agent with COLA changes
├── main.py               # Training entry
├── model.py              # Networks (policy/critic/encoders)
├── base.py               # Replay & utils
├── utils.py              # Helpers (logging, eval, etc.)
├── hypervolume.py        # Hypervolume computation
├── compute_hv.py         # HV/UT evaluation utilities
├── multi_step.py         # Multi-step learning utils
├── environments/         # MuJoCo multi-objective tasks & assets
├── run.sh                # Repro scripts
├── requirements.txt
├── environment.yml
└── README.md
```

## 🧪 Baselines

We compare against representative MORL baselines used in the paper:
- **PGMORL**
- **Envelope SAC**
- **CAPQL**
- **Q-Pensive**

## 📈 Results at a Glance

Across a range of multi-objective continuous-control tasks (2–5 objectives), COLA exhibits **higher sample efficiency** and **better final HV/UT** than state-of-the-art general-policy methods, owing to OADM (efficient knowledge sharing) and COR (conflict-aware value learning). See the paper for full plots and numbers.

## :heart: Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{Li2025COLA,
  title     = {COLA: Towards Efficient Multi-Objective Reinforcement Learning with Conflict Objective Regularization in Latent Space},
  author    = {Pengyi Li and Hongyao Tang and Yifu Yuan and Jianye Hao and Zibin Dong and Yan Zheng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=Cldpn7H3NN}
}
```

## 📄 License

This project is released under the terms in `LICENSE`.

## 📫 Contact

For questions or issues, please open a GitHub issue or contact me.
