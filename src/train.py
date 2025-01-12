from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

import os
import argparse
import yaml
import warnings
warnings.filterwarnings('ignore')

from typing import *
import logging
from datetime import datetime, timedelta

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from numba import njit
from numba import int32, float32
from numba.experimental import jitclass

os.environ['KMP_WARNINGS'] = 'off'

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
# class ProjectAgent:
#     def act(self, observation, use_random=False):
#         return 0

#     def save(self, path):
#         pass

#     def load(self):
#         pass


log_10 = np.log(10)

min_a1 = 0.0
max_a1 = 0.7
interval_a1 = 0.7
min_a2 = 0.0
max_a2 = 0.3
interval_a2 = 0.3

def make_HIV_env(**kwargs) -> gym.Env:
    register(
        id="hiv",
        entry_point=HIV_Dynamics,
        kwargs=kwargs,
    )
    hiv_env = gym.make("hiv")
    hiv_env.reset()
    return hiv_env


class HIV_Dynamics(gym.Env):
    """Gym Environment with HIV Infection Dynamics (with log10)"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        max_days: int,
        treatment_days: int,
        reward_scaler: float,
        init_state: Optional[np.ndarray] = None,
        is_test: bool = False,
        domain_randomization: bool = False,
    ) -> None:
        self.max_days = max_days
        self.treatment_days = treatment_days
        self.max_episode_steps = max_days // treatment_days
        self.reward_scaler = reward_scaler
        self.INIT_STATE = np.log10(np.array([163573, 5, 11945, 46, 63919, 24], dtype=np.float32))
        if init_state is not None:
            self.init_state = init_state
        else:
            self.init_state = self.INIT_STATE
        self.is_test = is_test

        self.t_interval = (0, treatment_days)
        self.t_eval = np.array(
            [
                treatment_days,
            ]
        )

        self.action_space = spaces.Discrete(4)
        self.controls = self.make_controls()
        self.observation_space = spaces.Box(
            low=-20.0,
            high=20,
            shape=(6,),
            dtype=np.float32,
        )

        self.Q = 0.1
        self.R1 = 20000
        self.R2 = 20000
        self.S = 1000
        self.domain_randomization = domain_randomization

        self.params = np.empty(20)
        self.reset()

    def _reset_patient_parameters(self):
        if self.domain_randomization:
            # randomly changing patient parameters
            self.k1 = np.random.uniform(low=5e-7, high=8e-7)
            # cell2
            self.k2 = np.random.uniform(low=0.1e-4, high=1.0e-4)
            self.f = np.random.uniform(low=0.29, high=0.34)
        else:
            self.k1 = 8e-7  # infection rate (mL per virions and per day)
            self.k2 = 1e-4  # infection rate (mL per virions and per day)
            self.f = 0.34  # treatment efficacy reduction for type 2 cells
        # cell type 1
        self.lmbd1 = 1e4  # production rate (cells per mL and per day)
        self.d1 = 1e-2  # death rate (per day)
        self.m1 = 1e-5  # immune-induced clearance rate (mL per cells and per day)
        self.rho1 = 1  # nb virions infecting a cell (virions per cell)
        # cell type 2
        self.lmbd2 = 31.98  # production rate (cells per mL and per day)
        self.d2 = 1e-2  # death rate (per day)
        self.m2 = 1e-5  # immune-induced clearance rate (mL per cells and per day)
        self.rho2 = 1  # nb virions infecting a cell (virions per cell)
        # infected cells
        self.delta = 0.7  # death rate (per day)
        self.n_T = 100  # virions produced (virions per cell)
        self.c = 13  # virus natural death rate (per day)
        # immune response (immune effector cells)
        self.lmbd_E = 1  # production rate (cells per mL and per day)
        self.b_E = 0.3  # maximum birth rate (per day)
        self.K_b = 100  # saturation constant for birth (cells per mL)
        self.d_E = 0.25  # maximum death rate (per day)
        self.K_d = 500  # saturation constant for death (cells per mL)
        self.delta_E = 0.1  # natural death rate (per day)

        self.params = np.empty(20)
        self.params[0] = self.lmbd1
        self.params[1] = self.lmbd2
        self.params[2] = self.d1
        self.params[3] = self.d2
        self.params[4] = self.m1
        self.params[5] = self.m2
        self.params[6] = self.k1
        self.params[7] = self.k2
        self.params[8] = self.delta
        self.params[9] = self.rho1
        self.params[10] = self.rho2
        self.params[11] = self.f
        self.params[12] = self.n_T
        self.params[13] = self.c
        self.params[14] = self.b_E
        self.params[15] = self.K_b
        self.params[16] = self.d_E
        self.params[17] = self.K_d
        self.params[18] = self.delta_E
        self.params[19] = self.lmbd_E

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset()
        self.state = self.init_state
        self.time = 0
        self._reset_patient_parameters()
        return self.state, {}

    def make_controls(self) -> np.ndarray:
        eps = 1e-12
        a1 = np.arange(min_a1, max_a1 + eps, interval_a1)
        a2 = np.arange(min_a2, max_a2 + eps, interval_a2)
        x, y = np.meshgrid(a1, a2)
        controls = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1, dtype=np.float32)
        return controls

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        self.state, reward, intermediate_sol = self.ode_step(self.state, self.controls[action])
        reward = reward / self.reward_scaler
        self.time += 1
        done = True if self.time > self.max_episode_steps else False
        return self.state, reward, done, False, {"intermediate_sol": intermediate_sol}

    def ode_step(
        self, state: np.ndarray, control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        reward = np.zeros((1,), dtype=np.float32)
        x = np.concatenate([state, control], axis=0)  # shape: (8,)
        reward = -(self.Q *(10 ** x[4]) + self.R1 * x[6] ** 2 + self.R2 * x[7] ** 2 - self.S * (10 ** x[5]))
        # sol = solve_ivp(
        #     _ode_ftn, self.t_interval, x, t_eval=self.t_eval, rtol=1e-6, atol=1e-6, method="RK45"
        # )
        # sol.y[:, -1][-1] /= self.t_interval
        nb_steps = int(self.t_interval[1] // 1e-3)
        for i in range(nb_steps):
            der = _ode_ftn(None, x, self.params)
            x += der * 1e-3
        # y = sol.y[:, -1]
        y = x[:]
        if self.is_test and len(self.t_eval) > 1:
            intermediate_sol = None
            # intermediate_sol = sol.y[:, :-1]
        else:
            intermediate_sol = None
        assert x.shape == y.shape
        next_state = y[:6]
        # reward = y[-1]
        return next_state, reward, intermediate_sol

@njit(cache=True)
def _ode_ftn(t: float, x: np.ndarray, params: np.ndarray) -> np.ndarray:
    dx = np.zeros((8,), dtype=np.float32)
    _x0 = 10 ** x[0]
    _x1 = 10 ** x[1]
    _x2 = 10 ** x[2]
    _x3 = 10 ** x[3]
    _x4 = 10 ** x[4]
    _x5 = 10 ** x[5]
    lmbd1 = params[0]
    lmbd2 = params[1]
    d1 = params[2]
    d2 = params[3]
    m1 = params[4]
    m2 = params[5]
    k1 = params[6]
    k2 = params[7]
    delta = params[8]
    rho1 = params[9]
    rho2 = params[10]
    f = params[11]
    n_T = params[12]
    c = params[13]
    b_E = params[14]
    K_b = params[15]
    d_E = params[16]
    K_d = params[17]
    delta_E = params[18]
    lmbd_E = params[19]

    dx[0] = lmbd1 - d1 * _x0 - (1 - x[6]) * k1 * _x4 * _x0
    dx[0] *= 10 ** (-x[0]) / log_10
    dx[1] = lmbd2 - d2 * _x1 - (1 - f * x[6]) * k2 * _x4 * _x1
    dx[1] *= 10 ** (-x[1]) / log_10
    dx[2] = (1 - x[6]) * k1 * _x4 * _x0 - delta * _x2 - m1 * _x5 * _x2
    dx[2] *= 10 ** (-x[2]) / log_10
    dx[3] = (1 - f * x[6]) * k2 * _x4 * _x1 - delta * _x3 - m2 * _x5 * _x3
    dx[3] *= 10 ** (-x[3]) / log_10
    dx[4] = (
        (1 - x[7]) * n_T * delta * (_x2 + _x3)
        - c * _x4
        - ((1 - x[6]) * rho1 * k1 * _x0 + (1 - f * x[6]) * rho2 * k2 * _x1) * _x4
    )
    dx[4] *= 10 ** (-x[4]) / log_10
    _I = _x2 + _x3
    _E_first = b_E * _I / (_I + K_b) * _x5
    _E_second = d_E * _I / (_I + K_d) * _x5
    dx[5] = lmbd_E + _E_first - _E_second - delta_E * _x5
    dx[5] *= 10 ** (-x[5]) / log_10
    return dx

###
spec_SumSegmentTree = [
    ('capacity', int32),
    ('tree', float32[:]),
]
@jitclass(spec=spec_SumSegmentTree)
class SumSegmentTree(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)

    def sum_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        return _sum_helper(self.tree, start, end, node, node_start, node_end)

    def sum(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1
        return self.sum_helper(start, end, 1, 0, self.capacity - 1)

    def retrieve(self, upperbound: float) -> int:
        return _sum_retrieve_helper(self.tree, 1, self.capacity, upperbound)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        _sum_setter_helper(self.tree, idx)

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]


spec_MinSegmentTree = [
    ('capacity', int32),
    ('tree', float32[:]),
]
INF = float('inf')
@jitclass(spec=spec_MinSegmentTree)
class MinSegmentTree(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.array([INF for _ in range(2 * capacity)], dtype=np.float32)

    def min_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        return _min_helper(self.tree, start, end, node, node_start, node_end)

    def min(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1
        return self.min_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        _min_setter_helper(self.tree, idx)

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]


@njit(cache=True)
def _sum_helper(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int) -> np.float32:
    if start == node_start and end == node_end:
        return tree[node]
    mid = (node_start + node_end) // 2
    if end <= mid:
        return _sum_helper(tree, start, end, 2 * node, node_start, mid)
    else:
        if mid + 1 <= start:
            return _sum_helper(tree, start, end, 2 * node + 1, mid + 1, node_end)
        else:
            a = _sum_helper(tree, start, mid, 2 * node, node_start, mid)
            b = _sum_helper(tree, mid + 1, end, 2 * node + 1, mid + 1, node_end)
            return a + b


@njit(cache=True)
def _min_helper(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int) -> np.float32:
    if start == node_start and end == node_end:
        return tree[node]
    mid = (node_start + node_end) // 2
    if end <= mid:
        return _min_helper(tree, start, end, 2 * node, node_start, mid)
    else:
        if mid + 1 <= start:
            return _min_helper(tree, start, end, 2 * node + 1, mid + 1, node_end)
        else:
            a = _min_helper(tree, start, mid, 2 * node, node_start, mid)
            b = _min_helper(tree, mid + 1, end, 2 * node + 1, mid + 1, node_end)
            if a < b:
                return a
            else:
                return b


@njit(cache=True)
def _sum_setter_helper(tree: np.ndarray, idx: int) -> None:
    while idx >= 1:
        tree[idx] = tree[2 * idx] + tree[2 * idx + 1]
        idx = idx // 2


@njit(cache=True)
def _min_setter_helper(tree: np.ndarray, idx: int) -> None:
    while idx >= 1:
        a = tree[2 * idx]
        b = tree[2 * idx + 1]
        if a < b:
            tree[idx] = a
        else:
            tree[idx] = b
        idx = idx // 2


@njit(cache=True)
def _sum_retrieve_helper(tree: np.ndarray, idx: int, capacity: int, upperbound: float) -> int:
    while idx < capacity: # while non-leaf
        left = 2 * idx
        right = left + 1
        if tree[left] > upperbound:
            idx = 2 * idx
        else:
            upperbound -= tree[left]
            idx = right
    return idx - capacity

###
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self,
        obs_dim: int,
        size: int,
        batch_size: int = 32,
        n_step_return: int = 3,
        stack_n_prev_frames: int = 3,
        gamma: float = 0.99,
    ):
        self.obs_buf = np.zeros((size, (stack_n_prev_frames + 1) * obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, (stack_n_prev_frames + 1) * obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size), dtype=np.float32)
        self.rews_buf = np.zeros((size), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr = 0
        self.size = 0
        self.n_step_return = n_step_return
        self.stack_n_prev_frames = stack_n_prev_frames
        self.traj_obs = np.zeros((n_step_return, (stack_n_prev_frames + 1) * obs_dim), dtype=np.float32)
        self.traj_actions = np.zeros(n_step_return, dtype=np.float32)
        self.traj_rewards = np.zeros(n_step_return, dtype=np.float32)
        self.traj_next_idx = 0
        self.is_traj_full = False
        self.prev_frames = np.zeros((stack_n_prev_frames, obs_dim), dtype=np.float32)
        self.next_prev_frame_idx = 0
        self.gamma = gamma

    def reset_traj(self):
        self.traj_obs = np.zeros_like(self.traj_obs)
        self.traj_actions = np.zeros_like(self.traj_actions)
        self.traj_rewards = np.zeros_like(self.traj_rewards)
        self.traj_next_idx = 0
        self.is_traj_full = False
        self.prev_frames = np.zeros_like(self.prev_frames)
        self.next_prev_frame_idx = 0

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool):
        if not self.is_traj_full and self.traj_next_idx == 0:  # traj just got reset
            self.prev_frames[:] = obs  # we set past frames to the initial state, assuming it's a fixed point
        augmented_obs = np.concatenate(
            [
                self.prev_frames[self.next_prev_frame_idx:].reshape(-1),
                self.prev_frames[:self.next_prev_frame_idx].reshape(-1),
                obs,
            ],
            axis=-1,
        )
        self.prev_frames[self.next_prev_frame_idx, :] = obs
        self.next_prev_frame_idx = (self.next_prev_frame_idx + 1) % self.stack_n_prev_frames
        self.traj_obs[self.traj_next_idx] = augmented_obs
        self.traj_actions[self.traj_next_idx] = act
        self.traj_rewards[self.traj_next_idx] = rew
        self.traj_next_idx = (self.traj_next_idx + 1) % self.n_step_return
        self.is_traj_full = self.is_traj_full or (self.traj_next_idx == 0)
        if self.is_traj_full:
            n_step_reward = 0
            discounted_gamma = 1
            for r in itertools.chain(
                self.traj_rewards[self.traj_next_idx :], self.traj_rewards[: self.traj_next_idx]
            ):
                n_step_reward += discounted_gamma * r
                discounted_gamma *= self.gamma

            self.obs_buf[self.ptr] = self.traj_obs[self.traj_next_idx]
            augmented_next_obs = np.concatenate(
                [
                    self.prev_frames[self.next_prev_frame_idx:].reshape(-1),
                    self.prev_frames[:self.next_prev_frame_idx].reshape(-1),
                    next_obs,
                ],
                axis=-1,
            )
            self.next_obs_buf[self.ptr] = augmented_next_obs
            self.acts_buf[self.ptr] = self.traj_actions[self.traj_next_idx]
            self.rews_buf[self.ptr] = n_step_reward
            self.done_buf[self.ptr] = done
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
        if done:
            self.reset_traj()
        return self.is_traj_full

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer."""

    def __init__(
        self,
        obs_dim: int,
        size: int,
        batch_size: int = 32,
        n_step_return: int = 3,
        stack_n_prev_frames: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 0.000005,
    ):
        """Initialization."""
        assert alpha >= 0

        super().__init__(obs_dim, size, batch_size, n_step_return, stack_n_prev_frames, gamma)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience and priority."""
        transition_added = super().store(obs, act, rew, next_obs, done)
        if transition_added:
            self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = self._calculate_weights(indices, self.beta)

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)
        _update_priorities_helper(indices, priorities, self.sum_tree, self.min_tree, self.alpha)
        self.max_priority = max(self.max_priority, priorities.max())

    def _sample_proportional(self) -> np.ndarray:
        """Sample indices based on proportions."""
        return _sample_proportional_helper(self.sum_tree, len(self), self.batch_size)

    def _calculate_weights(self, indices: np.ndarray, beta: float) -> np.ndarray:
        """Calculate the weights of the experiences"""
        return _calculate_weights_helper(indices, beta, self.sum_tree, self.min_tree, len(self))


@njit(cache=True)
def _sample_proportional_helper(
    sum_tree: SumSegmentTree,
    size: int,
    batch_size: int,
) -> np.ndarray:
    indices = np.zeros(batch_size, dtype=np.int32)
    p_total = sum_tree.sum(0, size - 1)
    segment = p_total / batch_size

    for i in range(batch_size):
        a = segment * i
        b = segment * (i + 1)
        upperbound = np.random.uniform(a, b)
        idx = sum_tree.retrieve(upperbound)
        indices[i] = idx

    return indices


@njit(cache=True)
def _calculate_weights_helper(
    indices: np.ndarray,
    beta: float,
    sum_tree: SumSegmentTree,
    min_tree: MinSegmentTree,
    size: int,
) -> np.ndarray:

    weights = np.zeros(len(indices), dtype=np.float32)

    for i in range(len(indices)):

        idx = indices[i]

        # get max weight
        p_min = min_tree.min() / sum_tree.sum()
        max_weight = (p_min * size) ** (-beta)

        # calculate weights
        p_sample = sum_tree[idx] / sum_tree.sum()
        weight = (p_sample * size) ** (-beta)
        weight = weight / max_weight

        weights[i] = weight

    return weights


@njit(cache=True)
def _update_priorities_helper(
    indices: np.ndarray,
    priorities: np.ndarray,
    sum_tree: SumSegmentTree,
    min_tree: MinSegmentTree,
    alpha: float,
) -> None:

    for i in range(len(indices)):
        idx = indices[i]
        priority = priorities[i]
        sum_tree[idx] = priority**alpha
        min_tree[idx] = priority**alpha

###

class Network(nn.Module):
    def __init__(self, in_dim: int, nf: int, out_dim: int, n_hidden_layers: int = 2):
        """Initialization."""
        super(Network, self).__init__()
        self.initial = nn.Sequential(
            nn.Linear(in_dim, nf),
            nn.SiLU(),
            nn.LayerNorm(nf),
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(nf, nf),
                    nn.SiLU(),
                    nn.LayerNorm(nf),
                )
                for _ in range(n_hidden_layers)
            ]
        )
        self.final = nn.Linear(nf, out_dim)

        # self.layers = nn.Sequential(
        #     nn.Linear(in_dim, nf),
        #     nn.LeakyReLU(),
        #     nn.Linear(nf, nf),
        #     nn.LeakyReLU(),
        #     nn.Linear(nf, nf),
        #     nn.LeakyReLU(),
        #     nn.Linear(nf, out_dim)
        # )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        h = self.initial(x)
        for layer in self.layers:
          h = layer(h)
        return self.final(h)

class DuelingNetwork(nn.Module):
    def __init__(self, in_dim: int, nf: int, out_dim: int):
        """Initialization."""
        super(DuelingNetwork, self).__init__()
        self.encoder = Network(in_dim, nf, nf, 0)
        self.encoder_nonlins = nn.Sequential(nn.SiLU(), nn.LayerNorm(nf))
        self.value_net = Network(nf, nf, 1, 0)
        self.adv_net = Network(nf, nf, out_dim, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        h = self.encoder(x)
        h = self.encoder_nonlins(h)
        value = self.value_net(h)
        adv = self.adv_net(h)
        return value + adv - adv.mean(dim=-1, keepdim=True)
###
class ProjectAgent: #DQNAgent
    def __init__(
        self,
        # General parameters
        log_dir: str = 'last_exp',
        load_ckpt: bool = True,
        writer: SummaryWriter = SummaryWriter("last_exp/tb"),
        ckpt_dir: str = "last_exp/ckpt_best",
        # Environment parameters
        max_days: int = 1000,
        treatment_days: int = 5,
        reward_scaler: float = 1e8,
        # Training parameters
        memory_size: int = int(1e6),
        batch_size: int = 2048,
        n_step_return: int = 1,
        stack_n_prev_frames: int = 3,
        lr: float = 2e-4,
        l2_reg: float = 1e-4,
        grad_clip: float = 1000.0,
        target_update: int = (1000//5)*5,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 1 / 300,
        decay_option: str = "logistic",
        discount_factor: float = 0.997,
        n_train: int = 1,
        # Network parameters
        hidden_dim: int = 512,
        # PER parameters
        per: bool = True,
        alpha: float = 0.2,
        beta: float = 0.6,
        beta_increment_per_sampling: float = 3e-6,
        prior_eps: float = 1e-6,
        # Double DQN
        double_dqn: bool = True,
        domain_randomization: bool = False,
    ):
        """Initialization.
        Args:
            log_dir (str): Logging directory path (root for the experiment)
            ckpt_dir (str): checkpoint directory path
            load_ckpt (bool): Load checkpoint? or not?
            writer (SummaryWriter): Tensorboard SummaryWriter

            max_days (int): Time length of one episode (days)
            treatment_days (int): Treatment interval days
            reward_scaler (float): Scaling factor for the instantaneous rewards

            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            lr (float): learning rate
            l2_reg (float) : L2-regularization (weight decay)
            grad_clip (float) : gradient clipping
            target_update (int): period for target model's hard update
            max_epsilon (float): Maximum value of epsilon
            min_epsilon (float): Minimum value of epsilon
            epsilon_decay (float): Epsilon decaying rate
            decay_option (str): Epsilon decaying schedule option (`linear`, `logistic`)
            discount_factor (float): Discounting factor
            n_train (int): Number of training per each step

            hidden_dim (int): hidden dimension in network

            per (bool): If true, PER is activated. Otherwise, the replay buffer of original DQN is used.
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            beta_increment_per_sampling (float): to increase beta per each sampling
            prior_eps (float): guarantees every transition can be sampled

            double_dqn (bool): Activate dqn or not
        """

        # Make environments
        self.max_days = max_days
        self.treatment_days = treatment_days
        self.reward_scaler = reward_scaler

        # (0) Prepare initial
        UNHEALTHY_STEADY_INIT_STATE = np.log10(
            np.array([163573, 5, 11945, 46, 63919, 24], dtype=np.float32)
        )
        HIGH_T_LOW_V_INIT_STATE = np.log10(
            np.array([1.0e6, 3198, 1.0e-4, 1.0e-4, 1, 10], dtype=np.float32)
        )
        HIGH_T_HIGH_V_INIT_STATE = np.log10(
            np.array([1.0e6, 3198, 1.0e-4, 1.0e-4, 1000000, 10], dtype=np.float32)
        )
        LOW_T_HIGH_V_INIT_STATE = np.log10(
            np.array([1000, 10, 10000, 100, 1000000, 10], dtype=np.float32)
        )
        self.domain_randomization = domain_randomization
        # (1) Make Envs
        self.envs = {
            "train": self.make_env(UNHEALTHY_STEADY_INIT_STATE),
            # "HTLV": self.make_env(HIGH_T_LOW_V_INIT_STATE),
            # "HTHV": self.make_env(HIGH_T_HIGH_V_INIT_STATE),
            # "LTHV": self.make_env(LOW_T_HIGH_V_INIT_STATE),
        }
        self.set_train_domain_randomization(self.domain_randomization)
        obs_dim = self.envs["train"].observation_space.shape[0]
        action_dim = self.envs["train"].action_space.n

        # Parameters
        self.writer = writer
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.best_ckpt_dir = f"{ckpt_dir}_best"
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.target_update = target_update
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.decay_option = decay_option
        self.discount_factor = discount_factor
        self.n_train = n_train
        self.stack_n_prev_frames = stack_n_prev_frames

        # Device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {self.device}")
        print(f"device: {self.device}")

        # PER memory
        self.per = per
        if per:
            self.prior_eps = prior_eps
            self.memory = PrioritizedReplayBuffer(
                obs_dim=obs_dim,
                size=memory_size,
                batch_size=batch_size,
                n_step_return=n_step_return,
                stack_n_prev_frames=stack_n_prev_frames,
                gamma=discount_factor,
                alpha=alpha,
                beta=beta,
                beta_increment_per_sampling=beta_increment_per_sampling,
            )
        else:
            self.memory = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step_return, stack_n_prev_frames, discount_factor
            )

        # Double DQN
        self.double_dqn = double_dqn

        # Networks: DQN, DQN_target
        dqn_config = dict(
            in_dim=(stack_n_prev_frames + 1) * obs_dim,
            nf=hidden_dim,
            out_dim=action_dim,
        )
        # self.dqn = Network(**dqn_config).to(self.device)
        # self.dqn_target = Network(**dqn_config).to(self.device)
        self.dqn = DuelingNetwork(**dqn_config).to(self.device)
        self.dqn_target = DuelingNetwork(**dqn_config).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn.compile()
        self.dqn_target.compile()
        self.dqn_target.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr, weight_decay=l2_reg)

        # Mode: train / test
        self.is_test = False
        self.max_cum_reward = -1.0

        # Record (archive train / test results)
        self.record = []

        # Initial episode (default: 1)
        self.init_episode = 1

        # Benchmark (no_drug, full_drug for each environment)
        logging.info("Computing Benchmark... (Wait for few seconds)")
        self.bm_info = {
            "no_drug": {"states": {}, "actions": {}, "rewards": {}},
            "full_drug": {"states": {}, "actions": {}, "rewards": {}},
        }
        for opt in self.bm_info.keys():
            for name, _env in self.envs.items():
                _states, _actions, _rewards = self._test(_env, opt)
                self.bm_info[opt]["states"][name] = _states
                self.bm_info[opt]["actions"][name] = _actions
                self.bm_info[opt]["rewards"][name] = _rewards
        logging.info("Done!")

        if load_ckpt:
            self.load()
    
    def make_env(self, init_state: Optional[np.ndarray] = None) -> gym.Env:
        env = make_HIV_env(
            max_days=self.max_days,
            treatment_days=self.treatment_days,
            reward_scaler=self.reward_scaler,
            init_state=init_state,
        )
        return env

    def set_train_domain_randomization(self, val) -> None:
        self.envs["train"].unwrapped.domain_randomization = val

    def save_ckpt(self, episode: int, path: str) -> None:
        if self.per:
            _memory = _gather_per_buffer_attr(self.memory)
        else:
            _memory = _gather_replay_buffer_attr(self.memory)
        ckpt = dict(
            episode=episode,
            dqn=self.dqn.state_dict(),
            dqn_target=self.dqn_target.state_dict(),
            optim=self.optimizer.state_dict(),
            memory=_memory,
        )
        torch.save(ckpt, path)

    def load(self, device='cpu') -> None:
        if device == 'cpu':
            ckpt = torch.load(os.path.join(self.ckpt_dir, "ckpt.pt"), map_location=torch.device('cpu')) 
        else:
            ckpt = torch.load(os.path.join(self.ckpt_dir, "ckpt.pt"))
        self.init_episode = ckpt["episode"] + 1
        self.dqn.load_state_dict(ckpt["dqn"])
        self.dqn_target.load_state_dict(ckpt["dqn_target"])
        self.optimizer.load_state_dict(ckpt["optim"])
        for key, value in ckpt["memory"].items():
            if key not in ["sum_tree", "min_tree"]:
                setattr(self.memory, key, value)
            else:
                tree = getattr(self.memory, key)
                setattr(tree, "capacity", value["capacity"])
                setattr(tree, "tree", value["tree"])

        logging.info(f"Success: Checkpoint loaded (start from Episode {self.init_episode})!")
        print(f"Success: Checkpoint loaded (start from Episode {self.init_episode})!")

    def act(self, state: np.ndarray, authorize_random: bool = True, step: int = None) -> int:
        """Select an action from the input state."""
        # epsilon greedy policy (only for training)
        if authorize_random and self.epsilon > np.random.random() and not self.is_test:
            selected_action = self.envs["train"].action_space.sample()
            # print(f"{step=}, {authorize_random=} {selected_action=}")
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def step(
        self, env: gym.Env, action: int
    ) -> Tuple[np.ndarray, np.float64, bool, Optional[np.ndarray]]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _, info = env.step(action)
        return next_state, reward, done, info["intermediate_sol"]

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        if self.per:
            # PER needs beta to calculate weights
            samples = self.memory.sample_batch()
            weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
            indices = samples["indices"]
        else:
            # Vanilla DQN does not require any weights
            samples = self.memory.sample_batch()

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples)
        if self.per:
            loss = torch.mean(elementwise_loss * weights)
        else:
            loss = torch.mean(elementwise_loss)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.per:
            # PER: update priorities
            loss_for_prior = elementwise_loss.squeeze().detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def train(
        self, max_episodes: int, log_freq: int, test_freq: int, save_freq: int, img_dir: str, domain_randomization: bool = False
    ) -> None:
        """Train the agent."""
        self.is_test = False

        max_steps = self.envs["train"].unwrapped.max_episode_steps
        update_cnt = 0
        start = datetime.now()
        self.set_train_domain_randomization(domain_randomization)
        print(f"{self.envs['train'].unwrapped.domain_randomization=}")


        for episode in range(self.init_episode, max_episodes + 1):
            state = self.envs["train"].reset()[0]
            train_traj_cum_reward = 0
            augmented_state = np.concatenate([state] * (self.stack_n_prev_frames + 1), axis=0)
            losses = []
            for step in range(max_steps):
                authorize_random = step <= 80 and step >= 3
                action = self.act(augmented_state.reshape(-1), authorize_random, step)
                next_state, reward, done, _ = self.step(self.envs["train"], action)
                train_traj_cum_reward += reward
                transition = [state, action, reward, next_state, done]
                self.memory.store(*transition)
                augmented_state = augmented_state.reshape(self.stack_n_prev_frames + 1, -1)
                augmented_state[:self.stack_n_prev_frames, :] = augmented_state[1:]
                augmented_state[-1, :] = next_state
                state = next_state

                # If training is available:
                if len(self.memory) >= self.batch_size:
                    for _ in range(self.n_train):
                        loss = self.update_model()
                    losses.append(loss)
                    self.writer.add_scalar("loss", loss, update_cnt)
                    update_cnt += 1

                    # # epsilon decaying
                    # if self.decay_option == "linear":
                    #     self.epsilon = max(
                    #         self.min_epsilon,
                    #         self.epsilon
                    #         - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                    #     )
                    # elif self.decay_option == "logistic":
                    #     self.epsilon = self.min_epsilon + (
                    #         self.max_epsilon - self.min_epsilon
                    #     ) * sigmoid(1 / self.epsilon_decay - episode)
                    if episode < 1 / self.epsilon_decay:
                        self.epsilon = self.max_epsilon
                    else:
                        self.epsilon = max(
                            self.min_epsilon,
                            self.epsilon
                            - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                        )
                    # self.epsilon = self.max_epsilon

                    # Target network update
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                # If episode ends:
                if done:
                    break

            avg_step_train_loss = np.array(losses).sum() * self.batch_size / max_steps
            # print(f"Epi {episode:>4d} | Cum Reward (train) {train_traj_cum_reward * self.reward_scaler:.3e}")
            # Test
            if test_freq > 0 and episode % test_freq == 0:
                last_treatment_day, max_E, last_E, cum_reward = self.test(episode, img_dir)
                self.record.append(
                    {
                        "episode": episode,
                        "last_treatment_day": last_treatment_day,
                        "max_E": max_E,
                        "last_E": last_E,
                        "cum_reward": cum_reward,
                        "train_loss": avg_step_train_loss,
                    }
                )
                self._save_record_df()

                # Logging
                if log_freq > 0 and episode % log_freq == 0:
                    self._track_results(
                        episode,
                        datetime.now() - start,
                        train_loss=avg_step_train_loss,
                        max_E=max_E,
                        last_E=last_E,
                        cum_reward=cum_reward,
                    )

            # Save
            if save_freq > 0 and episode % save_freq == 0:
                path = os.path.join(self.ckpt_dir, "ckpt.pt")
                self.save_ckpt(episode, path)

        self.envs["train"].close()

    def test(self, episode: int, img_dir: str) -> Tuple[int, float, float, float]:
        """Test the agent (computation & plotting)"""

        # Compute state/action/reward sequence for train env
        _states, _actions, _rewards = self._test(self.envs["train"], "policy")
        states = {"train": _states}
        actions = {"train": _actions}
        rewards = {"train": _rewards}

        # cum_reward = discounted_sum(
        #     rewards["train"], self.discount_factor
        # )  # total discounted reward
        cum_reward = rewards['train'].sum() # Original total reward
        if cum_reward > max(1e0, self.max_cum_reward):
            # save best agent so far
            path = os.path.join(self.best_ckpt_dir, "ckpt.pt")
            self.save_ckpt(episode, path)

            # Compute state/action/reward sequence for other envs, too
            for name, _env in self.envs.items():
                if name == "train":
                    continue
                states[name], actions[name], rewards[name] = self._test(_env, "policy")

            # FIGURE 1 (6-states & 2-actions - one figure per each env)
            for env_name in self.envs.keys():
                self._plot_6_states_2_actions(
                    episode,
                    img_dir,
                    states[env_name],
                    actions[env_name],
                    self.bm_info["no_drug"]["states"][env_name],
                    self.bm_info["full_drug"]["states"][env_name],
                    env_name,
                )

            # FIGURE 2 (V, E phase-plane - one figure for all envs)
            self._plot_VE_phase_plane(episode, img_dir, states, actions)

        for env in self.envs.values():
            env.close()

        self.dqn.train()

        last_a1_day = get_last_treatment_day(actions["train"][:, 0])
        last_a2_day = get_last_treatment_day(actions["train"][:, 1])
        last_treatment_day = max(last_a1_day, last_a2_day) * (
            self.max_days // len(actions["train"])
        )
        max_E = 10 ** (states["train"][:, 5].max())
        last_E = 10 ** (states["train"][-1, 5])
        if cum_reward > self.max_cum_reward:
            self.max_cum_reward = cum_reward

        return last_treatment_day, max_E, last_E, cum_reward

    def _test(
        self, env: gym.Env, mode: str = "policy"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Test the agent (dynamics propagation only)"""
        assert mode in ["policy", "no_drug", "full_drug"]
        self.is_test = True
        self.dqn.eval()
        max_steps = env.unwrapped.max_episode_steps
        states = []
        actions = []
        rewards = []

        with torch.no_grad():
            state = env.reset()[0]
            state = np.concatenate([state] * (self.stack_n_prev_frames + 1), axis=0)
            for _ in range(max_steps):
                if mode == "policy":
                    action = self.act(state.reshape(-1))
                elif mode == "no_drug":
                    action = 0
                elif mode == "full_drug":
                    action = 3
                next_state, reward, _, intermediate_sol = self.step(env, action)
                _action = env.unwrapped.controls[action].reshape(1, -1)
                _reward = (
                    np.array(
                        [
                            reward,
                        ]
                    )
                    * env.unwrapped.reward_scaler
                )
                if intermediate_sol is not None:  # i.e, treatment days > 1
                    intermediate_states = intermediate_sol[:6, :].transpose()
                    _state = np.concatenate([state.reshape(1, -1), intermediate_states], axis=0)
                    _action = np.repeat(_action, _state.shape[0], axis=0)
                else:  # i.e, treatment days = 1
                    _state = state.reshape(1, -1)
                states.append(_state)
                actions.append(_action)
                rewards.append(_reward)
                state = state.reshape(self.stack_n_prev_frames + 1, -1)
                state[:self.stack_n_prev_frames, :] = state[1:]
                state[-1, :] = next_state

        states = np.concatenate(states, axis=0, dtype=np.float32)  # shape (N, 6)
        actions = np.concatenate(actions, axis=0, dtype=np.float32)  # shape (N, 2)
        rewards = np.concatenate(rewards, axis=0, dtype=np.float32).reshape(
            -1, 1
        )  # shape (N//T, 1)
        self.is_test = False
        return states, actions, rewards

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        curr_q_value = self.dqn(state).gather(1, action)
        if not self.double_dqn:
            next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        else:
            next_q_value = (
                self.dqn_target(next_state)
                .gather(1, self.dqn(next_state).argmax(dim=1, keepdim=True))
                .detach()
            )
        mask = 1 - done
        target = (reward + self.discount_factor * next_q_value * mask).to(self.device)

        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _track_results(
        self,
        episodes: int,
        elapsed_time: timedelta,
        train_loss: float,
        max_E: float,
        last_E: float,
        cum_reward: float,
    ):
        elapsed_time = str(timedelta(seconds=elapsed_time.seconds))
        logging.info(
            f"Epi {episodes:>4d} | {elapsed_time} | LastE {last_E:8.1f} | CumR {cum_reward:.3e} | "
            f"Loss (Train) {train_loss:.2e} | Buffer {self.memory.size}"
        )
        print(
            f"Epi {episodes:>4d} | {elapsed_time} | LastE {last_E:8.1f} | CumR {cum_reward:.3e} | "
            f"Loss (Train) {train_loss:.2e} | Buffer {self.memory.size}"
        )

    def _save_record_df(self):
        """Save self.record as a pandas dataframe."""
        df = pd.DataFrame(self.record).set_index("episode")
        df.to_csv(os.path.join(self.log_dir, "records.csv"))

    def _plot_6_states_2_actions(
        self,
        episode: int,
        img_dir: str,
        policy_states: np.ndarray,
        policy_actions: np.ndarray,
        no_drug_states: np.ndarray,
        full_drug_states: np.ndarray,
        env_name: str,
    ) -> None:
        """Draw a figure with 6-states and 2-actions (our policy & no drug & full drug)"""
        fig = plt.figure(figsize=(14, 18))
        plt.axis("off")
        state_names = [
            r"$\log_{10}(T_{1}$)",
            r"$\log_{10}(T_{2})$",
            r"$\log_{10}(T_{1}^{*})$",
            r"$\log_{10}(T_{2}^{*})$",
            r"$\log_{10}(V)$",
            r"$\log_{10}(E)$",
        ]
        action_names = [
            rf"RTI $\epsilon_{1}$",
            rf"PI $\epsilon_{2}$",
        ]
        axis_t = np.arange(policy_states.shape[0]) * treatment_days
        label_fontdict = {
            "size": 13,
        }

        for i in range(6):
            ax = fig.add_subplot(4, 2, i + 1)
            ax.plot(axis_t, policy_states[:, i], label="ours", color="crimson", linewidth=2)
            ax.plot(
                axis_t,
                no_drug_states[:, i],
                label="no drug",
                color="royalblue",
                linewidth=2,
                linestyle="--",
            )
            ax.plot(
                axis_t,
                full_drug_states[:, i],
                label="full drug",
                color="black",
                linewidth=2,
                linestyle="-.",
            )
            ax.set_xlabel("Days", labelpad=0.8, fontdict=label_fontdict)
            ax.set_ylabel(state_names[i], labelpad=0.5, fontdict=label_fontdict)
            if i == 0:
                ax.set_ylim(min(4.8, policy_states[:, i].min() - 0.2), 6)

        last_a1_day = get_last_treatment_day(policy_actions[:, 0])
        last_a2_day = get_last_treatment_day(policy_actions[:, 1])
        last_treatment_day = max(last_a1_day, last_a2_day) * (self.max_days // len(policy_actions))
        for i in range(2):
            ax = fig.add_subplot(4, 2, i + 7)
            if last_treatment_day < 550:
                if last_a1_day >= last_a2_day:
                    if i == 0:
                        ax.text(
                            last_a1_day * treatment_days,
                            -0.07,
                            f"Day {last_a1_day * treatment_days}",
                        )
                else:
                    if i == 1:
                        ax.text(
                            last_a2_day * treatment_days,
                            -0.03,
                            f"Day {last_a2_day * treatment_days}",
                        )
            _a = np.repeat(policy_actions[:, i], treatment_days, axis=0)
            ax.plot(
                np.arange(policy_states.shape[0] * treatment_days),
                _a,
                color="forestgreen",
                linewidth=2,
            )
            if i == 0:
                ax.set_ylim(0.7 * (-0.2), 0.7 * 1.2)
                ax.set_yticks([0.0, 0.7])
            else:
                ax.set_ylim(0.3 * (-0.2), 0.3 * 1.2)
                ax.set_yticks([0.0, 0.3])
            ax.set_xlabel("Days", labelpad=0.8, fontdict=label_fontdict)
            ax.set_ylabel(action_names[i], labelpad=0.5, fontdict=label_fontdict)

        fig.savefig(
            os.path.join(img_dir, f"Epi{episode}_{env_name}_{last_treatment_day}.png"),
            bbox_inches="tight",
            pad_inches=0.2,
        )
        return

    def _plot_VE_phase_plane(
        self,
        episode: int,
        img_dir: str,
        states: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
    ) -> None:
        """Draw a figure of logV - logE phase plane for each environment"""
        label_fontdict = {
            "size": 13,
        }
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        meta_info = {
            "train": dict(color="navy", alpha=0.8, label="train (initial: unhealthy steady state)"),
            # "HTLV": dict(
            #     color="forestgreen",
            #     alpha=0.8,
            #     label="test (initial: early infection with one virus)",
            # ),
            # "HTHV": dict(
            #     color="darkorange",
            #     alpha=0.8,
            #     label=r"test (initial: early infection with $10^6$ virus)",
            # ),
            # "LTHV": dict(
            #     color="indianred",
            #     alpha=0.8,
            #     label=r"test (initial: small T-cells with $10^6$ virus)",
            # ),
        }
        init_labels = ["A", "B", "C", "D"]
        for i, (env_name, kwargs) in enumerate(meta_info.items()):
            _s = states[env_name]
            x = _s[:, 4]  # log(V)
            y = _s[:, 0]  # log(T1)
            z = _s[:, 5]  # log(E)
            ax.plot(x, y, z, **kwargs)
            ax.scatter(x[0], y[0], z[0], color="black", marker="o", s=70)
            ax.text(
                x[0],
                y[0],
                z[0] - 0.4,
                init_labels[i],
                fontdict=dict(
                    size=13,
                ),
            )

        # End point (only for training env)
        ax.scatter(
            states["train"][-1, 4],
            states["train"][-1, 0],
            states["train"][-1, 5],
            color="red",
            marker="*",
            s=120,
        )
        ax.text(
            states["train"][-1, 4],
            states["train"][-1, 0],
            states["train"][-1, 5] + 0.4,
            "End",
            fontdict=dict(
                size=14,
            ),
        )

        ax.view_init(15, 45)
        ax.set_xlabel(r"$\log_{10}(V)$", labelpad=2, fontdict=label_fontdict)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylabel(r"$\log_{10}(T_{1})$", labelpad=2, fontdict=label_fontdict)
        ax.set_ylim(3, 7)
        ax.set_zlabel(r"$\log_{10}(E)$", labelpad=2, fontdict=label_fontdict)
        ax.set_zlim(0, 6.5)
        ax.legend(loc="upper right")
        fig.savefig(
            os.path.join(img_dir, f"Epi{episode}_VE.png"),
            bbox_inches="tight",
            pad_inches=0.2,
        )
        return


def _gather_replay_buffer_attr(memory: Optional[ReplayBuffer]) -> dict:
    if memory is None:
        return {}
    replay_buffer_keys = [
        "obs_buf",
        "next_obs_buf",
        "acts_buf",
        "rews_buf",
        "done_buf",
        "max_size",
        "batch_size",
        "ptr",
        "size",
    ]
    result = {key: getattr(memory, key) for key in replay_buffer_keys}
    return result


def _gather_per_buffer_attr(memory: Optional[PrioritizedReplayBuffer]) -> dict:
    if memory is None:
        return {}
    per_buffer_keys = [
        "obs_buf",
        "next_obs_buf",
        "acts_buf",
        "rews_buf",
        "done_buf",
        "max_size",
        "batch_size",
        "ptr",
        "size",
        "max_priority",
        "tree_ptr",
        "alpha",
    ]
    result = {key: getattr(memory, key) for key in per_buffer_keys}
    result["sum_tree"] = dict(
        capacity=memory.sum_tree.capacity,
        tree=memory.sum_tree.tree,
    )
    result["min_tree"] = dict(
        capacity=memory.min_tree.capacity,
        tree=memory.min_tree.tree,
    )
    return result


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


@njit(cache=True)
def get_last_treatment_day(action: np.ndarray) -> int:
    """Find the last treatment day (i.e, nonzero actions) for a given action sequence."""
    n = len(action)
    for i in range(n - 1, -1, -1):
        if action[i] != 0:
            return i + 1
    return 0


@njit(cache=True)
def discounted_sum(rewards: np.ndarray, discount_factor: float = 0.99, n: int = 5) -> float:
    _sum = 0.0
    _factor = 1.0
    _cnt = 0
    for r in rewards[:, 0]:
        _sum += r * _factor
        _cnt += 1
        if _cnt % n == 0:
            _factor *= discount_factor
    return _sum

###
cfg = {}


# Environment Configuration
max_days = 1000
treatment_days = 5
reward_scaler = 1e8  # IMPORTANT
MAX_EPISODE_STEPS = max_days // treatment_days


# Agent Configuration
cfg["dqn_agent"] = dict(
    max_days=max_days,
    treatment_days=treatment_days,
    reward_scaler=reward_scaler,
    memory_size=int(1e6),
    batch_size=2048,
    n_step_return=1,
    stack_n_prev_frames=3,
    lr=2e-4,
    l2_reg=1e-4, # 0
    grad_clip=1000.0,
    target_update=MAX_EPISODE_STEPS * 5,
    max_epsilon=1.0,
    min_epsilon=0.05,
    epsilon_decay=1 / 300,
    decay_option="logistic",
    discount_factor=0.997,
    n_train=1,
    hidden_dim=512,  # 1024
    per=True,  # IMPORTANT
    alpha=0.2,
    beta=0.6,  # the episode after which the bias induced by importance sampling is fully corrected is (1-beta)/(200 * beta_increment_per_sampling) ~ 670 for .6 and 3e-6
    beta_increment_per_sampling=3e-6,
    prior_eps=1e-6,
    double_dqn=True,  # IMPORTANT
    domain_randomization=True,
)

def train(args):

    # Define agent
    writer = SummaryWriter(args.tb_dir)
    _agent = ProjectAgent

    agent = _agent(
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        load_ckpt=(args.resume) or (args.mode == "test"),
        writer=writer,
        **cfg[f"dqn_agent"],
    )

    # Training Configuraiton
    cfg["train"] = dict(
        max_episodes=args.max_episodes,
        log_freq=10,
        test_freq=10,
        save_freq=50,
        img_dir=args.img_dir,
        domain_randomization=True,
    )

    # Save configs
    with open(os.path.join(args.log_dir, "configs.yml"), "w") as f:
        yaml.dump(cfg, f)

    # Train agent
    if args.mode == "train":
        agent.train(**cfg["train"])

    # Test agent
    elif args.mode == "test":
        agent.test(agent.init_episode - 1, args.img_dir)

    return

if __name__ == "__main__":
    args = argparse.Namespace(
        exp="test3",
        gpu=0,
        mode="train",
        max_episodes=2000,
        resume=False,
    )

    # Set GPU num
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    # Logging
    args.log_dir = os.path.join("logs", args.exp)
    os.makedirs(args.log_dir, exist_ok=True)
    args.tb_dir = os.path.join(args.log_dir, "tb")
    os.makedirs(args.tb_dir, exist_ok=True)
    args.ckpt_dir = os.path.join(args.log_dir, "ckpt")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "ckpt_best"), exist_ok=True)
    args.img_dir = os.path.join(args.log_dir, "img")
    os.makedirs(args.img_dir, exist_ok=True)

    if args.mode == "train":
        logging.basicConfig(
            format="%(asctime)s %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{os.path.join(args.log_dir, "train_log.txt")}', mode="w"),
            ],
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
    train(args)