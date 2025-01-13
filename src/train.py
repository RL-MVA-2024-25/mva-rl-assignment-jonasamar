import os
from typing import *

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient
from memory import PrioritizedReplayBuffer, ReplayBuffer
from network import Network

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

MAX_NUM_STEPS = 200


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.dqn_agent = DQNAgent()

    def act(self, observation, use_random=False):
        return self.dqn_agent.act(observation, use_random)

    def save(self, path):
        pass

    def load(self):
        pass


class DQNAgent:
    def __init__(
        self,
        reward_scaler: float = 1e8,
        memory_size: int = int(1e6),
        batch_size: int = 2048,
        stack_n_prev_frames: int = 3,
        lr: float = 2e-4,
        l2_reg: float = 0.0,
        grad_clip: float = 1000.0,
        target_update: int = 3000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 1 / 200,
        discount_factor: float = 0.99,
        n_train: int = 1,
        domain_randomization: bool = False,
        hidden_dim: int = 1024,
        per: bool = True,
        alpha: float = 0.2,
        beta: float = 0.6,
        beta_increment_per_sampling: float = 0.000005,
        prior_eps: float = 1e-6,
        double_dqn: bool = False,
    ):
        self.reward_scaler = reward_scaler
        self.domain_randomization = domain_randomization
        self.env = TimeLimit(
            env=HIVPatient(domain_randomization=self.domain_randomization),
            max_episode_steps=MAX_NUM_STEPS,
        )
        self.set_train_domain_randomization(self.domain_randomization)
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        # Parameters
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.target_update = target_update
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.n_train = n_train
        self.stack_n_prev_frames = stack_n_prev_frames
        self.device = "cpu"

        # PER memory
        self.per = per
        if per:
            self.prior_eps = prior_eps
            self.memory = PrioritizedReplayBuffer(
                obs_dim=obs_dim,
                size=memory_size,
                batch_size=batch_size,
                # n_step_return=n_step_return,
                # stack_n_prev_frames=stack_n_prev_frames,
                # gamma=discount_factor,
                alpha=alpha,
                beta=beta,
                beta_increment_per_sampling=beta_increment_per_sampling,
            )
        else:
            self.memory = ReplayBuffer(
                obs_dim,
                memory_size,
                batch_size,
                # n_step_return,
                # stack_n_prev_frames,
                # discount_factor,
            )

        self.double_dqn = double_dqn
        dqn_config = dict(
            in_dim=(stack_n_prev_frames + 1) * obs_dim, nf=hidden_dim, out_dim=action_dim
        )
        self.dqn = Network(**dqn_config).to(self.device)
        self.dqn_target = Network(**dqn_config).to(self.device)
        # self.dqn = DuelingNetwork(**dqn_config).to(self.device)
        # self.dqn_target = DuelingNetwork(**dqn_config).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr, weight_decay=l2_reg)

        self.is_test = True
        self.init_episode = 1
        self.t = 0
        self.augmented_obs = None
        self.load_ckpt()

    def act(self, observation, use_random):
        log_observation = np.log10(observation)
        if self.t == 0:
            self.augmented_obs = np.stack(
                [log_observation] * (self.stack_n_prev_frames + 1), axis=0
            )
        else:
            self.augmented_obs[: self.stack_n_prev_frames, :] = self.augmented_obs[1:]
            self.augmented_obs[-1, :] = log_observation
        self.t = (self.t + 1) % 200
        return self.select_action(self.augmented_obs.reshape(-1))

    def set_train_domain_randomization(self, val) -> None:
        self.env.unwrapped.domain_randomization = val

    def load_ckpt(self) -> None:
        ckpt = torch.load(os.path.join("src", "ckpt.pt"), map_location=torch.device("cpu"))
        self.init_episode = ckpt["episode"] + 1
        self.dqn.load_state_dict(ckpt["dqn"])
        self.dqn_target.load_state_dict(ckpt["dqn_target"])
        print(f"Success: Checkpoint loaded (start from Episode {self.init_episode})!")

    def select_action(self, state: np.ndarray) -> int:
        """Select an action from the input state."""
        state = state.reshape((-1, 6))[:, np.array([0, 2, 1, 3, 4, 5])].reshape(-1)
        # epsilon greedy policy (only for training)
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return np.array([0, 2, 1, 3])[selected_action]

    def step(
        self, env: gym.Env, action: int
    ) -> Tuple[np.ndarray, np.float64, bool, Optional[np.ndarray]]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _, info = env.step(action)
        return next_state, reward / self.reward_scaler, done, info["intermediate_sol"]

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

    def train(self, max_episodes: int, domain_randomization: bool = False) -> None:
        """Train the agent."""
        self.is_test = False

        update_cnt = 0
        self.set_train_domain_randomization(domain_randomization)
        print(f"{self.envs['train'].unwrapped.domain_randomization=}")

        for episode in range(self.init_episode, max_episodes + 1):
            state = self.env.reset()[0]
            augmented_state = np.concatenate([state] * (self.stack_n_prev_frames + 1), axis=0)
            for _ in range(MAX_NUM_STEPS):
                action = self.select_action(augmented_state.reshape(-1))
                next_state, reward, done, _ = self.step(self.env, action)
                transition = [state, action, reward, next_state, done]
                self.memory.store(*transition)
                augmented_state = augmented_state.reshape(self.stack_n_prev_frames + 1, -1)
                augmented_state[: self.stack_n_prev_frames, :] = augmented_state[1:]
                augmented_state[-1, :] = next_state
                state = next_state

                # If training is available:
                if len(self.memory) >= self.batch_size:
                    for _ in range(self.n_train):
                        _ = self.update_model()
                    update_cnt += 1

                    # # epsilon decaying
                    if episode < 1 / self.epsilon_decay:
                        self.epsilon = self.max_epsilon
                    else:
                        self.epsilon = max(
                            self.min_epsilon,
                            self.epsilon
                            - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                        )

                    # Target network update
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                # If episode ends:
                if done:
                    break

        self.env.close()

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
        self.dqn_target.load_state_dict(self.dqn.state_dict())