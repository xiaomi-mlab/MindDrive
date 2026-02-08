import warnings
from abc import ABC, abstractmethod
# from collections.abc import Generator
from typing import Any, Optional, Union, Dict
from typing import Generator, Optional
import numpy as np
import torch
import copy
# from gymnasium import spaces

# from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    # RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Protocol, SupportsFloat, Union
import pickle

class RolloutBufferSamples(NamedTuple):
    # observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    # log_probs: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    episode_starts: torch.Tensor
    meta_action_info: torch.Tensor
    # action_language_log_probs: torch.Tensor
    # value_language_log_probs: torch.Tensor

class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.action_dim = 2
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True):
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if array.dtype == np.str_ or np.issubdtype(array.dtype, np.str_) or array.dtype == object:
            return array
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    # observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    
    def __init__(
        self,
        buffer_size: int,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        cache_obs: bool = True,
    ):
        super().__init__(buffer_size, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.cache_obs = cache_obs
        self.reset()

    def reset(self) -> None:

        # self.observations = np.zeros((self.buffer_size, self.n_envs, 1), dtype=object)
        self.actions = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32) # (5, 1, 2)
        self.rewards = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.values = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32) 
        self.log_probs = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.ref_log_probs = np.zeros((self.buffer_size, self.n_envs, 7), dtype=np.float32)
        # self.action_language_log_probs = [[] for _ in range(self.buffer_size)]
        # self.value_language_log_probs = [[] for _ in range(self.buffer_size)]
        self.advantages = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        # self.infos = np.zeros((self.buffer_size, self.n_envs, 1), dtype=object)
        self.meta_action_info = np.zeros((self.buffer_size, self.n_envs, 1), dtype=object)
        self.episode_lengths = []  # 记录每次episode的step数
        self.generator_ready = False
        self.episode_infos = {}
        super().reset()

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray, episode_steps = None) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        # @Note: 当前场景一定是done才计算gae
        # last_values = last_values.clone().cpu().numpy().flatten() 
        last_gae_lam = 0
        # if episode_steps is None:
        #     buffer_size = self.buffer_size
        # else:
        #     buffer_size = episode_steps
        buffer_size = self.pos
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - np.float32(dones)
                next_values = 0
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        ref_log_probs:torch.Tensor,
        meta_action_info: np.ndarray,
        # action_language_log_probs:torch.Tensor,
        # value_language_log_probs:torch.Tensor,
    ) -> None:
        """
        :param action: Action 执行的动作
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """

        # self.observations[self.pos] = obs
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        # self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.ref_log_probs[self.pos] = ref_log_probs.clone().cpu().numpy()
        # self.action_language_log_probs[self.pos].append(action_language_log_probs.clone().cpu().numpy())
        # self.value_language_log_probs[self.pos].append(value_language_log_probs.clone().cpu().numpy())

        self.meta_action_info[self.pos] = meta_action_info
        # self.infos[self.pos] = np.array(info)
        self.pos += 1
        # if self.pos == self.buffer_size:
        #     self.full = True
    def crop_to_pos(self):
        self.actions = self.actions[:self.pos]
        self.rewards = self.rewards[:self.pos]
        self.returns = self.returns[:self.pos]
        self.episode_starts = self.episode_starts[:self.pos]
        self.values = self.values[:self.pos]
        self.log_probs = self.log_probs[:self.pos]
        self.ref_log_probs = self.ref_log_probs[:self.pos]
        self.advantages = self.advantages[:self.pos]
        self.meta_action_info = self.meta_action_info[:self.pos]

    def save_current_episode(self, buffer, save_path):

        temp_buffer = copy.deepcopy(buffer)
        temp_buffer.actions = temp_buffer.actions[:temp_buffer.pos]
        temp_buffer.rewards = temp_buffer.rewards[:temp_buffer.pos]
        temp_buffer.returns = temp_buffer.returns[:temp_buffer.pos]
        temp_buffer.episode_starts = temp_buffer.episode_starts[:temp_buffer.pos]
        temp_buffer.values = temp_buffer.values[:temp_buffer.pos]
        # temp_buffer.log_probs = temp_buffer.log_probs[:temp_buffer.pos]
        temp_buffer.ref_log_probs = temp_buffer.ref_log_probs[:temp_buffer.pos]
        temp_buffer.advantages = temp_buffer.advantages[:temp_buffer.pos]
        temp_buffer.meta_action_info = temp_buffer.meta_action_info[:temp_buffer.pos]
        with open(save_path, "wb") as f:
            pickle.dump(temp_buffer, f)
            
    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                # "observations",
                "actions",
                "values",
                "ref_log_probs",
                "advantages",
                "returns",
                "episode_starts",
                "meta_action_info",
                # "action_language_log_probs",
                # "value_language_log_probs",
            ]

            for tensor in _tensor_names:
                if tensor in ["ref_language_log_probs"]:
                    self.__dict__[tensor] = np.array(self.__dict__[tensor])
                    continue
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    
    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            # self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            # self.log_probs[batch_inds].flatten(),
            self.ref_log_probs[batch_inds],
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.episode_starts[batch_inds].flatten(),
            self.meta_action_info[batch_inds],
            # self.action_language_log_probs[batch_inds],
            # self.value_language_log_probs[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def summarize_episode_infos(self):
        """Summarize the episode infos."""
        # NOTE@Jianfeng: num_envs == 1
        infos = self.infos[:, 0, 0]
        episode_starts = self.episode_starts[:, 0]
        if len(infos) == 0:
            return
        episode_starts_idx = np.where(episode_starts)[0]
        episode = []
        reward_info_keys = infos[0]['REWARD_INFO_KEYS']
        done_info_keys = infos[0]['DONE_INFO_KEYS']

        for i in range(len(episode_starts_idx)):

            start_idx = episode_starts_idx[i]
            end_idx = episode_starts_idx[i+1] if i < len(episode_starts_idx) - 1 else min(self.pos, self.buffer_size)
            if not end_idx - start_idx > 0:
                continue
            episode_len = end_idx - start_idx + 1
            episode_info = {}

            episode_info["TOTAL_REWARD"] = 0
            episode_info["STEPS"] = episode_len
            for k in reward_info_keys:
                episode_info[f"TOTAL_{k}"] = sum([info[k] for info in infos[start_idx:end_idx]])
                episode_info["TOTAL_REWARD"] += episode_info[f"TOTAL_{k}"]
            for k in done_info_keys:
                episode_info[f"COUNT_{k}"] = sum([info[k] for info in infos[start_idx:end_idx]])
            
            episode_info['IS_SUCCESS'] = np.any(np.array([info['SUCCESS'] for info in infos[start_idx:end_idx]]))

            episode.append(episode_info)

        episode_infos = {
            'NUM_EPISODES': len(episode),
            'EPISODE': episode,
            'SUCCESS_RATE': sum([episode_info['IS_SUCCESS'] for episode_info in episode]) / len(episode)
        }
        episode_infos[f"MEAN_EPISODE_REWARD"] = sum([episode_info["TOTAL_REWARD"] for episode_info in episode]) / len(episode)
        episode_infos[f"MEAN_EPISODE_STEPS"] = sum([episode_info["STEPS"] for episode_info in episode]) / len(episode)
        for k in reward_info_keys:
            episode_infos[f"MEAN_EPISODE_{k}"] = sum([episode_info[f"TOTAL_{k}"] for episode_info in episode]) / len(episode)
        for k in done_info_keys:
            episode_infos[f"MEAN_EPISODE_COUNT_{k}"] = sum([episode_info[f"COUNT_{k}"] for episode_info in episode]) / len(episode)

        self.episode_infos = episode_infos

        return

    def print_episode_infos(self):
        """Print the episode infos."""
        episode_infos = {k: v for k, v in self.episode_infos.items() if k != 'EPISODE'}
        print_str = "===> Rollout infos: "
        for k, v in episode_infos.items():
            print_str += f"{k.lower()}: {v}, "
        print(print_str)