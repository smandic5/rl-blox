import gymnasium as gym
import higher
import numpy as np
import scipy.special
import torch
import torch.optim as optim
from agent import Agent

from rl_blox.logging.logger import LoggerBase

from .ins.higher_to_torch import copy_from_fast
from .ins.ins import compare_agents


class TaskSelector:
    def __init__(self, envs_set: list[gym.vector.SyncVectorEnv], **kwargs):
        self.envs_set = envs_set
        self.sampled_env: int = 0
        self.iteration: int = 0
        self.waiting: bool = False

    def sample(self, **kwargs):
        if self.waiting:
            raise Exception(f"Sampling env while waiting.")
        self.waiting = True
        self.iteration += 1
        return self.envs_set[self.sampled_env]

    def feedback(self, **kwargs):
        self.waiting = False


class LoggingSelector(TaskSelector):
    def __init__(self, envs_set, logger: LoggerBase, **kwargs):
        super().__init__(envs_set, **kwargs)
        self.logger = logger
        self.stats_sample = dict()
        self.stats_feedback = dict()

    def sample(self, **kwargs):
        selected = super().sample(**kwargs)
        self.logger.record_stat(
            "Selected Env", self.sampled_env, step=self.iteration
        )
        for k, v in self.stats_sample.items():
            self.logger.record_stat(k, v, step=self.iteration)
        self.stats_sample.clear()
        return selected

    def feedback(self, to_log: dict = None, **kwargs):
        super().feedback(**kwargs)
        for k, v in self.stats_feedback.items():
            self.logger.record_stat(k, v, step=self.iteration)
        for k, v in to_log.items():
            self.logger.record_stat(
                f"Env {self.sampled_env} {k}", v, step=self.iteration
            )
        self.stats_feedback.clear()


class ProbabilitySelector(LoggingSelector):
    def __init__(self, envs_set, weights: np.ndarray, **kwargs):
        super().__init__(envs_set, **kwargs)
        self.weights = weights

    def sample(self, **kwargs):
        for i, e in enumerate(self.weights):
            self.stats_sample[f"Probability Env {i}"] = e
        self.sampled_env = np.random.choice(len(self.envs_set), p=self.weights)
        return super().sample(**kwargs)

    def feedback(self, **kwargs):
        return super().feedback(**kwargs)


class UniformSelector(ProbabilitySelector):
    def __init__(self, envs_set, **kwargs):
        l = len(envs_set)
        super().__init__(envs_set, np.ones(l) / l, **kwargs)


class HardTaskSelector(ProbabilitySelector):
    def __init__(self, envs_set, progress_weight: float = 0.5, **kwargs):
        l = len(envs_set)
        super().__init__(envs_set, np.ones(l) / l, **kwargs)
        self.last_progress = np.zeros(l)
        self.learning_speed = np.zeros(l)
        self.progress_weight = progress_weight

    def feedback(self, reward: float = None, **kwargs):
        progress = (reward + 1500) / 1500
        self.learning_speed[self.sampled_env] = (
            progress - self.last_progress[self.sampled_env]
        )
        self.last_progress[self.sampled_env] = progress
        t = 1

        p_progress = scipy.special.softmax((-self.learning_speed) / t)
        p_speed = scipy.special.softmax((1 - self.last_progress) / t)
        self.weights = p_progress * self.progress_weight + p_speed * (
            1 - self.progress_weight
        )
        return super().feedback(**kwargs)


class MatrixProbabilitySelector(ProbabilitySelector):
    def __init__(
        self,
        envs_set,
        from_last: bool,
        recalculate_on_feedback: bool,
        cost_matrix: np.ndarray = None,
        **kwargs,
    ):
        l = len(envs_set)
        if cost_matrix is None:
            cost_matrix = np.ones((l, l)) / l
        assert cost_matrix.shape[0] == cost_matrix.shape[1]
        self.cost_matrix = cost_matrix
        self.from_last = from_last
        self.recalculate_on_feedback = recalculate_on_feedback
        super().__init__(envs_set, weights=np.zeros((l, l)), **kwargs)
        self.weights = self.recalculate_weights()

    def recalculate_weights(self) -> np.ndarray:
        if self.from_last:
            w = self.cost_matrix[self.sampled_env]
        else:
            w = np.mean(self.cost_matrix, axis=0)
        w = np.clip(w, -10, 10)
        w = scipy.special.softmax(w)
        return w

    def feedback(self, **kwargs):
        if self.recalculate_on_feedback:
            self.weights = self.recalculate_weights()
        return super().feedback(**kwargs)


class InsSelector(MatrixProbabilitySelector):
    def __init__(
        self,
        envs_set,
        from_last,
        agents: list[Agent],
        disimilarity: bool,
        **kwargs,
    ):
        self.agents = agents
        self.disimilarity = disimilarity
        l = len(envs_set)
        super().__init__(
            envs_set, from_last, True, cost_matrix=np.zeros((l, l)), **kwargs
        )

    def feedback(self, used_model: higher.patch._MonkeyPatchBase, **kwargs):
        module = copy_from_fast(Agent(self.envs_set[0]), used_model)
        self.agents[self.sampled_env] = module
        for i, agent in enumerate(self.agents):
            if i == self.sampled_env:
                continue
            diff = compare_agents(agent, module) * (
                1 if self.disimilarity else -1
            )
            self.cost_matrix[self.sampled_env, i] = diff
            self.cost_matrix[i, self.sampled_env] = diff

        return super().feedback(**kwargs)
