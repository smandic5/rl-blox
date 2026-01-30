import gymnasium as gym
import higher
import numpy as np
import torch
import torch.optim as optim

from rl_blox.logging.logger import LoggerBase


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
