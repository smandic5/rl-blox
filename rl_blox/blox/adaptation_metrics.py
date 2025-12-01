from functools import partial
from math import isnan

import jax
import jax.numpy as jnp
from scipy.stats import linregress


class AdaptationTracker:
    def __init__(self, window_size: int = 5, slope_treshold: float = 1e-3):
        self.values = []
        self.steps = []
        self.window_size = window_size
        self.slope_treshold = slope_treshold
        self.significance_treshold = 0.05

    def update_progress(self, value: float, step: int):
        self.values.append(value)
        self.steps.append(step)

    def _calculate_performance_trend(self) -> tuple[bool, float, float]:
        rewards = jnp.array(self.values)
        slope, _, _, p, _ = linregress(
            jnp.arange(self.window_size), rewards[-self.window_size :]
        )
        is_trend_abscent = abs(slope) < self.slope_treshold
        is_trend_insignificant = (
            p > self.significance_treshold if not isnan(p) else False
        )
        plateaued = is_trend_abscent or is_trend_insignificant
        return plateaued, slope, p

    def is_finished(self) -> bool:
        if len(self.values) < self.window_size:
            return False

        plateaued, slope, p = self._calculate_performance_trend()
        return plateaued


@partial(jax.jit, static_argnames="start_steps")
def jumpstart(values: jnp.ndarray, start_steps: int = 5) -> float:
    return jnp.average(values[:start_steps]).item()


@partial(jax.jit, static_argnames="end_steps")
def asymptotic_performance(values: jnp.ndarray, end_steps: int = 5) -> float:
    return jnp.average(values[-end_steps:]).item()


@jax.jit
def total_reward(values: jnp.ndarray) -> float:
    return jnp.sum(values).item()


def area_ratio(values: jnp.ndarray, values_base: jnp.ndarray) -> float:
    auc_base = jnp.trapezoid(values_base) + 1e-8
    return (jnp.trapezoid(values) - auc_base) / auc_base


def time_to_treshold(values: jnp.ndarray, treshold_performance: float) -> int:
    return jnp.argmax(values > treshold_performance).item()
