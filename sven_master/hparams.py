from functools import partial

from rl_blox.blox.env_util import (
    AppendHistoryWrapper,
    NegativePerStepWrapper,
    OneHotObservationWrapper,
)
from rl_blox.blox.multitask import (
    HardTaskPrioritizationTaskSelector,
    PolicySimilarityTaskSelector,
    UniformTaskSelector,
)
from rl_blox.blox.similarity_metrics.model_similarity.bisimulation import (
    ModelBasedTaskSelector,
)
from rl_blox.blox.vec_env_util import (
    AppendHistoryVecEnvWrapper,
    NegativePerStepVecWrapper,
    OneHotVecObservationWrapper,
)

from .frozen_lake_factory import create_vectorized_fl_from_hparams

MAML_NAME = "MAML_PPO"
RL2_NAME = "RL2_PPO"
algorithm_to_use = MAML_NAME
save_frequency = 250
checkpoint_path = "./sven_master/checkpoints/"

hparams_model = dict(
    actor_hidden_layers=[64, 64],
    actor_activation="relu",
    actor_learning_rate=3e-4,
    critic_hidden_layers=[64, 64],
    critic_activation="relu",
    critic_learning_rate=1e-3,
)
hparams_algorithm = dict(
    num_envs=32,
    batch_size=128,
    iterations=502,
    set_size_train=3,
    set_size_test=1,
    epochs=2,
    seed=1,
)
hparams_eval = dict(
    iterations=5000,
    epochs=1,
)

params_frozen_lake = dict(
    env_name="FrozenLake-v1",
    lake_size=4,
)
max_reward = (
    (
        1 / (params_frozen_lake["lake_size"] - 1) ** 2
    )  # max potential reward per step (not realistic)
    * hparams_algorithm["batch_size"]
    * hparams_algorithm["num_envs"]  # max steps made
    * 0.3  # mulitplier
)
frozenLlake_vec_wrappers = [
    OneHotVecObservationWrapper,
    NegativePerStepVecWrapper,
]
vec_env_wrappers = frozenLlake_vec_wrappers + (
    [AppendHistoryVecEnvWrapper] if algorithm_to_use == RL2_NAME else []
)
frozenLlake_wrappers = [OneHotObservationWrapper, NegativePerStepWrapper]
env_wrappers = frozenLlake_wrappers + (
    [AppendHistoryWrapper] if algorithm_to_use == RL2_NAME else []
)

create_vec_fl_env = partial(
    create_vectorized_fl_from_hparams,
    hparams_algorithm=hparams_algorithm,
    wrappers=vec_env_wrappers,
    lake_size=params_frozen_lake["lake_size"],
)
create_fl_env = partial(
    create_vectorized_fl_from_hparams,
    hparams_algorithm=hparams_algorithm,
    wrappers=env_wrappers,
    is_vec=False,
    lake_size=params_frozen_lake["lake_size"],
)

create_vec_env = create_vec_fl_env
create_env = create_fl_env
params_env = params_frozen_lake

selector_configurations = [
    dict(
        selector_class=HardTaskPrioritizationTaskSelector,
        prefer_similar=None,
        choose_from_last_pick=None,
        max_reward=max_reward,
        progress_weight=0.8,
    ),
    dict(
        selector_class=PolicySimilarityTaskSelector,
        prefer_similar=True,
        choose_from_last_pick=False,
        max_reward=None,
        progress_weight=None,
    ),
    dict(
        selector_class=PolicySimilarityTaskSelector,
        prefer_similar=True,
        choose_from_last_pick=True,
        max_reward=None,
        progress_weight=None,
    ),
    dict(
        selector_class=PolicySimilarityTaskSelector,
        prefer_similar=False,
        choose_from_last_pick=False,
        max_reward=None,
        progress_weight=None,
    ),
    dict(
        selector_class=PolicySimilarityTaskSelector,
        prefer_similar=False,
        choose_from_last_pick=True,
        max_reward=None,
        progress_weight=None,
    ),
    dict(
        selector_class=UniformTaskSelector,
        prefer_similar=None,
        choose_from_last_pick=None,
        max_reward=None,
        progress_weight=None,
    ),
    """dict(
        selector_class = ModelBasedTaskSelector,
        prefer_similar = True,
        choose_from_last_pick = False,
        max_reward = None,
        progress_weight = None,
    ),
    dict(
        selector_class = ModelBasedTaskSelector,
        prefer_similar = True,
        choose_from_last_pick = True,
        max_reward = None,
        progress_weight = None,
    ),
    dict(
        selector_class = ModelBasedTaskSelector,
        prefer_similar = False,
        choose_from_last_pick = False,
        max_reward = None,
        progress_weight = None,
    ),
    dict(
        selector_class = ModelBasedTaskSelector,
        prefer_similar = False,
        choose_from_last_pick = True,
        max_reward = None,
        progress_weight = None,
    ),""",
]
