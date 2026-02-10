from args import Args
from logger_base import AIMLogger, LoggerBase, LoggerList, StandardLogger


def init_logger(args: Args, run_name: str) -> LoggerBase:
    logger = AIMLogger()
    logger.define_experiment(
        env_name="MamlTorchCheetah",
        algorithm_name="TorchMamlPPO",
        hparams=vars(args) | {run_name: run_name},
    )
    logger.start_new_episode()
    return logger
