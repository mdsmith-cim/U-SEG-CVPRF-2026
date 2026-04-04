import os
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger
from pathlib import Path
from datetime import datetime

# Based on detectron2 implementation
def default_setup(args, name="offline_eval"):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up logger
    2. Log basic information about environment, cmdline arguments

    Args:
        args (argparse.NameSpace): the command line arguments to be logged
        name: name of script/program
    """
    output_log_file = os.path.join(str(args.experiment_directory), f'log-{datetime.now().strftime("%d-%m-%Y-%I-%M-%S%p")}.txt')
    logger = setup_logger(output_log_file, name=name)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Command line arguments: " + str(args))
    return logger

def loadConfig(filename: Path):
    """
    Loads detectron2-format config file containing run information.
    :param filename: Filename of config file
    :return: detectron2.config.config.CfgNode object
    """
    from detectron2.config import CfgNode as CN
    return CN.load_yaml_with_base(filename, True)

