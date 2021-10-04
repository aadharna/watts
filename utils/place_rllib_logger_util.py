import os
import tempfile

from ray.tune.logger import UnifiedLogger


def custom_log_creator(custom_path, custom_str):

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=custom_str, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator
