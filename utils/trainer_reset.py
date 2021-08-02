from ray.rllib.utils.annotations import override
from ray.tune import Trainable


class ResetConfigOverride:
    @override(Trainable)
    def reset_config(self, new_config):
        # do something with new_config
        return True  # <- signals successful reset

