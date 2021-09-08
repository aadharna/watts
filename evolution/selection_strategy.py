import numpy as np


class SelectionStrategy:
    def select(self, pair_list) -> list:
        raise NotImplementedError()


class SelectRandomly(SelectionStrategy):
    def __init__(
            self,
            max_children: int = 10,
            rand: np.random.RandomState = np.random.RandomState()
    ):
        self._max_children = max_children
        self._rand = rand

    def select(self, pair_list) -> list:
        return [pair_list[i] for i in self._rand.choice(len(pair_list), size=self._max_children)]
