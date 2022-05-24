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
        """
        @param pair_list: the list from which self.max_children items will be randomly selected
        """
        # select self.max_children items randomly
        # select their IDs and then pull those elements out into a new list
        return [pair_list[i] for i in self._rand.choice(len(pair_list), size=self._max_children)]
