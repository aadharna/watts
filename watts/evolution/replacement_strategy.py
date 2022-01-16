from collections import OrderedDict


def _release(archive: dict, finished_list: list):
    """helper function to clean up pair objects that are no longer being used going forward

    :param archive: container to save meta-data in for the objects being destroyed
    :param finished_list: list of objects to release
    :return: n/a
    """
    for p in finished_list:
        archive[p.id] = p.get_picklable_state()
        p.solver.release.remote()


class ReplacementStrategy:
    def __init__(self, max_pairings: int = 10):
        self.max_pairings = max_pairings
        self.archive_history = OrderedDict()

    def update(self, archive) -> list:
        raise NotImplementedError()


class ReplaceOldest(ReplacementStrategy):
    def __init__(self, max_pairings: int = 10):
        super().__init__(max_pairings)

    def update(self, archive) -> list:
        if len(archive) > self.max_pairings:
            aged_pairs = sorted(archive, key=lambda x: x.id, reverse=True)
            archive = aged_pairs[:self.max_pairings]
            finished_pairs = aged_pairs[self.max_pairings:]
            _release(self.archive_history, finished_pairs)

        return archive


class KeepTopK(ReplacementStrategy):
    def __init__(self, max_pairings: int):
        super().__init__(max_pairings)

    def update(self, archive) -> list:
        if self.max_pairings >= len(archive):
            return archive
        sorted_pairs = sorted(archive, key=lambda x: x.get_eval_metric(), reverse=True)
        keep = sorted_pairs[:self.max_pairings]
        finished_pairs = sorted_pairs[self.max_pairings:]
        _release(self.archive_history, finished_pairs)

        return keep
