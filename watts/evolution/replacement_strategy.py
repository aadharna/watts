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
    def __init__(self, archive, max_pairings: int = 10):
        self.max_pairings = max_pairings
        self.archive_history = archive

    def update(self, archive) -> list:
        raise NotImplementedError()


class ReplaceOldest(ReplacementStrategy):
    def __init__(self, archive, max_pairings: int = 10):
        super().__init__(archive=archive, max_pairings=max_pairings)

    def update(self, active_population) -> list:
        if len(active_population) > self.max_pairings:
            aged_pairs = sorted(active_population, key=lambda x: x.id, reverse=True)
            active_population = aged_pairs[:self.max_pairings]
            finished_pairs = aged_pairs[self.max_pairings:]
            _release(self.archive_history, finished_pairs)

        return active_population


class KeepTopK(ReplacementStrategy):
    def __init__(self, archive, max_pairings: int = 10):
        super().__init__(archive=archive, max_pairings=max_pairings)

    def update(self, active_population) -> list:
        if self.max_pairings >= len(active_population):
            return active_population
        sorted_pairs = sorted(active_population, key=lambda x: x.get_eval_metric(), reverse=True)
        active_population = sorted_pairs[:self.max_pairings]
        finished_pairs = sorted_pairs[self.max_pairings:]
        _release(self.archive_history, finished_pairs)

        return active_population
