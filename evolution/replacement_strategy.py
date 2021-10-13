import ray
import numpy as np
import time


class ReplacementStrategy:
    def __init__(self, max_pairings: int = 10):
        self.max_pairings = max_pairings
        self.archive_history = {}

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
            for p in finished_pairs:
                self.archive_history[p.id] = p.get_picklable_state()
                time.sleep(1)
                p.solver.release.remote()

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
        print(f'keep: {[p.id for p in keep]}')
        print(f'kill: {[p.id for p in finished_pairs]}')
        for p in finished_pairs:
            self.archive_history[p.id] = p.get_picklable_state()
            time.sleep(5)
            p.solver.release.remote()
        return keep
