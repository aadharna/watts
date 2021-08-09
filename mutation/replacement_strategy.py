

class ReplacementStrategy:
    def __init__(self, max_pairings: int = 10):
        self.max_pairings = max_pairings
        self.overflow = []

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
            [p.solver.release.remote() for p in finished_pairs]
            self.overflow.extend(finished_pairs)
            del aged_pairs

        return archive
