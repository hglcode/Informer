class EarlyStopper:
    def __init__(self, patience: int = 18) -> None:
        self._best_loss = float('inf')
        self._best_score = float('-inf')
        self._patience = patience
        self._counter = 0

    def __call__(self, loss: float | None = None, score: float | None = None) -> bool:
        return self.stoppable(loss, score)

    def stoppable(self, loss: float | None = None, score: float | None = None) -> bool:
        if loss and score:
            return self.stop(loss, score)
        if score:
            return self.stop_with_score(score)
        if loss:
            return self.stop_with_loss(loss)
        raise ValueError('loss and score cannot be None at the same time')

    def _better_loss(self, loss: float) -> bool:
        return loss < self._best_loss

    def _better_score(self, score: float) -> bool:
        return score > self._best_score

    def stop_with_score(self, score: float) -> bool:
        if self._better_score(score):
            self._best_score = score
            self._counter = 0
            return False
        self._counter += 1
        return self._counter > self._patience

    def stop_with_loss(self, loss: float) -> bool:
        if self._better_loss(loss):
            self._best_loss = loss
            self._counter = 0
            return False
        self._counter += 1
        return self._counter > self._patience

    def stop(self, loss: float, score: float) -> bool:
        if self._better_score(score):
            self._best_loss = loss
            self._best_score = score
            self._counter = 0
            return False
        self._counter += 1
        if self._counter > self._patience:
            return True
        return False

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def best_score(self) -> float:
        return self._best_score

    def better(self) -> bool:
        return self.counter == 0

    def best(self) -> bool:
        return abs(self.best_loss) < 1e-9 and abs(abs(self.best_score) - 1.0) < 1e-9
