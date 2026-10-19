from typing import Any, Callable
import equinox as eqx
import optax
import optax.tree_utils as otu
from src.api.v5.hmm.params import Params
from src.api.v5.solvers.base_solver import BaseSolver


class GradientSolver(BaseSolver):
    """First-order optax optimiser (Adam by default); stops when the gradient norm drops below tol."""

    def __init__(self, optimizer=None, n_iter: int = 500, tol: float = 1e-6, verbose: bool = False):
        self.optimizer = optimizer or optax.adam(1e-3)
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    def _minimise(self, trainable: Params, objective: Callable, data: Any) -> tuple[Params, float, bool]:
        step, state = self._make_step(objective), self.optimizer.init(trainable) # type: ignore
        for i in range(self.n_iter):
            trainable, state, loss, grad_norm = step(trainable, state, data)
            self._log(i, loss)
            if grad_norm < self.tol:
                return trainable, float(loss), True
        return trainable, float(loss), False # type: ignore

    def _make_step(self, objective: Callable) -> Callable:
        @eqx.filter_jit
        def step(trainable, state, data):
            loss, grads = eqx.filter_value_and_grad(objective)(trainable, data)
            updates, state = self.optimizer.update(grads, state, trainable)
            return eqx.apply_updates(trainable, updates), state, loss, otu.tree_norm(grads)
        return step

    def _log(self, i: int, loss) -> None:
        if self.verbose and (i % 50 == 0 or i == self.n_iter - 1):
            print(f"iter {i:4d}  loss={float(loss):.6f}")
