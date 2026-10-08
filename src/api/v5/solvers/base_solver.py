from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable
import equinox as eqx
from src.api.v5.hmm.params import Params
from src.api.v5.solvers.freezer import Freezer, Frozen


@dataclass(frozen=True)
class SolverResult:
    params: Params
    loss: float
    converged: bool


class BaseSolver(ABC):
    """
    Stateless ABC for HMM solvers.

    fit() minimises loss_fn(params, data) over the trainable part of params.
    data is passed through untouched, since its structure is defined by loss_fn.
    """

    def fit(self, params: Params, loss_fn: Callable, data: Any, frozen: Frozen | None = None) -> SolverResult:
        freezer = Freezer(frozen)
        trainable, static = eqx.partition(params, freezer.filter_spec(params))
        objective = self._objective(loss_fn, static, freezer, params)
        trainable, loss, converged = self._minimise(trainable, objective, data)
        return SolverResult(freezer.pin(eqx.combine(trainable, static), params), loss, converged) # type: ignore

    @staticmethod
    def _objective(loss_fn: Callable, static: Params, freezer: Freezer, original: Params) -> Callable:
        """Returns objective(trainable, data) -> scalar loss."""
        def objective(trainable, data):
            return loss_fn(freezer.pin(eqx.combine(trainable, static), original), data)
        return objective

    @abstractmethod
    def _minimise(self, trainable: Params, objective: Callable, data: Any) -> tuple[Params, float, bool]:
        """Minimise objective(trainable, data); return (trainable, loss, converged)."""
