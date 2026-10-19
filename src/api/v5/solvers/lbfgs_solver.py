from functools import partial
from typing import Any, Callable
import jax
import optax
import optax.tree_utils as otu
from src.api.v5.hmm.params import Params
from src.api.v5.solvers.base_solver import BaseSolver


class LBFGSSolver(BaseSolver):
    """L-BFGS with line search; stops when the gradient norm drops below tol or after n_iter steps."""

    def __init__(self, n_iter: int = 200, tol: float = 1e-6):
        self.n_iter = n_iter
        self.tol = tol
        self.optimizer = optax.lbfgs()

    def _minimise(self, trainable: Params, objective: Callable, data: Any) -> tuple[Params, float, bool]:
        params, state = self._run(objective, trainable, data)
        loss, grad = otu.tree_get(state, "value"), otu.tree_get(state, "grad")
        return params, float(loss), bool(otu.tree_norm(grad) < self.tol)

    @partial(jax.jit, static_argnums=(0, 1))
    def _run(self, objective: Callable, params: Params, data: Any):
        value_fn, carry = partial(objective, data=data), (params, self.optimizer.init(params)) # type: ignore
        return jax.lax.while_loop(self._should_continue, partial(self._step, value_fn), carry)

    def _step(self, value_fn: Callable, carry):
        params, state = carry
        value, grad = optax.value_and_grad_from_state(value_fn)(params, state=state)
        updates, state = self.optimizer.update(grad, state, params, value=value, grad=grad, value_fn=value_fn)
        return optax.apply_updates(params, updates), state

    def _should_continue(self, carry) -> jax.Array:
        _, state = carry
        count, grad = otu.tree_get(state, "count"), otu.tree_get(state, "grad")
        return (count == 0) | ((count < self.n_iter) & (otu.tree_norm(grad) >= self.tol))
