import optax
import equinox as eqx
import jax
from typing import Callable
from src.base.base_solver import BaseSolver


class LBFGSSolver(BaseSolver):
    def __init__(self, n_iter: int = 200):
        self.n_iter = n_iter
        self.params = None
        self.opt_loss_val = None

    def fit(self, hmm_params, ys, xs=None, u_pre=None,
            frozen=None, loss_fn: Callable | None = None) -> None:
        whole_frozen, element_frozen = self._parse_frozen(frozen)
        filter_spec = self._build_filter_spec(hmm_params, whole_frozen)
        trainable, static = eqx.partition(hmm_params, filter_spec)
        _loss_fn = self._build_loss_fn(static, u_pre, ys, xs, loss_fn=loss_fn,
                                       element_frozen=element_frozen,
                                       original_params=hmm_params)

        arrays, non_arrays = eqx.partition(trainable, eqx.is_array)

        def array_loss_fn(arrays):
            return _loss_fn(eqx.combine(arrays, non_arrays))

        optimizer = optax.lbfgs()
        opt_state = optimizer.init(arrays)

        @jax.jit
        def run(arrays, opt_state):
            
            def body(_, carry):
                arrays, opt_state = carry
                val, grads = jax.value_and_grad(array_loss_fn)(arrays)
                updates, new_opt_state = optimizer.update(
                    grads, opt_state, arrays,
                    value=val, grad=grads, value_fn=array_loss_fn
                )
                return optax.apply_updates(arrays, updates), new_opt_state
            return jax.lax.fori_loop(0, self.n_iter, body, (arrays, opt_state))

        arrays, opt_state = run(arrays, opt_state)
        val = array_loss_fn(arrays)

        self.params = eqx.combine(eqx.combine(arrays, non_arrays), static)
        self.params = self._restore_frozen_elements(self.params, element_frozen, hmm_params)
        self.opt_loss_val = float(val)
