import optax
import equinox as eqx
from typing import Callable
from src.base.base_solver import BaseSolver


class GradientSolver(BaseSolver):
    def __init__(self, optimizer=None, n_iter: int = 500, verbose: bool = False):
        self.optimizer = optimizer or optax.adam(1e-3)
        self.n_iter = n_iter
        self.verbose = verbose
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

        optimizer = self.optimizer
        opt_state = optimizer.init(eqx.filter(trainable, eqx.is_array))

        @eqx.filter_jit
        def make_step(trainable, opt_state):
            val, grads = eqx.filter_value_and_grad(_loss_fn)(trainable)
            updates, new_opt_state = optimizer.update(
                grads, opt_state, eqx.filter(trainable, eqx.is_array)
            )
            new_trainable = eqx.apply_updates(trainable, updates)
            return new_trainable, new_opt_state, val

        for i in range(self.n_iter):
            trainable, opt_state, val = make_step(trainable, opt_state)
            if self.verbose and (i % 50 == 0 or i == self.n_iter - 1):
                print(f"iter {i:4d}  loss={float(val):.6f}")

        self.params = eqx.combine(trainable, static)
        self.params = self._restore_frozen_elements(
            self.params, element_frozen, hmm_params)
        self.opt_loss_val = float(val)
