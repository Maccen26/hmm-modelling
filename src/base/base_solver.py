from abc import ABC, abstractmethod
from typing import Callable
import equinox as eqx
import jax


class BaseSolver(ABC):
    """
    Stateless ABC for HMM solvers.
    After calling fit(), the fitted HMMParams are stored in self.params.
    """

    @abstractmethod
    def fit(self, hmm_params, ys, xs=None, u_pre=None,
            frozen=None, loss_fn: Callable | None = None) -> None:
        """Fit hmm_params to data. Result is stored in self.params."""
        ...

    def _parse_frozen(self, frozen):
        """Split frozen into whole-leaf freezes and element-level freezes.

        Returns (whole_frozen, element_frozen) where:
          whole_frozen: dict like {"mu0": False} or None
          element_frozen: dict like {"phi_tilde": (0, 3)} or {}
        """
        if frozen is None:
            return None, {}
        whole = {}
        element = {}
        for k, v in frozen.items():
            if v is False:
                whole[k] = False
            elif isinstance(v, tuple):
                element[k] = v
        return whole or None, element

    def _build_filter_spec(self, params: eqx.Module, frozen: dict | None):
        """Returns a boolean pytree: True = trainable, False = frozen.

        Only handles whole-leaf freezes (value is False).
        Element-level freezes (value is tuple) are handled in _build_loss_fn.

        frozen examples:
          {"mu0": False}  — freeze the whole mu0 leaf
        """
        if frozen is None:
            return eqx.is_array

        def filter_fn(path, leaf):
            if not path:
                return eqx.is_array(leaf)

            last = path[-1]
            name = getattr(last, "name", getattr(last, "key", None))

            if name in frozen and frozen[name] is False:
                return False

            return eqx.is_array(leaf)

        return jax.tree_util.tree_map_with_path(filter_fn, params)

    @staticmethod
    def _freeze_elements(params, element_frozen, original_params):
        """Replace frozen array elements with stop_gradient'd originals.

        The index tuple is adapted to the array's dimensionality:
          scalar  → freeze the whole scalar
          1D      → freeze element at idx[-1]
          2D+     → freeze element at idx tuple
        """
        if not element_frozen:
            return params

        def replace_fn(path, current, original):
            if not eqx.is_array(current) or not path:
                return current
            name = getattr(path[-1], "name", getattr(path[-1], "key", None))
            if name not in element_frozen:
                return current
            idx = element_frozen[name]
            ndim = current.ndim
            if ndim == 0:
                return jax.lax.stop_gradient(original)
            elif ndim == 1:
                return current.at[idx[-1]].set(
                    jax.lax.stop_gradient(original[idx[-1]]))
            else:
                return current.at[idx].set(
                    jax.lax.stop_gradient(original[idx]))

        return jax.tree_util.tree_map_with_path(
            replace_fn, params, original_params)

    @staticmethod
    def _restore_frozen_elements(params, element_frozen, original_params):
        """Restore frozen elements to their original values after optimisation."""
        if not element_frozen:
            return params

        def replace_fn(path, current, original):
            if not eqx.is_array(current) or not path:
                return current
            name = getattr(path[-1], "name", getattr(path[-1], "key", None))
            if name not in element_frozen:
                return current
            idx = element_frozen[name]
            ndim = current.ndim
            if ndim == 0:
                return original
            elif ndim == 1:
                return current.at[idx[-1]].set(original[idx[-1]])
            else:
                return current.at[idx].set(original[idx])

        return jax.tree_util.tree_map_with_path(
            replace_fn, params, original_params)

    def _build_loss_fn(self, static, u_pre, ys, xs,
                       loss_fn: Callable | None = None,
                       element_frozen=None,
                       original_params=None) -> Callable:
        """Returns scalar loss(trainable) -> scalar."""
        from src.api.v4.likelihoods import negative_log_likelihood
        from src.api.v4.algorithms.forward_algorithm import ForwardAlgorithm

        _loss = loss_fn if loss_fn is not None else negative_log_likelihood
        _freeze = self._freeze_elements

        def loss(trainable):
            full_params = eqx.combine(trainable, static)
            full_params = _freeze(full_params, element_frozen, original_params)
            output = ForwardAlgorithm().run(full_params, u_pre, ys, xs)
            return _loss(output, full_params)

        return loss
