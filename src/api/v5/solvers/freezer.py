from typing import Any
import equinox as eqx
import jax

Index = tuple[int, ...] | int
Frozen = list[tuple[str, Index | None]]


class Freezer:
    """
    Decides which parameters are held fixed during fitting.

    frozen is a list of (leaf name, index) tuples:
      [("log_sigma", None)]                           — freeze the whole leaf
      [("phi_tilde", (0, 3)), ("phi_tilde", (1, 2))]  — freeze single elements
    """

    def __init__(self, frozen: Frozen | None = None):
        frozen = frozen or []
        self.leaves = {name for name, idx in frozen if idx is None}
        self.elements: dict[str, list[Index]] = {}
        for name, idx in (f for f in frozen if f[1] is not None):
            self.elements.setdefault(name, []).append(idx) # type: ignore

    def filter_spec(self, params: eqx.Module) -> Any:
        """Boolean pytree: True = trainable, False = frozen leaf or non-array."""
        def is_trainable(path, leaf):
            return eqx.is_array(leaf) and self._leaf_name(path) not in self.leaves
        return jax.tree_util.tree_map_with_path(is_trainable, params)

    def pin(self, params: eqx.Module, original: eqx.Module) -> eqx.Module:
        """Reset frozen elements to their original values (zero gradient through them)."""
        if not self.elements:
            return params
        return jax.tree_util.tree_map_with_path(self._pin_leaf, params, original)

    def _pin_leaf(self, path, current, original):
        for idx in self.elements.get(self._leaf_name(path), []): # type: ignore
            current = current.at[idx].set(original[idx])
        return current

    @staticmethod
    def _leaf_name(path) -> str | None:
        return getattr(path[-1], "name", getattr(path[-1], "key", None)) if path else None
