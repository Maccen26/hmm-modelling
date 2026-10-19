# The base classes live next to their implementations (hmm/, inference/, solvers/).
# They are re-exported lazily: the transitions import `src.api.v5.base.utils`, which
# runs this file first, so importing them eagerly here would be circular.
import importlib

_EXPORTS = {
    "BaseTransition": "src.api.v5.hmm.transitions.base_transition",
    "BaseEmission": "src.api.v5.hmm.emissions.base_emission",
    "BaseHMM": "src.api.v5.hmm.base_hmm",
    "BaseInference": "src.api.v5.inference.base_inference",
    "BaseSolver": "src.api.v5.solvers.base_solver",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
