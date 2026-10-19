from dataclasses import dataclass
import jax
import jax.numpy as jnp
import pandas as pd
from jax.scipy.stats import chi2, norm
from src.api.v5.hmm.params import Params
from src.api.v5.inference.forward_algorithm import ForwardAlgorithm


def uniform_initial_dist(params: Params) -> jnp.ndarray:
    """Uniform (1, N) initial distribution. N is the (augmented) number of states."""
    N = params.transition.transition_logits.shape[0]
    return jnp.ones((1, N)) / N


@dataclass(frozen=True)
class ModelEntry:
    params: Params
    initial_dist: jnp.ndarray
    ys: jnp.ndarray
    xs: jnp.ndarray | None
    n_params: int


class Diagnostics:
    """
    In-sample diagnostics (LL, AIC, BIC, pseudo-residuals, LRT) for several named models.

    `ys`/`xs` given here are the default data for every model; `add_model` can
    override them per model (e.g. a multivariate model next to univariate ones).
    """

    def __init__(self, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None):
        self.ys = ys
        self.xs = xs
        self.models: dict[str, ModelEntry] = {}
        self.lrt_pairs: list[tuple[str, str]] = []

    def add_model(self, name: str, params: Params, initial_dist: jnp.ndarray | None = None,
                  ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None,
                  n_params: int | None = None) -> "Diagnostics":
        """Register a fitted model. `n_params` overrides len(params), e.g. when parameters were frozen."""
        ys = self.ys if ys is None else ys
        xs = self.xs if xs is None else xs
        if ys is None:
            raise ValueError(f"No data for model '{name}': pass ys here or to Diagnostics(ys=...).")
        initial_dist = uniform_initial_dist(params) if initial_dist is None else initial_dist
        n_params = len(params) if n_params is None else n_params
        self.models[name] = ModelEntry(params, jnp.asarray(initial_dist), jnp.asarray(ys), xs, n_params)
        return self

    def _forward(self, name: str):
        m = self.models[name]
        return ForwardAlgorithm.run(params=m.params, initial_dist=m.initial_dist, ys=m.ys, xs=m.xs)

    # --- information criteria ---------------------------------------------

    def log_likelihood(self, name: str) -> float:
        _, fts, _ = self._forward(name)
        return float(jnp.sum(jnp.log(fts)))

    def aic(self, name: str) -> float:
        return 2 * self.models[name].n_params - 2 * self.log_likelihood(name)

    def bic(self, name: str) -> float:
        m = self.models[name]
        return m.n_params * float(jnp.log(len(m.ys))) - 2 * self.log_likelihood(name)

    def summary(self) -> pd.DataFrame:
        rows = {}
        for name, m in self.models.items():
            ll = self.log_likelihood(name)
            k, T = m.n_params, len(m.ys)
            rows[name] = {"LL": ll, "n_params": k, "AIC": 2 * k - 2 * ll, "BIC": k * float(jnp.log(T)) - 2 * ll}
        return pd.DataFrame.from_dict(rows, orient="index")

    # --- residuals ----------------------------------------------------------

    def residuals(self, name: str) -> jnp.ndarray:
        """
        Forecast pseudo-residuals Phi^{-1}(sum_i u_t[i] F_i(y_t)), with u_t the
        one-step-ahead predictive state distribution.

        Multivariate emissions get one residual per dimension, shape (T, K), from the
        marginal CDFs: the product of marginal CDFs is not a valid PIT. For an AR
        emission the first k residuals are NaN, since their lags don't exist.
        """
        if not jax.config.jax_enable_x64:
            raise RuntimeError("Pseudo-residuals require float64: set "
                               "jax.config.update('jax_enable_x64', True) before importing anything.")
        m = self.models[name]
        _, _, uts = self._forward(name)
        ut = uts.reshape(len(m.ys), -1)                                           # (T, N)
        emission, indices = m.params.emission, jnp.arange(len(m.ys))

        if hasattr(emission, "marginal_cdfs"):
            Gs = jax.vmap(lambda i: emission.marginal_cdfs(i, m.ys))(indices)     # (T, K, N)
        else:
            Gs = m.params.cdfs(ys=m.ys).reshape(len(m.ys), 1, -1)                 # (T, 1, N)
        u = jnp.clip(jnp.sum(ut[:, None, :] * Gs, axis=-1), 1e-6, 1.0 - 1e-6)    # (T, K)
        res = norm.ppf(u)

        if hasattr(emission, "phi_tilde"):
            res = res.at[:len(emission.phi_tilde)].set(jnp.nan)
        return res[:, 0] if res.shape[1] == 1 else res

    def residuals_all(self) -> dict[str, jnp.ndarray]:
        return {name: self.residuals(name) for name in self.models}

    # --- likelihood-ratio tests ---------------------------------------------

    def add_lrt(self, base: str, expanded: str) -> "Diagnostics":
        """Register a nested pair: `base` must be a restricted version of `expanded`."""
        for name in (base, expanded):
            if name not in self.models:
                raise KeyError(f"Unknown model '{name}'. Add it with add_model first.")
        b, e = self.models[base], self.models[expanded]
        if e.n_params <= b.n_params:
            raise ValueError(f"'{expanded}' must have more parameters than '{base}' "
                             f"(got {e.n_params} vs {b.n_params}).")
        if b.ys.shape != e.ys.shape or not bool(jnp.array_equal(b.ys, e.ys)):
            raise ValueError(f"'{base}' and '{expanded}' were not given the same data, so the LRT is undefined.")
        self.lrt_pairs.append((base, expanded))
        return self

    def lrt(self) -> pd.DataFrame:
        """LR = 2(LL_expanded - LL_base) ~ chi2(df = k_expanded - k_base) under the base model."""
        rows = []
        for base, expanded in self.lrt_pairs:
            ll_b, ll_e = self.log_likelihood(base), self.log_likelihood(expanded)
            df = self.models[expanded].n_params - self.models[base].n_params
            lr = 2 * (ll_e - ll_b)
            rows.append({"base": base, "expanded": expanded, "LL_base": ll_b, "LL_expanded": ll_e,
                         "LR": lr, "df": df, "p_value": float(chi2.sf(lr, df))})
        return pd.DataFrame(rows)
