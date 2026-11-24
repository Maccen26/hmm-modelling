"""Simulation-based sample size for occupant feedback: how many votes before P(feedback | state) and the comfort curve
are estimated precisely, and the comfort-vote bands can be told apart.

R datasets are simulated from a saved AR(1)-HMM, each until N_MAX votes. Every step gets a temperature
T_t ~ U(15, 25), feedback f_t ~ Bernoulli(p[z_t]) and, if feedback is given, the vote is the bin of the perceived
temperature T_t + s·Logistic(0, 1) on the cut-points 17 | 19 | 21 | 23:

    opt0 (< 17) | opt1 (17-19) | opt2 (19-21, fine) | opt3 (21-23) | opt4 (> 23)

The true p defaults to the P(feedback | state) estimated by drivers.simulation.feedback_states. On the grid
N = n_step, 2·n_step, ..., N_MAX each dataset is cut at its N-th vote (t <= t_N) and

    p_hat_s = sum_t f_t γ_t(s) / W_s, W_s = sum_t γ_t(s)    γ_t(s) = P(z_t = s | y_1:t_N), smoothed with the true model
    c, s    ordered-logit MLE (statsmodels OrderedModel) on the (T_t, vote_t) pairs of the first N votes
    z_k     = w_k / se(w_k)                                            w_k = c_{k+1} - c_k, the width of band opt_k

The smoothing only uses the CO2 seen up to the N-th vote, as an experiment stopped there would.

Power at N is the share of the datasets with z_k > 1.645 (the band is wider than 0 at the 5 % level). The N needed is
the first N from which the power stays >= 80 %.

    uv run python -m drivers.simulation.sample_size --room 15 -K 4
    uv run python -m drivers.simulation.sample_size --room 15 -K 4 --R 4 --n-max 30 --n-step 10   # smoke run
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

from drivers.analysis.state_confusion import load_saved
from drivers.models.utils import PLOTS_DIR, RESULTS_DIR, TAG, covariates, room_name
from drivers.simulation.state_analysis import model_label
from drivers.utils import save_thesis_table
from src.api.v5 import ForwardAlgorithm, Params
from src.api.v5.inference.backward import BackwardAlgorithm

WEEK = 7 * 48
ROOM_WEEK = "2023-11-27"  # an ordinary Mon-Sun week in the training split, shown next to the simulated weeks
T_BUCKET = 2048           # smoothed series are padded to a multiple of this, so only a few shapes get compiled
CHUNK = 4 * WEEK          # steps simulated between checks of the vote count
TEMP_RANGE = (15.0, 25.0)
T_CENTRE = 20.0           # temperatures are centred here for the logit fit
CUTPOINTS = np.array([17.0, 19.0, 21.0, 23.0])
N_CATS = len(CUTPOINTS) + 1
CAT_LABELS = [f"opt{k}" for k in range(N_CATS)]
CUT_NAMES = [f"c_{k + 1}" for k in range(N_CATS - 1)]
COMFORT = CUT_NAMES + ["scale", "centre", "width"]  # centre and width of the comfort band opt2
BANDS = [f"z_{k}" for k in range(1, N_CATS - 1)]    # Wald z of the widths of the inner bands opt1..opt3
ALPHA, POWER = 0.05, 0.8
Z_CRIT = stats.norm.ppf(1 - ALPHA)
TABLE_N = (10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300)  # rows of the thesis estimate tables
TEST_LABELS = {"z_1": "opt1 width > 0", "z_2": "opt2 (comfort band) width > 0", "z_3": "opt3 width > 0",
               "bands_all": "all bands distinguishable"}


# --- Helpers ------------------------------------------------------------------------

def week_covariates(N: int, cols, start: str = "2024-03-04") -> jax.Array:
    """(N, C) covariates for the β models: one ordinary Mon-Sun week (30 min steps, no holidays) stacked until N rows."""
    week = covariates(pd.date_range(start, periods=WEEK, freq="30min"), cols)
    return jnp.tile(week, (N // len(week) + 1, 1))[:N]


def vote_colors():
    return plt.cm.coolwarm(np.linspace(0, 1, N_CATS))


def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den; NaN for a state with no weight yet."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def true_values(p: np.ndarray, scale: float) -> dict[str, float]:
    return {**{f"p_{s}": p[s] for s in range(len(p))}, **dict(zip(CUT_NAMES, CUTPOINTS)), "scale": scale,
            "centre": (CUTPOINTS[1] + CUTPOINTS[2]) / 2, "width": CUTPOINTS[2] - CUTPOINTS[1]}


# --- True feedback rates ------------------------------------------------------------

def feedback_csv(room: str, states: int, model: str, tag: str) -> str:
    return os.path.join(RESULTS_DIR, room, tag, f"states_{states}", "simulation", f"{model}_feedback_given_state.csv")


def estimated_feedback(path: str, K: int) -> tuple[np.ndarray, int]:
    """P(feedback | state) estimated by feedback_states (its `state` column is 1-based) and the room's feedback count."""
    df = pd.read_csv(path).sort_values("state")
    if len(df) != K:
        raise SystemExit(f"{path} has {len(df)} states, expected {K}")
    return df["p_feedback_mean"].to_numpy(), int(round(df["events_mean"].sum()))


def true_feedback(args) -> tuple[np.ndarray, int | None]:
    """The true p (--p, else the feedback_states estimate) and the room's number of feedbacks (None without that csv)."""
    csv = feedback_csv(room_name(args.room), args.states, args.model, args.tag)
    if not os.path.exists(csv) and args.p is None:
        raise SystemExit(f"No {csv}; run drivers.simulation.feedback_states {args.model} --room {args.room} "
                         f"-K {args.states} first, or pass --p")
    p_est, n_room = estimated_feedback(csv, args.states) if os.path.exists(csv) else (None, None)
    return (p_est if args.p is None else np.asarray(args.p)), n_room


# --- Simulation ---------------------------------------------------------------------

class FeedbackSimulator:
    """States and CO2 from a saved AR(1)-HMM, a uniform temperature per step and noisy categorical feedback.

    Follows the forward algorithm's convention: z_0 ~ u00 · Γ_0 and z_t ~ Γ_t[z_{t-1}]. The covariates repeat
    weekly, so one week of transition matrices is computed and reused.
    """
    def __init__(self, params: Params, init_dist, xs_fn: Callable[[int], jax.Array], scale: float):
        self.mu = np.asarray(params.emission.mu_vals(t=0, ys=0))
        self.std = np.asarray(params.emission.sigma(t=0, ys=0))
        self.phi = np.asarray(params.emission.phi())[0]
        self.scale = scale
        gammas = self._weekly_gammas(params, xs_fn)
        self.cum_gammas, self.cum_u0 = np.cumsum(gammas, axis=-1), np.cumsum(np.asarray(init_dist).ravel() @ gammas[0])

    @staticmethod
    def _weekly_gammas(params: Params, xs_fn) -> np.ndarray:
        """(WEEK, K, K) transition matrices of one week, checked to repeat in the next."""
        gammas = np.asarray(params.transition_matrices(N=2 * WEEK, xs=xs_fn(2 * WEEK)))
        if not np.allclose(gammas[:WEEK], gammas[WEEK:]):
            raise ValueError("The transition matrices do not repeat weekly")
        return gammas[:WEEK]

    def simulate(self, p: np.ndarray, seed: int, n_max: int, max_steps: int = 10_000_000) -> dict:
        """Simulate until n_max votes; every array ends at the n_max-th vote."""
        rng = np.random.default_rng(seed)
        path, emissions, feedback = self._until_votes(rng, p, n_max, max_steps)
        T = np.flatnonzero(feedback)[n_max - 1] + 1
        temps, votes = self._votes(rng, feedback[:T])
        return {"emissions": emissions[:T], "path": path[:T], "temps": temps, "feedback": feedback[:T].astype(int),
                "votes": votes, "seed": seed}

    def _until_votes(self, rng, p: np.ndarray, n_max: int, max_steps: int) -> tuple[np.ndarray, ...]:
        """(path, emissions, feedback), drawn CHUNK steps at a time until the feedback holds n_max votes."""
        chunks, last = [], (None, None)
        while (n_votes := sum(int(f.sum()) for *_, f in chunks)) < n_max:
            if len(chunks) * CHUNK >= max_steps:
                raise RuntimeError(f"Only {n_votes}/{n_max} votes after {max_steps} steps")
            path, ys = self._chunk(rng, len(chunks) * CHUNK, *last)
            chunks.append((path, ys, rng.random(CHUNK) < p[path]))
            last = (path[-1], ys[-1])
        return tuple(np.concatenate(a) for a in zip(*chunks))

    def _chunk(self, rng, t0: int, s, y) -> tuple[np.ndarray, np.ndarray]:
        """CHUNK states and emissions from step t0 on, continuing from state s and emission y (None at t = 0)."""
        z, u = rng.standard_normal(CHUNK), rng.random(CHUNK)
        path, ys = np.empty(CHUNK, dtype=int), np.empty(CHUNK)
        for i in range(CHUNK):
            s, y = self._step(t0 + i, s, y, u[i], z[i])
            path[i], ys[i] = s, y
        return path, ys

    def _step(self, t: int, s, y, u: float, z: float) -> tuple[int, float]:
        """z_t by inverse CDF from the uniform u, then its AR(1) emission from the standard normal z."""
        cum = self.cum_u0 if s is None else self.cum_gammas[t % WEEK, s]
        s = min(int(np.searchsorted(cum, u * cum[-1])), len(cum) - 1)
        mean = self.mu[s] if y is None else self.mu[s] + self.phi[s] * (y - self.mu[s])
        return s, mean + self.std[s] * z

    def _votes(self, rng, feedback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Temperatures T_t ~ U(15, 25) and, where f_t = 1, the bin of T_t + s·Logistic(0, 1); -1 elsewhere."""
        temps = rng.uniform(*TEMP_RANGE, size=len(feedback))
        perceived = temps + self.scale * rng.logistic(size=len(feedback))
        return temps, np.where(feedback, np.searchsorted(CUTPOINTS, perceived, side="right"), -1)


class DatasetDrawer:
    """One dataset per seed, each until n_max votes; every one is saved to datasets/ as soon as it is drawn."""
    def __init__(self, simulator: FeedbackSimulator, n_max: int, out_dir: str):
        self.simulator, self.n_max = simulator, n_max
        self.dir = os.path.join(out_dir, "datasets")
        os.makedirs(self.dir, exist_ok=True)

    def draw(self, p: np.ndarray, seeds: list[int]) -> list[dict]:
        return [self._draw_one(p, k, seed, len(seeds)) for k, seed in enumerate(seeds)]

    def _draw_one(self, p: np.ndarray, k: int, seed: int, R: int) -> dict:
        t0 = time.time()
        sim = self.simulator.simulate(p, seed, self.n_max)
        np.savez(os.path.join(self.dir, f"dataset_{k:03d}.npz"), **sim)
        print(f"Simulated dataset {k + 1}/{R}: {len(sim['path'])} steps ({time.time() - t0:.1f} s)")
        return sim


# --- P(feedback | state) ------------------------------------------------------------

class StateSmoother:
    """Smoothed state probabilities P(z_t | y_1:t_N), t <= t_N, under the true model, for each cut t_N.

    The forward pass is shared by all cuts. The backward pass treats every step after t_N as unobserved (density 1),
    which leaves it exact for the prefix, so the padding to T_BUCKET does not change the result either.
    """
    def __init__(self, params: Params, init_dist, xs_fn: Callable[[int], jax.Array]):
        self.params, self.init_dist, self.xs_fn = params, init_dist, xs_fn
        self._run = jax.jit(self._smooth)

    def __call__(self, ys: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
        """(len(t_ns), T, K); row i is zero after t_ns[i]."""
        T_pad = -(-len(ys) // T_BUCKET) * T_BUCKET
        ys_pad = jnp.pad(jnp.asarray(ys), (0, T_pad - len(ys)), mode="edge")
        return np.asarray(self._run(ys_pad, self.xs_fn(T_pad), jnp.asarray(t_ns)))[:, :len(ys)]

    def _smooth(self, ys: jax.Array, xs: jax.Array, t_ns: jax.Array) -> jax.Array:
        gammas = self.params.transition_matrices(N=len(ys), xs=xs)
        gs = self.params.densities(ys=ys)[:, 0, :]
        utt = ForwardAlgorithm.run(params=self.params, initial_dist=self.init_dist, ys=ys, xs=xs)[0][:, 0, :]
        return jax.vmap(lambda t_n: self._posterior(utt, gammas, gs, t_n))(t_ns)

    def _posterior(self, utt: jax.Array, gammas: jax.Array, gs: jax.Array, t_n) -> jax.Array:
        """(T, K) P(z_t | y_1:t_n), zero after t_n."""
        seen = (jnp.arange(len(gs)) <= t_n)[:, None]
        post = jnp.where(seen, utt * self._backward(gammas, jnp.where(seen, gs, 1.0)), 0.0)
        return post / jnp.where(seen, post.sum(axis=1, keepdims=True), 1.0)

    @staticmethod
    def _backward(gammas: jax.Array, gs: jax.Array) -> jax.Array:
        """(T, K) normalised backward probabilities, b_t = Γ_{t+1} (g_{t+1} · b_{t+1}) from b_{T-1} = 1/K."""
        b_last, later = jnp.full(gs.shape[-1], 1.0 / gs.shape[-1]), lambda a: a[1:][::-1]
        step_inputs = (later(gammas), later(gs), jnp.ones((len(gs) - 1, 1)))
        _, bs = jax.lax.scan(BackwardAlgorithm.step, b_last, step_inputs)
        return jnp.concatenate([bs[::-1], b_last[None]])


@dataclass(frozen=True)
class StateRates:
    """Weighted feedback counts F_s = sum_{t<=t_N} f_t w_t(s) and weights W_s = sum_{t<=t_N} w_t(s), (cuts, K) each."""
    F: np.ndarray
    W: np.ndarray

    @classmethod
    def from_cuts(cls, weights: np.ndarray, feedback: np.ndarray) -> "StateRates":
        """From (cuts, T, K) weights that are zero after each cut's t_N, e.g. the smoothed states."""
        return cls(np.einsum("t,ntk->nk", feedback, weights), weights.sum(axis=1))

    @classmethod
    def from_path(cls, path: np.ndarray, K: int, feedback: np.ndarray, t_ns: np.ndarray) -> "StateRates":
        """With the one-hot true states as the weights: the known-states best case."""
        onehot = np.eye(K)[path]
        return cls(np.cumsum(feedback[:, None] * onehot, axis=0)[t_ns], np.cumsum(onehot, axis=0)[t_ns])

    @property
    def p_hat(self) -> np.ndarray:
        return ratio(self.F, self.W)


# --- Comfort curve ------------------------------------------------------------------

class ComfortCurve:
    """Ordered-logit fit: cut-points c_1..c_4 [°C], scale s, comfort band (centre and width of opt2) and the Wald z of
    each inner band.

    statsmodels fits P(vote <= k | T) = F(τ_k - β (T - 20)) with params (β, τ_1, log(τ_2 - τ_1), ...), so
    c_k = 20 + τ_k / β, s = 1 / β and w_k = exp(θ_k) / β. By the delta method se(w_k) = w_k · se(log w_k), so
    z_k = 1 / se(log w_k). NaN until every category has a vote, and when the fit does not converge or its Hessian is
    singular.
    """
    FAILED = dict.fromkeys(COMFORT + BANDS, np.nan)

    @classmethod
    def fit(cls, temps: np.ndarray, votes: np.ndarray) -> dict:
        if len(np.unique(votes)) < N_CATS:
            return dict(cls.FAILED)
        res = cls._mle(temps, votes)
        theta = np.asarray(res.params)
        if not res.mle_retvals["converged"] or theta[0] <= 0 or res.normalized_cov_params is None:  # None: singular
            return dict(cls.FAILED)
        return {**cls._curve(theta), **cls._band_z(np.asarray(res.cov_params()), theta[0])}

    @staticmethod
    def _mle(temps: np.ndarray, votes: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return OrderedModel(votes, (temps - T_CENTRE)[:, None], distr="logit").fit(method="bfgs", disp=False)

    @staticmethod
    def _curve(theta: np.ndarray) -> dict:
        c = T_CENTRE + (theta[1] + np.concatenate([[0.0], np.cumsum(np.exp(theta[2:]))])) / theta[0]
        return {**dict(zip(CUT_NAMES, c)), "scale": 1 / theta[0], "centre": (c[1] + c[2]) / 2, "width": c[2] - c[1]}

    @staticmethod
    def _band_z(cov: np.ndarray, beta: float) -> dict:
        var_log_w = np.diag(cov)[2:] + cov[0, 0] / beta ** 2 - 2 * cov[0, 2:] / beta
        with np.errstate(invalid="ignore"):
            return dict(zip(BANDS, 1 / np.sqrt(var_log_w)))


# --- Estimation ---------------------------------------------------------------------

class DatasetEstimates:
    """The estimates and test statistics of one dataset at every N of the grid."""
    def __init__(self, sim: dict, k: int, grid: np.ndarray, smoother: StateSmoother):
        self.vote_times = np.flatnonzero(sim["feedback"])
        self.t_ns, self.grid = self.vote_times[grid - 1], grid
        self.base = {"dataset": k, "seed": sim["seed"]}
        self.hmm, self.known = self._rates(sim, smoother)
        self.comfort = [self._comfort(sim, n) for n in grid]

    def _rates(self, sim: dict, smoother: StateSmoother) -> tuple[StateRates, StateRates]:
        """P(feedback | state) weighted by the smoothed states and by the true states."""
        smoothed = smoother(sim["emissions"], self.t_ns)
        known = StateRates.from_path(sim["path"], smoothed.shape[-1], sim["feedback"], self.t_ns)
        return StateRates.from_cuts(smoothed, sim["feedback"]), known

    def _comfort(self, sim: dict, n: int) -> dict:
        first = self.vote_times[:n]
        return ComfortCurve.fit(sim["temps"][first], sim["votes"][first])

    def rows(self) -> tuple[list[dict], list[dict]]:
        """Rows of the estimates (one per N, parameter and estimator) and of the tests (one per N)."""
        estimates = [row for i in range(len(self.grid)) for row in self._estimate_rows(i)]
        return estimates, [self._test_row(i) for i in range(len(self.grid))]

    def _base(self, i: int) -> dict:
        return {**self.base, "n_votes": int(self.grid[i]), "t_N": int(self.t_ns[i])}

    def _test_row(self, i: int) -> dict:
        return {**self._base(i), **{b: self.comfort[i][b] for b in BANDS}}

    def _estimate_rows(self, i: int) -> list[dict]:
        rows = [("hmm", f"p_{s}", x) for s, x in enumerate(self.hmm.p_hat[i])]
        rows += [("known", f"p_{s}", x) for s, x in enumerate(self.known.p_hat[i])]
        rows += [("logit", name, self.comfort[i][name]) for name in COMFORT]
        return [{**self._base(i), "parameter": p, "estimator": e, "estimate": x} for e, p, x in rows]


class Estimator:
    """Estimates every dataset on the grid of N."""
    def __init__(self, smoother: StateSmoother, grid: np.ndarray):
        self.smoother, self.grid = smoother, grid

    def analyse(self, sims: list[dict]) -> tuple[list[dict], list[dict]]:
        rows = [self._estimate(sim, k, len(sims)).rows() for k, sim in enumerate(sims)]
        return [r for est, _ in rows for r in est], [r for _, tests in rows for r in tests]

    def _estimate(self, sim: dict, k: int, R: int) -> DatasetEstimates:
        t0 = time.time()
        estimates = DatasetEstimates(sim, k, self.grid, self.smoother)
        print(f"Estimated dataset {k + 1}/{R} ({time.time() - t0:.1f} s)")
        return estimates


# --- Power --------------------------------------------------------------------------

class PowerAnalysis:
    """Per (band test, N): the share of the datasets that reject at the 5 % level, and the mean weeks to N."""
    def __init__(self, tests: pd.DataFrame):
        self.tests = tests
        self.table = self._table(BANDS + ["bands_all"])

    def _table(self, order: list[str]) -> pd.DataFrame:
        rows = [row for n, at_n in self.tests.groupby("n_votes") for row in self._rows_at(n, at_n)]
        rank = {t: i for i, t in enumerate(order)}
        return pd.DataFrame(rows).sort_values(["test", "n_votes"], key=lambda c: c.map(rank) if c.name == "test" else c)

    def _rows_at(self, n: int, at_n: pd.DataFrame) -> list[dict]:
        base = {"n_votes": n, "weeks": (at_n["t_N"].mean() + 1) / WEEK}
        return [{**base, "test": test, "power": x} for test, x in self._bands(at_n).items()]

    @staticmethod
    def _bands(at_n: pd.DataFrame) -> dict:
        reject = at_n[BANDS].to_numpy() > Z_CRIT  # a failed fit (NaN) does not reject
        return {**dict(zip(BANDS, reject.mean(axis=0))), "bands_all": reject.all(axis=1).mean()}

    def needed(self, n_room: int | None) -> pd.DataFrame:
        """Per test: the first N from which the power stays >= 80 %, the weeks to reach it and the power at the room's N."""
        return pd.DataFrame([self._needed_row(t, g, n_room) for t, g in self.table.groupby("test", sort=False)])

    @staticmethod
    def _needed_row(test: str, g: pd.DataFrame, n_room: int | None) -> dict:
        ok, ns = g["power"].to_numpy() >= POWER, g["n_votes"].to_numpy()
        i = next((i for i in range(len(ok)) if ok[i:].all()), None)
        room = np.interp(n_room, ns, g["power"], left=np.nan, right=np.nan) if n_room else np.nan
        return {"test": test, "label": TEST_LABELS[test], "n_80": None if i is None else int(ns[i]),
                "weeks_80": np.nan if i is None else g["weeks"].iloc[i], "power_room": room}


# --- Tables -------------------------------------------------------------------------

class ResultTables:
    """Per N: the comfort curve and P(feedback | state), each as mean [2.5 %, 97.5 %] over the datasets, with the
    mean weeks to reach N, and the SE, CI of the mean and quantile summary of every parameter."""
    def __init__(self, estimates: pd.DataFrame, p: np.ndarray, scale: float):
        self.estimates, self.p, self.scale = estimates, p, scale
        self.weeks = ((estimates.groupby("n_votes")["t_N"].mean() + 1) / WEEK).round(1)

    def comfort(self) -> pd.DataFrame:
        truth, table = true_values(self.p, self.scale), self._cells(COMFORT, "logit", 2)
        table.columns = [f"c{k + 1} [°C] (true {c:g})" for k, c in enumerate(CUTPOINTS)] + [
            f"s [°C] (true {self.scale:g})", f"comfort centre (true {truth['centre']:g})",
            f"comfort width (true {truth['width']:g})"]
        return self._with_weeks(table)

    def p_feedback(self) -> pd.DataFrame:
        table = self._cells([f"p_{s}" for s in range(len(self.p))], "hmm", 4)
        table.columns = [f"state {s} (true {x:.4f})" for s, x in enumerate(self.p)]
        return self._with_weeks(table)

    def _with_weeks(self, table: pd.DataFrame) -> pd.DataFrame:
        table.insert(0, "weeks", self.weeks)
        table.index.name = "votes N"
        return table

    def _cells(self, params: list[str], estimator: str, digits: int) -> pd.DataFrame:
        """(N, params) cells "mean [2.5 %, 97.5 %]" over the datasets: where one experiment's estimate lands."""
        q = self._quantiles(params, estimator)
        q["cell"] = [f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]" for m, lo, hi in zip(q["mean"], q["lo"], q["hi"])]
        return q.pivot(index="n_votes", columns="parameter", values="cell").reindex(columns=params).fillna("–")

    def _quantiles(self, params: list[str], estimator: str) -> pd.DataFrame:
        e = self.estimates[(self.estimates["estimator"] == estimator) & self.estimates["parameter"].isin(params)]
        return e.dropna(subset=["estimate"]).groupby(["n_votes", "parameter"])["estimate"].agg(
            mean="mean", lo=lambda x: x.quantile(0.025), hi=lambda x: x.quantile(0.975)).reset_index()

    def summary(self) -> pd.DataFrame:
        """Per (parameter, estimator, N): mean, sd, the SE and CI of the mean and the 2.5 / 97.5 % quantiles over the
        datasets."""
        groups = self.estimates.groupby(["parameter", "estimator", "n_votes"], sort=False)[["estimate", "true"]]
        return groups.apply(self._summary_row).reset_index()

    @classmethod
    def _summary_row(cls, group: pd.DataFrame) -> pd.Series:
        x = group["estimate"].dropna().to_numpy()
        nan = dict.fromkeys(["mean", "sd", "se", "ci_lo", "ci_hi", "q_lo", "q_hi"], np.nan)
        return pd.Series({"R": len(x), **(cls._intervals(x) if len(x) >= 2 else nan), "true": group["true"].iloc[0]})

    @staticmethod
    def _intervals(x: np.ndarray) -> dict:
        mean, sd, t = x.mean(), x.std(ddof=1), stats.t.ppf(0.975, df=len(x) - 1)
        se = sd / np.sqrt(len(x))
        q_lo, q_hi = np.quantile(x, [0.025, 0.975])
        return {"mean": mean, "sd": sd, "se": se, "ci_lo": mean - t * se, "ci_hi": mean + t * se, "q_lo": q_lo,
                "q_hi": q_hi}


class ThesisTables:
    """The headline tables as LaTeX in Thesis/Tables: votes needed for 80 % power, the comfort curve and p_hat by N."""
    def __init__(self, room: str, model: str, K: int, n_max: int, n_room: int | None):
        self.stem, self.n_max, self.n_room = f"{room}_{model}_K{K}", n_max, n_room
        self.who = f"{room.replace('_', ' ').title()}, {model_label(model)}, $K={K}$"

    def save(self, needed: pd.DataFrame, curve: pd.DataFrame, p_table: pd.DataFrame) -> list[str]:
        return [self._power(needed), self._by_n(curve, "feedback_comfort", "Ordered-logit comfort curve"),
                self._by_n(p_table, "feedback_p", "HMM-smoothed P(feedback | state)")]

    def _power(self, needed: pd.DataFrame) -> str:
        caption = (f"Votes needed before 80\\,\\% of simulated experiments find the band wider than 0 at the 5\\,\\% level, "
                   f"and the weeks of monitoring that takes. {self.who}.")
        name = f"feedback_power_{self.stem}"
        return save_thesis_table(self._power_rows(needed), name, caption, f"tab:{name}")

    def _power_rows(self, needed: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "test": needed["label"],
            "votes for 80% power": [f"{int(n)}" if pd.notna(n) else f"$>{self.n_max}$" for n in needed["n_80"]],
            "weeks": [f"{w:.0f}" if pd.notna(w) else "–" for w in needed["weeks_80"]],
            f"power at N = {self.n_room}": [f"{x:.2f}" if pd.notna(x) else "–" for x in needed["power_room"]],
        })

    def _by_n(self, table: pd.DataFrame, name: str, what: str) -> str:
        rows, name = table[table.index.isin(TABLE_N)].reset_index().astype(str), f"{name}_{self.stem}"
        caption = f"{what} by number of votes, mean [2.5\\,\\%, 97.5\\,\\%] over the simulated experiments. {self.who}."
        return save_thesis_table(rows, name, caption, f"tab:{name}", max_width=True)


# --- Plots --------------------------------------------------------------------------

class WeekPlot:
    """First week of the first three datasets (states coloured, votes marked by category) above the observed room week."""
    def __init__(self, room_data, cols, K: int):
        self.dt = pd.date_range(ROOM_WEEK, periods=WEEK, freq="30min")
        self.room = pd.Series(np.asarray(room_data.ys).ravel(), index=room_data.datetime).reindex(self.dt)
        self.work = self._working_hours(cols)
        self.K, self.colors, self.vcolors = K, plt.cm.tab10(np.arange(K)), vote_colors()

    @staticmethod
    def _working_hours(cols) -> np.ndarray:
        if "working_hours" not in cols:
            return np.zeros(WEEK, dtype=bool)
        return np.asarray(week_covariates(WEEK, cols)[:, list(cols).index("working_hours")]).astype(bool)

    def save(self, sims: list[dict], title: str, path: str):
        fig, axes = plt.subplots(len(sims[:3]) + 1, 1, figsize=(12, 10), sharex=True)
        for ax, sim in zip(axes, sims[:3]):
            self._simulated(ax, sim)
        self._observed(axes[-1])
        self._finish(fig, axes, title)
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _simulated(self, ax, sim: dict):
        y, s, v = sim["emissions"][:WEEK], sim["path"][:WEEK], sim["votes"][:WEEK]
        dt = self.dt[:len(y)]  # a run can reach n_max votes before a full week
        ax.plot(dt, y, color="0.6", lw=0.8)
        self._markers(ax, dt, y, s, v)
        ax.set_title(f"Simulated, seed {sim['seed']} ({int((v >= 0).sum())} votes this week, "
                     f"{len(sim['path'])} steps in total)", loc="left", fontsize=10)

    def _markers(self, ax, dt, y: np.ndarray, s: np.ndarray, v: np.ndarray):
        for k in range(self.K):
            ax.scatter(dt[s == k], y[s == k], color=self.colors[k], s=8, label=f"state {k}", zorder=3)
        for c in range(N_CATS):
            ax.scatter(dt[v == c], y[v == c], marker="X", color=self.vcolors[c], edgecolor="k", lw=0.5, s=60,
                       label=CAT_LABELS[c], zorder=4)

    def _observed(self, ax):
        ax.plot(self.dt, self.room, color="k", lw=1)
        ax.set_title("Observed room", loc="left", fontsize=10)

    def _finish(self, fig, axes, title: str):
        for ax in axes:
            ax.fill_between(self.dt, 0, 1, where=self.work, transform=ax.get_xaxis_transform(), color="0.92",
                            step="post", zorder=0)
            ax.set_ylabel("CO$_2$ [ppm]")
        axes[0].legend(ncol=self.K + N_CATS, loc="upper right", fontsize=7, frameon=False)
        self._date_axis(axes[-1])
        fig.suptitle(f"{title}; shaded = working hours")
        fig.tight_layout()

    @staticmethod
    def _date_axis(ax):
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))


# --- Study --------------------------------------------------------------------------

class SampleSizeStudy:
    """One run: simulate the datasets, estimate them on the grid of N and save the results."""
    def __init__(self, args: argparse.Namespace):
        self.args, self.room, self.K = args, room_name(args.room), args.states
        self.p, self.n_room = true_feedback(args)
        self.params, self.cols, self.room_data = self._load_model(args)
        self.out_dir, self.plot_dir = run_dirs(args)
        for d in (self.out_dir, self.plot_dir):
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def _load_model(args) -> tuple:
        params, cfg, room_data = load_saved(args.model, args.room, args.states, args.tag)
        if not hasattr(params.emission, "phi"):
            raise SystemExit(f"{args.model} has no AR emission; the simulator needs an AR(1) model")
        return params, cfg.covariate_cols, room_data

    def xs(self, T: int) -> jax.Array:
        return week_covariates(T, self.cols)

    @property
    def title(self) -> str:
        p = ", ".join(f"{x:.4f}" for x in self.p)
        return f"{self.args.model}, {self.room}, K={self.K}, p=({p}), s={self.args.scale:g}"

    @property
    def grid(self) -> np.ndarray:
        return np.unique(np.append(np.arange(self.args.n_step, self.args.n_max + 1, self.args.n_step), self.args.n_max))

    def run(self):
        print(f"True P(feedback | state) = {np.round(self.p, 5)}; the room has {self.n_room} feedbacks")
        sims = self.simulate()
        estimates, tests = self.estimate(sims)
        self.save(estimates, tests, PowerAnalysis(tests))

    def simulate(self) -> list[dict]:
        simulator = FeedbackSimulator(self.params, self.room_data.init_dist, self.xs, self.args.scale)
        drawer, a = DatasetDrawer(simulator, self.args.n_max, self.out_dir), self.args
        sims = drawer.draw(self.p, [a.seed + k for k in range(a.R)])
        self.share = self._state_share(sims)
        WeekPlot(self.room_data, self.cols, self.K).save(sims, self.title, os.path.join(self.plot_dir, "simulated_week.png"))
        return sims

    def _state_share(self, sims: list[dict]) -> np.ndarray:
        """The share of time the datasets spend in each state."""
        share = np.bincount(np.concatenate([s["path"] for s in sims]), minlength=self.K) / sum(len(s["path"]) for s in sims)
        print(f"State time share {np.round(share, 3)}")
        return share

    def estimate(self, sims: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
        estimator = Estimator(StateSmoother(self.params, self.room_data.init_dist, self.xs), self.grid)
        rows, tests = estimator.analyse(sims)
        estimates, truth = pd.DataFrame(rows), true_values(self.p, self.args.scale)
        estimates["true"] = estimates["parameter"].map(truth)
        return estimates, pd.DataFrame(tests)

    def save(self, estimates: pd.DataFrame, tests: pd.DataFrame, power: PowerAnalysis):
        tables, needed = ResultTables(estimates, self.p, self.args.scale), power.needed(self.n_room)
        curve, p_table = tables.comfort(), tables.p_feedback()
        self._csvs({"estimates": estimates, "tests": tests, "power": power.table, "needed_n": needed,
                    "summary": tables.summary()}, {"comfort_curve": curve, "p_feedback": p_table})
        self._run_json()
        tex = ThesisTables(self.room, self.args.model, self.K, self.args.n_max, self.n_room).save(needed, curve, p_table)
        self._report(needed, curve, p_table, tex)

    def _csvs(self, flat: dict[str, pd.DataFrame], indexed: dict[str, pd.DataFrame]):
        for name, df in flat.items():
            df.to_csv(os.path.join(self.out_dir, f"{name}.csv"), index=False)
        for name, df in indexed.items():
            df.to_csv(os.path.join(self.out_dir, f"{name}.csv"))

    def _run_json(self):
        a = self.args
        run = {"room": self.room, "model": a.model, "states": self.K, "p": self.p.tolist(), "share": self.share.tolist(), "n_room": self.n_room, "scale": a.scale, "cutpoints": CUTPOINTS.tolist(),
               "n_max": a.n_max, "R": a.R, "title": self.title}
        with open(os.path.join(self.out_dir, "run.json"), "w") as f:
            json.dump(run, f, indent=2)

    def _report(self, needed: pd.DataFrame, curve: pd.DataFrame, p_table: pd.DataFrame, tex: list[str]):
        print("\nVotes needed for 80 % power:\n" + needed.to_string(index=False))
        print("\nComfort curve, mean [2.5 %, 97.5 % over the datasets]:\n" + curve.to_string())
        print("\nP(feedback | state), smoothed HMM, mean [2.5 %, 97.5 % over the datasets]:\n" + p_table.to_string())
        print(f"\nSaved results to {self.out_dir}\nSaved thesis tables to {', '.join(tex)}\nPlot with "
              f"drivers.simulation.plot_sample_size --room {self.args.room} -K {self.K}")


# --- CLI ----------------------------------------------------------------------------

def add_run_args(parser: argparse.ArgumentParser):
    """The arguments that name a run; shared with plot_sample_size."""
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 15 for room_015")
    parser.add_argument("--states", "-K", type=int, default=4)
    parser.add_argument("--model", default="ar1_hmm_beta_1", help="slug of the saved AR(1) fit to simulate from")
    parser.add_argument("--tag", default=TAG, help=f"data tag (default {TAG})")
    parser.add_argument("--p", type=float, nargs="+", default=None,
                        help="true P(feedback | state), one value per state; defaults to the feedback_states estimate")
    parser.add_argument("--scale", type=float, default=0.55,
                        help="logistic scale s [°C] of the perceived-temperature noise (std = s·π/√3)")
    parser.add_argument("--R", type=int, default=200, help="number of simulated datasets")
    parser.add_argument("--seed", type=int, default=0, help="dataset k uses seed + k")


def run_dirs(args) -> tuple[str, str]:
    p = "pest" if args.p is None else "p" + "-".join(f"{x:g}" for x in args.p)
    run = os.path.join(room_name(args.room), args.tag, f"states_{args.states}", "sample_size",
                       f"{args.model}_{p}_scale{args.scale:g}_R{args.R}_seed{args.seed}")
    return os.path.join(RESULTS_DIR, run), os.path.join(PLOTS_DIR, run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    parser.add_argument("--n-max", type=int, default=300, help="simulate every dataset until this many votes")
    parser.add_argument("--n-step", type=int, default=5, help="estimate at N = n_step, 2·n_step, ..., n_max")
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if args.R < 2:
        parser.error("--R must be at least 2 to estimate the spread")
    if args.p is not None and (len(args.p) != args.states or not all(0 <= x <= 1 for x in args.p) or not any(args.p)):
        parser.error(f"--p needs {args.states} values in [0, 1], at least one positive")
    if args.scale <= 0:
        parser.error("--scale must be positive")


if __name__ == "__main__":
    SampleSizeStudy(parse_args()).run()
