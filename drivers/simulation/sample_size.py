"""Simulation-based sample size for occupant feedback: P(feedback | state) and the temperature cut-points of the votes.

R datasets are simulated from a saved AR(1)-HMM, each until N_MAX votes. Every step gets a temperature
T_t ~ U(15, 25), feedback f_t ~ Bernoulli(p[z_t]) and, if feedback is given, the vote is the bin of the perceived
temperature T_t + N(0, sigma²) on the cut-points 17 | 19 | 21 | 23:

    opt0 (15-17) | opt1 (17-19) | opt2 (19-21, fine) | opt3 (21-23) | opt4 (23-25)

On the grid N = n_step, 2·n_step, ..., N_MAX each dataset is cut at its N-th vote (t <= t_N) and

    p_hat_s = sum_t f_t γ_t(s) / sum_t γ_t(s)      γ_t(s) = P(z_t = s | y_1:t), filtered with the true model
    p_known_s                                      the same with γ the one-hot true path (states observed)
    c_hat, sigma_hat                               ordered-probit MLE on the (T_t, vote_t) pairs of the first N votes

Across the R datasets, at each N and for each parameter:

    95 % CI of E[estimate]       mean ± t_{R-1} · sd / sqrt(R)
    95 % prediction interval     mean ± t_{R-1} · sd · sqrt(1 + 1/R)   (where one experiment's estimate lands)

    uv run python -m drivers.simulation.sample_size --room 9 -K 5 --p 0 0 0 0.05 0.05
    uv run python -m drivers.simulation.sample_size --room 9 -K 5 --p 0 0 0 0.05 0.05 --R 4 --n-max 30 --n-step 10   # smoke run
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os
import time
from typing import Callable

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax.scipy.special import log_ndtr
from scipy import optimize, stats

from drivers.analysis.state_confusion import load_saved
from drivers.models.utils import PLOTS_DIR, RESULTS_DIR, TAG, covariates
from src.api.v5 import ForwardAlgorithm, Params

WEEK = 7 * 48
ROOM_WEEK = "2023-11-27"  # an ordinary Mon-Sun week in the training split, shown next to the simulated weeks
T_BUCKET = 2048           # filtered series are padded to a multiple of this, so only a few shapes get compiled
CHUNK = 4 * WEEK          # steps simulated between checks of the vote count
TEMP_RANGE = (15.0, 25.0)
CUTPOINTS = np.array([17.0, 19.0, 21.0, 23.0])
N_CATS = len(CUTPOINTS) + 1
CAT_LABELS = [f"opt{k}" for k in range(N_CATS)]
SIGMA_BOUNDS = (0.05, 10.0)  # °C, the range the probit noise is fitted in
BIG = 1e4                 # stands in for ±inf as the outer edges, so the probit gradients stay finite


def week_covariates(N: int, cols, start: str = "2024-03-04") -> jax.Array:
    """(N, C) covariates for the β models: one ordinary Mon-Sun week (30 min steps, no holidays) stacked until N rows."""
    week = covariates(pd.date_range(start, periods=WEEK, freq="30min"), cols)
    return jnp.tile(week, (N // len(week) + 1, 1))[:N]


# --- Simulation ---------------------------------------------------------------------

class FeedbackSimulator:
    """States and CO2 from a saved AR(1)-HMM, a uniform temperature per step and noisy categorical feedback.

    Follows the forward algorithm's convention: z_0 ~ u00 · Γ_0 and z_t ~ Γ_t[z_{t-1}]. The covariates repeat
    weekly, so one week of transition matrices is computed and reused.
    """
    def __init__(self, params: Params, init_dist, xs_fn: Callable[[int], jax.Array], p: np.ndarray, sigma: float):
        self.mu = np.asarray(params.emission.mu_vals(t=0, ys=0))
        self.std = np.asarray(params.emission.sigma(t=0, ys=0))
        self.phi = np.asarray(params.emission.phi())[0]
        self.u00 = np.asarray(init_dist).ravel()
        self.p, self.sigma = np.asarray(p), sigma
        gammas = np.asarray(params.transition_matrices(N=2 * WEEK, xs=xs_fn(2 * WEEK)))
        if not np.allclose(gammas[:WEEK], gammas[WEEK:]):
            raise ValueError("The transition matrices do not repeat weekly")
        self.cum_gammas = np.cumsum(gammas[:WEEK], axis=-1)
        self.cum_u0 = np.cumsum(self.u00 @ gammas[0])

    def simulate(self, seed: int, n_max: int, max_steps: int = 10_000_000) -> dict:
        """Simulate until n_max votes; every array ends at the n_max-th vote."""
        rng = np.random.default_rng(seed)
        states, ys, feedback, n_votes = [], [], [], 0
        s, y = None, None
        while n_votes < n_max:
            if len(states) * CHUNK >= max_steps:
                raise RuntimeError(f"Only {n_votes}/{n_max} votes after {max_steps} steps")
            z, u = rng.standard_normal(CHUNK), rng.random(CHUNK)
            chunk_s, chunk_y = np.empty(CHUNK, dtype=int), np.empty(CHUNK)
            t0 = len(states) * CHUNK
            for i in range(CHUNK):
                cum = self.cum_u0 if s is None else self.cum_gammas[(t0 + i) % WEEK, s]
                s = min(int(np.searchsorted(cum, u[i] * cum[-1])), len(cum) - 1)
                mean = self.mu[s] if y is None else self.mu[s] + self.phi[s] * (y - self.mu[s])
                y = mean + self.std[s] * z[i]
                chunk_s[i], chunk_y[i] = s, y
            states.append(chunk_s)
            ys.append(chunk_y)
            feedback.append(rng.random(CHUNK) < self.p[chunk_s])
            n_votes += int(feedback[-1].sum())

        feedback = np.concatenate(feedback)
        T = np.flatnonzero(feedback)[n_max - 1] + 1
        path, emissions, feedback = np.concatenate(states)[:T], np.concatenate(ys)[:T], feedback[:T]
        temps = rng.uniform(*TEMP_RANGE, size=T)
        perceived = temps + self.sigma * rng.standard_normal(T)
        votes = np.where(feedback, np.searchsorted(CUTPOINTS, perceived, side="right"), -1)
        return {"emissions": emissions, "path": path, "temps": temps, "feedback": feedback.astype(int), "votes": votes,
                "seed": seed}


def simulate(simulator: FeedbackSimulator, args, out_dir: str) -> list[dict]:
    """R datasets, each until n_max votes; every one is saved to datasets/ as soon as it is drawn."""
    os.makedirs(os.path.join(out_dir, "datasets"), exist_ok=True)
    sims = []
    for k in range(args.R):
        t0 = time.time()
        sim = simulator.simulate(args.seed + k, args.n_max)
        np.savez(os.path.join(out_dir, "datasets", f"sim_{k:03d}.npz"), **sim)
        sims.append(sim)
        print(f"Simulated dataset {k + 1}/{args.R}: {len(sim['path'])} steps ({time.time() - t0:.1f} s)")
    return sims


# --- P(feedback | state) ------------------------------------------------------------

def build_filter(params: Params, init_dist, xs_fn) -> Callable[[np.ndarray], np.ndarray]:
    """(T,) emissions -> (T, K) filtered state probabilities P(z_t | y_1:t) under the true model."""
    run = jax.jit(lambda ys, xs: ForwardAlgorithm.run(params=params, initial_dist=init_dist, ys=ys, xs=xs)[0][:, 0, :])

    def filtered(ys: np.ndarray) -> np.ndarray:
        T_pad = -(-len(ys) // T_BUCKET) * T_BUCKET  # filtering only looks back, so the padding leaves 0..T-1 untouched
        return np.asarray(run(jnp.pad(jnp.asarray(ys), (0, T_pad - len(ys)), mode="edge"), xs_fn(T_pad)))[:len(ys)]
    return filtered


def p_hats(weights: np.ndarray, feedback: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    """(len(t_ns), K) sum_{t<=t_N} f_t w_t(s) / sum_{t<=t_N} w_t(s); NaN for a state with no weight yet."""
    num, den = np.cumsum(feedback[:, None] * weights, axis=0)[t_ns], np.cumsum(weights, axis=0)[t_ns]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# --- Cut-points ---------------------------------------------------------------------

def unpack(theta) -> tuple[jax.Array, jax.Array]:
    """theta = (c_1, softplus^-1 of the cut-point gaps, log sigma) -> (ordered cut-points, sigma)."""
    c = theta[0] + jnp.concatenate([jnp.zeros(1), jnp.cumsum(jax.nn.softplus(theta[1:N_CATS - 1]))])
    return c, jnp.exp(theta[-1])


def pack(c: np.ndarray, sigma: float) -> np.ndarray:
    gaps = np.maximum(np.diff(c), 0.1)
    return np.concatenate([[c[0]], np.log(np.expm1(gaps)), [np.log(sigma)]])


@jax.jit
def probit_nll(theta, temps, votes, mask):
    """Ordered probit: P(vote = k | T) = Φ((c_{k+1} - T)/σ) - Φ((c_k - T)/σ), summed over the unmasked votes."""
    c, sigma = unpack(theta)
    edges = jnp.concatenate([jnp.array([-BIG]), c, jnp.array([BIG])])
    log_hi, log_lo = log_ndtr((edges[votes + 1] - temps) / sigma), log_ndtr((edges[votes] - temps) / sigma)
    log_p = log_hi + jnp.log1p(-jnp.exp(jnp.minimum(log_lo - log_hi, -1e-12)))
    return -jnp.sum(jnp.where(mask, log_p, 0.0))


probit_grad = jax.jit(jax.value_and_grad(probit_nll))


def initial_cutpoints(temps: np.ndarray, votes: np.ndarray) -> np.ndarray:
    """Midpoints between the mean temperatures of neighbouring vote categories."""
    means = np.array([temps[votes == k].mean() for k in range(N_CATS)])
    return (means[:-1] + means[1:]) / 2


def fit_probit(temps, votes, n: int, start: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    """(c_1, ..., c_4, sigma) from the first n votes and the optimiser's theta, for warm starts.

    NaN (and no theta) until every category has a vote, since a missing category leaves its cut-points unidentified.
    Every fit is also started from the data, and the better optimum kept, so one bad fit cannot carry over to later N.
    sigma is bounded to SIGMA_BOUNDS: with near-separable votes the likelihood keeps growing as sigma -> 0.
    """
    if len(np.unique(np.asarray(votes[:n]))) < N_CATS:
        return np.full(N_CATS, np.nan), None
    starts = [pack(initial_cutpoints(np.asarray(temps[:n]), np.asarray(votes[:n])), 1.0)] + ([start] if start is not None else [])
    mask = jnp.arange(len(votes)) < n
    fun = lambda theta: tuple(np.asarray(v, dtype=float) for v in probit_grad(theta, temps, votes, mask))
    bounds = [(None, None)] * (N_CATS - 1) + [tuple(np.log(SIGMA_BOUNDS))]
    res = min((optimize.minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds) for x0 in starts),
              key=lambda r: r.fun)
    c, sigma = unpack(jnp.asarray(res.x))
    return np.append(np.asarray(c), float(sigma)), res.x


# --- Estimation ---------------------------------------------------------------------

def estimate(sims: list[dict], filtered: Callable, grid: np.ndarray, p: np.ndarray, sigma: float) -> pd.DataFrame:
    """Long table: one row per (dataset, N, parameter, estimator)."""
    K = len(p)
    truth = {**{f"p_{s}": p[s] for s in range(K)}, **{f"c_{k + 1}": CUTPOINTS[k] for k in range(N_CATS - 1)},
             "sigma": sigma}
    rows = []
    for k, sim in enumerate(sims):
        t0 = time.time()
        vote_times = np.flatnonzero(sim["feedback"])
        t_ns = vote_times[grid - 1]
        hmm = p_hats(filtered(sim["emissions"]), sim["feedback"], t_ns)
        known = p_hats(np.eye(K)[sim["path"]], sim["feedback"], t_ns)

        temps, votes = jnp.asarray(sim["temps"][vote_times]), jnp.asarray(sim["votes"][vote_times])
        theta = None
        for i, (n, t_n) in enumerate(zip(grid, t_ns)):
            probit, theta = fit_probit(temps, votes, int(n), theta)
            base = {"dataset": k, "seed": sim["seed"], "n_votes": int(n), "t_N": int(t_n)}
            rows += [{**base, "parameter": f"p_{s}", "estimator": "hmm", "estimate": hmm[i, s]} for s in range(K)]
            rows += [{**base, "parameter": f"p_{s}", "estimator": "known", "estimate": known[i, s]} for s in range(K)]
            rows += [{**base, "parameter": name, "estimator": "probit", "estimate": probit[j]}
                     for j, name in enumerate([f"c_{j + 1}" for j in range(N_CATS - 1)] + ["sigma"])]
        print(f"Estimated dataset {k + 1}/{len(sims)} ({time.time() - t0:.1f} s)")
    df = pd.DataFrame(rows)
    df["true"] = df["parameter"].map(truth)
    return df


# --- Intervals ----------------------------------------------------------------------

def summarise(estimates: pd.DataFrame) -> pd.DataFrame:
    """Per (parameter, estimator, N): mean, sd, the CI of the mean and the prediction interval over the datasets."""
    def one(group: pd.DataFrame) -> pd.Series:
        x = group["estimate"].dropna().to_numpy()
        R, true = len(x), group["true"].iloc[0]
        if R < 2:
            return pd.Series({"R": R, "mean": np.nan, "sd": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                              "pi_lo": np.nan, "pi_hi": np.nan, "pi_width": np.nan, "covers": False, "true": true})
        mean, sd = x.mean(), x.std(ddof=1)
        t = stats.t.ppf(0.975, df=R - 1)
        ci, pi = t * sd / np.sqrt(R), t * sd * np.sqrt(1 + 1 / R)
        return pd.Series({"R": R, "mean": mean, "sd": sd, "ci_lo": mean - ci, "ci_hi": mean + ci,
                          "pi_lo": mean - pi, "pi_hi": mean + pi, "pi_width": 2 * pi,
                          "covers": bool(mean - ci <= true <= mean + ci), "true": true})
    keys = ["parameter", "estimator", "n_votes"]
    return estimates.groupby(keys, sort=False)[["estimate", "true"]].apply(one).reset_index()


def first_n_covering(summary: pd.DataFrame) -> int | None:
    """Smallest N after which the CI keeps containing the true value."""
    covers = summary["covers"].to_numpy()
    return next((int(n) for i, n in enumerate(summary["n_votes"]) if covers[i:].all()), None)


def first_n_narrow(summary: pd.DataFrame, width: float) -> int | None:
    narrow = summary[summary["pi_width"] <= width]
    return int(narrow["n_votes"].iloc[0]) if len(narrow) else None


def needed_n(summary: pd.DataFrame, target_width: dict[str, float]) -> pd.DataFrame:
    rows = []
    for (param, est), group in summary.groupby(["parameter", "estimator"], sort=False):
        width = target_width["p" if param.startswith("p_") else "temp"]
        rows.append({"parameter": param, "estimator": est, "n_ci_covers": first_n_covering(group),
                     "target_width": width, "n_pi_narrow": first_n_narrow(group, width)})
    return pd.DataFrame(rows)


def interval_cells(estimates: pd.DataFrame, params: list[str], estimator: str, digits: int) -> pd.DataFrame:
    """(N, params) cells "mean [2.5 %, 97.5 %]" over the datasets: where one experiment's estimate lands."""
    e = estimates[(estimates["estimator"] == estimator) & estimates["parameter"].isin(params)].dropna(subset=["estimate"])
    stats_ = e.groupby(["n_votes", "parameter"])["estimate"].agg(
        mean="mean", lo=lambda x: x.quantile(0.025), hi=lambda x: x.quantile(0.975)).reset_index()
    stats_["cell"] = [f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
                      for m, lo, hi in zip(stats_["mean"], stats_["lo"], stats_["hi"])]
    return stats_.pivot(index="n_votes", columns="parameter", values="cell").reindex(columns=params).fillna("–")


def result_tables(estimates: pd.DataFrame, p: np.ndarray, sigma: float):
    """Per N: the comfort curve (cut-points and sigma) and P(feedback | state), each as mean [2.5 %, 97.5 %] over the datasets."""
    cut_names = [f"c_{k + 1}" for k in range(N_CATS - 1)]
    curve = interval_cells(estimates, cut_names + ["sigma"], "probit", 2)
    curve.columns = [f"opt{k}|opt{k + 1} °C (true {c:g})" for k, c in enumerate(CUTPOINTS)] + [f"σ °C (true {sigma:g})"]
    steps = estimates.groupby("n_votes")["t_N"].mean() + 1
    curve.insert(0, "avg emissions", steps.round().astype(int))

    p_table = interval_cells(estimates, [f"p_{s}" for s in range(len(p))], "hmm", 4)
    p_table.columns = [f"state {s} (true {p[s]:g})" for s in range(len(p))]
    for table in (curve, p_table):
        table.index.name = "Votes N"
    return curve, p_table


# --- Plots --------------------------------------------------------------------------

def vote_colors():
    return plt.cm.coolwarm(np.linspace(0, 1, N_CATS))


def plot_week(sims: list[dict], room_data, cols, states: int, title: str, path: str):
    """First week of the first three datasets (states coloured, votes marked by category) above the observed room week."""
    dt = pd.date_range(ROOM_WEEK, periods=WEEK, freq="30min")
    work = np.asarray(week_covariates(WEEK, cols)[:, list(cols).index("working_hours")]).astype(bool) \
        if "working_hours" in cols else np.zeros(WEEK, dtype=bool)
    colors, vcolors = plt.cm.tab10(np.arange(states)), vote_colors()

    shown = sims[:3]
    fig, axes = plt.subplots(len(shown) + 1, 1, figsize=(12, 10), sharex=True)
    for ax, sim in zip(axes, shown):
        y, s, v = sim["emissions"][:WEEK], sim["path"][:WEEK], sim["votes"][:WEEK]
        dt_i = dt[:len(y)]  # a run can reach n_max votes before a full week
        ax.plot(dt_i, y, color="0.6", lw=0.8)
        for k in range(states):
            ax.scatter(dt_i[s == k], y[s == k], color=colors[k], s=8, label=f"state {k}", zorder=3)
        for c in range(N_CATS):
            ax.scatter(dt_i[v == c], y[v == c], marker="X", color=vcolors[c], edgecolor="k", lw=0.5, s=60,
                       label=CAT_LABELS[c], zorder=4)
        ax.set_title(f"Simulated, seed {sim['seed']} ({int((v >= 0).sum())} votes this week, "
                     f"{len(sim['path'])} steps in total)", loc="left", fontsize=10)

    room = pd.Series(np.asarray(room_data.ys).ravel(), index=room_data.datetime).reindex(dt)
    axes[-1].plot(dt, room, color="k", lw=1)
    axes[-1].set_title("Observed room", loc="left", fontsize=10)

    for ax in axes:
        ax.fill_between(dt, 0, 1, where=work, transform=ax.get_xaxis_transform(), color="0.92", step="post", zorder=0)
        ax.set_ylabel("CO$_2$ [ppm]")
    axes[0].legend(ncol=states + N_CATS, loc="upper right", fontsize=7, frameon=False)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    fig.suptitle(f"{title}; shaded = working hours")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bands(ax, estimates: pd.DataFrame, summary: pd.DataFrame, param: str, est: str, color: str, label: str,
               lines: bool = True):
    s = summary[(summary["parameter"] == param) & (summary["estimator"] == est)]
    if lines:
        wide = estimates[(estimates["parameter"] == param) & (estimates["estimator"] == est)] \
            .pivot(index="n_votes", columns="dataset", values="estimate")
        ax.plot(wide.index, wide.to_numpy(), color="0.85", lw=0.5, zorder=1)
    ax.fill_between(s["n_votes"], s["pi_lo"], s["pi_hi"], color=color, alpha=0.15, label=f"95% PI, {label}")
    ax.fill_between(s["n_votes"], s["ci_lo"], s["ci_hi"], color=color, alpha=0.4, label=f"95% CI of mean, {label}")
    ax.plot(s["n_votes"], s["mean"], color=color, lw=1.5, label=f"mean, {label}")


def plot_p_hat(estimates: pd.DataFrame, summary: pd.DataFrame, p: np.ndarray, title: str, path: str):
    """One panel per state: p_hat_s per dataset (HMM), its CI/PI, the known-states PI and the true p_s."""
    K = len(p)
    ncols = min(K, 3)
    nrows = -(-K // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), sharex=True, squeeze=False)
    for s, ax in enumerate(axes.flat):
        if s >= K:
            ax.set_visible(False)
            continue
        plot_bands(ax, estimates, summary, f"p_{s}", "hmm", "tab:blue", "HMM")
        known = summary[(summary["parameter"] == f"p_{s}") & (summary["estimator"] == "known")]
        ax.plot(known["n_votes"], known["pi_lo"], "k--", lw=1, label="95% PI, known states")
        ax.plot(known["n_votes"], known["pi_hi"], "k--", lw=1)
        ax.axhline(p[s], color="tab:red", lw=1, label=f"true p = {p[s]:g}")
        ax.set_ylim(0, max(3 * p.max(), 1e-3))
        ax.set_title(f"state {s}", loc="left", fontsize=10)
        ax.set_ylabel(r"$\hat{p}_s$")
    for ax in axes[-1]:
        ax.set_xlabel("N, number of votes")
    axes.flat[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(f"P(feedback | state): {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cutpoints(estimates: pd.DataFrame, summary: pd.DataFrame, sigma: float, title: str, path: str):
    """The four cut-points and sigma against N, with CI/PI and the true values."""
    names = [f"c_{k + 1}" for k in range(N_CATS - 1)] + ["sigma"]
    truth = list(CUTPOINTS) + [sigma]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), sharex=True)
    for ax, name, true in zip(axes, names, truth):
        plot_bands(ax, estimates, summary, name, "probit", "tab:blue", "probit")
        ax.axhline(true, color="tab:red", lw=1, label=f"true = {true:g}")
        ax.set_ylim(true - 2, true + 2)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel("N, number of votes")
    axes[0].set_ylabel("Temperature [°C]")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(f"Vote cut-points: {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def category_probs(c: np.ndarray, sigma: float, temps: np.ndarray) -> np.ndarray:
    """(len(temps), N_CATS) ordered-probit P(vote = k | T)."""
    cdf = stats.norm.cdf((np.concatenate([[-np.inf], c, [np.inf]])[None, :] - temps[:, None]) / sigma)
    return np.diff(cdf, axis=1)


def plot_feedback_by_temp(sim: dict, estimates: pd.DataFrame, sigma: float, title: str, path: str):
    """One dataset at N_MAX: its votes against temperature, and the fitted against the true category probabilities."""
    votes_at = sim["votes"] >= 0
    temps, votes = sim["temps"][votes_at], sim["votes"][votes_at]
    last = estimates[(estimates["seed"] == sim["seed"]) & (estimates["estimator"] == "probit")]
    last = last[last["n_votes"] == last["n_votes"].max()].set_index("parameter")["estimate"]
    grid = np.linspace(*TEMP_RANGE, 300)
    colors = vote_colors()

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(votes))
    top.scatter(temps, votes + jitter, c=colors[votes], s=12, edgecolor="k", lw=0.3)
    top.set_yticks(range(N_CATS), CAT_LABELS)
    top.set_title(f"Seed {sim['seed']}, {len(votes)} votes", loc="left", fontsize=10)
    true_p = category_probs(CUTPOINTS, sigma, grid)
    for k in range(N_CATS):
        bottom.plot(grid, true_p[:, k], color=colors[k], ls="--", lw=1)
    if last.notna().all():
        fit_p = category_probs(last[[f"c_{k + 1}" for k in range(N_CATS - 1)]].to_numpy(), last["sigma"], grid)
        for k in range(N_CATS):
            bottom.plot(grid, fit_p[:, k], color=colors[k], lw=1.5, label=CAT_LABELS[k])
    for ax in (top, bottom):
        for c in CUTPOINTS:
            ax.axvline(c, color="0.8", lw=0.8, zorder=0)
    bottom.set(xlabel="Temperature [°C]", ylabel="P(vote | temperature)")
    bottom.legend(frameon=False, fontsize=8, title="solid: fitted, dashed: true", title_fontsize=8, loc="center right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Main ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 9 for room_009")
    parser.add_argument("--states", "-K", type=int, default=5)
    parser.add_argument("--model", default="ar1_hmm_beta_1", help="slug of the saved AR(1) fit to simulate from")
    parser.add_argument("--tag", default=TAG, help=f"data tag (default {TAG})")
    parser.add_argument("--p", type=float, nargs="+", required=True,
                        help="true P(feedback | state), one value per state, e.g. --p 0 0 0 0.05 0.05")
    parser.add_argument("--sigma", type=float, default=1.0, help="std [°C] of the perceived-temperature noise")
    parser.add_argument("--R", type=int, default=100, help="number of simulated datasets")
    parser.add_argument("--n-max", type=int, default=150, help="simulate every dataset until this many votes")
    parser.add_argument("--n-step", type=int, default=5, help="estimate at N = n_step, 2·n_step, ..., n_max")
    parser.add_argument("--seed", type=int, default=0, help="dataset k is simulated with seed + k")
    parser.add_argument("--target-width-p", type=float, default=0.02, help="PI width wanted for p_s")
    parser.add_argument("--target-width-temp", type=float, default=1.0, help="PI width [°C] wanted for cut-points and sigma")
    args = parser.parse_args()
    if args.R < 2:
        parser.error("--R must be at least 2 to estimate the spread")
    if len(args.p) != args.states:
        parser.error(f"--p needs {args.states} values (one per state), got {len(args.p)}")
    if not all(0 <= x <= 1 for x in args.p) or not any(args.p):
        parser.error("--p must lie in [0, 1] and be positive in at least one state")
    if args.sigma <= 0:
        parser.error("--sigma must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    params, cfg, room_data = load_saved(args.model, args.room, args.states, args.tag)
    if not hasattr(params.emission, "phi"):
        raise SystemExit(f"{args.model} has no AR emission; the simulator needs an AR(1) model")
    cols, init_dist = cfg.covariate_cols, room_data.init_dist
    xs_fn = lambda T: week_covariates(T, cols)
    p = np.asarray(args.p)

    name = f"{args.model}_p{'-'.join(f'{x:g}' for x in p)}_sigma{args.sigma:g}_R{args.R}_seed{args.seed}"
    out_dir = os.path.join(cfg.run_dir(RESULTS_DIR), "sample_size", name)
    plot_dir = os.path.join(cfg.run_dir(PLOTS_DIR), "sample_size", name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    title = f"{args.model}, {cfg.room}, K={cfg.states}, p=({', '.join(f'{x:g}' for x in p)}), σ={args.sigma:g}"

    sims = simulate(FeedbackSimulator(params, init_dist, xs_fn, p, args.sigma), args, out_dir)
    plot_week(sims, room_data, cols, cfg.states, title, os.path.join(plot_dir, "simulated_week.png"))

    grid = np.unique(np.append(np.arange(args.n_step, args.n_max + 1, args.n_step), args.n_max))
    estimates = estimate(sims, build_filter(params, init_dist, xs_fn), grid, p, args.sigma)
    summary = summarise(estimates)
    needed = needed_n(summary, {"p": args.target_width_p, "temp": args.target_width_temp})
    estimates.to_csv(os.path.join(out_dir, "estimates.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    needed.to_csv(os.path.join(out_dir, "needed_n.csv"), index=False)

    plot_p_hat(estimates, summary, p, title, os.path.join(plot_dir, "p_hat_vs_N.png"))
    plot_cutpoints(estimates, summary, args.sigma, title, os.path.join(plot_dir, "cutpoints_vs_N.png"))
    plot_feedback_by_temp(sims[0], estimates, args.sigma, title, os.path.join(plot_dir, "feedback_by_temp.png"))

    curve, p_table = result_tables(estimates, p, args.sigma)
    curve.to_csv(os.path.join(out_dir, "comfort_curve.csv"))
    p_table.to_csv(os.path.join(out_dir, "p_feedback.csv"))
    print("\nComfort curve, mean [2.5 %, 97.5 % over the datasets]:\n" + curve.to_string())
    print("\nP(feedback | state), HMM, mean [2.5 %, 97.5 % over the datasets]:\n" + p_table.to_string())
    print(f"\nSaved results to {out_dir}\nSaved plots to {plot_dir}")
