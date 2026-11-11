"""Simulation-based sample size for the complaint probability p = P(complaint | occupied).

R datasets are simulated from a saved AR(1)-HMM fit, each until N_MAX complaints. On the grid N = n_step, 2·n_step,
..., N_MAX each dataset is refitted on its data up to the N-th complaint only (later steps are masked out of the loss),
warm-started from its own fit at the previous grid point, and

    p_hat = N / sum_{t <= t_N} P(O_t = 1 | y_1:t)

with the filtered probabilities of the refitted model. Across the R datasets, at each N:

    95 % CI of E[p_hat]          mean ± t_{R-1} · sd / sqrt(R)
    95 % prediction interval     mean ± t_{R-1} · sd · sqrt(1 + 1/R)   (where one experiment's p_hat lands)

The models, p_hat rows and summary are saved after every N, so a long run can be inspected while it goes.

    uv run python -m drivers.analysis.sample_size --room 9 -K 5
    uv run python -m drivers.analysis.sample_size --room 9 -K 5 --R 4 --n-max 6 --n-step 2 --n-iter 500   # smoke run
    uv run python -m drivers.analysis.sample_size --room 9 -K 5 --init-noise 0.1               # refits not started at the truth
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os
import pickle
import time
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import optax.tree_utils as otu
import pandas as pd
from scipy import stats

from drivers.analysis.state_confusion import load_saved
from drivers.models.utils import PLOTS_DIR, RESULTS_DIR, TAG, covariates
from drivers.utils import save_model
from src.api.v5 import ForwardAlgorithm, LBFGSSolver, Params
from src.api.v5.solvers.base_solver import BaseSolver
from src.api.v5.solvers.freezer import Freezer

WEEK = 7 * 48
ROOM_WEEK = "2023-11-27"  # an ordinary Mon-Sun week in the training split, shown next to the simulated weeks
T_BUCKET = 2048           # refit lengths are rounded up to a multiple of this, so only a few shapes get compiled


class DataGenerater:
    def __init__(self, feedback_prob: float, occupied_states: tuple[int, ...] | None = None):
        self.gammas = None
        self.mu_vals = None
        self.std_vals = None
        self.emission_list = []
        self.path = []
        self.key = None
        self.sub_key = None
        self.num_states = None
        self.phi_vals = None
        self.params = None
        self.xs_fn = None
        self.feedback_prob = feedback_prob
        self.occupied_states = occupied_states  # None -> the last (highest-mu) state
        self.feedback_list = []

    def set_new_keys(self) -> None:
        self.key, self.sub_key = jax.random.split(self.key) #type: ignore

    def generate_emissions(self, key: int, params: Params, u00: jax.Array, N: int,
                           xs_fn: Callable[[int], jax.Array] | None = None, max_steps: int = 1_000_000):
        """Simulate until N feedback complaints have been drawn. Returns (emissions, path, feedbacks), all of equal length."""
        self.reset()
        self.key = jax.random.PRNGKey(key)
        self.params, self.xs_fn = params, xs_fn
        self.set_tranisition_matrix(7 * 48)
        self.set_std_vals(params)
        self.set_mu_vals(params)
        self.set_phi_vals(params)

        s0 = self.chose_state_0(u00.flatten())
        st = s0
        self.add_state_to_path(s0)

        y0 = self.get_first_emission(s0)
        self.add_emission(y0)
        n_feedback = self.add_feedback(st)

        i = 0
        while n_feedback < N:
            if i >= max_steps:
                raise RuntimeError(f"Only {n_feedback}/{N} complaints after {max_steps} steps")
            st = self.get_next_state(i, st)
            yt = self.draw_emission(st, i)
            self.add_state_to_path(st)
            self.add_emission(yt)
            n_feedback += self.add_feedback(st)
            i += 1

        emissions = jnp.asarray(self.emission_list)
        paths = jnp.asarray(self.path)
        feedbacks = jnp.asarray(self.feedback_list)
        self.reset()
        return emissions, paths, feedbacks

    def reset(self) -> None:
        self.emission_list = []
        self.path = []
        self.feedback_list = []

    def set_tranisition_matrix(self, T: int) -> None:
        xs = None if self.xs_fn is None else self.xs_fn(T)
        self.gammas = self.params.transition.transition_matrices(N = T, xs=xs) #type: ignore

    def set_std_vals(self, params: Params) -> None:
        self.std_vals = params.emission.sigma(t=0, ys=0)
    def set_mu_vals(self, params: Params) -> None:
        self.mu_vals = params.emission.mu_vals(t=0, ys=0)
    def set_phi_vals(self, params: Params) -> None:
        self.phi_vals = params.emission.phi()

    def chose_state_0(self, u00: jax.Array) -> jax.Array:
        self.num_states = len(u00)
        if self.occupied_states is None:
            self.occupied_states = (self.num_states - 1,)
        self.set_new_keys()
        s0 = jax.random.choice(self.sub_key, self.num_states, p=u00) #type: ignore
        return s0

    def add_state_to_path(self, st: jax.Array) -> None:
        self.path.append(st)

    def add_feedback(self, st: jax.Array) -> int:
        f = self.draw_feedback(st)
        self.feedback_list.append(f)
        return f

    def draw_feedback(self, st: jax.Array) -> int:
        if int(st) not in self.occupied_states: #type: ignore
            return 0
        self.set_new_keys()
        return int(jax.random.bernoulli(self.sub_key, self.feedback_prob)) #type: ignore

    def get_first_emission(self, s0: jax.Array) -> jax.Array:
        s0 = jnp.int16(s0)
        z = self.draw_standard_normal()

        y0 = self.mu_vals[s0] + z*self.std_vals[s0]
        return y0

    def draw_standard_normal(self) -> jax.Array:
        self.set_new_keys()
        z = jax.random.normal(self.sub_key, shape=(1,)) #type: ignore
        return z

    def add_emission(self, yt: jax.Array) -> None:
        self.emission_list.append(yt)

    def get_next_state(self, i, st):
        if i >= len(self.gammas): #type: ignore
            self.set_tranisition_matrix(2 * len(self.gammas)) #type: ignore
        gamma = self.gammas[i, : , :]
        ut = gamma[st, :]
        self.set_new_keys()
        st = jax.random.choice(self.sub_key, self.num_states, p=ut) #type: ignore
        return st

    def draw_emission(self, st, i) -> jax.Array:
        z = self.draw_standard_normal()
        y_pre = self.emission_list[i]
        ar = jnp.sum(self.phi_vals * (y_pre - self.mu_vals[None, :]), axis=0)

        mu = self.mu_vals + ar

        return mu[st] + self.std_vals[st]*z


def week_covariates(N: int, cols, start: str = "2024-03-04") -> jax.Array:
    """(N, C) covariates for the β models: one ordinary Mon-Sun week (30 min steps, no holidays) stacked until N rows."""
    week = covariates(pd.date_range(start, periods=WEEK, freq="30min"), cols)
    return jnp.tile(week, (N // len(week) + 1, 1))[:N]


# --- Simulation ---------------------------------------------------------------------

def simulate(params: Params, init_dist, xs_fn, args, out_dir: str) -> tuple[list[dict], tuple[int, ...]]:
    """R datasets, each until n_max complaints; every one is saved to datasets/ as soon as it is drawn."""
    os.makedirs(os.path.join(out_dir, "datasets"), exist_ok=True)
    sims, occupied = [], None
    for k in range(args.R):
        t0, seed = time.time(), args.seed + k
        generator = DataGenerater(args.p)
        emissions, path, feedbacks = generator.generate_emissions(key=seed, params=params, u00=init_dist,
                                                                  N=args.n_max, xs_fn=xs_fn)
        occupied = generator.occupied_states
        sim = {"emissions": np.asarray(emissions).ravel(), "path": np.asarray(path), "feedbacks": np.asarray(feedbacks),
               "seed": seed}
        np.savez(os.path.join(out_dir, "datasets", f"sim_{k:03d}.npz"), **sim)
        sims.append(sim)
        print(f"Simulated dataset {k + 1}/{args.R}: {len(sim['path'])} steps ({time.time() - t0:.0f} s)")
    return sims, occupied


def stack(sims: list[dict], n_max: int) -> tuple[jnp.ndarray, np.ndarray]:
    """(R, T_pad) edge-padded series, T_pad a multiple of T_BUCKET, and the (R, n_max) complaint time steps."""
    T_pad = -(-max(len(s["emissions"]) for s in sims) // T_BUCKET) * T_BUCKET
    ys = jnp.stack([jnp.pad(jnp.asarray(s["emissions"]), (0, T_pad - len(s["emissions"])), mode="edge") for s in sims])
    complaint_times = np.stack([np.flatnonzero(s["feedbacks"])[:n_max] for s in sims])
    return ys, complaint_times


# --- Refitting ----------------------------------------------------------------------

def masked_loss(init_dist, beta_lam: float) -> Callable:
    """Driver loss (NLL + β-ridge, as `penalised_negative_likelihood`) with the masked steps left out of the NLL."""
    def loss_fn(params: Params, data) -> jax.Array:
        ys, xs, mask = data
        _, f_t, _ = ForwardAlgorithm.run(params=params, initial_dist=init_dist, ys=ys, xs=xs)
        loss = -jnp.sum(jnp.where(mask, jnp.log(f_t), 0.0))
        if beta_lam:
            loss = loss + beta_lam * jnp.sum(params.transition.beta ** 2)
        return loss
    return loss_fn


def perturbed_starts(trainable, seeds: list[int], noise: float):
    """(R, ...) start points: the saved fit plus noise · N(0, 1) on every float leaf, one key per dataset."""
    def perturb(key):
        leaves, treedef = jax.tree_util.tree_flatten(trainable)
        keys = jax.random.split(key, len(leaves))
        return jax.tree_util.tree_unflatten(treedef, [
            leaf + noise * jax.random.normal(k, leaf.shape) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf
            for leaf, k in zip(leaves, keys)])
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(s), 1) for s in seeds])
    return jax.vmap(perturb)(keys)


def warm_starts(fitted, initial):
    """Each dataset's next start is its last fit, or its initial start if that fit went non-finite."""
    leaves = jax.tree_util.tree_leaves(fitted)
    finite = np.all([np.isfinite(leaf.reshape(len(leaf), -1)).all(axis=1) for leaf in leaves], axis=0)
    pick = lambda f, s: np.where(finite.reshape(-1, *[1] * (f.ndim - 1)), f, s)
    return jax.tree_util.tree_map(pick, fitted, initial), finite


def build_refit(params: Params, cfg, init_dist, occupied, n_iter: int, tol: float) -> tuple[Callable, Callable, Params]:
    """Jitted, vmapped refit of a batch of datasets, a function rebuilding Params from one refitted slice, and the start point.

    The objective is built once: `LBFGSSolver._run` takes it as a static argument, so a new closure would recompile.
    """
    solver = LBFGSSolver(n_iter=n_iter, tol=tol)
    freezer = Freezer(None)
    trainable, static = eqx.partition(params, freezer.filter_spec(params))
    loss_fn = masked_loss(init_dist, cfg.loss.beta_lam)
    objective = BaseSolver._objective(loss_fn, static, freezer, params)
    occupied = jnp.asarray(occupied)

    def refit_one(start, ys, xs, t_n, n):
        mask = jnp.arange(len(ys)) <= t_n  # only the data up to the n-th complaint
        fitted, state = solver._run(objective, start, (ys, xs, mask))
        utt, _, _ = ForwardAlgorithm.run(params=eqx.combine(fitted, static), initial_dist=init_dist, ys=ys, xs=xs)
        expected_occupied = jnp.cumsum(utt[:, 0, occupied].sum(axis=-1))[t_n]  # sum_{s<=t_n} P(O_s = 1 | y_1:s)
        grad_norm = otu.tree_norm(otu.tree_get(state, "grad"))
        converged = (grad_norm < tol) & LBFGSSolver._finite(fitted)
        return fitted, n / expected_occupied, expected_occupied, grad_norm, converged

    refit_batch = jax.jit(jax.vmap(refit_one, in_axes=(0, 0, None, 0, None)))
    to_params = lambda fitted: freezer.pin(eqx.combine(fitted, static), params)
    return refit_batch, to_params, trainable


def refit_at(n: int, refit_batch, starts, ys, xs, complaint_times, batch: int):
    """Refit every dataset on its data up to its n-th complaint, `batch` datasets at a time."""
    R = len(ys)
    outs = []
    for lo in range(0, R, batch):
        idx = np.arange(lo, lo + batch).clip(max=R - 1)  # the last batch is padded with its last dataset
        t_n = complaint_times[idx, n - 1]
        T = min(-(-(int(t_n.max()) + 1) // T_BUCKET) * T_BUCKET, ys.shape[1])
        start = jax.tree_util.tree_map(lambda leaf: leaf[idx], starts)
        out = refit_batch(start, ys[idx, :T], xs[:T], jnp.asarray(t_n), n)
        keep = slice(0, min(batch, R - lo))
        outs.append(jax.tree_util.tree_map(lambda leaf: np.asarray(leaf)[keep], out))
    return jax.tree_util.tree_map(lambda *leaves: np.concatenate(leaves), *outs)


# --- Intervals ----------------------------------------------------------------------

def summarise(p_hats: np.ndarray, Ns: np.ndarray, p: float) -> pd.DataFrame:
    """Per N (columns of p_hats, (R, len(Ns))): mean, sd, the CI of the mean and the prediction interval."""
    R = p_hats.shape[0]
    mean, sd = p_hats.mean(axis=0), p_hats.std(axis=0, ddof=1)
    t = stats.t.ppf(0.975, df=R - 1)
    ci, pi = t * sd / np.sqrt(R), t * sd * np.sqrt(1 + 1 / R)
    known = 1.96 * p * np.sqrt((1 - p) / Ns)  # prediction interval if the states were known
    return pd.DataFrame({
        "n_complaints": Ns, "mean": mean, "sd": sd, "t": t,
        "ci_lo": mean - ci, "ci_hi": mean + ci, "pi_lo": mean - pi, "pi_hi": mean + pi, "pi_width": 2 * pi,
        "known_lo": p - known, "known_hi": p + known,
        "covers": (mean - ci <= p) & (p <= mean + ci),
    })


def first_n_covering(summary: pd.DataFrame) -> int | None:
    """Smallest N after which the CI keeps containing p."""
    covers = summary["covers"].to_numpy()
    return next((int(n) for i, n in enumerate(summary["n_complaints"]) if covers[i:].all()), None)


def first_n_narrow(summary: pd.DataFrame, width: float) -> int | None:
    narrow = summary[summary["pi_width"] <= width]
    return int(narrow["n_complaints"].iloc[0]) if len(narrow) else None


# --- Plots --------------------------------------------------------------------------

def plot_series(sims: list[dict], title: str, path: str):
    fig, axes = plt.subplots(min(3, len(sims)), 1, figsize=(12, 8), squeeze=False)
    for ax, sim in zip(axes[:, 0], sims):
        ax.plot(sim["emissions"], lw=0.6)
        ax.set_title(f"Simulated, seed {sim['seed']} ({len(sim['path'])} steps)", loc="left", fontsize=10)
        ax.set_ylabel("CO$_2$ [ppm]")
    axes[-1, 0].set_xlabel("Time step (30 min)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_week(sims: list[dict], room_data, cols, states: int, title: str, path: str):
    """First week of the first three datasets (states coloured, complaints marked) above the observed room week."""
    dt = pd.date_range(ROOM_WEEK, periods=WEEK, freq="30min")
    work = np.asarray(week_covariates(WEEK, cols)[:, list(cols).index("working_hours")]).astype(bool) \
        if "working_hours" in cols else np.zeros(WEEK, dtype=bool)
    colors = plt.cm.tab10(np.arange(states))

    shown = sims[:3]
    fig, axes = plt.subplots(len(shown) + 1, 1, figsize=(12, 10), sharex=True)
    for ax, sim in zip(axes, shown):
        y, s, f = sim["emissions"][:WEEK], sim["path"][:WEEK], sim["feedbacks"][:WEEK].astype(bool)
        dt_i = dt[:len(y)]  # a run can reach n_max complaints before a full week
        ax.plot(dt_i, y, color="0.6", lw=0.8)
        for k in range(states):
            ax.scatter(dt_i[s == k], y[s == k], color=colors[k], s=8, label=f"state {k}", zorder=3)
        ax.scatter(dt_i[f], y[f], marker="x", color="k", s=30, label="complaint", zorder=4)
        ax.set_title(f"Simulated, seed {sim['seed']} ({int(f.sum())} complaints this week, "
                     f"{len(sim['path'])} steps in total)", loc="left", fontsize=10)

    room = pd.Series(np.asarray(room_data.ys).ravel(), index=room_data.datetime).reindex(dt)
    axes[-1].plot(dt, room, color="k", lw=1)
    axes[-1].set_title("Observed room", loc="left", fontsize=10)

    for ax in axes:
        ax.fill_between(dt, 0, 1, where=work, transform=ax.get_xaxis_transform(), color="0.92", step="post", zorder=0)
        ax.set_ylabel("CO$_2$ [ppm]")
    axes[0].legend(ncol=states + 1, loc="upper right", fontsize=8, frameon=False)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    fig.suptitle(f"{title}; shaded = working hours")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_p_hat(p_hats: np.ndarray, summary: pd.DataFrame, p: float, target_width: float, title: str, path: str):
    """Top: p_hat per dataset with the CI and prediction interval. Bottom: mean - p with the CI, green where it covers p."""
    R = p_hats.shape[0]
    Ns = summary["n_complaints"].to_numpy()
    mean, covers = summary["mean"].to_numpy(), summary["covers"].to_numpy()
    n_cover, n_narrow = first_n_covering(summary), first_n_narrow(summary, target_width)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    top.plot(Ns, p_hats.T, color="0.8", lw=0.5, marker=".", ms=2)
    top.fill_between(Ns, summary["pi_lo"], summary["pi_hi"], color="tab:blue", alpha=0.15,
                     label="95% prediction interval (one experiment)")
    top.fill_between(Ns, summary["ci_lo"], summary["ci_hi"], color="tab:blue", alpha=0.45,
                     label=f"95% CI of the mean ({R} simulations)")
    top.plot(Ns, mean, color="tab:blue", lw=1.5, label=r"mean $\hat{p}$")
    top.plot(Ns, summary["known_lo"], "k--", lw=1, label="prediction interval, known states")
    top.plot(Ns, summary["known_hi"], "k--", lw=1)
    top.axhline(p, color="tab:red", lw=1, label=f"true p = {p:g}")
    if n_narrow is not None:
        top.axvline(n_narrow, color="tab:purple", lw=1, label=f"prediction interval width ≤ {target_width:g} at N = {n_narrow}")
    top.set_ylabel(r"$\hat{p}$")
    top.set_ylim(0, 3 * p)
    top.legend(frameon=False, fontsize=8, loc="upper right")
    top.set_title(title, loc="left", fontsize=10)

    half = (summary["ci_hi"] - summary["ci_lo"]).to_numpy() / 2
    for mask, color, label in ((covers, "tab:green", "CI contains p"), (~covers, "tab:red", "CI misses p")):
        bottom.errorbar(Ns[mask], mean[mask] - p, yerr=half[mask], fmt="o", ms=3, lw=0.8, color=color, label=label)
    bottom.axhline(0, color="k", lw=0.8)
    for ax in (top, bottom):
        if n_cover is not None:
            ax.axvline(n_cover, color="tab:green", lw=1.2, ls=":")
    bottom.set_ylabel(r"mean $\hat{p}$ − p")
    bottom.set_xlabel("N, number of complaints the model was trained on")
    bottom.set_title(f"CI contains p for every N ≥ {n_cover}" if n_cover is not None else "CI does not yet keep containing p",
                     loc="left", fontsize=10)
    bottom.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Main ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 9 for room_009")
    parser.add_argument("--states", "-K", type=int, default=5)
    parser.add_argument("--model", default="ar1_hmm_beta_1", help="slug of the saved AR(1) fit to simulate from and refit")
    parser.add_argument("--tag", default=TAG, help=f"data tag (default {TAG})")
    parser.add_argument("--p", type=float, default=0.05, help="true P(complaint | occupied state)")
    parser.add_argument("--R", type=int, default=100, help="number of simulated datasets")
    parser.add_argument("--n-max", type=int, default=150, help="simulate every dataset until this many complaints")
    parser.add_argument("--n-step", type=int, default=5, help="refit at N = n_step, 2·n_step, ..., n_max")
    parser.add_argument("--batch", type=int, help="datasets refitted at once (default R); lower it to save memory")
    parser.add_argument("--n-iter", type=int, default=5_000, help="L-BFGS step cap per refit")
    parser.add_argument("--tol", type=float, default=1e-3, help="L-BFGS stops once the gradient norm is below this")
    parser.add_argument("--init-noise", type=float, default=0.0,
                        help="std of the Gaussian noise added to the saved fit before each refit (default 0: start at the truth)")
    parser.add_argument("--seed", type=int, default=0, help="dataset k is simulated with seed + k")
    parser.add_argument("--target-width", type=float, default=0.02, help="full width wanted for the prediction interval")
    args = parser.parse_args()
    if args.R < 2:
        parser.error("--R must be at least 2 to estimate the spread of p_hat")
    return args


if __name__ == "__main__":
    args = parse_args()
    params, cfg, room_data = load_saved(args.model, args.room, args.states, args.tag)
    if not hasattr(params.emission, "phi"):
        raise SystemExit(f"{args.model} has no AR emission; the generator needs an AR(1) model")
    cols, init_dist = cfg.covariate_cols, room_data.init_dist
    xs_fn = lambda T: week_covariates(T, cols)

    name = f"{args.model}_p{args.p:g}_R{args.R}_seed{args.seed}" + (f"_noise{args.init_noise:g}" if args.init_noise else "")
    out_dir = os.path.join(cfg.run_dir(RESULTS_DIR), "sample_size", name)
    plot_dir = os.path.join(cfg.run_dir(PLOTS_DIR), "sample_size", name)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    csv_path, summary_path = os.path.join(out_dir, "p_hat.csv"), os.path.join(out_dir, "summary.csv")
    for stale in (csv_path, summary_path):
        if os.path.exists(stale):
            os.remove(stale)
    title = f"{args.model}, {cfg.room}, K={cfg.states}, loss={cfg.loss.slug}, p={args.p:g}"

    sims, occupied = simulate(params, init_dist, xs_fn, args, out_dir)
    plot_series(sims, title, os.path.join(plot_dir, "simulated_series.png"))
    plot_week(sims, room_data, cols, cfg.states, title, os.path.join(plot_dir, "simulated_week.png"))

    ys, complaint_times = stack(sims, args.n_max)
    xs = xs_fn(ys.shape[1])
    seeds = [s["seed"] for s in sims]
    refit_batch, to_params, trainable = build_refit(params, cfg, init_dist, occupied, args.n_iter, args.tol)
    initial = perturbed_starts(trainable, seeds, args.init_noise)
    starts = initial
    grid = np.unique(np.append(np.arange(args.n_step, args.n_max + 1, args.n_step), args.n_max))
    print(f"Occupied state(s): {occupied}; padded length {ys.shape[1]}; refitting at N = {grid.tolist()}")

    p_hats = np.full((args.R, len(grid)), np.nan)
    for i, n in enumerate(grid):
        t0 = time.time()
        fitted, p_hat, expected, grad_norm, converged = refit_at(n, refit_batch, starts, ys, xs, complaint_times,
                                                                 args.batch or args.R)
        p_hats[:, i] = p_hat
        starts, finite = warm_starts(fitted, initial)

        save_model({"n_complaints": n, "seeds": seeds, "converged": converged, "grad_norm": grad_norm,
                    "params": [to_params(jax.tree_util.tree_map(lambda leaf: leaf[k], fitted)) for k in range(args.R)]},
                   os.path.join(out_dir, "models", f"N{n:03d}.pkl"))
        t_n = complaint_times[:, n - 1]
        pd.DataFrame({
            "dataset": np.arange(args.R), "seed": seeds, "n_complaints": n, "t_n": t_n, "steps_trained": t_n + 1,
            "expected_occupied": expected, "p_hat": p_hat, "grad_norm": grad_norm, "converged": converged,
            "true_p": args.p,
        }).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

        summary = summarise(p_hats[:, :i + 1], grid[:i + 1], args.p)
        summary.to_csv(summary_path, index=False)
        plot_p_hat(p_hats[:, :i + 1], summary, args.p, args.target_width, title, os.path.join(plot_dir, "p_hat_vs_N.png"))
        row = summary.iloc[-1]
        print(f"N = {n:3d}: mean {row['mean']:.4f}, CI [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}], "
              f"PI [{row['pi_lo']:.4f}, {row['pi_hi']:.4f}], covers p: {bool(row['covers'])}, "
              f"converged {int(converged.sum())}/{args.R}, non-finite (restarted) {int((~finite).sum())} "
              f"({time.time() - t0:.0f} s)")

    print(f"CI contains p for every N ≥ {first_n_covering(summary)}; "
          f"prediction interval width ≤ {args.target_width:g} from N = {first_n_narrow(summary, args.target_width)}")
    print(f"Saved results to {out_dir}\nSaved plots to {plot_dir}")
