# Simulation drivers

All of them read a saved `random_search` fit from `results/{room}/{tag}/states_{K}/`. The tag defaults to `calibrated_anchor`.

## Sampled state paths

```bash
uv run python -m drivers.simulation.sample_paths ar1_hmm_beta_1 --room 15 -K 4      # 100,000 paths
uv run python -m drivers.simulation.state_analysis ar1_hmm_beta_1 --room 15 -K 4
```

- `sample_paths` runs the forward filter over the whole observed series (train + val + test) and draws state paths from P(z_1:T | y_1:T) with `BackwardSampler`. It saves the paths, CO2, timestamps and off-day flag to `results/.../states_{K}/simulation/{model}_paths.npz` (about 37 MB for 100,000 paths).
- `state_analysis` reads that file. For each state it computes the mean observed CO2, the mean drift y_t - y_{t-1} and the share of time spent in the state, per path (mean and 2.5-97.5 % range across paths) → `{model}_state_summary.csv`. The path means, with the 25 % and 75 % quantiles of the drifts of all paths pooled into one series, also go to `Thesis/Tables/state_summary_<room>_<tag>_<model>_K<K>.tex`. It also plots the mean state probability over the day, for working days (working hours shaded) and for off days → `plots/.../states_{K}/simulation/`.

## Feedback by state

```bash
uv run python -m drivers.simulation.feedback_states ar1_hmm_beta_1 --room 15 -K 4
```

`feedback_states` reads the same paths file and bins the room's raw occupant feedback onto the series. For each state and path it counts the feedback events in the state and P(feedback | state), the share of the path's steps in the state that received feedback. The mean and 2.5-97.5 % range across paths go to `{model}_feedback_given_state.csv`, and a bar plot (labelled with the feedback count per state) goes to `plots/.../states_{K}/simulation/{model}_feedback_given_state.png`.

## Feedback sample size

```bash
uv run python -m drivers.simulation.sample_size --room 15 -K 4        # needs feedback_states to have been run
uv run python -m drivers.simulation.plot_sample_size --room 15 -K 4
```

`sample_size` answers: how many occupant votes are needed before (a) P(feedback | state) and the comfort-curve cut-points are estimated precisely and (b) the thermal-comfort vote bands can be told apart? It simulates `--R` experiments (default 200) from a saved AR(1) fit and estimates them at a growing number of votes N. Results go to `results/.../states_{K}/sample_size/<run>/`, plots to `plots/.../sample_size/<run>/`, and the headline tables to `Thesis/Tables/feedback_{power,comfort,p}_<room>_<model>_K<K>.tex`.

### How a dataset is simulated

Each dataset is drawn step by step until it holds `--n-max` votes (default 300):

1. **State and CO2.** The state follows the fitted chain, z_0 ~ u_0 · Γ_0 and z_t ~ Γ_t[z_{t-1}], and CO2 follows the AR(1) emission of that state:

       y_t = μ_s + φ_s (y_{t-1} − μ_s) + σ_s ε_t,    ε_t ~ N(0, 1)

2. **Feedback or not.** f_t ~ Bernoulli(p[z_t]). By default p is the P(feedback | state) that `feedback_states` estimated for the room (`p_feedback_mean`); `--p` overrides it. The temperature has no effect on whether feedback is given.
3. **Temperature.** T_t ~ U(15, 25) °C, independent of the state and the CO2.
4. **The vote.** When f_t = 1, the occupant perceives T_t + s · Logistic(0, 1) (`--scale`, default s = 0.55 °C, so the std is ≈ 1 °C) and votes the bin it falls in:

       opt0 (< 17) | opt1 (17-19) | opt2 (19-21, fine) | opt3 (21-23) | opt4 (> 23)

   This is an ordered logit with cut-points c = (17, 19, 21, 23):

       P(vote ≤ k | T) = F((c_{k+1} − T) / s),    F(x) = 1 / (1 + e^{−x})

### What is estimated

Each dataset is cut at its N-th vote for N = n_step, 2·n_step, ..., n_max (`--n-step`, default 5).

- **p̂_s.** The state is not observed, so each step's feedback is weighted by the smoothed state probability γ_t(s) = P(z_t = s | y_1:t_N) under the true model. It uses only the CO2 seen up to the N-th vote, as an experiment stopped there would. With W_s = Σ_t γ_t(s):

      p̂_s = Σ_t f_t γ_t(s) / W_s

  The same formula with γ_t the one-hot true state gives the known-states best case.
- **Comfort curve.** An ordered logit from `statsmodels` (`OrderedModel`, `distr="logit"`) on the (T, vote) pairs of the first N votes. Its parameters (β, τ_1, log gaps) are turned into cut-points c_k = 20 + τ_k / β [°C] and a scale s = 1/β. The comfort band is opt2: its centre is (c_2 + c_3)/2 and its width is c_3 − c_2. The fit counts as unusable until all five categories have a vote.

### The tests

| test | statistic | rejects when |
|---|---|---|
| band opt_k has width > 0 (k = 1, 2, 3) | z_k = ŵ_k / se(ŵ_k), with w_k = c_{k+1} − c_k and se from the delta method | z_k > 1.645 |
| all bands distinguishable | all three z_k | all three reject |

In the band test, se(ŵ) = ŵ · se(log ŵ), so z_k = 1 / se(log ŵ_k). In practice the test asks whether each band's width is known to a relative standard error below about 0.6. A failed logit fit counts as not rejecting.

**Power** at N is the share of the datasets that reject. The **N needed** is the first N from which the power stays ≥ 80 %. It is converted to weeks of monitoring through the mean time to the N-th vote. The vote temperatures do not depend on the state, so the tests depend only on N; p only sets how many weeks that takes. The p̂_s are not tested; their precision is read off `p_feedback_vs_N.png`.

### Outputs

| file | content |
|---|---|
| `needed_n.csv`, `Thesis/Tables/feedback_power_*.tex` | per test: N for 80 % power, the weeks that takes, and the power at the room's observed number of feedbacks |
| `comfort_curve.csv`, `feedback_comfort_*.tex` | per N: cut-points, s and comfort band as mean [2.5 %, 97.5 %] over the datasets |
| `p_feedback.csv`, `feedback_p_*.tex` | per N: p̂_s as mean [2.5 %, 97.5 %] |
| `power.csv`, `tests.csv`, `estimates.csv`, `summary.csv`, `run.json` | power per test and N, the statistics and estimates per dataset and N, mean, sd, SE and CI of the mean and 2.5 / 97.5 % quantiles per parameter, the run's settings |
| `p_feedback_vs_N.png` | p̂_s against N, one panel per state: per-dataset lines, mean, 95 % CI of the mean (mean ± t·SE, SE = sd/√R), 2.5–97.5 % quantiles over the datasets, truth |
| `cutpoints_vs_N.png` | the same for each cut-point ĉ_k, one panel per cut-point. N where under half the datasets have a usable logit fit get no mean or bands |
| `simulated_week.png` | the first week of three simulated datasets above an observed room week |
