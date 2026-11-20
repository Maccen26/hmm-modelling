# Random-restart search: room_015, `calibrated_anchor`, K = 3–7

All metrics before the final section are on the **validation set** (724 points, 2024-01-05 → 2024-01-20).
The selection score is the **mean validation RMSE over horizons h = 1, 2, 4, 6, 8** (30 min – 4 h ahead).
Test metrics are reported only in the last section.

## Top 4 models (validation)

All four are AR(1)-HMM(β,1) from random starts. Their scores are within 2.7 ppm of each other, so they
are effectively tied on validation.

| Rank | Model | K | Init | Val score | Loss |
|---|---|---|---|---|---|
| 1 | AR(1)-HMM(β,1) | 5 | random_6 | 262.1 | 15837.1 |
| 2 | AR(1)-HMM(β,1) | 3 | random_2 | 262.6 | 16036.3 |
| 3 | AR(1)-HMM(β,1) | 6 | random_8 | 264.5 | 15805.2 |
| 4 | AR(1)-HMM(β,1) | 4 | random_9 | 264.8 | 15900.8 |

Validation metrics (RMSE / R² / MAPE %):

| Model | h = 1 | h = 2 | h = 4 | h = 6 | h = 8 |
|---|---|---|---|---|---|
| K = 5 | 151.9 / 0.935 / 5.9 | **215.3 / 0.870** / 9.0 | **278.8 / 0.782** / 14.4 | **312.6 / 0.726** / 19.0 | 351.8 / 0.653 / 23.6 |
| K = 3 | **150.6 / 0.936 / 5.7** | 215.8 / 0.869 / **8.1** | 281.6 / 0.777 / **12.3** | 315.5 / 0.721 / **16.2** | 349.7 / 0.657 / 19.9 |
| K = 6 | 151.7 / 0.935 / 5.7 | 219.9 / 0.864 / 8.6 | 288.9 / 0.766 / 13.5 | 316.2 / 0.720 / 17.0 | **345.8 / 0.665** / 20.7 |
| K = 4 | 151.2 / 0.936 / 5.9 | 220.3 / 0.864 / 9.0 | 287.8 / 0.768 / 13.4 | 317.7 / 0.717 / 16.6 | 346.9 / 0.662 / **19.8** |

- **RMSE:** K = 5 is best at 1–3 h ahead and K = 6 at 4 h ahead.
- **MAPE:** K = 3 is best from 30 min to 3 h ahead. K = 4 edges it at 4 h ahead (19.8 vs 19.9), and both are far
  below K = 5 (23.6).
- **K = 4 vs K = 3:** K = 4 is close to K = 3 everywhere and close to K = 6 at 4 h, but it is not best on RMSE at
  any horizon.

### 1. K = 5 (val score 262.1)

| State | μ (ppm) | σ | φ |
|---|---|---|---|
| 1 | 489.1 | 29.5 | 0.960 |
| 2 | 489.1 | 18.1 | 0.943 |
| 3 | 571.7 | 25.9 | 0.930 |
| 4 | 1483.9 | 262.4 | 0.664 |
| 5 | 5448.1 | 139.4 | 0.954 |

Base Γ (both covariates 0):
```
[[0.888 0.019 0.092 0.    0.   ]
 [0.004 0.990 0.003 0.    0.002]
 [0.    0.    0.997 0.002 0.001]
 [0.408 0.001 0.012 0.515 0.064]
 [0.081 0.    0.062 0.126 0.731]]
```
β (rows: from-state; columns: its 4 off-diagonal logits in order):
```
off_day                              working_hours
[[ 2.71 -3.18  0.   -1.42]           [[ 0.44 -3.06  0.    3.83]
 [ 1.38 -2.84  0.   -3.59]            [ 1.37  0.03  0.    2.68]
 [ 0.    0.   -1.66  4.11]            [ 0.    0.   -0.48  7.19]
 [-1.15 -0.39  2.93 -2.36]            [-5.65  1.76  1.35  0.48]
 [-1.99  0.   -1.49  1.78]]           [-1.93  0.   -4.55 -0.02]]
```
- **States:** two background states at 489 ppm that differ only in noise (σ 30 vs 18), a low-occupancy level (572),
  a build-up state (1484, low φ = 0.66, large σ) and high occupancy (5448).
- **Redundancy:** states 1 and 2 share μ, and their σ differ by a factor of only 1.6, just above the 1.5 duplicate
  threshold, so one of them is nearly redundant.
- **Covariates:** working hours make the jump from the quiet states into high occupancy much more likely (3→5 logit
  +7.19, 1→5 +3.83) and the drop from build-up back to background less likely (4→1 −5.65).

### 2. K = 3 (val score 262.6)

| State | μ (ppm) | σ | φ |
|---|---|---|---|
| 1 | 514.9 | 23.2 | 0.942 |
| 2 | 1869.8 | 265.6 | 0.863 |
| 3 | 5820.3 | 122.5 | 0.955 |

Base Γ (both covariates 0):
```
[[0.997 0.002 0.002]
 [0.299 0.701 0.   ]
 [0.    0.214 0.786]]
```
β (rows: from-state; columns: its 2 off-diagonal logits in order):
```
off_day                working_hours
[[ 1.32 -4.41]         [[-2.39  4.36]
 [-0.81  0.  ]          [-2.56  0.  ]
 [ 0.    0.46]]         [ 0.   -0.44]]
```
- **States:** background (515, calm), build-up / decay (1870, large σ, lowest φ) and high occupancy (5820).
  All three means are distinct.
- **Covariates:** during working hours the background state jumps straight to high occupancy far more often
  (1→3 logit +4.36) and the build-up state is less likely to fall back to background (2→1 −2.56). On off days the jump
  to high occupancy is strongly suppressed (1→3 −4.41).
- **Transitions:** high occupancy can only decay through the build-up state (3→1 is 0), which matches CO₂ ventilating
  out gradually.

### 3. K = 6 (val score 264.5)

| State | μ (ppm) | σ | φ |
|---|---|---|---|
| 1 | 480.9 | 18.8 | 0.943 |
| 2 | 583.6 | 345.9 | 0.774 |
| 3 | 583.6 | 158.6 | 0.974 |
| 4 | 583.6 | 26.4 | 0.929 |
| 5 | 783.9 | 27.1 | 0.948 |
| 6 | 7358.5 | 139.3 | 0.965 |

Base Γ (both covariates 0):
```
[[0.988 0.    0.    0.009 0.    0.002]
 [0.003 0.326 0.573 0.089 0.    0.01 ]
 [0.    0.    0.469 0.    0.439 0.092]
 [0.    0.002 0.    0.997 0.    0.001]
 [0.068 0.    0.    0.094 0.837 0.   ]
 [0.    0.071 0.116 0.07  0.    0.743]]
```
β (rows: from-state; columns: its 5 off-diagonal logits in order):
```
off_day                                   working_hours
[[ 0.    0.   -1.84 -3.5  -3.71]           [[ 0.    0.   -2.4   4.85  3.04]
 [-0.01  0.3   0.14  0.   -0.02]            [ 1.44 -2.09  0.26  0.    2.34]
 [ 0.    0.    0.   -1.32 -3.03]            [ 0.    0.   -0.01 -3.54 -4.1 ]
 [ 0.   -1.52  0.    0.    4.15]            [ 0.   -1.    0.    0.    6.97]
 [ 1.72  0.    0.   -0.95 -0.12]            [-2.98  0.    0.   -4.05  3.96]
 [ 0.   -1.68  0.96  1.3   0.  ]]           [ 0.    0.24 -0.95 -3.98 -0.01]]
```
- **States:** three states share μ = 584 and differ only in σ (346 / 159 / 26) and φ. They act as noisy, medium
  and calm versions of one level, rather than as distinct occupancy levels.
- **Reliability:** only 2 of 11 starts were admissible, so this fit is the least robust of the four.

### 4. K = 4 (val score 264.8)

| State | μ (ppm) | σ | φ |
|---|---|---|---|
| 1 | 480.3 | 20.4 | 0.957 |
| 2 | 599.0 | 273.6 | 0.825 |
| 3 | 599.0 | 28.2 | 0.931 |
| 4 | 4618.6 | 150.4 | 0.943 |

Base Γ (both covariates 0):
```
[[0.967 0.    0.033 0.   ]
 [0.219 0.544 0.15  0.086]
 [0.006 0.001 0.991 0.003]
 [0.088 0.107 0.099 0.706]]
```
β (rows: from-state; columns: its 3 off-diagonal logits in order):
```
off_day                    working_hours
[[ 0.   -3.89  3.44]       [[ 0.   -0.3  -1.84]
 [-3.08  1.24 -2.55]        [-4.89  0.13  0.83]
 [ 3.57 -0.44  3.22]        [-0.89 -0.25  5.83]
 [-2.57  1.65 -2.  ]]       [-2.59 -0.05 -4.13]]
```
- **States:** background (480, calm) and high occupancy (4619). In between are two states at the same 599 ppm level:
  a noisy, fast-moving one (σ 274, φ 0.83) and a calm one (σ 28, φ 0.93). The noisy one acts as the build-up/decay
  state that K = 3 has at 1870, here placed at the low level with a large σ.
- **Working hours:** these drive the jump from the calm 599 state to high occupancy (3→4 logit +5.83). They also keep
  the high state from decaying to the calm level (4→3 −4.13), and the noisy state from falling back to background
  (2→1 −4.89).
- **Off days:** these raise the chance of the calm 599 state dropping back to background (3→1 +3.57).
- **Reliability:** 4 of 11 starts were admissible. The seeded start was rejected as a duplicate.

## How the search was run

**Data.** room_015, tag `calibrated_anchor`, 30-minute bins, observation `co2`, transition covariates
`off_day` and `working_hours` (1 on Mon–Fri 08:00–17:00). Train: 3137 points (2023-11-01 → 2024-01-05) for fitting.
Validation: 724 points (→ 2024-01-20) for choosing the start. Test: 966 points (→ 2024-02-09), used for reporting only.

**Chain.** HMM(1) → HMM(β,1) → AR(1)-HMM(1) → AR(1)-HMM(β,1), fitted in that order for each K. φ is **not capped**:
it is free in (−1, 1). The β models minimise NLL + 0.01·Σβ² and the others plain NLL, as in the per-model drivers.

**Starts.** Each model has 11 candidate starts:

- **seeded**: the drivers' start, built from the models *chosen earlier in this search*:
  - HMM(1) starts from data quantiles.
  - HMM(β,1) starts from the chosen HMM(1), with β = 0.05.
  - AR(1)-HMM(1) starts from the chosen HMM(1) emission and transition, with φ = 0.05.
  - AR(1)-HMM(β,1) starts from the chosen AR(1)-HMM(1) emission and the chosen HMM(β,1) transition.
- **random_0 … random_9**: no previous fit is used.
  - μ at random data quantiles drawn from U(0.02, 0.98).
  - σ = std(y)·U(0.1, 1).
  - φ ~ U(0, 0.95).
  - Γ = 0.8·I + 0.2·Dirichlet(1).
  - β ~ N(0, 0.5²).

**Rejected as collapsed:**

- φ ≥ 0.999: a unit root, so the state is a random walk.
- σ < 1 ppm, where the state has collapsed to a point, or σ > 10⁴ ppm, where it has blown up.
- Two states with means within 1 % **and** σ within a factor of 1.5.

**Selection.** The admissible start with the lowest validation score is saved and seeds the next model.

**Files.** The script is `drivers/models/random_search.py`. Each `states_K/` folder holds:
- the chosen `{slug}.pkl` files,
- `candidates.csv` with all 44 starts,
- `val_metrics.csv` and `test_metrics.csv`.

## The chain for each number of states (validation)

Columns: init = chosen start; seeded = the seeded start's score (or why it was rejected); admissible = starts that
passed the collapse checks, out of 11; val RMSE at h = 1 / 4 / 8; val MAPE at h = 1 / 4 / 8.

### K = 3

| Model | Init | Loss | Val score | Seeded | Admissible | RMSE h1 / h4 / h8 | MAPE h1 / h4 / h8 |
|---|---|---|---|---|---|---|---|
| HMM(1) | random_3 | 20552.6 | 418.4 | 418.4 | 11 | 333.6 / 420.6 / 507.4 | 20.0 / 29.2 / 39.7 |
| HMM(β,1) | random_7 | 20458.8 | 365.4 | 365.4 | 11 | 325.6 / 368.9 / 405.2 | 19.3 / 23.8 / 28.7 |
| AR(1)-HMM(1) | random_6 | 16094.9 | 290.2 | 294.9 | 5 | 151.9 / 303.4 / 414.2 | 5.7 / 14.7 / 27.1 |
| **AR(1)-HMM(β,1)** | random_2 | 16036.3 | **262.6** | 268.2 | 7 | 150.6 / 281.6 / 349.7 | 5.7 / 12.3 / 19.9 |

- **Seeded vs random:** for the two plain HMMs, the seeded and random starts reach the same optimum, so the scores
  are identical. For both AR models a random start beats the seeded one.
- **Loss vs forecasting:** the AR(1)-HMM(β,1) start with the lowest loss (`random_0`, 16005.8) was not the best
  forecaster. The chosen fit has a loss about 30 higher.

### K = 4

| Model | Init | Loss | Val score | Seeded | Admissible | RMSE h1 / h4 / h8 | MAPE h1 / h4 / h8 |
|---|---|---|---|---|---|---|---|
| HMM(1) | random_3 | 19916.1 | 390.5 | 420.2 | 11 | 285.9 / 395.1 / 495.1 | 14.7 / 24.2 / 34.4 |
| HMM(β,1) | random_2 | 19797.5 | 337.8 | 337.8 | 11 | 282.0 / 343.3 / 389.7 | 13.7 / 19.9 / 25.7 |
| AR(1)-HMM(1) | random_5 | 16012.9 | 294.8 | rejected: φ = 1 | 4 | 151.9 / 311.0 / 419.2 | 5.6 / 14.2 / 24.7 |
| **AR(1)-HMM(β,1)** | random_9 | 15900.8 | **264.8** | rejected: duplicate (μ 462/462) | 4 | 151.2 / 287.8 / 346.9 | 5.9 / 13.4 / 19.8 |

- **Chosen AR(1)-HMM(β,1):** see Top 4, rank 4.
- **Seeded start:** the seeded AR(1)-HMM(1) start fell into the unit-root solution found earlier (φ = 1, μ 3150/3150).

### K = 5

| Model | Init | Loss | Val score | Seeded | Admissible | RMSE h1 / h4 / h8 | MAPE h1 / h4 / h8 |
|---|---|---|---|---|---|---|---|
| HMM(1) | random_0 | 19463.3 | 365.8 | 396.1 | 10 | 244.0 / 372.6 / 483.6 | 13.0 / 22.3 / 32.0 |
| HMM(β,1) | **seeded** | 19317.2 | 311.6 | 311.6 | 10 | 243.4 / 324.2 / 365.2 | 11.3 / 17.5 / 22.9 |
| AR(1)-HMM(1) | random_5 | 15912.3 | 289.1 | 293.6 | 3 | 151.0 / 305.2 / 408.4 | 5.5 / 13.9 / 23.5 |
| **AR(1)-HMM(β,1)** | random_6 | 15837.1 | **262.1** | 266.4 | 3 | 151.9 / 278.8 / 351.8 | 5.9 / 14.4 / 23.6 |

- **Chosen AR(1)-HMM(1):** μ = 493, 512, 512, 907, 5290. Two states share μ = 512, with σ 293 vs 22.
- **Chosen AR(1)-HMM(β,1):** see Top 4, rank 1.

### K = 6

| Model | Init | Loss | Val score | Seeded | Admissible | RMSE h1 / h4 / h8 | MAPE h1 / h4 / h8 |
|---|---|---|---|---|---|---|---|
| HMM(1) | **seeded** | 18960.9 | 367.0 | 367.0 | 8 | 249.1 / 374.9 / 481.6 | 13.0 / 22.4 / 32.3 |
| HMM(β,1) | random_1 | 18939.7 | 288.5 | 311.9 | 10 | 203.2 / 300.3 / 356.1 | 10.0 / 16.2 / 21.6 |
| AR(1)-HMM(1) | random_5 | 15919.6 | 293.7 | rejected: φ = 1 | 2 | 151.0 / 307.7 / 421.9 | 5.6 / 16.3 / 30.9 |
| **AR(1)-HMM(β,1)** | random_8 | 15805.2 | **264.5** | rejected: φ = 1 | 2 | 151.7 / 288.9 / 345.8 | 5.7 / 13.5 / 20.7 |

- **Re-run:** a first run chose an HMM(1) with σ = 10⁸¹ in one state. I added the σ > 10⁴ check and reran K = 5 and 6;
  the clean seeded HMM(1) now wins.
- **HMM(β,1):** gets close to the AR models at long horizons here (h = 8 RMSE 356 vs 346).

### K = 7

| Model | Init | Loss | Val score | Seeded | Admissible | RMSE h1 / h4 / h8 | MAPE h1 / h4 / h8 |
|---|---|---|---|---|---|---|---|
| HMM(1) | random_1 | 18819.7 | 351.8 | 355.3 | 9 | 215.4 / 360.1 / 478.9 | 10.4 / 20.3 / 29.9 |
| HMM(β,1) | **seeded** | 18638.9 | 289.9 | 289.9 | 10 | 204.2 / 300.4 / 357.9 | 9.3 / 15.3 / 20.7 |
| AR(1)-HMM(1) | random_6 | 15899.5 | 297.5 | rejected: φ = 1 | 1 | 151.8 / 312.9 / 426.1 | 5.6 / 16.6 / 31.0 |
| **AR(1)-HMM(β,1)** | random_1 | 15794.0 | **270.6** | rejected: φ = 1 | 1 | 151.1 / 292.1 / 360.1 | 5.6 / 12.4 / 18.8 |

- **Chosen AR(1)-HMM(β,1):** μ = 489 ×3, 618, 1535, 4834 ×2. It has three background states and two high states
  that differ only in σ.
- **Reliability:** only **one** of 11 starts was admissible, so this choice is the most fragile.
- **Score:** it has the worst validation score of the five AR(1)-HMM(β,1) models, despite the lowest loss.

## Best model per number of states (validation)

| K | Best model | Init | Val score | Val RMSE h8 | Val R² h8 | Val MAPE h8 | Admissible starts | Distinct AR means? |
|---|---|---|---|---|---|---|---|---|
| 3 | AR(1)-HMM(β,1) | random_2 | 262.6 | 349.7 | 0.657 | **19.9** | **7** | **yes** |
| 4 | AR(1)-HMM(β,1) | random_9 | 264.8 | 346.9 | 0.662 | 19.8 | 4 | no (599 ×2) |
| 5 | AR(1)-HMM(β,1) | random_6 | **262.1** | 351.8 | 0.653 | 23.6 | 3 | no (489 ×2) |
| 6 | AR(1)-HMM(β,1) | random_8 | 264.5 | **345.8** | **0.665** | 20.7 | 2 | no (584 ×3) |
| 7 | AR(1)-HMM(β,1) | random_1 | 270.6 | 360.1 | 0.636 | **18.8** | 1 | no (489 ×3, 4834 ×2) |

## What the search shows

1. **Random starts matter.** 17 of the 20 chosen models come from random starts. At K = 4, 6 and 7 the seeded AR
   starts ended in a unit-root state (φ = 1) or a duplicate, because seeding carries the previous model's solution
   forward.
2. **Collapse grows with K.** For AR(1)-HMM(β,1), the number of admissible starts out of 11 falls 7 → 4 → 3 → 2 → 1
   for K = 3 → 7. From K = 4 up, every chosen AR model has states that share a mean and differ only in σ.
3. **Covariates help most at long horizons.** Adding β cuts the h = 8 validation RMSE at every K:
   - AR models: by 57–76 ppm (e.g. K = 3: 414 → 350).
   - Plain HMMs: by 102–126 ppm.
4. **AR(1) helps most at short horizons.** The h = 1 validation RMSE is about 151 for every AR model, against
   204–334 without AR.
5. **Validation does not separate K = 3–6.** Their scores are within 2.7 ppm of each other, so the choice between them
   has to rest on parsimony, interpretability, or the test set.
6. **The lowest loss is not the best forecaster.** Within a K, the lowest-loss start is not always chosen (K = 3).
   Across K, the loss keeps falling with more states while the validation score does not improve.

## Caveats

- **Convergence:** most AR(1)-HMM(β,1) fits, and many HMM(β,1) fits at K ≥ 4, stopped at the iteration limit
  without reaching the tolerance.
- **Duplicate rule:** same-mean states with clearly different σ are kept, following the brief ("same mean with almost
  the same std").
- **Starts:** each model had only 10 random starts. At K = 6–7 just 1–2 survived, so those choices are fragile.
- **Validation size:** the validation set is short (about 2 weeks).

## Test set results

The test set has 966 points (2024-01-20 → 2024-02-09). The parameters are fixed, and the filter runs through train
and validation first.

Test metrics for each chosen model are below (RMSE / R² / MAPE %).

| K | Model | h = 1 | h = 4 | h = 8 |
|---|---|---|---|---|
| 3 | HMM(1) | 263.0 / 0.613 / 21.6 | 352.1 / 0.307 / 30.8 | 419.8 / 0.015 / 38.9 |
| 3 | HMM(β,1) | 282.5 / 0.554 / 22.9 | 340.6 / 0.352 / 29.1 | 381.4 / 0.187 / 33.9 |
| 3 | AR(1)-HMM(1) | 145.7 / 0.881 / 6.6 | 296.5 / 0.509 / 17.3 | 371.0 / 0.231 / 29.7 |
| 3 | **AR(1)-HMM(β,1)** | 145.1 / 0.882 / 6.5 | 284.1 / 0.549 / 15.0 | **320.3 / 0.427** / 21.8 |
| 4 | HMM(1) | 221.9 / 0.725 / 15.7 | 323.2 / 0.416 / 24.4 | 403.6 / 0.090 / 33.2 |
| 4 | HMM(β,1) | 241.0 / 0.676 / 16.3 | 304.1 / 0.483 / 23.1 | 353.3 / 0.302 / 29.0 |
| 4 | AR(1)-HMM(1) | 146.8 / 0.880 / 6.6 | 306.1 / 0.476 / 16.8 | 375.3 / 0.213 / 27.0 |
| 4 | **AR(1)-HMM(β,1)** | 145.6 / 0.882 / 6.7 | 291.8 / 0.524 / 16.0 | 331.7 / 0.385 / 21.7 |
| 5 | HMM(1) | 198.6 / 0.780 / 14.1 | 328.1 / 0.398 / 24.0 | 410.6 / 0.058 / 33.2 |
| 5 | HMM(β,1) | 212.6 / 0.747 / 13.6 | 295.1 / 0.513 / 20.7 | 331.4 / 0.386 / 26.1 |
| 5 | AR(1)-HMM(1) | 146.4 / 0.880 / 6.6 | 304.6 / 0.481 / 17.4 | 368.3 / 0.242 / 27.4 |
| 5 | **AR(1)-HMM(β,1)** | 146.0 / 0.881 / 7.0 | 287.7 / 0.538 / 17.9 | 348.1 / 0.323 / 27.0 |
| 6 | HMM(1) | 198.8 / 0.779 / 12.7 | 326.9 / 0.403 / 22.6 | 411.5 / 0.054 / 31.9 |
| 6 | HMM(β,1) | 171.9 / 0.835 / 11.5 | 284.1 / 0.549 / 19.3 | 320.8 / 0.425 / 24.4 |
| 6 | AR(1)-HMM(1) | 144.8 / 0.883 / 6.6 | 301.5 / 0.492 / 18.9 | 392.5 / 0.139 / 34.0 |
| 6 | **AR(1)-HMM(β,1)** | 146.7 / 0.880 / 6.9 | 304.3 / 0.482 / 17.6 | 350.3 / 0.314 / 24.7 |
| 7 | HMM(1) | 182.4 / 0.814 / 11.7 | 315.0 / 0.446 / 21.4 | 401.7 / 0.098 / 30.5 |
| 7 | HMM(β,1) | 166.1 / 0.846 / 10.5 | **283.3 / 0.552** / 18.1 | **320.1 / 0.427** / 23.0 |
| 7 | AR(1)-HMM(1) | 145.7 / 0.881 / 6.6 | 310.2 / 0.462 / 18.9 | 392.9 / 0.137 / 32.3 |
| 7 | **AR(1)-HMM(β,1)** | **143.3 / 0.885 / 6.4** | 289.1 / 0.533 / 15.3 | 337.3 / 0.364 / **21.5** |

Test metrics for the top 4 validation models, at every horizon:

| Model | h = 1 | h = 2 | h = 4 | h = 6 | h = 8 |
|---|---|---|---|---|---|
| K = 5 (val rank 1) | 146.0 / 0.881 / 7.0 | 218.3 / 0.734 / 11.3 | 287.7 / 0.538 / 17.9 | 321.0 / 0.424 / 22.7 | 348.1 / 0.323 / 27.0 |
| **K = 3 (val rank 2)** | **145.1 / 0.882 / 6.5** | **218.0 / 0.734 / 10.0** | **284.1 / 0.549 / 15.0** | **308.0 / 0.470 / 18.5** | **320.3 / 0.427** / 21.8 |
| K = 6 (val rank 3) | 146.7 / 0.880 / 6.9 | 227.7 / 0.710 / 11.4 | 304.3 / 0.482 / 17.6 | 331.8 / 0.385 / 21.4 | 350.3 / 0.314 / 24.7 |
| K = 4 (val rank 4) | 145.6 / 0.882 / 6.7 | 223.6 / 0.721 / 10.8 | 291.8 / 0.524 / 16.0 | 317.5 / 0.437 / 19.0 | 331.7 / 0.385 / **21.7** |

**What the test set adds:**

- **K = 3 generalises best of the top 4.** On test it is the best of the four on every metric at every horizon, with
  two near-ties: K = 5 matches its R² at h = 2, and K = 4 has a 0.1-point lower MAPE at h = 8 (21.7 vs 21.8).
- **K = 5 falls behind.** The validation winner has an h = 8 R² of 0.323 against 0.427 for K = 3.
- **K = 4 is the runner-up on test from 3 h ahead.** At h = 6 and h = 8 its RMSE (317.5, 331.7) is second only to K = 3.
- **Recommendation: K = 3, AR(1)-HMM(β,1), random start `random_2`.**
  - It is tied for best on validation.
  - It is the best of the top 4 on test.
  - It is the simplest of them.
  - It is the only K where every AR state has a distinct mean.
  - It has the most admissible starts.
- **Plain HMM(β,1) at K = 6–7 matches it at long horizons on test** (h = 8 RMSE 320.1–320.8 vs 320.3). The
  AR models' advantage is therefore mainly at short horizons.
