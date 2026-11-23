# Sample Size Determination

1. Simulating data (R = 100 datasets, each until 300 votes)
- A vector of probs $p_j$ for observing feedback in each state is given.
- A state path is drawn from the fitted chain: $z_0 \sim u_0 \Gamma_0$, $z_t \sim \Gamma_t[z_{t-1}]$.
- A CO2 emission is generated from the state's AR(1): $y_t = \mu_s + \phi_s (y_{t-1} - \mu_s) + \sigma_s \varepsilon_t$.
- At each step, a Bernoulli trial gives feedback or not: $f_t \sim \text{Bernoulli}(p_{z_t})$.
- At each step, a temperature is drawn from U(15,25), independent of the state and the CO2.
- If feedback, an error is drawn from the logistic distribution with s = 0.55, giving a standard deviation of $0.55 \cdot \pi / \sqrt{3} \approx 1$ °C.
  The perceived temperature $T_t + s \cdot \text{Logistic}(0,1)$ is given a category: <17, 17-19, 19-21, 21-23, >23.
- Each dataset is cut at its N-th vote, N = 5, 10, ..., 300, and everything is re-estimated at each N.

2. Estimating state probs
State probs are estimated as
$$
\hat p_j = P(\text{Feedback} \mid S = j) = \frac{\sum_t f_t \, \gamma_t(j)}{\sum_t \gamma_t(j)}, \qquad \gamma_t(j) = P(S_t = j \mid y_{1:t_N})
$$
where $f_t \in \{0, 1\}$ is feedback at step $t$ and $\gamma_t(j)$ uses only the CO2 seen up to the N-th vote, under the true model.

3. Estimated cutoff temps for categories
An ordered logit model from statsmodels (`OrderedModel`, `distr="logit"`) is fitted on the (T, vote) pairs of the first N votes.
It estimates the cutoff temps $c_1, \dots, c_4$ between the categories and the scale $s = 1/\beta$.
A fit counts only once all five categories have a vote.
