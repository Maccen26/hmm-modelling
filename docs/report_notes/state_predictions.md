# State predictions

Let $z_t \in \{1, \dots, K\}$ be the hidden state at time $t$, $Y_{1:t}$ the observations up to time $t$, and $\Gamma$ the transition matrix with entries $\Gamma_{ij} = P(z_{t+1} = j \mid z_t = i)$.

## Predicted state

The **filtered state distribution** is our belief about the state at time $t$ using only the observations up to time $t$:
$$
u_{t,t}(i) = P(z_t = i \mid Y_{1:t}), \qquad i = 1, \dots, K.
$$

To predict $k$ steps ahead, we take the filtered distribution and move it $k$ steps forward with the transition matrix. No new observations are used:
$$
u_{t+k,t} = u_{t,t}\, \Gamma^k,
$$
where $u_{t,t}$ is a row vector. The $i$-th entry is
$$
u_{t+k,t}(i) = P(z_{t+k} = i \mid Y_{1:t}).
$$

The **predicted state** $k$ steps ahead is the most probable state under this distribution:
$$
\hat{z}_{t+k \mid t} = \operatorname*{arg\,max}_{i} \; u_{t+k,t}(i).
$$

## Inferred state

The **smoothed state distribution** from the forward–backward algorithm uses *all* observations $Y_{1:T}$, both before and after time $t$:
$$
\gamma_t(i) = P(z_t = i \mid Y_{1:T}).
$$

The **inferred state** is the most probable state under the smoothed distribution:
$$
\tilde{z}_t = \operatorname*{arg\,max}_{i} \; \gamma_t(i).
$$

## Difference

- The **predicted** state uses only data up to time $t$ and looks $k$ steps into the future.
- The **inferred** state uses the full data set, so it is our best estimate of what the state actually was at time $t$.