# Theorems behind the models 

## Transformation of mu

$$
u_i = \tilde{u_0} + \sum_{k = 1}^{i} e^{\tilde{u_i}} | i \neq 0
$$
$s.t. u_0 = \tilde{u_0}$

$$

$$




## Autoregressive HMM 

We set 
$$
Y_t | ( C = i ) = u_i + \phi_i (u_i -y_{t-1}) + \epsilon_t^{i}
$$
where we define 
$$
\phi_i = g(\tilde{\phi_i})
$$
s.t. 
$$
\tilde{\phi_i} =  \frac{2e^{\tilde{\phi}}}{1 + e^{\tilde{\phi}}} - 1
$$
we use the change of variables and find the log likelihood as where $\tilde{\phi_i} \sim \mathcal{N}(0, \sigma)$
$$
l(\theta|\phi) = \sum_{i = 1}^k {\log {f_N(0, \sigma|\phi_i)}} - \sum_{i = 1}^K {\log {\frac{2e^{\tilde{\phi}}}{(1 + e^{\tilde{\phi}})^{2}}}}
$$



