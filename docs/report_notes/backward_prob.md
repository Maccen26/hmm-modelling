# Backward Probabilities


We need to find the prob that 

$$
P(C_t = i | C_{t+1} = j, Y_{1:T})
$$
Where $Y_{1:T}$ is the entire sequence of observations.

We can find this by using the forward probabilities and the transition matrix, where $u_{t,t} = P(C_t = i | Y_{1:t})$ is the forward probability, and $\Gamma_{ij}^t = P(C_{t} = j | C_{t-1} = i)$ is the transition matrix. 

We then do a backward sampling by the formula: 

$$
P(C_t = i | C_{t+1} = j, Y_{1:T}) = \frac{P(C_{t+1} = j|C_i = i) P(C_{t})}{\sum_{k=1}^{K} P(C_{t+1} = j|C_i = k) P(C_{t} = k)}
$$
where $K$ is the number of states. This can be simplified to by finding a state vector $z_t$ where $z_t = P(C_t = i | C_{t+1} = j, Y_{1:T})$ as: 
$$
z_t = \frac{u_{t,t} \odot \Gamma^{t+1}_{:, j}}{\sum ( u_{t,t} \odot \Gamma^{t+1}_{:, j})}
$$


# Backward Sampling

We need to find the probability

$$
P(C_t = i \mid C_{t+1} = j, Y_{1:T})
$$

where $Y_{1:T}$ is the entire sequence of observations.

We can find this using the filtered probabilities and the transition matrix, where $u_{t,t}$ is the vector of filtered probabilities with entries $(u_{t,t})_i = P(C_t = i \mid Y_{1:t})$, and $\Gamma_{ij}^t = P(C_{t} = j \mid C_{t-1} = i)$ is the transition matrix.

Given $C_{t+1}$, the future observations $Y_{t+1:T}$ carry no further information about $C_t$, so $P(C_t = i \mid C_{t+1} = j, Y_{1:T}) = P(C_t = i \mid C_{t+1} = j, Y_{1:t})$. Using Bayes' rule, we get:

$$
P(C_t = i \mid C_{t+1} = j, Y_{1:T}) = \frac{P(C_{t+1} = j \mid C_t = i)\, P(C_t = i \mid Y_{1:t})}{\sum_{k=1}^{K} P(C_{t+1} = j \mid C_t = k)\, P(C_t = k \mid Y_{1:t})}
$$

where $K$ is the number of states. This can be written in vector form by defining the vector $p_t$ with entries $(p_t)_i = P(C_t = i \mid C_{t+1} = j, Y_{1:T})$:

$$
p_t = \frac{u_{t,t} \odot \Gamma^{t+1}_{:, j}}{\sum_{k=1}^{K} \left( u_{t,t} \odot \Gamma^{t+1}_{:, j} \right)_k}
$$

We start by sampling $C_T$ from $u_{T,T}$, and then for $t = T-1, \dots, 1$ we sample $C_t$ from $p_t$, using the sampled value of $C_{t+1}$ as $j$.