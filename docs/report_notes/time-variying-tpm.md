# Time Varying Transition Probability Matrices (TPMs)

The TPM is denoted $\Gamma$ and has so far been a Homogeneous matrix, meaning that the transition probabilities are constant over time. However, in many real-world scenarios, the transition probabilities may vary with time or depend on external covariates. To include covarites in the transition probabilities, $\Gamma$ is now a function of covarites. 

$$
\Gamma: \mathbb{R}^p \to \mathbb{R}^{K \times K}
$$ 

The covarites we wil focus on is the time varying covarites, where there is a dependence on time.   
We can write the transition probability matrix to the time t, as: 

$$
\Gamma^t_{i,j}(\xi, \beta, x_t) = 
    \begin{cases}
    \dfrac{\exp(\bar{\xi_{ij}} )}{1 + \sum_{k \neq i} \exp(\bar{\xi_{ik}})}, & j \neq i, \\[6pt]
    \dfrac{1}{1 + \sum_{k \neq i} \exp(\bar{\xi_{ik}})}, & j = i.
    \end{cases}$$
where $\bar{\xi_{ij}}$ is a linear combination of the covarites $x_t$ and the parameters $\beta_{i,j,l}$, which can be written as:
$$
\bar{\xi_{ij}} = \xi_{ij} + \sum_{l=1}^p \beta_{i,j,l} x_{t,l}
$$