# Multivariate Hidden Markov Models


Instead of using a single sequence of observations to infer a hidden state sequence, we can use multiple sequences of observations. This is useful when we have multiple features or modalities that are related to the same underlying hidden states.


In this project, we will also use huminity as a feature to infer the underlying hidden states. 

Given 
$$
X = \{X_1, X_2, ..., X_T\}
$$
where $X_t$ is a vector of observations at time $t$, we can model the hidden states the same way we used in the univariate case. The main difference is that the emission probabilities will now be multivariate distributions, such as multivariate Gaussian distributions, where the proberbility matrix of being in a state is now defined as: 

$$
P(X_t) = diag( p_1^{t}(x_t), p_2^{t}(x_t), ..., p_k^{t}(x_t) )
$$
where $p_i^{t}(x_t)$ is the probability of observing $x_t$ given that the hidden state at time $t$ is $i$.
We can then find the joint distribution of the marginal distribitions of $X_t$ being in state $i$ given time $t$, also denoted $P(X_t | S_t = i, T=t) $ as 
$$
P(X_t | S_t = i, T=t) = \prod_{j=1}^{d} p_{ij}^{t}(x_{tj})
$$

Practically, we can use the same algorithms as in the univariate case, such as the forward-backward algorithm and the Viterbi algorithm, to infer the hidden states and estimate the model parameters. However, we need to modify the emission probabilities to account for the multivariate nature of the observations.

This is very nice, because this allows to only introduce 1 new class and refactor our HMM class to handle multivariate observations. 























