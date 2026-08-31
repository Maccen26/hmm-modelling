import jax
from src.api.v4.hmm_models.hmm_params import HMMParams
import jax.numpy as jnp
from typing import Callable
from src.api.v4.algorithms.forward_algorithm import ForwardAlgorithm

from src.base.base_inference import BaseInference
from src.base.base_emission import BaseEmission
from src.base.base_transition import BaseTransition
from typing import Tuple
from src.api.v4.likelihoods import negative_log_likelihood
from src.api.v4.hmm_models.results import HMMResults, StateResults 


class HMM:
    def __init__(self, transition: BaseTransition, emission: BaseEmission,
                 inital_distribution=None):
        
        self.params = HMMParams(transition=transition, emission=emission)
        self.u_pre = self._set_initial_distribution(inital_distribution)
        self.ll_fits = []  
        self.negative_log_likelihood : Callable = negative_log_likelihood 
        self.hmm_results: HMMResults | None = None
        self.state_results: StateResults | None = None 

    def set_negative_log_likelihood(self, loss_fn: Callable):
        self.negative_log_likelihood = loss_fn


    def _set_initial_distribution(self, inital_distribution):
        if inital_distribution is not None:
            return self._validate_initial_distribution(inital_distribution)
        return self._compute_stationary_distribution()

    def _validate_initial_distribution(self, u: jnp.ndarray) -> jnp.ndarray:
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        num_states = self.transition.transition_logits.shape[0]
        if u.ndim == 1:
            if u.shape[0] != num_states:
                raise ValueError(
                    f"inital_distribution has {u.shape[0]} states but transition has {num_states}."
                )
            u = u[jnp.newaxis, :]  # reshape (num_states,) -> (1, num_states)
        if u.ndim == 2:
            if u.shape != (1, num_states):
                raise ValueError(
                    f"inital_distribution must have shape (1, {num_states}), got {u.shape}."
                )
        else:
            raise ValueError(
                f"inital_distribution must be 1-D or 2-D, got {u.ndim}-D array."
            )
        return u

    def _compute_stationary_distribution(self):
        num_states = self.transition.transition_logits.shape[0]
        I = jnp.eye(num_states)
        E = jnp.ones((num_states, num_states))
        e = jnp.ones((num_states, 1))

        try:
            Gamma = self.transition.transition_matrix()
            delta = e.T @ jnp.linalg.inv(I - Gamma + E)
            return delta  # shape (1, num_states)
        except Exception as e:
            raise ValueError(
                f"Error computing inital state distribution. "
                f"Maybe the Stationary Transition matrix is not invertible? {e}"
            )

    @property
    def transition(self):
        return self.params.transition

    @property
    def emission(self):
        return self.params.emission

    def _set_inference_algorithm(self, inference: str) -> BaseInference:
        if inference == "forward":
            return ForwardAlgorithm()
        raise ValueError(f"Inference method {inference} could not be set")

    def fit(self, ys: jnp.ndarray,
            xs: jnp.ndarray | None = None,
            solver=None,
            frozen=None,
            num_iters: int = 200,
            tol: float = 1e-6) -> None:
        if solver is None:
            from src.api.v4.solvers import GradientSolver
            solver = GradientSolver()

        convergence = False
        prev_ll = float('-inf')

        for _ in range(num_iters):
            solver.fit(self.params, ys, xs, u_pre=self.u_pre,
                   frozen=frozen, loss_fn=self.negative_log_likelihood)
            self.params = solver.params
            current_ll = -solver.opt_loss_val if solver.opt_loss_val is not None else float('-inf')
            self.ll_fits.append(current_ll)

            if abs(current_ll - prev_ll) / (abs(prev_ll) + 1e-10) < tol:
                convergence = True
                break
            prev_ll = current_ll

        self.hmm_results = HMMResults(convergence=convergence, log_likelihood=self.ll_fits[-1], num_params=len(self.params))
        self.state_results = self._compute_state_results(ys, xs) 

    def _compute_state_results(self, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> StateResults:
        from jax.scipy.stats import norm
        inference_alg = self._set_inference_algorithm("forward")
        output = inference_alg.run(self.params, self.u_pre, ys, xs)
        z_list = []
        for t in range(0, len(ys) - 1):
            G_t = self.emission.cdf(t, ys, xs)  # shape (1, num_states)
            z_t = norm.ppf(jnp.sum(output.ut[t+1] * G_t))
            z_list.append(z_t)
        
        return StateResults(utt=output.utt, ut=output.ut, time_index=jnp.arange(len(ys)), pseudo_residuals=jnp.asarray(z_list))

    def log_likelihood(self, ys: jnp.ndarray| None = None, xs: jnp.ndarray | None = None) -> float:
        if (ys is None):
            return self.ll_fits[-1] if self.ll_fits else float('-inf')
        ll = self._compute_log_likelihood(ys, xs)
        return ll


    def _compute_log_likelihood(self, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> float:
        inference_alg = self._set_inference_algorithm("forward")
        output = inference_alg.run(self.params, self.u_pre, ys, xs)
        from src.api.v4.likelihoods import negative_log_likelihood
        return -float(negative_log_likelihood(output, self.params)) 
    

    def update_param(self, param_name: str, new_value: jax.Array, index: Tuple|float|None = None) -> None:
        self.params = self.params.update_param(param_name, new_value, index) 

    # Todo: Refactor this method to be part of fit maybe 
    def pseudo_residuals(self, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        from jax.scipy.stats import norm
        inference_alg = self._set_inference_algorithm("forward")
        output = inference_alg.run(self.params, self.u_pre, ys, xs) 
        ut = output.ut  # shape (T, num_states) 
        z_list = []
        for t in range(0, len(ys) - 1):
            G_t = self.emission.cdf(t, ys, xs)  # shape (1, num_states)
            z_t = norm.ppf(jnp.sum(ut[t+1] * G_t))
            z_list.append(z_t)
        
        return jnp.array(z_list)


    

