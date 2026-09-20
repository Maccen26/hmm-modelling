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
        self.no_of_free_params = len(self.params)  # Store the number of free parameters

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

    def fit(self, 
            ys: jnp.ndarray,
            ts: jnp.ndarray | None = None,
            xs: jnp.ndarray | None = None,
            solver=None,
            frozen=None,
            num_iters: int = 200,
            tol: float = 1e-6) -> None:
        
        if solver is None:
            from src.api.v4.solvers import LBFGSSolver
            solver = LBFGSSolver()

        convergence = False
        prev_ll = float('-inf')
        if (frozen is not None):
            self.no_of_free_params = self.no_of_free_params - len(frozen)

        for i in range(num_iters):
            solver.fit(self.params, ys, ts, xs, u_pre=self.u_pre,
                   frozen=frozen, loss_fn=self.negative_log_likelihood)
            self.params = solver.params
            current_ll = -solver.opt_loss_val if solver.opt_loss_val is not None else float('-inf')
            self.ll_fits.append(current_ll)
    
            print(f"Iteration {i}: Log-Likelihood = {current_ll:.6f}")

            if abs(current_ll - prev_ll) / (abs(prev_ll) + 1e-10) < tol:
                convergence = True
                break
            prev_ll = current_ll

        self.hmm_results = HMMResults(convergence=convergence, log_likelihood=self.ll_fits[-1], num_params=len(self.params))
        self.state_results = self._compute_state_results(ys, xs, ts)

    def _compute_state_results(self, ys: jnp.ndarray, xs: jnp.ndarray | None = None, ts: jnp.ndarray | None = None) -> StateResults:
        from jax.scipy.stats import norm
        inference_alg = self._set_inference_algorithm("forward")
        output = inference_alg.run(self.params, self.u_pre, ys=ys, ts=ts, xs=xs)
        z_list = []
        for t in range(0, len(ys)):
            G_t = self.emission.cdf(t, ys, xs)  # shape (1, num_states)
            # ut[t] is the one-step-ahead predictive state distribution for obs t,
            # so the forecast pseudo-residual for obs t pairs ut[t] with cdf(y_t).
            # Clip into the open interval so float saturation at the tails (cdf ~0/1)
            # yields large-but-finite residuals instead of +/-inf/NaN.
            u_t = jnp.clip(jnp.sum(output.ut[t] * G_t), 1e-6, 1.0 - 1e-6)
            z_list.append(norm.ppf(u_t))

        return StateResults(utt=output.utt, ut=output.ut, time_index=jnp.arange(len(ys)), pseudo_residuals=jnp.asarray(z_list))

    def log_likelihood(self, ys: jnp.ndarray| None = None, xs: jnp.ndarray | None = None, ts: jnp.ndarray | None = None) -> float:
        if (ys is None):
            return self.ll_fits[-1] if self.ll_fits else float('-inf')
        ll = self._compute_log_likelihood(ys, xs, ts)
        return ll


    def _compute_log_likelihood(self, ys: jnp.ndarray, xs: jnp.ndarray | None = None, ts: jnp.ndarray | None = None) -> float:
        inference_alg = self._set_inference_algorithm("forward")
        output = inference_alg.run(self.params, self.u_pre, ys=ys, ts=ts, xs=xs)
        from src.api.v4.likelihoods import negative_log_likelihood
        return -float(negative_log_likelihood(output, self.params)) 
        #return float(jnp.sum(jnp.log(output.ft[drop_first:])))
    

    def update_param(self, param_name: str, new_value: jax.Array, index: Tuple|float|None = None) -> None:
        self.params = self.params.update_param(param_name, new_value, index) 

    # Todo: Refactor this method to be part of fit maybe 
    def pseudo_residuals(self, ys: jnp.ndarray, xs: jnp.ndarray | None = None, ts: jnp.ndarray|None = None) -> jnp.ndarray:
        from jax.scipy.stats import norm
        inference_alg = self._set_inference_algorithm("forward")
        output = inference_alg.run(self.params, self.u_pre, ys=ys, xs=xs, ts=ts) 
        ut = output.ut  # shape (T, num_states)
        z_list = []
        for t in range(0, len(ys)):
            G_t = self.emission.cdf(t, ys, xs)  # shape (1, num_states)
            # ut[t] is the one-step-ahead predictive state distribution for obs t,
            # so the forecast pseudo-residual for obs t pairs ut[t] with cdf(y_t).
            # Clip into the open interval so float saturation at the tails (cdf ~0/1)
            # yields large-but-finite residuals instead of +/-inf/NaN.
            u_t = jnp.clip(jnp.sum(ut[t] * G_t), 1e-6, 1.0 - 1e-6)
            z_list.append(norm.ppf(u_t))

        return jnp.array(z_list)
    

    def predict_emission(self, t_pred: jnp.ndarray, ys: jnp.ndarray, xs: jnp.ndarray | None = None, x_pred: jnp.ndarray | None = None, ts: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Predict the next n_steps observations based on the fitted model and given observations, observed covariates, and optional future covariates.
        The prediction is based on the expected value of the emission distribution at each step, weighted by the state probabilities.
        `ts` are the waiting times of the observed sequence `ys`; they must be
        supplied for a continuous model so the anchor filtered state is computed
        with the correct spacing (otherwise the forward pass assumes unit steps).
        """
        self.check_predict_args(t_pred = t_pred, ys=ys, xs=xs, x_pred=x_pred)
        # Get the last state probabilities from the fitted model
        state_results = self._compute_state_results(ys, xs, ts)
        utt = state_results.utt[-1]  # shape (num_states,)
        self.prediction: jnp.ndarray = self._run_prediction(utt=utt, ys=ys, x_pred=x_pred, t_pred = t_pred)
        return self.prediction  # shape (len(t_pred),)

    def check_predict_args(self, t_pred: jnp.ndarray, ys: jnp.ndarray, xs: jnp.ndarray | None = None, x_pred: jnp.ndarray | None = None) -> None:
        if (t_pred is None) or (len(t_pred) == 0):
            raise ValueError("t_pred must be provided and cannot be empty.")
        if (t_pred.ndim != 1):
            raise ValueError(f"t_pred must be a 1-D array, got {t_pred.ndim}-D array.")
        # t_pred are absolute distances from the last observation (anchor = 0),
        # so every value must be positive and strictly increasing for the
        # per-step gaps (obtained by differencing) to be positive.
        if bool(jnp.any(t_pred <= 0)):
            raise ValueError(f"t_pred must contain only positive values, got {t_pred}.")
        if bool(jnp.any(jnp.diff(t_pred) <= 0)):
            raise ValueError(f"t_pred must be strictly increasing, got {t_pred}.")
        if ys.ndim != 1:
            raise ValueError(f"ys must be a 1-D array, got {ys.ndim}-D array.")
        if xs is not None and xs.shape[0] != ys.shape[0]:
            raise ValueError(f"xs must have the same number of samples as ys. Got xs shape {xs.shape} and ys shape {ys.shape}.")
        if hasattr(self.emission, "phi_tilde") and self.emission.phi_tilde is not None:
            k = len(self.emission.phi_tilde)
            if ys.shape[0] < k:
                raise ValueError(f"ys must have at least {k} samples for the autoregressive emission. Got ys shape {ys.shape}.")

    def _run_prediction(self, utt: jnp.ndarray,  t_pred: jnp.ndarray , ys: jnp.ndarray, x_pred: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Forecast the emission mean at each absolute time in `t_pred`, measured
        from the last observation (anchor = 0). The per-step gap drives the
        transition: for a continuous model the gap is the waiting time fed to
        expm(Q * gap); for a discrete model the fixed matrix is raised to the
        integer gap power.
        """
        from src.api.v4.transitions.continuous_static_transition import ContinuousStaticTransition
        is_continuous = isinstance(self.transition, ContinuousStaticTransition)

        predictions = []
        u = utt  # last filtered state distribution, shape (num_states,)
        prev_t = 0.0

        for t_abs in t_pred:
            gap = t_abs - prev_t
            if is_continuous:
                # expm(Q * gap); chaining the gaps reproduces expm(Q * t_abs).
                Gamma = self.transition.transition_matrix(t=gap, ys=ys, xs=x_pred)
            else:
                # Discrete: Gamma ignores time, so bridge the gap with Gamma^gap.
                base = self.transition.transition_matrix(ys=ys, xs=x_pred)
                Gamma = jnp.linalg.matrix_power(base, int(gap))

            u = u @ Gamma
            # Plain Gaussian mean ignores t; state-weighted expectation.
            mu = self.emission.mu(t=int(t_abs), ys=ys, xs=x_pred)
            next_obs = jnp.sum(u * mu)
            predictions.append(next_obs)
            prev_t = t_abs

        return jnp.array(predictions)
    

