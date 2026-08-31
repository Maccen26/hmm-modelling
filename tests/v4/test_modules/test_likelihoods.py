from unittest import TestCase
from src.api.v4 import HMMParams, StaticTransition, GaussEmission
import jax.numpy as jnp
from src.api.v4.algorithms import ForwardAlgorithm
from src.api.v4.likelihoods import negative_log_likelihood

class TestLikelihoods(TestCase):
    def setUp(self) -> None:
        self.transition_logits = jnp.array([[-0.84729786, -1.94591015],
                                            [-2.07944154, -1.09861229],
                                            [-1.60943791, -1.09861229]])

        self.transition_matrix = StaticTransition(self.transition_logits)

        self.emission_mean = jnp.array([0.0, 1.0, 2.0])
        self.emission_sigma = jnp.array([1.0, 1.0, 1.0])
        self.emission = GaussEmission.from_params(self.emission_mean, self.emission_sigma)

        self.hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        self.forward_alg = ForwardAlgorithm()
        self.num_states = self.transition_logits.shape[0]
        self.u0 = jnp.ones((1, self.num_states)) / self.num_states

    def _run(self, ys):
        return self.forward_alg.run(self.hmm, self.u0, ys)

    def test_negative_log_likelihood_computation(self):
        ys = jnp.array([[0.5], [1.5], [2.5]])
        try:
            output = self._run(ys)
            nll = negative_log_likelihood(output, self.hmm)
        except Exception as e:
            self.fail(f"NLL computation failed with error: {e}")

    def test_nll_equals_negative_log_likelihood(self):
        ys = jnp.array([[0.5], [1.5], [2.5]])
        output = self._run(ys)
        nll = negative_log_likelihood(output, self.hmm)
        self.assertTrue(jnp.isclose(nll, -output.log_likelihood()))

    def test_observations_at_state_means_have_lower_nll(self):
        ys_near_means = jnp.array([[0.0], [1.0], [2.0]])
        ys_far_from_means = jnp.array([[10.0], [10.0], [10.0]])

        nll_near = negative_log_likelihood(self._run(ys_near_means), self.hmm)
        nll_far = negative_log_likelihood(self._run(ys_far_from_means), self.hmm)

        self.assertTrue(nll_near < nll_far)
