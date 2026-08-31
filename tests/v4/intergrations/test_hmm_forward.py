from unittest import TestCase 
from src.api.v4 import ForwardAlgorithm, StaticTransition, GaussEmission, HMMParams
import jax.numpy as jnp


class TestForwardAlgorithmIntegrationBase(TestCase):
    def setUp(self) -> None:
        """
        Defining the hmm param to be used in the intergration test. 
        """
        self.transition_logits = jnp.array([[-0.84729786, -1.94591015], 
                                            [-2.07944154, -1.09861229], 
                                            [-1.60943791, -1.09861229]]) 
        
        self.transition_matrix = StaticTransition(self.transition_logits)
        
        self.emission_mean = jnp.array([0.0, 1.0, 2.0]) 
        self.emission_sigma = jnp.array([1.0, 1.0, 1.0])

        self.emission = GaussEmission.from_params(self.emission_mean, self.emission_sigma) 

        self.hmm_params = HMMParams(transition=self.transition_matrix, emission=self.emission)

    def test_forward_algorithm_integration(self):
        forward_alg = ForwardAlgorithm()
        
        ys = jnp.array([[0.0], [1.0], [2.0]])  # 3 timesteps, 1-dimensional obs
        u0 = jnp.array([[1.0, 0.0, 0.0]])  # Initial forward variable (start in state 0 with prob 1)
        xs = None  # No covariates

        try:
            outputs = forward_alg.run(self.hmm_params,u0, ys, xs)
        except Exception as e:
            self.fail(f"Forward algorithm integration test failed with error: {e}")