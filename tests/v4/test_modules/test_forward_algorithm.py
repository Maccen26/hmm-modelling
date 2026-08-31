from unittest import TestCase 
from src.api.v4 import ForwardAlgorithm, StaticTransition, GaussEmission, HMMParams
import jax.numpy as jnp

class TestForwardAlgorithm(TestCase):
    def setUp(self) -> None:
        self.transition_logits = jnp.array([[-0.84729786, -1.94591015], 
                                            [-2.07944154, -1.09861229], 
                                            [-1.60943791, -1.09861229]]) 
        
        self.transition_matrix = StaticTransition(self.transition_logits)
        
        self.emission_mean = jnp.array([0.0, 1.0, 2.0]) 
        self.emission_sigma = jnp.array([1.0, 1.0, 1.0])
        self.emission = GaussEmission.from_params(self.emission_mean, self.emission_sigma) 

        self.hmm_params = HMMParams(transition=self.transition_matrix, emission=self.emission)

    def test_build_forward_algorithm(self):
        try:
            forward_alg = ForwardAlgorithm()
        except Exception as e:
            self.fail(f"Building ForwardAlgorithm failed with error: {e}") 

    def test_forward_algorithm_step(self):
        forward_alg = ForwardAlgorithm()
        ys = jnp.array([[0.0], [1.0], [2.0]])  # 3 timesteps, 1-dimensional obs
        xs = None  # No covariates

        carry_0 = jnp.array([[1.0, 0.0, 0.0]])  # Initial forward variable (start in state 0 with prob 1)
        try:
            carry_1, output_1 = forward_alg.step(self.hmm_params, carry_0, t=0, ys=ys, xs=xs)
            carry_2, output_2 = forward_alg.step(self.hmm_params,carry_1, t=1, ys=ys, xs=xs)
            carry_3, output_3 = forward_alg.step(self.hmm_params,carry_2, t=2, ys=ys, xs=xs)
        except Exception as e:
            self.fail(f"Forward algorithm step failed with error: {e}")

        # Check output shapes
        self.assertTrue(carry_1.shape == (1, 3))  # Should be a vector of length num_states
        self.assertTrue(carry_2.shape == (1, 3))
        self.assertTrue(carry_3.shape == (1, 3))
        self.assertTrue(isinstance(output_1, float) or isinstance(output_1, tuple))  # f_t should be a scalar
        self.assertTrue(isinstance(output_2, float) or isinstance(output_2, tuple))
        self.assertTrue(isinstance(output_3, float) or isinstance(output_3, tuple))

    def test_forward_run(self):
        forward_alg = ForwardAlgorithm()
        ys = jnp.array([[0.0], [1.0], [2.0]])  # 3 timesteps, 1-dimensional obs
        u0 = jnp.array([[1.0, 0.0, 0.0]])  # Initial forward variable (start in state 0 with prob 1)
        xs = None  # No covariates

        try:
            outputs = forward_alg.run(self.hmm_params,u0, ys, xs)
        except Exception as e:
            self.fail(f"Forward algorithm run failed with error: {e}")

        # Check final carry shape and outputs length
        #self.assertTrue(carry.shape == (1, 3))  # Final forward variable should be a vector of length num_states
    def test_forward_run_output_length(self):
        forward_alg = ForwardAlgorithm()
        ys = jnp.array([[0.0], [1.0], [2.0]])  # 3 timesteps, 1-dimensional obs
        xs = None  # No covariates
        u0 = jnp.array([[1.0, 0.0, 0.0]])  # Initial forward variable (start in state 0 with prob 1)

        forward_output = forward_alg.run(self.hmm_params,u0, ys, xs)

        self.assertTrue(len(forward_output.ft) == len(ys))  # Should have one output per timestep

    def test_forward_variable_sums_to_one(self):
        forward_alg = ForwardAlgorithm()
        ys = jnp.array([[0.0], [1.0], [2.0]])
        u0 = jnp.array([[1.0, 0.0, 0.0]])
        xs = None

        forward_output = forward_alg.run(self.hmm_params, u0, ys, xs)

        # Each u_tt is a probability distribution and must sum to 1
        row_sums = forward_output.utt.sum(axis=-1)
        self.assertTrue(jnp.allclose(row_sums, 1.0, atol=1e-5))

    def test_ft_values_are_positive(self):
        forward_alg = ForwardAlgorithm()
        ys = jnp.array([[0.0], [1.0], [2.0]])
        u0 = jnp.array([[1.0, 0.0, 0.0]])
        xs = None

        forward_output = forward_alg.run(self.hmm_params,u0, ys, xs)

        self.assertTrue(jnp.all(forward_output.ft > 0))
