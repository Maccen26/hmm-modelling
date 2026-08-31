from unittest import TestCase
from src.api.v4 import StaticTransition, GaussEmission, HMMParams
import jax.numpy as jnp 

class TestHMMParams(TestCase):
    def setUp(self) -> None:
        self.transition_logits = jnp.array([[-0.84729786, -1.94591015], 
                                            [-2.07944154, -1.09861229], 
                                            [-1.60943791, -1.09861229]]) 
        
        self.transition_matrix = StaticTransition(self.transition_logits)
        
        self.emission_mean = jnp.array([0.0, 1.0, 2.0]) 
        self.emission_sigma = jnp.array([1.0, 1.0, 1.0])
        self.emission = GaussEmission.from_params(self.emission_mean, self.emission_sigma) 

    def test_build_hmm(self):
        try:
            hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        except Exception as e:
            self.fail(f"Building HMM failed with error: {e}")  

    def test_get_params(self):
        hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        try:
            transition = hmm.transition
            emission = hmm.emission
        except Exception as e:
            self.fail(f"Getting HMM parameters failed with error: {e}") 

        self.assertIsInstance(transition, StaticTransition)
        self.assertIsInstance(emission, GaussEmission) 

    def test_is_iterable(self):
        hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        try:
            params = list(hmm)
        except Exception as e:
            self.fail(f"Iterating over HMM parameters failed with error: {e}") 

        self.assertEqual(len(params), 2)  # Should have two parameters: transition and emission
        param_names = [name for name, _ in params]
        self.assertIn('transition', param_names)
        self.assertIn('emission', param_names)

    def test_hmm_equality_is_true(self):
        hmm1 = HMMParams(transition=self.transition_matrix, emission=self.emission)
        hmm2 = HMMParams(transition=self.transition_matrix, emission=self.emission)

        self.assertEqual(hmm1, hmm2)  # Should be equal since they have the same parameters

    def test_hmm_equality_is_false(self):
        different_transition_logits = jnp.array([[-0.5, -1.0], 
                                                [-1.5, -0.5], 
                                                [-1.0, -0.5]]) 
        different_transition_matrix = StaticTransition(different_transition_logits)
        different_emission_mean = jnp.array([1.0, 2.0, 3.0]) 
        different_emission_sigma = jnp.array([1.0, 1.0, 1.0])
        different_emission = GaussEmission.from_params(different_emission_mean, different_emission_sigma) 

        hmm1 = HMMParams(transition=self.transition_matrix, emission=self.emission)
        hmm2 = HMMParams(transition=different_transition_matrix, emission=different_emission)

        self.assertNotEqual(hmm1, hmm2)  # Should not be equal since they have different parameters

    def test_update_param(self):
        hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)


        try:
            updated_hmm = hmm.update_param('transition_logits', 0, index=(0,0))
        except Exception as e:
            self.fail(f"Updating HMM parameter failed with error: {e}") 

        self.assertIsInstance(updated_hmm, HMMParams)
        self.assertEqual(updated_hmm.emission, hmm.emission)  # Emission should be unchanged
        self.assertNotEqual(updated_hmm.transition, hmm.transition)  # Transition should be updated
        self.assertNotEqual(updated_hmm, hmm)

        self.assertEqual(updated_hmm.transition.transition_logits[0,0], 0)  # The updated parameter should be changed to the new value

    def test_get_transition_params(self):
        hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        try:
            transition_matrix = hmm.transition_matrix()
        except Exception as e:
            self.fail(f"Getting transition matrix from HMM failed with error: {e}") 

        self.assertTrue(jnp.allclose(transition_matrix, self.transition_matrix.transition_matrix()))


    def test_get_emission_params(self):
        hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        ys = jnp.array([[0.0], [1.0], [2.0]])  # 3 timesteps, 1-dimensional obs
        xs = None  # No covariates
        t = 0
        try:
            density = hmm.density(t=t, ys=ys[0], xs=xs)
            cdf = hmm.cdf(t=t, ys=ys[0], xs=xs)
        except Exception as e:
            self.fail(f"Getting emission parameters from HMM failed with error: {e}")

        self.assertTrue(density.shape == (1, 3))  # Should return a vector of length num_states
        self.assertTrue(cdf.shape == (1, 3))

    def test_density_values_are_non_negative(self):
        hmm = HMMParams(transition=self.transition_matrix, emission=self.emission)
        ys = jnp.array([[0.0], [1.0], [2.0]])
        xs = None
        density = hmm.density(t=0, ys=ys[0], xs=xs)
        self.assertTrue(jnp.all(density >= 0.0))
