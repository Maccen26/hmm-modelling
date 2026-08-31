from unittest import TestCase
from src.api.v4 import HMM  , StaticTransition, GaussEmission, AutoregressiveGaussEmission
from src.api.v4.algorithms import ForwardAlgorithm
import jax.numpy as jnp
import jax

class TestBaseHMM(TestCase):
    __test__ = False
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
            hmm = HMM(transition=self.transition_matrix, emission=self.emission)
        except Exception as e:
            self.fail(f"Building HMM failed with error: {e}") 
    
    def test_hmm_params_access(self):
        hmm = HMM(transition=self.transition_matrix, emission=self.emission)
        try:
            transition = hmm.transition
            emission = hmm.emission
        except Exception as e:
            self.fail(f"Accessing HMM parameters failed with error: {e}")

        #self.assertIsInstance(transition, StaticTransition)
        #self.assertIsInstance(emission, GaussEmission)  

    def test_update_param(self): 
        hmm = HMM(transition=self.transition_matrix, emission=self.emission)
        param_before = hmm.params
        try:
            hmm.update_param("mu0", 5.0)
        except Exception as e:
            self.fail(f"Updating HMM parameter failed with error: {e}")

        self.assertTrue(jnp.allclose(hmm.params.emission.mu0, 5)) 
        self.assertNotEqual(hmm.params, param_before)

    def test_default_hmm_fit(self):
        # This is a placeholder for a future test that would check if the HMM can be fitted to data
        ys = jnp.array([0.0, 1.0, 2.0])
        xs = None  # No covariates
        hmm = HMM(transition=self.transition_matrix, emission=self.emission)
        try:
            # This will eventually call a fit method that we haven't implemented yet
            hmm.fit(ys, xs)
        except Exception as e:
            self.fail(f"Fitting HMM to data failed with error: {e}")

    def test_custom_initial_distribution_is_stored(self):
        custom_u0 = jnp.array([0.5, 0.3, 0.2])
        hmm = HMM(transition=self.transition_matrix, emission=self.emission, inital_distribution=custom_u0)
        self.assertTrue(jnp.allclose(hmm.u_pre, custom_u0.reshape(1, -1)))

    def test_set_inference_algorithm_returns_forward_algorithm(self):
        hmm = HMM(transition=self.transition_matrix, emission=self.emission)
        inference_alg = hmm._set_inference_algorithm("forward")
        self.assertIsInstance(inference_alg, ForwardAlgorithm)

    def test_fit_improves_log_likelihood(self):
        import jax
        ys = jax.random.normal(jax.random.PRNGKey(42), shape=(100, ))
        hmm = HMM(transition=self.transition_matrix, emission=self.emission)
        ll_before = hmm.log_likelihood(ys)
        hmm.fit(ys)
        ll_after = hmm.log_likelihood(ys)
        self.assertGreater(ll_after, ll_before)

    def test_mu0_stays_constant_during_fit(self):
        ys = jax.random.normal(jax.random.PRNGKey(42), shape=(100, ))
        hmm = HMM(transition=self.transition_matrix, emission=self.emission)
        mu0_before = hmm.emission.mu0
        hmm.fit(ys, frozen={"mu0": False})
        mu0_after = hmm.emission.mu0
        self.assertTrue(jnp.allclose(mu0_before, mu0_after))



class TestStaticHMM(TestBaseHMM):
    __test__ = True 
    def setUp(self) -> None:
        return super().setUp() 
    

class TestAutoRegressiveHMM(TestBaseHMM):
    __test__ = True 
    def setUp(self) -> None:
        self.transition_logits = jnp.array([[-0.84729786, -1.94591015], 
                                            [-2.07944154, -1.09861229], 
                                            [-1.60943791, -1.09861229]]) 
        
        self.transition_matrix = StaticTransition(self.transition_logits)
        
        self.emission_mean = jnp.array([0.0, 1.0, 2.0]) 
        self.emission_sigma = jnp.array([1.0, 1.0, 1.0])
        self.phi = jnp.array([0.5, 0.5, 0.5])  # AR(1) coefficient for each state
        self.emission = AutoregressiveGaussEmission.from_params(self.emission_mean, self.emission_sigma, self.phi) 



