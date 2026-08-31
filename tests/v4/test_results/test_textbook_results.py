from unittest import TestCase
import jax
import jax.numpy as jnp
from src.api.v4 import HMM, StaticTransition, GaussEmission, AutoregressiveGaussEmission
from src.api.v4.utils import load_y_data


class TestModelsYieldJansLikelihood(TestCase):

    def setUp(self) -> None:
        self.results = {
            "static": {
                "3_state": 
                    -10602.0,
                "4_state": 
                    -10389.0
            },
            "autoregressive": {
                "4_state": 
                    -9354  # Placeholder value, replace with actual result from Jans book
            }
        }

        self.ys = load_y_data() 

        self.ERROR_TOLERANCE = 1.0  # Allowable error tolerance for log-likelihood comparisons




    def _make_static_hmm(self, result_type: int) -> HMM:  
        """
        Page 329 to 332 in Jans Book
        """
        if (result_type == 4): 
            mean = jnp.array([400, 536, 890, 1307]) 
            sd = jnp.array([22, 77, 111, 197])  
            u0 = jnp.array([0.19, 0.35, 0.15, 0.32])
            Gamma = jnp.array([
            [8.5e-01, 1.5e-01, 0.0028, 9.7e-09],
            [8.3e-02, 8.5e-01, 0.0679, 6.8e-09],
            [4.3e-04, 1.6e-01, 0.6842, 1.5e-01],
            [2.6e-08, 6.1e-09, 0.0723, 9.3e-01],]
            )

        elif (result_type == 3):
            mean = jnp.array([400, 547, 1188]) 
            sd = jnp.array([24, 85, 252])
            u0 = jnp.array([0.2, 0.35, 0.45])

            Gamma = jnp.array([
                [8.5e-01, 0.15,    4.8e-51],
                [8.2e-02, 0.86,    6.3e-02],
                [3.4e-09, 0.05,    9.5e-01],])
            
        emission = GaussEmission.from_params(mean, sd)
        transition = StaticTransition.from_params(Gamma) 
        hmm = HMM(transition=transition, emission=emission, inital_distribution=u0) 
        return hmm 
    
    def _make_autoregressive_hmm(self, result_type: int) -> HMM:
        if (result_type == 1): 
            mean = jnp.array([400, 453, 760, 1590]) 
            sd = jnp.array([13, 42, 108, 72]) 
            u0 = jnp.array([0.29, 0.21, 0.17, 0.34]) 
            Gamma = jnp.array([
            [8.2e-01, 0.0946, 0.087, 9.3e-09],
            [1.8e-01, 0.6875, 0.136, 5.5e-09],
            [9.6e-02, 0.2104, 0.414, 2.8e-01],
            [2.3e-08, 0.0049, 0.133, 8.6e-01],]
            )
            phi = jnp.array([0.97, 0.30, 0.62, 0.85])
        
        hmm = HMM(transition=StaticTransition.from_params(Gamma), emission=AutoregressiveGaussEmission.from_params(mean, sd, phi), inital_distribution=u0)
        return hmm




    def test_setup(self):
        try:
            hmm = self._make_static_hmm(result_type=3) 
            hmm = self._make_static_hmm(result_type=4)
        except Exception as e:
            self.fail(f"Setting up HMM for textbook results failed with error: {e}") 

    def test_3_state_hmm(self):
        hmm = self._make_static_hmm(result_type=3)  
        self.assertAlmostEqual(hmm.log_likelihood(self.ys), float(self.results["static"]["3_state"]), delta=self.ERROR_TOLERANCE) 

    def test_4_state_hmm(self):
        hmm = self._make_static_hmm(result_type=4)  
        self.assertAlmostEqual(hmm.log_likelihood(self.ys), float(self.results["static"]["4_state"]), delta=self.ERROR_TOLERANCE)


    def test_4_state_autoregressive_hmm_lag_1(self):
        # This is a placeholder for a future test that would check the 4-state autoregressive HMM results from Jans book
        hmm = self._make_autoregressive_hmm(result_type=1)
        self.assertAlmostEqual(hmm.log_likelihood(self.ys), float(self.results["autoregressive"]["4_state"]), delta=self.ERROR_TOLERANCE)
            

        



class TestPhiIsCovergent(TestCase):

    def _make_autoregressive_hmm(self, result_type: int) -> HMM:
        if (result_type == 1):
            mean = jnp.array([400.0, 453.0, 760.0, 1590.0])
            sd = jnp.array([13.0, 42.0, 108.0, 72.0])
            u0 = jnp.array([0.29, 0.21, 0.17, 0.34])
            Gamma = jnp.array([
            [8.2e-01, 0.0946, 0.087, 9.3e-09],
            [1.8e-01, 0.6875, 0.136, 5.5e-09],
            [9.6e-02, 0.2104, 0.414, 2.8e-01],
            [2.3e-08, 0.0049, 0.133, 8.6e-01],]
            )
            phi = jnp.array([0.97, 0.30, 0.62, 0.85])
        
        hmm = HMM(transition=StaticTransition.from_params(Gamma), emission=AutoregressiveGaussEmission.from_params(mean, sd, phi), inital_distribution=u0)
        return hmm
    
    def setUp(self) -> None:
        self.hmm = self._make_autoregressive_hmm(result_type=1) 
        self.ys = load_y_data() 

    def test_phi_convergence(self):
        """After fitting, all phi values should be below 0.99 (stationary AR process, phi < 1)."""
        from src.api.v4.solvers import LBFGSSolver
        self.hmm.fit(self.ys, solver=LBFGSSolver())
        phi = self.hmm.emission.phi()
        self.assertTrue(
            jnp.all(phi < 0.985),
            f"All phi values should be < 0.98 after fitting, got {phi}"
        )

        self.assertTrue(
            jnp.all(phi > -0.985),
            f"All phi values should be > -0.985 after fitting, got {phi}"
        )


    def test_phi_convergence_2_lags(self):
        """With 2 lags, sum of phi per state should be < 1 (stationarity condition)."""
        mean = jnp.array([400.0, 453.0, 760.0, 1590.0])
        sd = jnp.array([13.0, 42.0, 108.0, 72.0])
        u0 = jnp.array([0.29, 0.21, 0.17, 0.34])
        Gamma = jnp.array([
            [8.2e-01, 0.0946, 0.087, 9.3e-09],
            [1.8e-01, 0.6875, 0.136, 5.5e-09],
            [9.6e-02, 0.2104, 0.414, 2.8e-01],
            [2.3e-08, 0.0049, 0.133, 8.6e-01],
        ])
        phi_2lag = jnp.array([
            [0.5, 0.2, 0.4, 0.5],   # lag-1 coefficients per state
            [0.3, 0.1, 0.2, 0.3],   # lag-2 coefficients per state
        ])  # shape (2, 4)
        emission = AutoregressiveGaussEmission.from_params(mean, sd, phi_2lag)
        hmm = HMM(
            transition=StaticTransition.from_params(Gamma),
            emission=emission,
            inital_distribution=u0,
        )
        from src.api.v4.solvers import LBFGSSolver
        hmm.fit(self.ys, solver=LBFGSSolver())
        phi = hmm.emission.phi()               # shape (2, 4)
        phi_sum_per_state = jnp.sum(phi, axis=0)  # sum over lags for each state
        self.assertTrue(
            jnp.all(phi_sum_per_state < 1.0),
            f"Sum of phi per state should be < 1.0 (stationarity), got {phi_sum_per_state}"
        ) 

        self.assertTrue(
            jnp.all(phi_sum_per_state > -1.0),
            f"Sum of phi per state should be > -1.0 (stationarity), got {phi_sum_per_state}"
        )


#    def test_phi_convergence_random_init(self):
#        """With randomly initialized params, phi should still converge below 0.97 after fitting."""
#        key = jax.random.PRNGKey(42)
#        k1, k2, k3, k4 = jax.random.split(key, 4)
#        mean = jnp.sort(400.0 + jax.random.uniform(k1, shape=(4,)) * 1200.0)
#        sd = 10.0 + jax.random.uniform(k2, shape=(4,)) * 200.0
#        phi_rand = jax.random.uniform(k3, shape=(4,)) * 0.9   # random in (0, 0.9)
#        Gamma_rand = jax.nn.softmax(jax.random.normal(k4, shape=(4, 4)), axis=1)
#        u0 = jnp.ones(4) / 4.0
#        emission = AutoregressiveGaussEmission.from_params(mean, sd, phi_rand)
#        hmm = HMM(
#            transition=StaticTransition.from_params(Gamma_rand),
#            emission=emission,
#            inital_distribution=u0,
#        )
#        hmm.fit(self.ys)
#        phi = hmm.emission.phi()
#        self.assertTrue(
#            jnp.all(phi < 0.985),
#            f"All phi values should be < 0.985 after fitting (random init), got {phi}"
#        )
#
#        self.assertTrue(
#            jnp.all(phi > -0.985),
#            f"All phi values should be > -0.985 after fitting (random init), got {phi}"
#        )

    def test_phi_convergence_notebooks_from_params(self): 
        RUN_TO_LOAD = {
        "tag": "week_2",
        "run": 1
        }
        from src.data import load_model_and_data 
        params, _ , _= load_model_and_data("ar_hmm_model", tag=RUN_TO_LOAD["tag"], run=RUN_TO_LOAD["run"])
        phi_clipped = jnp.clip(params.phi, -0.999, 0.999)
        emission = AutoregressiveGaussEmission.from_params(params.mu, params.sigma, phi_clipped)
        hmm = HMM(
            transition=StaticTransition(params.transition_logits),
            emission=emission,
            inital_distribution=jnp.array([0.29, 0.21, 0.17, 0.34]),
        )

        self.hmm.fit(self.ys, frozen={"mu0":False})
        phi = self.hmm.emission.phi()
        self.assertTrue(
            jnp.all(phi < 0.985),
            f"All phi values should be < 0.985 after fitting (notebook params), got {phi}"
        )

        self.assertTrue(
            jnp.all(phi > -0.985),
            f"All phi values should be > -0.985 after fitting (notebook params), got {phi}"
        )   


    def test_phi_convergence_static_mu0(self):
        """After fitting, all phi values should be below 0.99 (stationary AR process, phi < 1)."""
        from src.api.v4.solvers import LBFGSSolver
        self.hmm.fit(self.ys, solver=LBFGSSolver(), frozen={"mu0":False})
        phi = self.hmm.emission.phi()
        self.assertTrue(
            jnp.all(phi < 0.985),
            f"All phi values should be < 0.98 after fitting, got {phi}"
        )
        self.assertTrue(
            jnp.all(phi > -0.985),
            f"All phi values should be > -0.985 after fitting, got {phi}"
        )


    def test_phi_convergence_2_lags_static_mu0(self):
        """With 2 lags, sum of phi per state should be < 1 (stationarity condition)."""
        mean = jnp.array([400.0, 453.0, 760.0, 1590.0])
        sd = jnp.array([13.0, 42.0, 108.0, 72.0])
        u0 = jnp.array([0.29, 0.21, 0.17, 0.34])
        Gamma = jnp.array([
            [8.2e-01, 0.0946, 0.087, 9.3e-09],
            [1.8e-01, 0.6875, 0.136, 5.5e-09],
            [9.6e-02, 0.2104, 0.414, 2.8e-01],
            [2.3e-08, 0.0049, 0.133, 8.6e-01],
        ])
        phi_2lag = jnp.array([
            [0.5, 0.2, 0.4, 0.5],   # lag-1 coefficients per state
            [0.3, 0.1, 0.2, 0.3],   # lag-2 coefficients per state
        ])  # shape (2, 4)
        emission = AutoregressiveGaussEmission.from_params(mean, sd, phi_2lag)
        hmm = HMM(
            transition=StaticTransition.from_params(Gamma),
            emission=emission,
            inital_distribution=u0,
        )
        from src.api.v4.solvers import LBFGSSolver
        hmm.fit(self.ys, solver=LBFGSSolver(), frozen={"mu0":False})
        phi = hmm.emission.phi()               # shape (2, 4)
        phi_sum_per_state = jnp.sum(phi, axis=0)  # sum over lags for each state
        self.assertTrue(
            jnp.all(phi_sum_per_state < 1.0),
            f"Sum of phi per state should be < 1.0 (stationarity), got {phi_sum_per_state}"
        )
        self.assertTrue(
            jnp.all(phi_sum_per_state > -1.0),
            f"Sum of phi per state should be > -1.0 (stationarity), got {phi_sum_per_state}"
        )

#    def test_phi_convergence_random_init_static_mu0(self):
#        """With randomly initialized params, phi should still converge below 0.97 after fitting."""
#        key = jax.random.PRNGKey(42)
#        k1, k2, k3, k4 = jax.random.split(key, 4)
#        mean = jnp.sort(400.0 + jax.random.uniform(k1, shape=(4,)) * 1200.0)
#        sd = 10.0 + jax.random.uniform(k2, shape=(4,)) * 200.0
#        phi_rand = jax.random.uniform(k3, shape=(4,)) * 0.9   # random in (0, 0.9)
#        Gamma_rand = jax.nn.softmax(jax.random.normal(k4, shape=(4, 4)), axis=1)
#        u0 = jnp.ones(4) / 4.0
#        emission = AutoregressiveGaussEmission.from_params(mean, sd, phi_rand)
#        hmm = HMM(
#            transition=StaticTransition.from_params(Gamma_rand),
#            emission=emission,
#            inital_distribution=u0,
#        )
#        hmm.fit(self.ys, frozen={"mu0":False})
#        phi = hmm.emission.phi()
#        self.assertTrue(
#            jnp.all(phi < 0.985),
#            f"All phi values should be < 0.985 after fitting (random init), got {phi}"
#        )
#        self.assertTrue(
#            jnp.all(phi > -0.985),
#            f"All phi values should be > -0.985 after fitting (random init), got {phi}"
#        )

