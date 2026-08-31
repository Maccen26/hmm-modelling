from unittest import TestCase
from src.api.v4 import StaticTransition, StaticTransitionHigherOrder
import jax.numpy as jnp 
from src.base.utils import logits_to_transition_matrix, transition_matrix_to_logits


class TestStaticTransition(TestCase):
    """
    Should test build and computation of transition matrix and step function.
    """
    
    def setUp(self) -> None:
        self.transition_logits = jnp.array([[-0.84729786, -1.94591015], 
                                            [-2.07944154, -1.09861229], 
                                            [-1.60943791, -1.09861229]]) 
        
        self.transition_matrix = logits_to_transition_matrix(self.transition_logits)
        
        self.dim = self.transition_logits.shape[0]

    def test_build_from_params(self): 
        try:
            transition = StaticTransition.from_params(self.transition_matrix)
        except Exception as e:
            self.fail(f"Building StaticTransition from params failed with error: {e}") 

    def test_build_from_logits(self):
        try:
            transition = StaticTransition(self.transition_logits)
        except Exception as e:
            self.fail(f"Building StaticTransition from logits failed with error: {e}")

    def test_iterable(self):
        transition = StaticTransition(self.transition_logits)
        try:
            params = list(transition)
        except Exception as e:
            self.fail(f"Iterating over StaticTransition parameters failed with error: {e}") 

        self.assertEqual(len(params), 1)  # Should have one parameter: transition_logits
        param_names = [name for name, _ in params]
        self.assertIn('transition_logits', param_names)

    def test_equality_is_true(self):
        transition1 = StaticTransition(self.transition_logits)
        transition2 = StaticTransition(self.transition_logits)

        self.assertEqual(transition1, transition2)  # Should be equal since they have the same parameters

    def test_equality_is_false(self):
        different_transition_logits = jnp.array([[-0.5, -1.0], 
                                                [-1.5, -0.5], 
                                                [-1.0, -0.5]]) 
        different_transition = StaticTransition(different_transition_logits)

        transition = StaticTransition(self.transition_logits)
        self.assertNotEqual(transition, different_transition)  # Should not be equal since they have different parameters

    def test_build_gives_same_transition_matrix(self):
        transition_from_logits = StaticTransition(self.transition_logits)

        transition_from_matrix = StaticTransition.from_params(self.transition_matrix)

        matrix_from_logits = transition_from_logits.transition_matrix()
        matrix_from_matrix = transition_from_matrix.transition_matrix()

        self.assertTrue(jnp.allclose(matrix_from_logits, self.transition_matrix))
        self.assertTrue(jnp.allclose(matrix_from_matrix, self.transition_matrix))

    def test_mapping_logits_to_matrix_and_back(self):
        matrix_from_logits = logits_to_transition_matrix(self.transition_logits) 
        self.assertEqual(matrix_from_logits.shape, (self.dim, self.dim)) 
        self.assertTrue(jnp.allclose(transition_matrix_to_logits(matrix_from_logits), self.transition_logits))

    def test_transtion_matrix_is_probability_matrix(self):
        transition = StaticTransition.from_params(self.transition_matrix)
        computed_transition_matrix = transition.transition_matrix()
        self.assertTrue(jnp.allclose(computed_transition_matrix.sum(axis=1), 1.0))
        self.assertTrue(jnp.all(computed_transition_matrix >= 0.0))

    def test_step_returns_same_logits_for_any_time(self):
        transition = StaticTransition(self.transition_logits)
        ys = jnp.array([[0.0], [1.0]])
        logits_t0 = transition.step(t=0, ys=ys, xs=None)
        logits_t5 = transition.step(t=5, ys=ys, xs=None)
        self.assertTrue(jnp.allclose(logits_t0, logits_t5))

    def test_transition_matrix_has_correct_shape(self):
        transition = StaticTransition(self.transition_logits)
        matrix = transition.transition_matrix()
        self.assertEqual(matrix.shape, (self.dim, self.dim))
        



class TestHigherOrderStaticTransition(TestCase):
    """
    Should test build and computation of transition matrix and step function for higher order static transition.
    """
    
    def setUp(self) -> None:
        self.transition_logits = jnp.array([[-0.84729786], 
                                            [-2.07944154], 
                                            [-1.60943791], 
                                            [-0.84729786], 
                                            ]) 
        #self.transition_matrix = logits_to_transition_matrix(self.transition_logits)
        self.dim = 4

    def test_build_from_logits(self):
        try:
            transition = StaticTransitionHigherOrder(self.transition_logits, order=2, num_states=2)
        except Exception as e:
            self.fail(f"Building StaticTransitionHigherOrder from logits failed with error: {e}")
    def test_iterable(self):
        transition = StaticTransition(self.transition_logits)
        try:
            params = list(transition)
        except Exception as e:
            self.fail(f"Iterating over StaticTransition parameters failed with error: {e}") 
        self.assertEqual(len(params), 1)  # Should have one parameter: transition_logits
        param_names = [name for name, _ in params]
        self.assertIn('transition_logits', param_names)


    def test_equality_is_true(self):
        transition1 = StaticTransitionHigherOrder(self.transition_logits, order=2, num_states=2)
        transition2 = StaticTransitionHigherOrder(self.transition_logits, order=2, num_states=2)

        self.assertEqual(transition1, transition2)  # Should be equal since they have the same parameters

    def test_equality_is_false(self):
        different_transition_logits = jnp.array([[-0.5], 
                                                [-1.5], 
                                                [-1.0],
                                                [-0.5]]) 
        
        different_transition = StaticTransitionHigherOrder(different_transition_logits, order=2, num_states=2)
        transition = StaticTransitionHigherOrder(self.transition_logits, order=2, num_states=2)
        self.assertNotEqual(transition, different_transition)  # Should not be equal since they have different parameters


    def test_transtion_matrix_is_probability_matrix(self):
        transition = StaticTransitionHigherOrder(self.transition_logits, order=2, num_states=2)
        computed_transition_matrix = transition.transition_matrix()

        self.assertTrue(jnp.allclose(computed_transition_matrix.sum(axis=1), 1.0))
        self.assertTrue(jnp.all(computed_transition_matrix >= 0.0))

    def test_step_returns_same_logits_for_any_time(self):
        transition = StaticTransitionHigherOrder(self.transition_logits, order=2, num_states=2)
        ys = jnp.array([[0.0], [1.0]])
        logits_t0 = transition.step(t=0, ys=ys, xs=None)
        logits_t5 = transition.step(t=5, ys=ys, xs=None)

        self.assertTrue(jnp.allclose(logits_t0, logits_t5))
    def test_transition_matrix_has_correct_shape(self):
        transition = StaticTransitionHigherOrder(self.transition_logits, order=2, num_states=2)
        matrix = transition.transition_matrix()
        self.assertEqual(matrix.shape, (self.dim, self.dim))

    def test_transition_matrix_respects_augmented_state_structure(self):
        """
        In a 2nd-order chain with 2 base states, augmented states are (prev, curr):
          0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        From state i, the next state's 'prev' component must equal i's 'curr' component:
          States 0,2 (curr=0) can only go to states 0,1 (next prev=0)
          States 1,3 (curr=1) can only go to states 2,3 (next prev=1)
        """
        transition = StaticTransitionHigherOrder(self.transition_logits, order=2, num_states=2)
        matrix = transition.transition_matrix()

        # States 0=(0,0) and 2=(1,0): current is 0, so next must have prev=0 → states 0 or 1
        self.assertAlmostEqual(float(matrix[0, 2]), 0.0, places=5)
        self.assertAlmostEqual(float(matrix[0, 3]), 0.0, places=5)
        self.assertAlmostEqual(float(matrix[2, 2]), 0.0, places=5)
        self.assertAlmostEqual(float(matrix[2, 3]), 0.0, places=5)

        # States 1=(0,1) and 3=(1,1): current is 1, so next must have prev=1 → states 2 or 3
        self.assertAlmostEqual(float(matrix[1, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(matrix[1, 1]), 0.0, places=5)
        self.assertAlmostEqual(float(matrix[3, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(matrix[3, 1]), 0.0, places=5)