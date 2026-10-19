import jax
jax.config.update("jax_enable_x64", True)

import unittest
import equinox as eqx
import jax.numpy as jnp
from src.api.v5.solvers import Freezer, GradientSolver, LBFGSSolver, SolverResult


class Toy(eqx.Module):
    a: jnp.ndarray
    b: jnp.ndarray


def squared_error(params: Toy, data) -> jnp.ndarray:
    ta, tb = data
    return jnp.sum((params.a - ta) ** 2) + jnp.sum((params.b - tb) ** 2)


INIT = Toy(a=jnp.array([0.0, 0.0, 0.0]), b=jnp.array(0.0))
DATA = (jnp.array([1.0, -2.0, 3.0]), jnp.array(4.0))


class TestFreezer(unittest.TestCase):

    def test_filter_spec_marks_whole_leaf_frozen(self):
        spec = Freezer([("b", None)]).filter_spec(INIT)
        self.assertTrue(spec.a)
        self.assertFalse(spec.b)

    def test_filter_spec_without_frozen_trains_everything(self):
        spec = Freezer(None).filter_spec(INIT)
        self.assertTrue(spec.a and spec.b)

    def test_pin_resets_listed_elements_only(self):
        moved = Toy(a=jnp.array([5.0, 5.0, 5.0]), b=jnp.array(5.0))
        pinned = Freezer([("a", 0), ("a", (2,))]).pin(moved, INIT)
        self.assertTrue(jnp.array_equal(pinned.a, jnp.array([0.0, 5.0, 0.0])))
        self.assertEqual(float(pinned.b), 5.0)

    def test_pin_blocks_gradient(self):
        freezer = Freezer([("a", 1)])
        grads = jax.grad(lambda p: squared_error(freezer.pin(p, INIT), DATA))(INIT)
        self.assertEqual(float(grads.a[1]), 0.0)
        self.assertNotEqual(float(grads.a[0]), 0.0)


class SolverContract:
    """Shared tests run against every solver."""

    def make_solver(self):
        raise NotImplementedError

    def fit(self, frozen=None) -> SolverResult:
        return self.make_solver().fit(INIT, squared_error, DATA, frozen=frozen)

    def test_converges_to_targets(self):
        result = self.fit()
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.loss, 0.0, places=6)
        self.assertTrue(jnp.allclose(result.params.a, DATA[0], atol=1e-4))

    def test_whole_leaf_freeze(self):
        result = self.fit([("b", None)])
        self.assertEqual(float(result.params.b), 0.0)
        self.assertTrue(jnp.allclose(result.params.a, DATA[0], atol=1e-4))

    def test_element_freeze(self):
        result = self.fit([("a", (1,))])
        self.assertEqual(float(result.params.a[1]), 0.0)
        self.assertAlmostEqual(float(result.params.a[0]), 1.0, places=4)
        self.assertAlmostEqual(float(result.params.b), 4.0, places=4)

    def test_empty_frozen_equals_none(self):
        self.assertTrue(jnp.allclose(self.fit([]).params.a, self.fit(None).params.a))


class TestLBFGSSolver(SolverContract, unittest.TestCase):
    def make_solver(self):
        return LBFGSSolver()


class TestGradientSolver(SolverContract, unittest.TestCase):
    def make_solver(self):
        import optax
        return GradientSolver(optimizer=optax.adam(1e-1), n_iter=5000, tol=1e-6)


if __name__ == "__main__":
    unittest.main()
