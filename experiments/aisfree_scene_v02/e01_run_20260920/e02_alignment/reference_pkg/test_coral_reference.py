import unittest
import numpy as np
from coral_reference import (fit_target_to_source, fit_source_to_target,
                             symmetric_power)

class Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        a = np.array([[2., .4], [.3, 1.]])
        b = np.array([[1., -.6], [.4, 2.5]])
        self.s = rng.standard_normal((300, 2)) @ a + [3., -1.]
        self.t = rng.standard_normal((350, 2)) @ b + [-2., 4.]

    def test_identity_on_identical_domain(self):
        for ridge in (0., 1e-6, 1.):
            tr = fit_target_to_source(self.s, self.s, ridge)
            np.testing.assert_allclose(tr.transform(self.s), self.s, atol=1e-10)

    def test_target_to_source_empirical_covariance(self):
        tr = fit_target_to_source(self.s, self.t, 0.)
        out = tr.transform(self.t)
        np.testing.assert_allclose(np.cov(out, rowvar=False),
                                   np.cov(self.s, rowvar=False), atol=1e-10)
        np.testing.assert_allclose(out.mean(0), self.s.mean(0), atol=1e-10)

    def test_source_to_target_direction(self):
        tr = fit_source_to_target(self.s, self.t, 0.)
        out = tr.transform(self.s)
        np.testing.assert_allclose(np.cov(out, rowvar=False),
                                   np.cov(self.t, rowvar=False), atol=1e-10)

    def test_regularized_covariance_identity(self):
        tr = fit_target_to_source(self.s, self.t, .1)
        self.assertLess(tr.regularized_covariance_residual(), 1e-10)

    def test_old_formula_breaks_identity(self):
        c = np.cov(self.s, rowvar=False) + 1e-6*np.eye(2)
        wrong = symmetric_power(c, -.5) @ symmetric_power(c, -.5)
        out = (self.s - self.s.mean(0)) @ wrong + self.s.mean(0)
        self.assertGreater(np.max(np.abs(out-self.s)), .1)

    def test_noncommuting_order_is_important(self):
        cs, ct = np.cov(self.s, rowvar=False), np.cov(self.t, rowvar=False)
        wrong_order = symmetric_power(cs, .5) @ symmetric_power(ct, -.5)
        residual = np.linalg.norm(wrong_order.T @ ct @ wrong_order - cs)
        self.assertGreater(residual, .01)

    def test_eval_does_not_refit(self):
        tr = fit_target_to_source(self.s, self.t)
        before = tr.matrix.copy()
        x = self.t[:20] + [20., -10.]
        out = tr.transform(x)
        np.testing.assert_array_equal(tr.matrix, before)
        np.testing.assert_allclose(out, (x-self.t.mean(0))@before+self.s.mean(0))

    def test_invalid_inputs(self):
        for value in (np.ones((1,2)), np.array([[1.,np.nan],[2.,3.]])):
            with self.assertRaises(ValueError):
                fit_target_to_source(value, self.t)
        with self.assertRaises(ValueError):
            fit_target_to_source(self.s, self.t, -1.)
        with self.assertRaises(ValueError):
            fit_target_to_source(self.s, np.ones((10,3)))
        with self.assertRaises(ValueError):
            fit_target_to_source(self.s, self.t).transform(np.ones((2,3)))

    def test_rank_deficiency_needs_ridge(self):
        s, t = np.ones((20,2)), np.ones((30,2))*2
        with self.assertRaises(ValueError):
            fit_target_to_source(s, t, 0.)
        tr = fit_target_to_source(s, t, 1e-6)
        self.assertTrue(np.isfinite(tr.transform(t)).all())

if __name__ == '__main__':
    unittest.main(verbosity=2)
