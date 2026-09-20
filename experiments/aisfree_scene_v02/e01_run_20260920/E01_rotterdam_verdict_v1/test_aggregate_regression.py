"""Run shape/value regressions against an E01 e01_core.py. No real labels used."""
from __future__ import annotations
import argparse
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
import numpy as np


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location('e01_core_under_test', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def inputs(N=17, H=2, J=3, C=5, D=4, B=3, seed=23):
    rng = np.random.default_rng(seed)
    phi = rng.random((N, D))
    mask = rng.random((N, D)) > .2
    g = np.arange(N, dtype=np.int64) % B
    blocks = np.asarray([f'b{i}' for i in range(N)])
    p = rng.random((H, N, C)); p /= p.sum(-1, keepdims=True)
    q = rng.random((J, N, C, D))
    return phi, mask, g, blocks, p, q


def assert_reference(test, mod, data, B, enabled=None):
    phi, mask, g, blocks, p, q = data
    result = mod.aggregate(*data, B=B, min_cell=1, min_blocks=1,
                           enabled_dimensions=enabled)
    if enabled is None:
        enabled = np.ones(phi.shape[1], bool)
    for b in range(B):
        for d in range(phi.shape[1]):
            idx = np.flatnonzero((g == b) & mask[:, d])
            test.assertEqual(int(result.counts[b, d]), len(idx))
            if not enabled[d] or not len(idx):
                test.assertFalse(bool(result.active[b, d]))
                continue
            test.assertTrue(bool(result.active[b, d]))
            test.assertAlmostEqual(result.observed[b, d], float(phi[idx, d].mean()))
            for h in range(p.shape[0]):
                for j in range(q.shape[0]):
                    # Scalar-loop oracle avoids any mixed advanced indexing.
                    expected = sum(sum(float(p[h, i, c]) * float(q[j, i, c, d])
                                       for c in range(p.shape[2])) for i in idx) / len(idx)
                    test.assertAlmostEqual(float(result.predicted[h, j, b, d]), expected, places=12)


def suite_for(mod):
    class Tests(unittest.TestCase):
        def test_numpy_shape_mechanism(self):
            q = np.arange(3 * 7 * 5 * 4).reshape(3, 7, 5, 4)
            use = np.array([1,0,1,1,0,1,0], bool)
            self.assertEqual(q[:, use, :, 2].shape, (4, 3, 5))
            self.assertEqual(q[:, use][:, :, :, 2].shape, (3, 4, 5))

        def test_selected_N_different_from_J(self):
            d = list(inputs(N=7, B=1)); d[1][:] = True
            assert_reference(self, mod, d, B=1)

        def test_selected_N_equal_J_silent_axis_swap(self):
            d = list(inputs(N=3, J=3, B=1)); d[1][:] = True
            assert_reference(self, mod, d, B=1)

        def test_many_groups_variable_masks(self):
            assert_reference(self, mod, inputs(N=29, B=4), B=4)

        def test_unavailable_values_not_in_moments(self):
            d = list(inputs(N=29, B=4))
            d[0][~d[1]] = np.nan
            for i, dim in np.argwhere(~d[1]):
                d[5][:, i, :, dim] = np.nan
            assert_reference(self, mod, d, B=4)

        def test_disabled_dimension(self):
            d = list(inputs(N=19, B=3)); d[5][:, :, :, 1] = np.nan
            assert_reference(self, mod, d, B=3, enabled=np.array([1,0,1,1], bool))

        def test_all_unavailable(self):
            d = list(inputs(N=13, B=2)); d[1][:] = False; d[0][:] = np.nan; d[5][:] = np.nan
            assert_reference(self, mod, d, B=2)
    return unittest.defaultTestLoader.loadTestsFromTestCase(Tests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--core', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite_for(load_core(args.core)))
    report = {'core': str(args.core), 'numpy': np.__version__, 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'passed': result.wasSuccessful(), 'log': log.getvalue(),
              'scope': 'Synthetic axis/value regressions only; no SAR training or server artifact validation.'}
    print(log.getvalue())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()
