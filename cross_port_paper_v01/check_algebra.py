"""Finite checks of previous KIRC equations, not a trained SAR experiment.

The equalities and counterexamples are analytic; these checks only verify the
particular numerical examples. No originality or empirical performance claim.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def main() -> None:
    rng = np.random.default_rng(20260919)
    # Even estimated logits telescope if the same values are used in subtraction.
    ls, lsv, lsvk = (rng.normal(size=(1000, 6)) for _ in range(3))
    composed_log = ls + (lsv - ls) + (lsvk - lsv)
    telescoping_error = float(np.max(np.abs(composed_log - lsvk)))
    assert telescoping_error < 1e-12

    # Identical source and target observed distributions: infinitely many
    # semantic allocations are compatible with the same closure equality.
    source = np.array([0.5, 0.5])
    target = source.copy()
    ratio = source / target
    mixtures = []
    for beta in (0.0, 0.2, 0.5, 0.8, 1.0):
        residual = 1.0 - beta * ratio
        assert np.all(residual >= 0)
        assert np.allclose(beta * ratio + residual, 1.0)
        mixtures.append({'known_mass': beta, 'residual': residual.tolist()})

    # A knowledge variable independent of category, but port-dependent.
    # The increment is nonzero and identical for all classes, hence cancels in
    # fixed-weight, pointwise normalized class scores.
    pk_source, pk_target = 0.8, 0.2
    delta = np.array([np.log(pk_source / pk_target)] * 3)
    visual_scores = np.array([0.55, 0.30, 0.15])
    fused_scores = visual_scores * np.exp(delta)
    fused_scores /= fused_scores.sum()
    no_class_change_error = float(np.max(np.abs(fused_scores - visual_scores)))
    assert no_class_change_error < 1e-12

    # Even with the exact ratio R=1 and all observations known, regularized
    # closure can give positive unknown score. Set port/group penalties to 0.
    # J(beta,rho)=phi(beta+rho)+lambda_u*rho+eta*beta^2/2.
    lam, eta = 0.1, 1.0
    threshold = 1 / (1 + lam)
    beta_star = lam / eta
    assert beta_star <= threshold
    rho_star = threshold - beta_star
    unknown_score = rho_star / (beta_star + rho_star)
    phi = lambda value: value - np.log(value) - 1
    objective = float(phi(beta_star + rho_star) + lam * rho_star + eta * beta_star**2 / 2)
    grid_beta = np.linspace(0, 1, 10001)
    grid_rho = np.maximum(threshold - grid_beta, 0)
    grid_value = phi(grid_beta + grid_rho) + lam * grid_rho + eta * grid_beta**2 / 2
    assert abs(float(grid_value.min()) - objective) < 1e-10

    results = {
        'scope': 'Algebra checks and counterexamples only; no SAR model training.',
        'seed': 20260919,
        'telescoping_max_absolute_log_error': telescoping_error,
        'nonidentifiable_closure_examples': mixtures,
        'nonzero_knowledge_increment_without_class_discrimination': {
            'increment_for_each_class': delta.tolist(),
            'before': visual_scores.tolist(),
            'after': fused_scores.tolist(),
            'max_difference': no_class_change_error,
            'scope': 'Fixed mixture weights; this proves no pointwise class information, not invariance of every jointly refitted procedure.'
        },
        'penalty_selected_unknown_with_all_known_data': {
            'exact_ratio': 1.0,
            'true_known_fraction_in_constructed_world': 1.0,
            'lambda_unknown': lam,
            'eta': eta,
            'optimizer_known_mass': beta_star,
            'optimizer_residual': rho_star,
            'normalized_unknown_score': unknown_score,
            'objective': objective,
            'interpretation': 'Unique convex optimization solution is not statistical identification of unknown class mass.'
        },
        'not_established': [
            'A new algorithm with three independent innovations.',
            'Novel-class posterior calibration.',
            'Performance on the new AIS-free full-scene task.',
            'Impossibility of useful port knowledge or of a revised algorithm.'
        ]
    }
    out = Path(__file__).with_name('ALGEBRA_CHECKS.json')
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
