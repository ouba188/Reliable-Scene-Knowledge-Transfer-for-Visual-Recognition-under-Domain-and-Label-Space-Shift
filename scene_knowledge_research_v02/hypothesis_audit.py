"""Finite-hypothesis relation compatibility audit.

Research scaffold only. Not a trained SAR model, a novel-mathematics claim,
or an unknown-class posterior estimator. filter_hypotheses receives no labels.
A single relation-mechanism index must explain ALL observed visual groups.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

@dataclass(frozen=True)
class FilterResult:
    feasible_indices: tuple[int, ...]
    returned_indices: tuple[int, ...]
    mechanism_conflict: bool
    excess_by_hypothesis: tuple[float, ...]


def filter_hypotheses(observed: np.ndarray, predicted: np.ndarray,
                      tolerance: float | np.ndarray,
                      observed_mask: np.ndarray | None = None) -> FilterResult:
    """Test target relation moments against a FIXED finite hypothesis bank.

    observed: (visual_groups, relation_moments)
    predicted: (hypotheses, mechanism_worlds, visual_groups, relation_moments)
    tolerance: scalar or broadcastable to observed.shape
    observed_mask: True only where relation measurement is available.

    The tolerance must be fixed using source-development/calibration data or a
    declared statistical model, never target classification labels. No coverage
    guarantee follows from running this function alone. If no hypothesis passes,
    return the entire original bank AND a conflict flag; do not call it a new class.
    """
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if obs.ndim != 2 or pred.ndim != 4 or pred.shape[2:] != obs.shape:
        raise ValueError('Expected observed[B,D], predicted[H,J,B,D].')
    if pred.shape[0] < 1 or pred.shape[1] < 1:
        raise ValueError('At least one hypothesis and one mechanism required.')
    mask = np.ones(obs.shape, bool) if observed_mask is None else np.asarray(observed_mask, bool)
    if mask.shape != obs.shape:
        raise ValueError('Mask shape must match observed.')
    tol = np.broadcast_to(np.asarray(tolerance, dtype=float), obs.shape)
    if np.any(tol < 0) or np.isnan(tol).any():
        raise ValueError('Tolerance must be nonnegative and not NaN.')
    if not np.isfinite(obs[mask]).all() or not np.isfinite(pred[..., mask]).all():
        raise ValueError('Active moments must be finite.')
    all_indices = tuple(range(pred.shape[0]))
    if not mask.any():
        return FilterResult(all_indices, all_indices, False, tuple(0.0 for _ in all_indices))
    excess = np.abs(pred - obs[None, None]) - tol[None, None]
    # First require ONE mechanism to fit EVERY active group/moment, then allow
    # an existential choice over the frozen mechanism bank. Do not swap these.
    worst_by_world = excess[..., mask].max(axis=-1)  # [H,J]
    best_world = worst_by_world.min(axis=1)           # [H]
    feasible = tuple(np.flatnonzero(best_world <= 1e-12).tolist())
    return FilterResult(feasible, feasible if feasible else all_indices,
                        not bool(feasible), tuple(map(float, best_world)))


def label_sets(probabilities: np.ndarray, indices: tuple[int, ...]) -> list[list[int]]:
    """Return possible argmax labels; a singleton is agreement, NOT proof of truth."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 3 or not indices or not np.isfinite(p).all():
        raise ValueError('Expected finite probabilities[H,N,C] and retained indices.')
    if np.any(p < 0) or not np.allclose(p.sum(-1), 1):
        raise ValueError('Each probability vector must sum to one.')
    labels = p[list(indices)].argmax(-1)
    return [np.unique(labels[:, i]).tolist() for i in range(p.shape[1])]


def as_dict(r: FilterResult) -> dict:
    return dict(feasible=list(r.feasible_indices), returned=list(r.returned_indices),
                mechanism_conflict=r.mechanism_conflict,
                excess_by_hypothesis=list(r.excess_by_hypothesis))


def main() -> None:
    rng = np.random.default_rng(20260919)
    # Source relation law: P(R=1|Y=0)=0.1, P(R=1|Y=1)=0.9.
    # Target visual bins are X=0 and X=1. Hypotheses are Y=X vs Y=1-X.
    pred = np.array([[[[.1], [.9]]], [[[.9], [.1]]]])
    observed = np.array([[.9], [.1]])
    out = {}
    r = filter_hypotheses(observed, pred, .1)
    assert r.feasible_indices == (1,)
    out['stable_relation_disambiguates'] = as_dict(r)

    r = filter_hypotheses(observed, pred, .1, np.zeros((2,1), bool))
    assert r.returned_indices == (0,1) and not r.mechanism_conflict
    out['missing_knowledge_preserves_bank'] = as_dict(r)

    # Only an unconditional relation histogram is insufficient: both predict .5.
    r = filter_hypotheses(observed.mean(0, keepdims=True), pred.mean(2, keepdims=True), .1)
    assert r.feasible_indices == (0,1)
    out['marginal_prior_does_not_disambiguate'] = as_dict(r)

    # Independent/shuffled relation matches neither supported conditional law.
    r = filter_hypotheses(np.full((2,1), .5), pred, .1)
    assert r.mechanism_conflict and r.returned_indices == (0,1)
    out['unexplained_relation_is_conflict_not_unknown'] = as_dict(r)

    # Admit either stable or reversed relation law: labels are no longer identified.
    two_worlds = np.concatenate([pred, 1-pred], axis=1)
    r = filter_hypotheses(observed, two_worlds, .1)
    assert r.feasible_indices == (0,1)
    out['mechanism_uncertainty_restores_ambiguity'] = as_dict(r)

    # One global mechanism, NOT one convenient mechanism chosen for each group.
    r = filter_hypotheses(np.array([[.1],[.1]]), two_worlds, .05)
    assert r.mechanism_conflict
    out['one_mechanism_must_fit_all_groups'] = as_dict(r)

    # Declared failure: target labels flip AND relation law flips. Observations
    # agree with identity under the assumed source law. No test using these
    # observations can detect this particular hidden misspecification.
    r = filter_hypotheses(np.array([[.1],[.9]]), pred, .1)
    assert r.feasible_indices == (0,)
    out['declared_failure_hidden_relation_flip'] = {
        **as_dict(r), 'true_label_rule': 'Y=1-X',
        'selected_label_rule': 'Y=X', 'selected_accuracy_in_this_toy': 0.0,
        'interpretation': 'Misspecified relation support can yield confident wrong selection.'}

    reps=500; n_per_group=100; successes=0; conflicts=0
    for _ in range(reps):
        obs=np.array([[rng.binomial(n_per_group,.9)/n_per_group],
                      [rng.binomial(n_per_group,.1)/n_per_group]])
        r=filter_hypotheses(obs,pred,.2)
        successes += r.feasible_indices == (1,)
        conflicts += r.mechanism_conflict
    out['finite_sample_sanity_check'] = {
        'repetitions':reps, 'independent_observations_per_group':n_per_group,
        'fixed_tolerance':.2, 'unique_correct_hypothesis_runs':successes,
        'conflict_runs':conflicts,
        'scope':'Prespecified simulated source law, not calibration on real ports.'}

    report={'status':'Seven analytical unit-test situations and a finite-sample toy; no SAR training.',
            'seed':20260919,'results':out,
            'not_established':['Global novelty','Prediction correctness without relation assumptions',
                               'Novel-vs-low-quality identifiability','Real cross-port improvement',
                               'Probability calibration or universal safety']}
    Path(__file__).with_name('audit_results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
