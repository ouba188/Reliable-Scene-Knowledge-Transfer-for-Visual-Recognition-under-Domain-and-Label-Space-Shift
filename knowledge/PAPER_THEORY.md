# Theory — draft section (to merge into Method §2 and Results §3)

*The propositions below are the paper's theoretical core. They are deliberately small and provable; the
part that would need assumptions we cannot verify is stated as a measured relation instead of a
theorem.*

---

## A. The assumption class

Write the target port's score distribution as related to the source's by

$$\tilde s(x) \;=\; \phi\big(s(x)\big) \quad \text{with } \phi \text{ strictly increasing},
\qquad\text{and}\qquad \pi_{\text{tgt}} \;\neq\; \pi_{\text{src}} \ \text{(class priors)},$$

i.e. **monotone recalibration plus class-prior shift**, with the conditionals *not* assumed invariant —
the measured setting in this paper is close to exactly this class: the drop decomposes into a
−12.79 pp prior component, and the target's own conditionals are +6.54 pp *better* than the source's
rather than identical (Section 7).

Two facts hold in this class, and one fails outside it.

---

## B. Proposition 1 (rank invariance)

**Statement.** Under any strictly increasing $\phi$, the top-$k$% set of a score vector is unchanged,
for every $k$. Equivalently: the budget policy $\pi_k$ is *invariant* to monotone recalibration.

**Proof.** $\phi$ strictly increasing gives $s_i \ge s_j \iff \phi(s_i) \ge \phi(s_j)$ for all pairs;
hence the score-induced total preorder is identical, and so is every top-$k$% prefix (ties broken by a
fixed rule). ∎

**Verification on real scores.** For 12 ports, applying $\phi(s)=s^{0.3}$, $s^{2}$ and $s^{7}$ to the
measured score vector leaves the top-20 % set's symmetric difference at exactly **0** in every case.

**Contrast.** A rule that compares a score against a *fixed absolute* level is not invariant: after any
$\phi$ the accepted set changes, and if the threshold was calibrated on the source port the change is
uncontrolled. The empirical counterpart is measured and negative: source-calibrated conformal
thresholds hold their risk target on **0/23** ports.

---

## C. Proposition 2 (threshold fragility under prior shift)

**Statement.** Let $R$ be the row-normalised confusion matrix (the conditionals) and $\pi$ a class
prior. The expected error of accepting a set $A$ is
$\ \mathrm{Err}_\pi(A) = \sum_y \pi_y \sum_{\hat y \ne y} R_{y\hat y}\, \mathbb{1}[A \ni \text{instances of } y]$.
For a fixed accepted set determined by a threshold on the source distribution,
$\ \mathrm{Err}_{\pi_{\text{tgt}}} - \mathrm{Err}_{\pi_{\text{src}}} = \sum_y (\pi_{\text{tgt},y} - \pi_{\text{src},y})\, \rho_y$,
where $\rho_y$ is the class-$y$ contribution to the accepted error. The difference is bounded only by
$\|\pi_{\text{tgt}} - \pi_{\text{src}}\|_1 \cdot \max_y |\rho_y|$, which can be large: in our ports the
prior total-variation distance has median **0.568**.

**Consequence.** Any guarantee of the form "the target risk stays within $\alpha$" that is built from
source calibration requires the priors to be comparable or the risk to be observable at the target —
and observability is exactly what the AIS-free protocol removes. Measured: **0/23** ports for plain
conformal, 2/23 with importance weighting, 5/23 with a surrogate risk model.

---

## D. The measured regret relation

The natural remaining question is how much of the *oracle-quantile* policy (top-$k$% selected by the
true per-instance benefit) a practical budget policy attains, and what it depends on. We do not claim a
tight analytic bound; we measure the relation.

| quantity | value (24 ports) |
|---|---|
| gate AUC for "consulting the expert helps" | median **0.947** |
| $R_{\text{gate}}$ at a 20 % budget (consult-everywhere $=1.0$) | median **+1.00** |
| $R_{\text{oracle}}$ at the same budget | median +1.22 |
| **Spearman(gate AUC, $R_{\text{gate}}$)** | **$\rho = +0.592$, p = 0.0023** |

So the regret against the oracle-quantile policy is *governed by ranking quality*, measured rather than
assumed: with a gate whose benefit-AUC is ≈0.95, the 20 % budget already matches consulting everywhere.
This is the sense in which the module is the right shape for the law — its quality is a function of the
one thing that transfers.

---

## E. The failure boundary (falsifiability)

If the law were vacuous, it would hold under any perturbation. It does not. Destroying the conditionals
while keeping the priors — within-class full permutation, and a cross-class permutation — collapses the
rank policy's advantage over a matched-rate random arm to chance:

| intervention | rank median | random median | ports retaining the advantage |
|---|---:|---:|---:|
| within-class full permutation | **−0.0032** | +0.0049 | **9/24** |
| across-class permutation | +0.0264 | +0.0544 | **7/24** |

So the law's domain is the assumption class of Section A: monotone distortion plus prior shift, where
the order is meaningful; break $P(y\mid x)$ and the order stops carrying the decision, exactly as the
propositions say it should.

---

## F. What this theory does not claim

- It is **not** a causal identification. The evidence is controlled intervention with matched controls,
  not do-calculus with stated ignorability.
- It is **not** a tight bound: Section D is a measured relation ($\rho = 0.592$), not a theorem with a
  constant. A tight constant would need assumptions on the benefit-generating process that this data
  cannot verify.
- The prior-correction corollary of Section C does **not** rescue prior-corrected classifiers here: the
  inversion requires the conditionals to be domain-invariant, and they measurably are not
  (+6.54 pp better on the target), which is why a stabilised BBSE estimator still fails outright
  (0.4395 → 0.2416, 0/24 ports). The honest statement is that the prior is the main component of the
  drop **and** the conditionals cannot be treated as fixed.
