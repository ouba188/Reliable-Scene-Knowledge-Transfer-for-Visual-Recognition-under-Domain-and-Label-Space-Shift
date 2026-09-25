# Cross-port SAR vessel classification with scene knowledge: order transfers, value does not

*Draft prose — Introduction and Method. Every number below is tied to an entry in `EXPERIMENT_STATE.md`;
figures are listed at the end of each section as placeholders.*

---

## 1. Introduction

Vessel classification in SAR imagery is usually trained in one port and deployed in another, and the
transfer is rarely clean: the same ship type looks different under a different incidence geometry, a
different berth, and a different local traffic regime. Practitioners have long added *scene knowledge*
— facility layers, distance-to-quay, vessel-density context — as extra input features, and the
literature reports the mixed result that everyone recognises: sometimes it helps, sometimes it hurts.
What has been missing is not another way to inject that knowledge but a quantitative account of **when
the injection is reliable**. That is the question this paper answers, and it answers it with a
protocol built to be able to say no.

We assemble a 24-port benchmark from Sentinel-1 GRD imagery, AIS-derived vessel labels, and OSM
facility layers, in two pools: an initial 38,091-chip pool (128-dimensional visual projection plus 82
knowledge dimensions) and a rebuilt 71,801-chip pool (224-pixel chips, 17 classes, 24 ports). Both are
evaluated leave-one-port-out with an **AIS-free protocol**: target-port AIS identity may never enter
the input features, and the compliant knowledge block is restricted to its first 71 dimensions —
a restriction we enforce by enumerating feature permissions rather than by trusting a description, a
step that reversed two of our own results mid-study (Section 4).

The central finding is a law with an unusual form, because it is about what transfers rather than
about what helps:

> **Cross-port transfer preserves the order and destroys the value.** A source-fit model still ranks
> the target port's instances correctly — three independent measurements put the ranking at AUC
> 0.72, 0.555 and 0.679 — while any quantity that must be compared against a fixed level stops
> transferring: a source-calibrated threshold holds its risk target on 0 % of ports, and a
> source-fit error model mis-states the target's risk so badly that it meets its guarantee on 22 %.

Three consequences follow, and each is measured rather than argued.

**First, gates built on absolute decisions are a dead end here.** Thresholds, calibrated
probabilities, conformal risk control, importance-weighted conformal, and surrogate-risk control were
each evaluated and each failed, and the failure is structural rather than implementational: verifying
a risk requires target labels, which is precisely what the problem withholds. The decision-relevant
event is also rare — 1.2 % of instances are rescued or harmed by the knowledge — so the oracle ceiling
for the whole family is +2.91 pp. Deeper architectures do not escape this: three independent capacity
comparisons produced zero gain, and six separate "more structure" attempts were all negative.

**Second, the deployable form of the same idea is a budget, not a threshold.** If only the order
transfers, then the policy should consume the order and leave the operating point to the deployment
cost. Concretely, we consult the knowledge-augmented expert for the top-$k$% of the target port by a
source-trained gate score — pure rank, no calibration, no target labels. On the compliant expert, a
20 % budget recovers 106.8 % of what consulting the expert everywhere would give (p = 0.0194, 95 %
bootstrap CI [+0.0062, +0.0328]), and the core effect holds from 2 to 23 source ports, with the ranked
20 % worth 3–10× a random 20 %.

**Third, the law also prices the data the protocol forbids.** Restricting the input to be AIS-free is
not free: an AIS-derived context block carries +4.23 pp where the SAR-detection version of the same
eleven quantities carries +0.37 pp (paired p = 0.0096), the two are essentially uncorrelated
(−0.16 to −0.22), and pooling SAR detections across acquisitions to rebuild an equivalent port-wide
field adds nothing (−0.06 pp, p = 0.90, over 24 ports and 123,790 objects). The AIS advantage is not
more of the same coverage; it is a different kind of information, and the honest way to report a
restricted-input result is to quantify what the restriction removed.

We also report an attribution that changes how the transfer problem should be attacked. Decomposing
the port-to-port accuracy difference under one weighted functional, the entire drop comes from the
target port's class mix (−12.79 pp label-shift component), while the target port's *conditionals* are
actually better than the source's (+6.54 pp). The practical reading is that prior correction, not
representation learning, is the first thing to fix — and that our own earlier rejection of
prior-correction methods was an estimator problem, not a principled limit.

The contributions are therefore:

1. a 24-port SAR vessel benchmark with an AIS-free protocol and per-port leave-one-out evaluation
   (two pools, 38,091 and 71,801 chips), including matched-rate random and oracle control arms;
2. the order/value law, established by three independent measurements, plus the identifiability wall
   that explains why guarantee-style gates cannot work without target labels;
3. budgeted rank allocation, a compliant module that reaches ≥100 % of full-consultation value at a
   20 % budget, with its scope conditions and its measured cap;
4. a measured price for the AIS-free restriction, including the refutation that pooling SAR
   detections can rebuild the AIS field, and the attribution of the drop to label shift;
5. a negative taxonomy in which every refuted route (capacity, objective, thresholding, eight input
   axes, temporal/identity signals, conformal and surrogate risk control) carries its own measurement
   rather than an opinion.

---

## 2. Method

### 2.1 Data and protocol

**Pools.** The initial pool has 38,091 chips over 24 ports and 8 classes, with a 128-dimensional PCA
projection of ResNet-50 penultimate features and an 82-dimensional scene-knowledge vector. The rebuilt
pool has 71,801 chips over the same 24 ports and 17 classes (8 known, 9 label-space-novel) at 224 px,
with 64/112/224 field-of-view feature variants; the per-port counts range from 89 to 9,328 chips.
Scene knowledge is derived from OSM facility layers, quay/anchorage/fairway distances and
detection- and AIS-derived context fields.

**Feature permissions.** Every input dimension is checked against a written permission table before it
enters an arm. The protocol is AIS-free, so AIS-derived context is excluded; the compliant knowledge
block is therefore the first 71 of the 82 dimensions. An explicit enumeration is necessary because
the forbidden set is not identical to the documented one: the permission table names only two
explicitly forbidden columns, while the AIS-derivability of the tail dimensions is a property of how
they were built. Two of our own earlier results were withdrawn on this basis.

**Evaluation.** Leave-one-port-out over 24 ports: for target port $p$, source ports are all others.
Reported quantities are balanced accuracy over the known classes (primary), per-port × per-class
grids, and paired Wilcoxon tests over the 24 ports. Every policy arm ships with two controls: a
**matched-rate random arm** (the same accept/consult rate, drawn at random, averaged over 20–50
redraws) and an **oracle arm** for the same decision family, so that a policy is judged by its regret
against the ceiling rather than by its absolute score.

**Criteria.** Decision experiments are pre-registered: the criterion is written before the run, and
the result is reported whether or not it passes. Effect sizes are reported with 24-port bootstrap 95 %
intervals; p-values alone were found to flip between 0.095 and 0.0194 on identical data purely as a
function of inner-episode and redraw counts, so intervals are the primary instrument.

### 2.2 Arms

| arm | definition |
|---|---|
| **V** | visual only |
| **V+K** | visual plus compliant knowledge (the *expert*) |
| **gate** | $g(x) \rightarrow$ whether consulting the expert is the right call, trained on source-port episodes only |
| **budget policy** | consult the expert for the top-$k$% of the target port by $g$, otherwise use V |
| oracle (decision) | per-instance ideal call, marked `ORACLE_ONLY`, never deployable |

### 2.3 Budgeted rank allocation

The module is deliberately small, because its point is where the decision is taken, not how much
capacity it has. Let $s_i = g(x_i)$ be the gate score, computed from source ports only, and let
$k \in (0, 1]$ be the deployment's consultation budget. Then

$$\pi_k(x_i) = \begin{cases} \text{expert}(x_i), & s_i \ge q_{1-k}(s_{1:n}) \\ V(x_i), & \text{otherwise}\end{cases}$$

where $q_{1-k}$ is the $(1-k)$-th quantile of the *target port's own* scores. Nothing is calibrated:
$k$ is an exogenous cost parameter, the quantile is computed within the target port, and no target
label is used at any point. The corresponding quantity reported throughout is the **recovery**

$$R(k) = \frac{\mathrm{BA}(\pi_k) - \mathrm{BA}(V)}{\mathrm{BA}(V{+}K) - \mathrm{BA}(V)},$$

so $R(k) = 1$ means the budget captures the whole value of consulting the expert on every instance.

**Gate features.** Per-instance margins and entropies of both arms, their agreement, and — in the
sequence probe — leave-one-out vessel-bag statistics. Training uses nested source-port episodes
(3–6 held-out source ports) so that no gate ever sees the target port.

### 2.4 Compliant expert

The compliant expert is a ridge classifier on the visual projection concatenated with the first 71
knowledge dimensions, standardised inside the training split. Its strength bounds the module's
absolute value, so its limits matter: an exhaustive enumeration of all seven non-empty subsets of the
three knowledge parts (facility 0:42, chain 42:60, detection 60:71) crossed with two classifier
families (ridge, gradient-boosted trees) yields a best legal gain of +2.32 pp (p = 0.32) against the
current +2.16 pp (p = 0.0604) — i.e. the compliant expert is capped at roughly +2.2–2.3 pp, and the
ridge-on-full-block configuration remains the best deployable choice.

### 2.5 What is *not* claimed

The order/value law is supported by controlled interventions in which one factor changes while folds,
pools and metrics are held fixed; it is not a causal identification. Class-level correlations
(n = 8–9) are not treated as mechanisms: three such conclusions in this study collapsed under direct
instance-level measurement, so class-level tables are presented as displays of instance-level effects
once those exist. Where a construction turned out to be unstable — per-class thresholds on thin
calibration cells — the derived numbers are withdrawn rather than qualified.

---

### Figure/table plan

- **Fig. 1** Protocol schematic: pools, AIS-free boundary, LOO folds, control arms.
- **Fig. 2** Rank vs value: three AUC measurements against three calibration hold-rates.
- **Fig. 3** Budget–value curve on the compliant expert (5/10/20/50 %), with the random arm and CIs.
- **Table 1** Arms and their gains on both pools.
- **Table 2** Compliant expert: 7 subsets × 2 classifiers.
- **Table 3** AIS block vs detection block, per quantity, with correlations.
- **Table 4** Negative taxonomy with the measurement behind each entry.
