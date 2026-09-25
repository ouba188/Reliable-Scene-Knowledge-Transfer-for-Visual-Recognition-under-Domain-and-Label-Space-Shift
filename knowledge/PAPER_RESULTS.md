# Results — draft prose (Sections 3–9)

*Continues `PAPER_DRAFT.md` (Introduction + Method). Numbers are the measured values from
`EXPERIMENT_STATE.md`; artifact files are named per subsection.*

---

## 3. The order/value law

Table 1 collects three tasks, each measured the same way: an arm trained on source ports only,
evaluated on the target port, judged twice — once by how well it *ranks*, once by whether a *level*
transfers.

| task | ranking ability (transfers) | value / operating point (does not) |
|---|---|---|
| knowledge-gain gate | AUC **0.72** | thresholds and calibrators never transfer |
| novelty detection (Mahalanobis) | AUC **0.555** | per-class absolute levels do not port |
| error prediction | AUC **0.679** | guarantee met on **22 %** of ports (target: ≥80 %) |

**Table 1.** The same asymmetry in three independent measurements.

**Gate experiment.** A gate predicting whether consulting the knowledge expert is the right call
reaches AUC 0.72 on the target port, yet every threshold and calibrator built on it failed: a
pre-registered decision-risk objective did not beat a proxy target (0.2300 vs 0.2295, within noise),
and a per-port calibration analysis showed the decision label is algebraically equivalent to the
argmax, so calibration cannot move the boundary at all. The gate's own capture of the oracle ceiling
was 37 % against a selectivity curve predicting 67–86 % at its AUC.

**Novelty experiment.** Class-conditional Gaussians fitted on source ports separate 9 label-space-novel
classes from the known ones at AUC 0.555 averaged over classes, with an across-class standard
deviation of 0.0752 against a per-class bootstrap noise of 0.0324 (2.3×), all 24 ports and 16,699
novel instances. Detectability is a genuine class property. The energy score, by contrast, scores
0.429 — below chance — and is reported as misleading rather than neutral.

**Error-prediction experiment.** An error model fitted on source ports and applied to the target port
ranks the target's errors at AUC 0.679 (p10 0.584, p90 0.774, above 0.65 on 14/23 ports), but the
guarantee it is meant to support holds on 22 % of ports.

*(Artifacts: `e123b_out.txt` for the error model, `e117_out.txt` for novelty, `e121_out.txt`/`e112_out.txt`
for the gate and its calibration analysis.)*

### 3.1 A protocol violation caught mid-study

Two of our earlier results — a budget curve with p = 0.0000–0.025 — were withdrawn after enumerating
input dimensions against the permission table: the knowledge matrix's tail (dimensions 71:82) is
AIS-derived, so an expert built on all 82 dimensions is not compliant. The permission table names only
two explicitly forbidden columns, which is why the violation was silent. All results reported here use
the compliant block (dimensions 0:71), and the compliant numbers are weaker — a fact we report rather
than bury, because the gap is itself a measurement of what the restriction costs (Section 7).

---

## 4. The identifiability wall

The sign of the knowledge gain is a class × instance property, not a domain property: within a single
port, the same class flips sign between instances, and the discriminating variable correlates with the
class at r = 0.771. At the same time the decision-relevant event is rare — of 25,893 instances, 617 are
rescued and 313 harmed, 1.2 % in total — so the oracle ceiling for any per-instance decision rule is
+2.91 pp of balanced accuracy. A deeper model cannot manufacture a signal that is not there, and three
independent capacity comparisons plus six separate "more structure" attempts produced no gain.

Guarantee-style methods fail for a structural reason: verifying a risk requires target labels, which
the problem withholds.

| construction | outcome |
|---|---|
| source-calibrated conformal thresholds | risk target met on **0/23** ports |
| density-ratio (importance) weighted conformal | 2/23 ports; median realised risk 0.1705 → 0.1235, p = 0.0277 |
| surrogate-risk control (source-fit error model) | 5/23 ports |

**Table 2.** Every guarantee construction, and the reason to stop: the risk cannot be *verified*
without the labels the protocol forbids.

---

## 5. Budgeted rank allocation

Since only the order transfers, the policy consumes the order and leaves the operating point to the
deployment's consultation cost: consult the expert for the top-$k$% of the target port by the
source-trained gate score.

| budget | rank − random (95 % CI) | p | recovery vs consulting everywhere |
|---:|---|---:|---:|
| 5 % | +0.0076 [+0.0005, +0.0156] | 0.160 | 41.4 % |
| 10 % | +0.0149 [+0.0050, +0.0254] | 0.0269 | 80.5 % |
| **20 %** | **+0.0192 [+0.0062, +0.0328]** | **0.0194** | **106.8 %** |
| 50 % | +0.0113 [+0.0019, +0.0209] | 0.0457 | 103.0 % |

**Table 3.** Compliant expert (knowledge 0:71), 24 ports, random arm averaged over 50 matched-size
redraws. The pre-registered criterion (interval excluding zero and recovery ≥ 50 %) is met at 20 %.

Three properties matter as much as the number.

**Robustness to the source count.** At 2, 4, 8, 16 and 23 source ports the ranked 20 % is worth 3–10×
a random 20 % (0.415/0.114, 1.048/0.109, 2.881/0.500, 1.469/0.347, 2.211/0.459 in ΔBA percentage
points). The core effect does not depend on having many sources.

**Scope conditions.** The criterion does not apply where there is nothing to allocate: the three single
knowledge parts have gains of −0.22, +0.12 and +0.37 pp, and one is negative. Below a ~10 % budget the
paired test is underpowered on this data.

**A measured cap.** The module's absolute value is bounded by the compliant expert's strength, and that
strength is capped: an exhaustive enumeration of seven knowledge subsets × two classifier families
gives a best legal gain of +2.32 pp (p = 0.32) against the deployed +2.16 pp (p = 0.0604), with the
ridge-on-full-compliant-block configuration remaining both the best and the best-powered choice.

**A note on how this result was nearly missed.** Every earlier gate in this project used an absolute
threshold; "top-$k$%" was never tested, and the reason it works is exactly the distinction the law
draws: a gate must map a score onto a use/don't-use decision (a value-level object), while this policy
only needs the relative order of who deserves consultation.

---

## 6. The price of the AIS-free restriction

**Block level.** On identical folds and protocol:

| arm | ΔBA (pp) | ports positive | paired p |
|---|---:|---:|---:|
| V + AIS-derived context (non-compliant) | **+4.23** | 17/24 | 0.0022 |
| V + SAR-detection-derived context (compliant) | **+0.37** | 13/24 | 0.86 |

**Table 4.** The detection block carries a tenth of the AIS block's value; the two differ at p = 0.0096.

**Per quantity.** The eleven quantities exist in both versions. The five that carry value are
heading/queue/density interactions, and for each the detection version carries roughly a tenth of the
AIS version *and* the two are essentially uncorrelated:

| quantity | detection ΔBA | AIS ΔBA | corr |
|---|---:|---:|---:|
| heading² | +0.22 | +1.12 | −0.163 |
| queue² | +0.30 | +2.20 | −0.187 |
| heading×queue | +0.29 | +1.49 | −0.175 |
| density×heading | +0.08 | +2.21 | −0.220 |
| density×queue | +0.20 | +3.04 | −0.224 |

**Table 5.** Not a coarse proxy: the same-named quantities are uncorrelated, so they measure different
things.

**Can SAR rebuild the field?** We pooled SAR detections across acquisitions onto a 2 km port grid and
recomputed the same eleven quantities — feasible (2.9 M detections, 24 ports, headings populated).
Measured under adequate power, cross-acquisition pooling and per-scene pooling are indistinguishable:
0.1756 vs 0.1762, paired −0.06 pp, p = 0.90, over 24 ports and 123,790 objects. The AIS advantage is
therefore not wider coverage of the same kind; and the two heading-free quantities (`queue²`,
`density×queue`) carry most of the AIS value, which rules out "AIS headings are just more accurate" as
the explanation.

---

## 7. Attribution: the drop is the class mix

Decomposing the port-to-port difference under a single weighted functional
$a = \sum_t \pi_t[t] R[t,t] / \sum_t \pi_t[t]$:

| term | value | reading |
|---|---:|---|
| source conditionals, source priors | 0.5044 | reference |
| source conditionals, target priors | 0.3765 | **label/base-rate component −12.79 pp** |
| target conditionals, target priors | 0.4418 | **conditional component +6.54 pp** |

**Table 6.** 24 ports. Signed components can over-explain a net difference of 6.25 pp; report the
components, never a percentage share.

The practical consequence redirects the problem: the target port's conditionals are *better*, so prior
correction is the first thing to fix, and our own earlier rejection of prior-correction methods
attributes to their estimator (measured then as 0.012 better than the raw prediction histogram) rather
than to a principled obstacle.

---

## 8. Negative taxonomy

Every entry below was measured; none is an opinion, and each carries its artifact in
`EXPERIMENT_STATE.md`.

| route | verdict | measurement |
|---|---|---|
| deeper/wider capacity (3 encoder families, joint training) | refuted | zero gain |
| decision-risk objective | refuted | within noise of the proxy |
| threshold/calibration | refuted | algebraically cannot move the boundary |
| eight alternative input axes | refuted | all negative |
| multi-temporal consistency signal | refuted | AUC 0.484, coverage 34 % |
| conformal and weighted conformal | refuted | 0/23 and 2/23 ports |
| surrogate risk control | refuted | 5/23 ports |
| SAR-only port field to replace AIS | refuted | −0.06 pp, p = 0.90 |
| oracle-identity vessel bags (sequence) | refuted | gate AUC 0.905 → 0.910, p = 0.80 |
| more source ports | ineffectual | 8 ports (+2.71 pp) ≥ 23 ports (+2.16 pp) |
| stronger compliant expert | capped | best legal +2.32 pp at p = 0.32 |

**Table 7.** Positive survivors: the compliant expert (+2.16 pp, p = 0.0604), budgeted rank allocation
(20 % → 106.8 %, p = 0.0194), per-class novelty detectability (across-class SD 2.3× noise), and the
attribution of Section 7.
