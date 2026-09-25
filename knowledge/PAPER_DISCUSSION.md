# Abstract, Discussion, Conclusion, Related-Work outline — draft prose

*Completes `PAPER_DRAFT.md` (Intro + Method) and `PAPER_RESULTS.md` (Sections 3–9).*

---

## Abstract

Vessel classifiers trained in one port are deployed in another, and the scene knowledge practitioners
add to steady that transfer — facility layers, quay proximity, traffic context — is known to help
sometimes and hurt others. We build a 24-port Sentinel-1 benchmark with an AIS-free protocol (38,091
and 71,801 chips; leave-one-port-out; matched-rate random and oracle controls) and show that the
transfer has an asymmetric structure: **the ranking of instances transfers while any quantity compared
against a fixed level does not**. Three independent measurements agree (AUC 0.72, 0.555, 0.679 for
ranking; a guarantee met on 22 % of ports for the value side), and the failure of threshold-, conformal-
and surrogate-based gates is structural — verifying a risk needs the target labels the protocol
withholds — with a decision-relevant event rate of 1.2 % bounding the whole family at +2.91 pp of
balanced accuracy. The law implies a different deployment shape: rather than transferring a threshold,
consult the knowledge-augmented expert for the top-$k$% of the target port by a source-trained score.
On the compliant expert this reaches 106.8 % of full-consultation value at a 20 % budget
(p = 0.0194, 95 % CI [+0.0062, +0.0328]) and holds from 2 to 23 source ports. We also price the
AIS-free restriction (an AIS-derived context block carries +4.23 pp against +0.37 pp for its
SAR-derived counterpart, and pooling SAR detections across acquisitions rebuilds none of it,
−0.06 pp, p = 0.90) and show the port-to-port drop is a class-mix effect (−12.79 pp label-shift
component; target conditionals are +6.54 pp better). Eleven further routes are reported as measured
negatives with their evidence rather than as opinions.

---

## Discussion

**What the law changes in practice.** The usual recipe — learn a reliability score, calibrate it on
source data, threshold it — fails here for a reason that no amount of model capacity addresses: the
threshold's *position* is a value-level object and values do not cross ports. The operational
consequence is to stop tuning the threshold and start choosing a budget: how many instances can the
deployment afford to send to the expensive expert? The answer selects a quantile, not a level, and the
quantile is computed inside the target port. This is a small change with a large practical
difference: the module reaches full-consultation value at a fifth of the cost, and its randomised
control shows the value comes from the ranking and not from dilution.

**Why guarantees are structurally unavailable here.** Conformal, weighted conformal and surrogate-risk
control all require the realised risk to be observable at the target; the AIS-free protocol removes
exactly that. Importance weighting narrowed the gap (median realised risk 0.1705 → 0.1235,
p = 0.0277) but could not close it, and the residual is not a tuning parameter: it is the exchangeability
that cross-port transfer breaks. Reporting such numbers as "risk control" would be a category error;
we report them as calibration heuristics with their measured shortfall.

**The price of a restricted input is a result, not a caveat.** Protocols that forbid a data source are
usually justified by deployment reality, and the resulting numbers are usually reported without
quantifying what was given up. Here the forbidding is measurable: an AIS-derived context block is
worth six times its SAR-derived counterpart, the two are uncorrelated rather than redundant, and
reconstructing the missing structure from SAR detections across acquisitions fails with adequate
power. The useful generalisation for practitioners is to report the forbidden channel's *measured*
contribution, because that number decides whether the restriction is acceptable for the intended use.

**Where the wall could move.** The wall is a property of the observation, not of the architecture: the
decision-relevant signal is rare and its sign is a class × instance property. Two things would move it,
and both were tested here rather than assumed. Adding sequence or identity information does not help —
an oracle-identity probe that groups a vessel's acquisitions by its true MMSI leaves the gate unchanged
(AUC 0.905 → 0.910, p = 0.80) — which says the deficiency is not resolution but information. Adding
source ports does not help either (8 ports match 23). What would move it is a channel that supplies
information the imagery lacks, which is precisely the AIS field we are forbidden; the honest
formulation of this paper's negative result is therefore that the restricted-input setting has a
measurable information deficit, not a modelling deficit.

**On negative results with controls.** Every route we refute carries a measurement, a matched-rate
random arm and, wherever a decision is involved, an oracle bound for the same family. This discipline
is what makes it possible to state a cap — the compliant expert tops out at +2.3 pp — as a boundary of
the problem rather than a shortcoming of one attempt. Three of our own intermediate conclusions were
withdrawn during the study (a protocol violation, an underpowered pooling effect, an unstable
calibrator), and they are recorded in the artifacts rather than tidied away, because their withdrawal
is itself what bounds the claims.

**Limitations.** The pools cover 24 ports and 17 classes; the paired tests over 24 ports are the main
power limit, and several margins sit between p = 0.03 and p = 0.10. The 512-dimensional projection used
for the risk experiments was validated against the raw features (0.4722 vs 0.4735) but is a
simplification. The attribution in Section 7 rests on source-side conditionals assumed stable across
the held-out source port, and signed components can over-explain a net difference.

---

## Conclusion

We set out to quantify when scene knowledge is reliable across ports, and found a structure simple
enough to change the recipe and sharp enough to be falsifiable: order transfers, value does not. The
positive consequence is a budget-based allocation that reaches full-consultation value at a fifth of
the cost and does not depend on calibration; the negative consequence is that guarantee-style gates
cannot be built in this setting at all, because verifying a risk requires the labels the protocol
withholds. Both are measured, both come with controls, and the same law prices the data the protocol
forbids. The remaining improvement space on this benchmark is exhausted — every lever we could
identify is either refuted or capped — which we take as the appropriate end state for a paper whose
contribution is as much about what does not work, and why, as about what does.

---

## Related work — strands to cite (no references invented here)

1. **Cross-domain / unsupervised domain adaptation for remote sensing**, and the standard remedy of
   feature alignment; our finding that representation capacity is not the bottleneck speaks to that
   literature's assumptions.
2. **Label shift and prior correction** (BBSE-style estimators, maximum-likelihood priors); Section 7
   re-opens this line and attributes a previous failure to estimation rather than principle.
3. **Conformal prediction under distribution shift** (weighted/adaptive conformal, risk-controlling
   prediction sets); Section 4 supplies a cross-port empirical bound on what those methods can
   promise.
4. **SAR ship classification and SAR ATR benchmarks**, including the AIS-labelled dataset tradition
   and the practice of augmenting imagery with contextual layers.
5. **Selective classification and learning to defer**, whose risk–coverage curves we adopt, and whose
   thresholds we show cannot be transferred across ports.
6. **Negative-result methodology in ML**, as the framing for Section 8's taxonomy.
