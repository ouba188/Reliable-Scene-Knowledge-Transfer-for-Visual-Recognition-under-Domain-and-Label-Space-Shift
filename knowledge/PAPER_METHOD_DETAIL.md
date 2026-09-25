# Method — detailed version

*Supersedes and expands `PAPER_DRAFT.md` §2. Same notation; every quantity named here is implemented
in the scripts listed in §6.*

---

## 1. Data construction

### 1.1 Imagery and chips

Inputs are Sentinel-1 IW GRD products (VV and VH) over 24 ports, with 430 scenes retained locally as
8-bit UTM GeoTIFFs at 10 m. Detections come from a YOLO-OBB detector run at 1024 px with confidence
threshold 0.08; each detection carries an oriented bounding box, a detection id and its UTM centroid.

Two chip pools are used.

| | initial pool | rebuilt pool |
|---|---|---|
| chips | 38,091 | 71,801 |
| chip size | 128 px | 224 px |
| classes | 8 | 17 (8 known, 9 label-space-novel) |
| visual features | 128-dim PCA of ResNet-50 penultimate | 64 / 112 / 224 field-of-view variants (2048-dim each) |
| ports | 24 | 24 (89–9,328 chips per port) |
| per-class counts (known) | 11,128 / 8,599 / 5,929 / 5,827 / 2,208 / 1,757 / 1,478 / 1,165 | 3,647 / 3,159 / 3,004 / 2,888 / 2,606 / 2,534 / 2,387 / 2,120 … |

Chips are cut by mapping each detection's UTM centroid through the scene geotransform and cropping a
square window centred on it (row/column bounds clamped to the raster). The field-of-view variants let a
single experiment separate *what is in the window* from *what the model does with it*: the 64-px window
is a tight view of the hull, the 224-px window includes berth context. Everything reported in the
results uses the 64-px variant unless stated, after the tight view was found to be the stronger
baseline.

### 1.2 Labels and label provenance

Vessel labels are AIS-derived and carry a three-tier provenance: a gold tier (unique single-MMSI
spatial and temporal match), a coarse tier (parent-sum aggregation of fine type codes), and an
excluded tier. Coarse-tier labels never enter the evaluation set, `unknown` is never a training label,
and labels below 0.2 confidence are excluded from pretraining. Class support is counted in **labelled
objects**, not MMSIs; a class is present in a port at ≥ 20 objects.

### 1.3 Scene knowledge

The knowledge vector has 82 dimensions in four documented parts:

| part | dims | content |
|---|---|---|
| facility | 0:42 | 14 OSM facility classes × (proximity, local background contrast, near-vs-route) |
| chain | 42:60 | relative evidence along dry/general-logistics chains |
| detection | 60:71 | 11 quantities from SAR detections: three density-contrast windows, inner fraction, heading², queue², heading×queue, density×heading, density×queue, and two × quay-proximity terms |
| AIS context | 71:82 | the same eleven quantities computed from the AIS traffic field |

Values are stored with a per-dimension validity mask (`support`), and a dimension is zero-filled where
invalid.

### 1.4 Feature permissions — the compliance step

Every input is checked dimension-by-dimension against a written permission table before an arm is run.
The protocol is **P0 AIS-free**. Two consequences are easy to get wrong in practice:

1. the table enumerates only two explicitly forbidden columns (`support_i`, `support_missing`), so the
   AIS-derivability of the context part is a property of *how it was built*, not of what the table
   says. The compliant block is therefore **dims 0:71**, and an expert built on all 82 dims is
   non-compliant even though nothing in the table names it as such;
2. the SAR-derived columns that carry the analogous information (`local_ships_500m/1km`,
   `nn_distance_m`, `heading_consistency_deg`) are listed as features with source SAR and a
   post-deduplication condition — the legal channel exists, it is simply much weaker (Section 6 of the
   results).

Compliance is asserted in the code by slicing the knowledge matrix, never by describing it.

---

## 2. Protocol

### 2.1 Folds and roles

Evaluation is **leave-one-port-out over all 24 ports**: for target port $p$, the source pool is every
other port. Nothing from the target port participates in fitting — not labels, not scaling
statistics, not PCA, not thresholds. Target labels exist only in the evaluation code path. Where a
gate or a calibrator needs held-out data, it is taken from **source ports** (nested episodes over 2–6
held-out source ports), never from the target.

### 2.2 Metrics

Primary metric is **balanced accuracy over the known classes** on the target port. It is always
accompanied by per-port × per-class grids, because a port-level mean hides the structure that motivates
the paper: the sign of a knowledge gain is a class × instance property, so a single number can be
positive while individual port–class cells collapse.

Comparisons between arms are **paired over ports** (Wilcoxon signed-rank, $n = 24$), and effect sizes
are reported with **bootstrap 95 % confidence intervals** computed by resampling the 24 port-level
differences. Intervals, not p-values, are the primary instrument: on identical data the p-value for one
comparison moved between 0.095 and 0.0194 purely as a function of the number of inner episodes and
random redraws.

### 2.3 Control arms

Every decision policy ships with two controls.

- **Matched-rate random.** The same accept/consult rate, applied to a uniformly random subset of the
  target port, averaged over 20–50 redraws and paired with the policy. This separates *selection* from
  *dilution*: an intervention that merely dilutes a strong arm would show the random arm no worse.
- **Oracle for the same decision family** (`ORACLE_ONLY`, never deployable). For the knowledge
  decision, the per-instance ideal call; the difference between a policy and this bound is the
  reported regret.

### 2.4 Pre-registration

Each decision experiment writes its criterion before the run — a direction, a significance or interval
condition, and, where applicable, a recovery fraction. Results are reported whether or not the
criterion passes, failed constructions are recorded with their defect rather than silently re-run, and
numbers produced by a construction later found unstable are **withdrawn**, not qualified.

---

## 3. Arms and the module

### 3.1 Arms

| symbol | definition |
|---|---|
| $V$ | visual-only classifier (ridge on the standardised projection) |
| $V{+}K$ | the **expert**: ridge on visual concatenated with the compliant knowledge block |
| $g$ | the **gate**: predicts whether consulting the expert is the right call for an instance |
| $\pi_k$ | the **budget policy**: consult the expert for the top-$k$% of the target port by $g$ |
| oracle | per-instance ideal call over the arm set, `ORACLE_ONLY` |

Classifiers are fitted on the source pool with per-feature standardisation computed inside the training
split only. The expert is deliberately a ridge classifier: an exhaustive enumeration of knowledge
subsets and a boosted-tree family changed the gain by less than 0.2 pp while degrading the paired
p-value, so extra capacity buys nothing here (results Section 5).

### 3.2 Gate construction

The gate is trained on **source episodes**. For a target port $p$, pick 2–6 held-out source ports $q$;
fit $V$ and $V{+}K$ on the remaining source ports; predict on $q$; then each instance of $q$ yields a
training row

$$x_i = \big[\, m_V,\ m_{VK},\ H_{V},\ H_{VK},\ \mathbb{1}[\,\hat y_V = \hat y_{VK}\,]\,\big],\qquad
y_i = \mathbb{1}\big[\,\hat y_{VK} = y \ \wedge\ \hat y_V \neq y\,\big],$$

where $m$ are top-1 minus top-2 margins and $H$ are predictive entropies. The label is the
*decision-level* target "consulting the expert is what fixes this instance", not the correctness of
either arm — a distinction that matters because the two have different base rates (rescue events are
1.2 % of instances). A gradient-boosted classifier (120 iterations, depth 4) is fitted on the pooled
episodes and applied to the target port. No target label is used at any point.

### 3.3 Budget policy

Let $s_i = g(x_i)$ and let $q_{1-k}$ be the $(1-k)$-th empirical quantile of the target port's own
scores. Then

$$\pi_k(x_i) = \begin{cases} V{+}K(x_i), & s_i \ge q_{1-k}(s_{1:n}) \\ V(x_i), & \text{otherwise.}\end{cases}$$

Nothing is calibrated: $k$ is an exogenous cost parameter of the deployment, and the quantile is
computed within the target port, which is exactly the "use the order, not the value" move the law
prescribes. Results are reported as the **recovery**

$$R(k) = \frac{\mathrm{BA}(\pi_k) - \mathrm{BA}(V)}{\mathrm{BA}(V{+}K) - \mathrm{BA}(V)},$$

so $R(k) = 1$ means the budget captures the whole value of consulting the expert on every instance and
$R(k) > 1$ means selective consulting beats consulting everywhere.

### 3.4 Oracle-identity probe (non-deployable)

To test whether the gate's *order* could be sharpened by sequence information, bags are formed from the
gold MMSI (same port, ≥ 2 acquisitions) — legal only as an **upper-bound probe**, since a deployment
has no target identity. Per chip, leave-one-out bag statistics are added to the gate: bag size, number
of acquisitions, cross-bag agreement of the visual and knowledge arms, mean margin of the other
members, and the target's agreement with its bag mates. The probe is run with the same folds and the
same metric as the deployable arms.

---

## 4. Diagnostic procedures

The negative results are themselves measured with explicit procedures, so that a refutation is a
measurement rather than a failure to find something.

**Per-dimension ablation.** Each knowledge dimension is entered alone (`V` + one dim) under the same
folds, and the paired difference against `V` is reported. This is what separates "the block carries
value" from "the value sits in particular quantities": five of the eleven context quantities carry
+1.1 to +3.0 pp, six carry nothing.

**Shift decomposition.** With one weighted functional
$a = \sum_t \pi_t[t]\,R[t,t] \big/ \sum_t \pi_t[t]$, where $R$ is a row-normalised confusion matrix,
three quantities are computed on the same scale: source conditionals with source priors, source
conditionals with target priors, and target conditionals with target priors. The first step attributes
the drop to the class mix, the second to the conditionals. Target labels are used for evaluation only.
Because signed components can over-explain a net difference, components are reported and percentage
shares are not.

**Guarantee constructions.** Conformal risk control is implemented as the largest accepted set whose
empirical risk stays within $\alpha$ (checked at $\alpha \in \{0.01, 0.05, 0.10\}$), with the threshold
keyed on the predicted class — the test-time key — and compared against a true-class oracle key.
Importance-weighted conformal uses a domain classifier's density ratio, clipped for stability and
estimated from the target port's unlabelled features only. Surrogate-risk control fits *(score,
label-free surrogates) → P(error)* on source episodes and accepts the largest prefix whose mean
predicted risk stays within $\alpha$. All three are judged on the **realised** risk at the target port,
which is the only quantity that can falsify them.

**Pooling ablation.** To test whether the forbidden AIS context can be reconstructed, all detections
of a port are pooled onto a 2 km grid in a per-port metric frame, one polarisation only (so a vessel is
not double counted), and the same eleven context quantities are recomputed — once pooled over the whole
port (all acquisitions) and once pooled over the object's own product (a single acquisition). Both are
evaluated with identical folds and labels; the only difference is coverage.

---

## 5. Implementation notes

- Features are stored as float16 and upcast per experiment; ridge/boosted classifiers run in the local
  Anaconda environment with `KMP_DUPLICATE_LIB_OK=TRUE`.
- The risk-family experiments run on a cached 512-dimensional randomised PCA of the 64-px features
  (`pca512_all.npy`), validated against the raw features beforehand (0.4722 vs 0.4735 balanced
  accuracy) — a simplification that keeps the quadratic terms inside the machine's memory.
- Per-port covariance inversions (class-conditional Gaussians, Mahalanobis scores) use a fixed
  0.05 shrinkage because many ports are rank-deficient at these dimensions.
- Runs are single-draw in the source-count curve and are labelled as such; the budget-curve results
  average 50 random redraws for the control arm.
- Typical wall-clock: a 24-port ridge arm is minutes, a 24-port gate with inner episodes and a boosted
  classifier is 20–40 minutes, the pooling ablation ~10 minutes.

---

## 6. Reproducibility inventory

| stage | script | output |
|---|---|---|
| pool construction | `e84_build_224.py`, `e109_build_all.py` | `dataset244_q/`, `dataset_all/` + `index.csv` |
| field-of-view features | `e106_fov.py` | `resnet50_{c64,c112,full224}.float16.npy` |
| novelty per class | `e117_novelty_perclass.py` | `e117_out.txt` |
| gate/portfolio arms | `e124_rank_budget.py`, `e125_rank_budget_strong.py` | `e124_out.txt`, `e125_out.txt` |
| compliant module | `e128_budget_clean.py` | `e128_out.txt` |
| conformal / weighted / surrogate | `e119…e123b_*.py` | matching `_out.txt` |
| attribution | `e134_shift_decomp.py` | `e134b_out.txt` |
| AIS vs detection, per-dimension | `e129_det_vs_ais.py`, `e130_per_dim.py` | `e129_out.txt`, `e130_out.txt` |
| SAR field feasibility and pooling | `e131_field_feasibility.py`, `e132_field_build.py`, `e133_pooling_ablation.py` | matching outputs |
| source-count curve, identity probe | `e136_source_count_curve.py`, `e137_sequence_oracle.py` | `e136b_out.txt`, `e137_out.txt` |
| compliant-expert enumeration | `e138_stronger_legal_expert.py` | `e138_out.txt` |

All scripts, outputs and the full experiment log (`EXPERIMENT_STATE.md`, entries E12–E19b) are versioned
together; the log records each pre-registered criterion, each result, and each withdrawal.
