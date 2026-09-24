# Selective Knowledge Transfer — problem formalization (v1, draft for review)

Every claim below is paired with its evidence and its status: **[measured]**, **[proved]**, **[open]**.
Numbers are from the 8-port protocol unless marked otherwise.

---

## 1. Setting

Domains are **ports** $p \in \mathcal P$; the deployment target is one held-out port $t$, the sources are
$\mathcal S = \mathcal P \setminus \{t\}$. Instances $i = (x_i, k_i, y_i)$: a SAR chip $x_i$ (crop of a fixed metric
extent), a **scene-knowledge vector** $k_i$ (facility proximity, operational-chain composites, local context),
label $y_i \in \mathcal C$, $|\mathcal C| = 8$ known classes, plus a disjoint set of unseen types $\mathcal U$.

Two frozen arms:
$$f_V : x \mapsto \hat y, \qquad f_K : (x, k) \mapsto \hat y .$$

**Definition 1 (per-instance knowledge gain).**
$$G_i \;=\; \ell\big(y_i, f_V(x_i)\big) \;-\; \ell\big(y_i, f_K(x_i)\big), \qquad G_i \in \{-1, 0, +1\}$$
with sign classes *rescue* $(G=+1)$, *harm* $(G=-1)$, *neutral* $(G=0)$. Convention $\ell \in \{0,1\}$ (BA is the
outer metric, per protocol).

**Definition 2 (selective transfer policy and its regret).** A policy $\pi: (x,k) \mapsto \{V, K\}$ decides per
instance. Its realised gain and regret against the oracle policy are
$$\mathrm{Gain}(\pi) = \mathbb E[G_i \pi_i], \qquad
R(\pi) = \underbrace{\mathbb E[G_i]}_{\text{ceiling}} - \mathrm{Gain}(\pi),$$
where the ceiling is the oracle's gain, $\mathrm{Gain}^\star = \mathbb E[\max(0, G_i)]$.

---

## 2. The two propositions this paper rests on

**P1 (the sign is a *class* property, not a *domain* property).**
Across the 24-port pool, per-class sign consistency is $100\%$ for the extreme classes — bulk carrier positive in
**24/24** ports (mean $+41.0$ pp), crude oil tanker negative in **9/9** (mean $-16.2$ pp) — while the port-level
average is *actively misleading*: BA and composition-weighted accuracy disagree in **sign** on $3/24$ ports
(Santos $-0.5$ vs $+18.9$; Hamburg $+10.3$ vs $-4.8$). **[measured]**

**P2 (the sign is estimable without target labels).**
Trained on source-port leave-one-port-out episodes only, a selector over $\{f_V, f_K\}$ predicts the sign of $G_i$ at
AUC $0.71$–$0.74$ (median $0.72$; above $0.6$ in $17/23$ ports). **[measured]**

Together: *applicability is a $\mathcal C \times$ instance predicate, and it is learnable from sources.*

---

## 3. The obstruction that makes this non-trivial

**P3 (identifiability wall).** For the knowledge features used here, class information and port identity are
entangled: the per-dimension class-informativeness and port-identifiability scores correlate at
$r = 0.771$ across the 71 legal dimensions; the subset that is class-informative *and* port-stable is **empty**
(0 of 71). Consequently any **domain-level** selection rule is confounded by construction — which is exactly what the
negative results show: domain-conditioned localisation $-5.61$ pp, conservative $\lambda$-gating with
$\lambda^\star = 0$ in $5/24$ ports, region narrowing worse than a size-matched random subset, source-port selection
with its mean criterion **anti-predictive** of transfer ($r = -0.70$, $p = 10^{-3}$). **[measured]**
⇒ the allocation unit must be the **instance conditioned on the class**, not the domain.

---

## 4. The quantity that connects selector quality to realised gain

**Definition 3 (selectivity–regret curve).** For a selector of given ranking quality $a$ (AUC on the non-neutral
instances), let $R(a)$ be the mean regret achieved by a selector of that quality against the *real* distribution of
$(G_i)$.

Measured on 25,893 known-class instances (rescue 617 / harm 313 / neutral 24,963), ceiling $+2.91$ pp: **[measured]**

| $a$ | 0.50 | 0.60 | 0.70 | 0.75 | **0.80** |
|---|---|---|---|---|---|
| ceiling captured | 29 % | 42 % | 67 % | 86 % | **100 %** |

Two consequences, both non-obvious:
- **A perfect selector is not needed**: because harm is rare ($1.2\%$), AUC $0.80$ saturates the ceiling.
- **AUC is not sufficient**: the deployed gate has $a \approx 0.72$–$0.75$, which the curve maps to $67$–$86\%$, yet it
  realises $37\%$. **[open]** The missing link is *tail* judgement, and it is not fixable by thresholding: the
  correct decision label $(L_K \ge L_V)$ *is* the argmax boundary, so post-hoc calibration provably cannot move it
  (Platt arm $\equiv$ the visual arm, "choose $K$" fires with probability $0.000$). **[proved]** + **[measured]**

---

## 5. Method (what the paper would actually contribute)

A framework, not a network — the depth experiments are negative and are reported as such
(encoder swap $-2.7$ pp for a ViT; the deep module $0.197 <$ ridge $0.215$; joint training $-3.56$ pp, $p=0.013$).
The method is the triple:

1. **Representation**: report knowledge in the target's *own* coordinates (within-port rank), which drops
   port-identifiability to $0.000$ and raises class-informativeness $0.0172 \to 0.1386$. **[measured]**
2. **Estimator**: an applicability score trained on source-port leave-one-out episodes; the ceiling for any such
   selector is the oracle, which is empirically reachable in quality terms ($a \ge 0.80$ suffices).
3. **Protocol**: leave-one-port-out with per-class cross-port sign counts, BA **and** composition-weighted accuracy
   side by side, oracle and non-deployable arms marked as such.

**Falsifiable predictions** (each is a check a referee can run):
- P1: hold out any new port; per-class signs should reproduce on the classes with $n \ge 20$. **[open, testable]**
- P2: a selector trained on sources should stay above AUC $0.6$ on the held-out port. **[open, testable]**
- P3: a domain-conditioned variant should not beat a size-matched random subset. **[measured: it loses]**
- §4: a selector with $a \ge 0.80$ should capture $\ge 95\%$ of the ceiling. **[open, testable]**

---

## 6. Honest status

| claim | status |
|---|---|
| P1 class-conditional sign | measured, 24 ports |
| P2 label-free sign estimation | measured, $a \approx 0.72$ |
| P3 identifiability wall / domain rules confounded | measured (5 independent negative families) |
| selectivity–regret curve | measured |
| reaching the curve at a given $a$ | **open** — tail judgement; thresholding proved insufficient |
| any gain from deeper models | **refuted** (3 independent negative results) |
| open-set novelty | **absent in this data** (unseen types are ordinary ships: visual AUC 0.493, knowledge-side 0.509, size 0.474) |

**What the paper claims**: a mechanism-level, falsifiable account of *when auxiliary scene knowledge transfers*, with
a protocol and a benchmark, and explicit negative evidence on capacity, on domain-level allocation, and on open-set
novelty in this observation regime.
**What it does not claim**: a new architecture, an open-set solution, or a universal gain.
