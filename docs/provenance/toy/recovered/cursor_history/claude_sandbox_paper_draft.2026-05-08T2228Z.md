# Paper draft (outline)

Working title (pick one):
- **Schedule Modalities by Irreducible Content: A Conditional-Entropy Principle for Multi-Modality Flow Matching**
- **Why Structure First? Information-Theoretic Scheduling in Joint Flow Matching**
- **Conditional-Entropy Ordering for Multi-Modality Flow Matching**

Target venue: ICML SPIGM workshop. ~4–6 pages.

---

## Abstract (bullets)

- Multi-modality flow matching (continuous + categorical, e.g. molecules) requires a schedule per modality.
- Practitioners empirically find structure-first schedules (atoms/types denoised last) work best on QM9. No principled explanation in the literature.
- We show this is a consequence of a conditional-entropy asymmetry: $h(Y_1 \mid A_1)$ is large but $H(A_1 \mid Y_1) \approx 0$.
- We propose the **Conditional-Entropy Ordering Principle**: schedule modalities in decreasing order of $H(X^{(m)} \mid X^{(\neq m)})$ — modalities with high irreducible content given the others should denoise first.
- We compute these quantities directly from QM9 (no training), recover the empirically observed schedule order (structure → bonds → atoms+charges), and validate on a controlled toy where the asymmetry can be tuned.

---

## 1. Introduction

**Setup**
- Multi-modality flow matching for joint generation of continuous and categorical data.
- Concrete motivation: molecular generation — continuous atomic positions $Y$, categorical atom types $A$ (and bonds $E$, charges, etc.).
- Each modality has its own schedule $\alpha_m(t)$ controlling rate of denoising.

**The empirical puzzle**
- FlowMol-style architectures on QM9 reach best validity rates with **structure → bonds → atoms+charges** scheduling.
- Folk wisdom is that "discrete-late helps." No quantitative explanation; usually treated as a hyperparameter.

**Two competing intuitions in the literature**
- "Synchronize information clocks" — match channel-wise mutual-information rates about $A_1$ across $t$.
- "Discrete-ahead" — give the model categorical info early so geometry can use it.
- Neither predicts the empirical structure-first ordering.

**Our claim**
- Optimal schedule order is determined by a property of the data joint distribution: each modality's *irreducible information content given the others*.
- For QM9, $h(Y \mid A, E) \approx 53$ nats vs $H(A \mid Y, E) \approx 0.011$ nats — five orders of magnitude. This explains why structure-first dominates.

**Contributions**
1. A **Conditional-Entropy Ordering Principle** for multi-modality flow scheduling.
2. **Empirical analysis on QM9** confirming the principle predicts the observed schedule ordering.
3. **Toy validation** with a controllable manifold-per-class model that reproduces the structure-first preference and tracks the asymmetry magnitude.
4. **Negative result**: classical synchronization-via-MI is not predictive of optimal schedule choice.

---

## 2. Background

**Continuous–categorical flow matching**
- Source: $Y_0 \sim \mathcal N(0, I_d)$, $A_0 \sim \mathcal N(0, I_K)$.
- Endpoint: $Y_1$ = continuous data; $A_1 = e_{C_1}$ one-hot.
- Linear interpolant: $X_t^{(m)} = (1-\alpha_m(t)) X_0^{(m)} + \alpha_m(t) X_1^{(m)}$.
- Per-modality schedules $\alpha_m(t)$, typically of the form $t^{\gamma_m}$.
- Network predicts denoised endpoints $\hat X_1$; velocity derived as $b_t^{(m)} = \dot\alpha_m / (1-\alpha_m)\,(\hat X_1^{(m)} - X_t^{(m)})$.

**Existing scheduling heuristics**
- Shared $\alpha_y = \alpha_a = t$.
- Discrete-ahead ($\gamma_a < 1$): categorical resolves earlier.
- Discrete-delayed ($\gamma_a > 1$): categorical resolves later.

**Limits of the synchronization framing**
- The MI-mismatch metric $\mathcal M_{MI} = \int (I(Y_t; A_1)/H(A_1) - I(A_t; A_1)/H(A_1))^2 dt$ measures only one-sided informativeness about labels.
- Empirically not predictive of learning difficulty (preview of negative result).

---

## 3. Theory: Conditional-Entropy Ordering

**Setting**: joint distribution $p(X_1) = p(X_1^{(1)}, \dots, X_1^{(M)})$ over $M$ modalities.

**Definition (irreducible content):**
$$H_m^{\text{irr}} := H(X_1^{(m)} \mid X_1^{(\neq m)})$$
The entropy of modality $m$ remaining when all other modalities are clean. *Continuous case uses differential entropy.*

**Intuition:** $H_m^{\text{irr}}$ is the "work" that must be done by modality $m$'s own denoising channel, because no other modality can determine it. A modality with $H_m^{\text{irr}} \approx 0$ is "redundant given the rest" and can be denoised arbitrarily late.

### Theorem (Conditional-Entropy Ordering Principle, informal)

> *In multi-modality flow matching with linear-Gaussian interpolants, the schedule that minimizes time-integrated joint posterior uncertainty assigns earlier denoising (faster $\alpha_m$) to modalities with larger $H_m^{\text{irr}}$.*

Quantitative version (target):
$$\alpha_m^\star(t) \;\propto\; t^{1/\gamma_m^\star}, \qquad \gamma_m^\star \propto \frac{\bar H^{\text{irr}}}{H_m^{\text{irr}}}$$
i.e. modalities with large irreducible content get small $\gamma_m$ (faster schedule), modalities with small irreducible content get large $\gamma_m$ (slower).

**Proof sketch (planned):**
- Express joint posterior entropy at time $t$ as a sum over modalities of channel-specific information rates and conditional entropies.
- Show the optimal allocation of "schedule budget" minimizes residual joint entropy.
- This is a constrained optimization with KKT-type conditions; should give $\gamma_m^\star \propto 1/H_m^{\text{irr}}$ at leading order.

**Caveats**
- Theorem is for the *posterior optimum*, not the trained-network optimum. Architecture-dependent corrections exist (we observed these empirically) but the *ordering* is data-determined.
- Discrete trajectories (Gaussian-noised one-hot) have additional vertex-snap geometry near $t=1$; the irreducible-content recipe must be combined with adequate late-time budget for vertex commitment.

---

## 4. Empirical Analysis on QM9

**4.1 Setup**
- QM9 dataset: 97k molecules, atom types $A_1 \in \{H, C, N, O, F\}^L$, bond graphs $E_1$, structure $Y_1 \in \mathbb R^{3L}$.
- Plug-in mixture estimator for $I(\cdot; \cdot)$ using kernel density on the empirical dataset.
- Gaussian upper-bound for $h(Y_1)$ via sample covariance.

**4.2 Results (from your QM9 analysis)**

```
H(A_1)               5.41 nats
H(E_1)               8.11 nats
h(Y_1)              ≤ 61.1 nats (Gaussian UB)
H(A_1 | Y_1)         0.068
H(E_1 | Y_1)         0.068
H(A_1 | Y_1, E_1)    0.011    <-- atom irreducible content
H(E_1 | Y_1, A_1)    0.020    <-- bond irreducible content
h(Y_1 | A_1, E_1)   ≈ 53      <-- structure irreducible content
```

**4.3 Reading**

The irreducible-content ranking matches FlowMol's empirically optimal schedule order:

| modality | $H_m^{\text{irr}}$ (nats) | empirical schedule rank |
|---|---|---|
| Structure | ~53 | first |
| Bonds | 0.020 | second |
| Atoms | 0.011 | third |

- Structure has ~2500× more irreducible content than any single discrete modality → structure-first dominates.
- Bonds have ~1.8× more irreducible content than atoms → bonds before atoms.
- Both predicted orderings match the empirical optimum.

**4.4 Why classical synchronization fails here**
- The MI-mismatch metric measures only $I(\cdot; A_1)$ — class info, ignoring structural information.
- For QM9 the structural side is dominant; ignoring it produces incorrect γ recommendations.
- Concrete example: synchronization-recipe gives γ ≈ 0.4 (discrete-ahead); empirical optimum is γ > 1 (discrete-delayed).

---

## 5. Toy Validation: Manifold-per-Class

**5.1 Construction**
- $K$ classes, each represented by a 1D circle of radius $r_k$ in $\mathbb R^3$ at distinct centers.
- Sample $(C, Y)$: pick class $C$, sample uniformly along the circle, add small perpendicular Gaussian noise.
- Tunable parameters: $r_k$ (controls $h(Y \mid C)$); circle separation (controls $H(C \mid Y)$).

**5.2 Why this toy and not the previous ones**
- Earlier toys (Gaussian blobs) have $h(Y \mid C) \to 0$ in any high-SNR regime — no asymmetry to drive structure-first preference.
- Real molecules have non-trivial conformational entropy per atom-type set. Manifold-per-class captures this minimally.

**5.3 Predictions**
- $\gamma_a > 1$ (structure-first) wins for both MLP and DiT architectures, because the asymmetry is data-side.
- The optimal $\gamma_a$ tracks the asymmetry magnitude $h(Y\mid C) / H(C\mid Y)$.
- Schedules motivated by classical synchronization (γ < 1) lose, because they ignore structural information.

**5.4 Experimental design**
- $\gamma$-sweep across $[0.2, 12]$, both architectures (MLP, small DiT), 3 seeds.
- Metrics: ApproxError² against analytical Bayes posterior; generation quality (sample MMD on $Y$, class consistency on $A$).
- Vary asymmetry by changing circle radii: small radii → low $h(Y\mid C)$ → less benefit from structure-first; large radii → high $h(Y\mid C)$ → strong structure-first preference predicted.

**5.5 Negative-result baselines**
- Same toy with isotropic Gaussian classes (no per-class manifold): structure-first should *not* win, because asymmetry is absent.
- Architecture ablation (MLP vs DiT): same ordering predicted because principle is data-side.

---

## 6. Discussion

**What this paper does claim**
- A principled, data-driven recipe for schedule ordering in multi-modality flow matching.
- A diagnostic computable from data with no training (conditional entropies via plug-in or kNN estimators).
- Quantitative agreement with empirical FlowMol scheduling on QM9.

**What this paper does NOT claim**
- That synchronization (MI matching) is meaningless — it's a one-sided information measure that captures part of the picture but misses the structural dimension.
- That the matched γ is necessarily optimal end-to-end — architecture amplifications exist.
- That this generalizes to all multi-modality flow problems — the conditional-entropy asymmetry is a property of the data, and other domains might have different asymmetry structures.

**Limitations**
- Differential entropies estimated via Gaussian upper bounds; tighter estimates (kNN) would refine quantitative predictions but don't change the ordering conclusion.
- The toy is single-atom-per-class; real molecular flow has additional structure (bonds, charges, valences) that interact in ways not captured.
- The architecture–schedule interaction (we observe MLPs and DiTs prefer different γ in symmetric toys) is a separate phenomenon outside our principle's scope.

**Future work**
- Quantitative theorem with proof.
- Extension to discrete flow matching paths (CTMC, Dirichlet) where the categorical channel's geometry differs.
- Per-atom (rather than per-molecule) schedule allocation, where each atom's local conditional entropy determines its own schedule.

---

## 7. Related Work (placeholder)

- MultiFlow / FlowMol / EquiFM: multi-modality flow matching for molecules.
- Discrete diffusion: D3PM, MaskGIT, etc.
- Schedule design in single-modality diffusion: EDM, Karras et al.
- Information bottleneck and conditional entropy in generative models.

---

## Open items before submission

- [ ] Build manifold-per-class toy and run the predicted γ-sweep. Confirm both architectures show structure-first preference.
- [ ] Make the quantitative theorem rigorous, with proof in appendix (or a clean argument-sketch).
- [ ] Run the QM9 information clocks for the structural target ($Y_1$) — gives a complementary figure showing both clocks side-by-side.
- [ ] Tighten $h(Y_1)$ via kNN entropy estimator (optional; doesn't change qualitative claim but cleans up the asymmetry numbers).
- [ ] Pull figures from FlowMol experiment — γ-sweep validity rates as the empirical anchor.
- [ ] One Figure 1 candidate: side-by-side bar chart of $H_m^{\text{irr}}$ for QM9 modalities, next to the empirical FlowMol schedule rank. The visual alignment is the headline.

---

## Notes on framing

- We're claiming a *positive* principle (irreducible-content ordering predicts the schedule) and demonstrating it both theoretically and empirically.
- We acknowledge the synchronization-via-MI framing as one prior intuition; we explain why it's incomplete (one-sided about labels) but don't position the paper as a negative result against it.
- The QM9 evidence is the strongest piece: numbers computed directly from data align exactly with the empirically optimal schedule order. That's the take-home figure.
