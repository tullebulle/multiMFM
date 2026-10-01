# Research brief: ICML SPIGM workshop paper on scheduling in multimodal flow matching

You are an AI research assistant. Your task is to help produce a workshop-publishable research paper for the ICML SPIGM workshop on **scheduler design in multimodal flow matching**, especially motivated by mixed continuous–discrete molecular generation.

Work like a researcher, not like a passive assistant. Keep your own notes, maintain hypotheses, challenge weak assumptions, propose better experiments, and revise the theory when evidence contradicts the first idea. The goal is a coherent workshop paper with theory and experiments, not just code or plots.

---

## 1. Research objective

We want to understand the role of **modality-specific schedules** in multimodal flow matching for data of the form

\[
x=(y,a),
\]

where

\[
y\in \mathbb R^d
\]

is a continuous coordinate, such as molecular geometry, and

\[
a\in \{e_1,\dots,e_K\}\subset \mathbb R^K
\]

is a one-hot embedding of a semantic category, such as atom type, residue type, nucleotide identity, or color.

The motivating concern is that standard multimodal flow matching often uses a shared time schedule for both continuous and categorical coordinates:

\[
y_t=(1-t)y_0+t y_1,
\]

\[
a_t=(1-t)a_0+t a_1.
\]

This may be practically suboptimal because the continuous and categorical parts reveal endpoint information at different rates. Continuous uncertainty contracts smoothly, while categorical uncertainty collapses toward simplex vertices. A single shared time variable may therefore create posterior desynchronization.

The central research question is:

> How do modality-specific schedules affect learning difficulty, numerical integration, and guidance stability in continuous–categorical flow matching?

We do **not** want to claim that shared-time multimodal flow matching is mathematically invalid. With an ideal vector field and exact integration, any valid probability path gives the correct endpoint distribution. The claim should be more precise:

> Although an ideal multimodal flow can transport the source to the correct joint endpoint distribution, the schedule determines the geometry of intermediate conditional distributions. This affects finite-capacity learning, finite-step sampling, and inference-time guidance. Modality-specific schedules can improve practical generation by synchronizing posterior uncertainty across continuous and categorical modalities.

---

## 2. Desired paper positioning

Target: ICML SPIGM workshop.

Likely paper type: theory-motivated empirical study.

Possible titles:

1. **Posterior-Synchronized Scheduling for Multimodal Flow Matching**
2. **On Time Schedules for Continuous–Categorical Flow Matching**
3. **When Should Categories Denoise? Scheduling Mixed Discrete–Continuous Flow Models**
4. **Modality-Specific Schedules for Molecular Flow Matching**

The paper should be crisp, not overbroad. The strongest version has:

1. A clear mathematical toy model.
2. An analytical ideal flow.
3. A theory section explaining posterior desynchronization.
4. Experiments separating:
   - ideal transport,
   - vector-field approximation,
   - finite-step integration,
   - guidance stability.
5. A molecular motivation and, if feasible, a small molecule-inspired experiment.

---

## 3. Key conceptual framing

The data endpoint is

\[
X_1=(Y_1,A_1),
\]

but the category should be represented carefully.

Use notation:

\[
C_1\in\{1,\dots,K\}
\]

for the semantic category, and

\[
A_1=e_{C_1}\in\mathbb R^K
\]

for the one-hot Euclidean embedding used by the flow.

So the endpoint is

\[
X_1=(Y_1,A_1)=(Y_1,e_{C_1}).
\]

During interpolation,

\[
A_t=(1-\alpha_a(t))A_0+\alpha_a(t)A_1,
\]

where

\[
A_0\sim \mathcal N(0,I_K).
\]

Thus \(A_t\) is not itself a category during the path. It is an ambient Euclidean coordinate whose endpoint lies on the one-hot vertices.

The multimodal flow lives in

\[
\mathbb R^d\times \mathbb R^K.
\]

Its coordinatewise ideal velocity is

\[
b_t(x_t)
=
(b_t^y(y_t,a_t), b_t^a(y_t,a_t)),
\]

where

\[
b_t^y(y_t,a_t)
=
\frac{\dot \alpha_y(t)}{1-\alpha_y(t)}
\left(
\mathbb E[Y_1\mid Y_t=y_t,A_t=a_t]-y_t
\right),
\]

\[
b_t^a(y_t,a_t)
=
\frac{\dot \alpha_a(t)}{1-\alpha_a(t)}
\left(
\mathbb E[A_1\mid Y_t=y_t,A_t=a_t]-a_t
\right).
\]

Both are determined by the same joint posterior:

\[
p(Y_1,C_1\mid Y_t,A_t).
\]

This is the key object.

---

## 4. Main hypothesis

The main hypothesis is:

> Schedules matter because they control the relative rate at which \(Y_t\) and \(A_t\) reveal information about the endpoint. If the categorical posterior remains high-entropy while the continuous coordinate is already making large geometric decisions, the continuous velocity averages over incompatible branches. If the categorical coordinate collapses too early, the model may prematurely commit to the wrong branch. Good schedules synchronize posterior uncertainty across modalities.

Use posterior uncertainty measures:

\[
U_y(t)
=
\operatorname{Tr}\operatorname{Cov}(Y_1\mid Y_t,A_t),
\]

\[
U_a(t)
=
H(C_1\mid Y_t,A_t).
\]

A posterior-synchronized schedule should approximately satisfy:

\[
\frac{U_y(t)}{U_y(0)}
\approx
\frac{U_a(t)}{U_a(0)}.
\]

This is a principle, not necessarily a closed-form optimality theorem.

---

## 5. Analytical toy model

Use the following synthetic distribution:

\[
C_1\sim \mathrm{Categorical}(\pi),
\]

\[
A_1=e_{C_1},
\]

\[
Y_1\mid C_1=k\sim \mathcal N(\mu_k,\sigma^2 I_d).
\]

The source is

\[
Y_0\sim \mathcal N(0,I_d),
\qquad
A_0\sim \mathcal N(0,I_K).
\]

The path is

\[
Y_t=(1-\alpha_y(t))Y_0+\alpha_y(t)Y_1,
\]

\[
A_t=(1-\alpha_a(t))A_0+\alpha_a(t)A_1.
\]

Derive and implement the exact posterior and ideal velocity.

For fixed \(t\), define

\[
s_y^2(t)=(1-\alpha_y(t))^2+\alpha_y(t)^2\sigma^2,
\]

\[
s_a^2(t)=(1-\alpha_a(t))^2.
\]

Conditional on \(C_1=k\),

\[
Y_t\mid C_1=k
\sim
\mathcal N(\alpha_y(t)\mu_k,s_y^2(t)I_d),
\]

\[
A_t\mid C_1=k
\sim
\mathcal N(\alpha_a(t)e_k,s_a^2(t)I_K).
\]

Therefore the posterior class probability is

\[
w_k(t,y,a)
=
\mathbb P(C_1=k\mid Y_t=y,A_t=a)
\]

with

\[
w_k
=
\operatorname{softmax}_k
\left[
\log\pi_k
-\frac{\|y-\alpha_y(t)\mu_k\|^2}{2s_y^2(t)}
-\frac{\|a-\alpha_a(t)e_k\|^2}{2s_a^2(t)}
\right].
\]

Given \(C_1=k\), the posterior for \(Y_1\) is Gaussian:

\[
Y_1\mid Y_t=y,C_1=k
\sim
\mathcal N(m_k(t,y),C_y(t)),
\]

where

\[
C_y(t)
=
\left(
\sigma^{-2}I
+
\frac{\alpha_y(t)^2}{(1-\alpha_y(t))^2}I
\right)^{-1},
\]

and

\[
m_k(t,y)
=
C_y(t)
\left(
\sigma^{-2}\mu_k
+
\frac{\alpha_y(t)}{(1-\alpha_y(t))^2}y
\right).
\]

Then

\[
\mathbb E[Y_1\mid Y_t=y,A_t=a]
=
\sum_{k=1}^K w_k(t,y,a)m_k(t,y),
\]

and

\[
\mathbb E[A_1\mid Y_t=y,A_t=a]
=
\sum_{k=1}^K w_k(t,y,a)e_k.
\]

Thus the ideal velocity is analytic:

\[
b_t^y(y,a)
=
\frac{\dot \alpha_y(t)}
{1-\alpha_y(t)}
\left(
\sum_{k=1}^K w_k(t,y,a)m_k(t,y)-y
\right),
\]

\[
b_t^a(y,a)
=
\frac{\dot \alpha_a(t)}
{1-\alpha_a(t)}
\left(
\sum_{k=1}^K w_k(t,y,a)e_k-a
\right).
\]

This analytical model is the foundation of the paper.

---

## 6. Important theoretical nuance

Do not say:

> Shared schedules fail even for ideal flows.

That is wrong.

The correct statement is:

> With the exact ideal vector field and exact integration, the flow reaches the correct endpoint distribution for any valid schedule. However, schedules change the intermediate posterior geometry and can make the vector field harder to approximate, stiffer to integrate, and less reliable under guidance.

Use this distinction repeatedly.

---

## 7. Theory goals

Develop theory around the following ideas.

### 7.1 Branch averaging

For the mixture model,

\[
\mathbb E[Y_1\mid Y_t,A_t]
=
\sum_k w_k(t,Y_t,A_t)m_k(t,Y_t).
\]

If the class posterior \(w_k\) is high entropy, the continuous denoiser is a mixture average.

For separated means \(\mu_k\), this average may lie between modes. That region may have low endpoint data density.

This does not invalidate ideal transport, but it can make the learned vector field difficult because the model must represent sharp posterior transitions.

Prove a simple proposition:

> In a separated Gaussian mixture, if \(H(C_1\mid Y_t,A_t)\) is high while the conditional within-class posterior variance is small, then the continuous posterior mean lies near a convex combination of separated branch means. Unless one class dominates, this point is far from every branch mean.

A possible bound:

Let

\[
\bar \mu(w)=\sum_k w_k\mu_k.
\]

If for all \(k\), \(w_k\le 1-\epsilon\), and the means are separated by \(\Delta\), then under symmetric/simplex arrangements, \(\bar \mu(w)\) is bounded away from at least the selected branch mean. Work out a clean special case first, e.g. \(K=2\), \(\mu_1=-\mu\), \(\mu_2=\mu\).

For \(K=2\),

\[
\mathbb E[Y_1\mid\cdot]\approx (2w-1)\mu.
\]

When \(w\approx 1/2\), the denoiser points toward 0, the barycenter between modes.

### 7.2 Posterior synchronization

Define

\[
U_y(t)=\mathbb E\left[\operatorname{Tr}\operatorname{Cov}(Y_1\mid Y_t,A_t)\right],
\]

\[
U_a(t)=\mathbb E\left[H(C_1\mid Y_t,A_t)\right].
\]

The expectation over \(X_t\) is important if computing global curves.

A schedule is posterior-synchronized if the normalized curves decay together:

\[
\frac{U_y(t)}{U_y(0)}
\approx
\frac{U_a(t)}{U_a(0)}.
\]

Define a mismatch score:

\[
\mathcal M(\alpha_y,\alpha_a)
=
\int_0^1
\left[
\frac{U_y(t)}{U_y(0)}
-
\frac{U_a(t)}{U_a(0)}
\right]^2dt.
\]

Hypothesize that lower mismatch correlates with easier learning, lower finite-step error, and more stable guidance.

### 7.3 Stiffness

Schedules enter the velocity through:

\[
\frac{\dot\alpha_m(t)}{1-\alpha_m(t)}.
\]

Aggressive schedules can increase stiffness or create sharp posterior transitions. Analyze or measure a stiffness proxy:

\[
S_{\text{proxy}}
=
\mathbb E\int_0^1
\|\nabla_x b_t^\star(X_t)\|_F^2dt.
\]

Hypothesize a tradeoff:

- categorical-ahead helps branch selection,
- too much categorical-ahead creates stiffness or premature commitment.

This may produce an inverted-U curve for generation quality versus schedule lead.

---

## 8. Schedule families to test

Use smooth schedules.

### Shared

\[
\alpha_y(t)=t,\qquad \alpha_a(t)=t.
\]

### Discrete-ahead

\[
\alpha_y(t)=t,\qquad \alpha_a(t)=t^\gamma,
\qquad 0<\gamma<1.
\]

Since \(t^\gamma>t\) for \(t\in(0,1)\), the categorical coordinate is denoised earlier.

Try:

\[
\gamma\in\{0.25,0.5,0.75\}.
\]

### Discrete-delayed

\[
\alpha_y(t)=t,\qquad \alpha_a(t)=t^\gamma,
\qquad \gamma>1.
\]

Try:

\[
\gamma\in\{1.5,2,3\}.
\]

### Continuous-ahead

\[
\alpha_y(t)=t^\gamma,\qquad \alpha_a(t)=t,
\qquad 0<\gamma<1.
\]

### Posterior-synchronized

Search over a parametric family:

\[
\alpha_y(t)=t,
\qquad
\alpha_a(t)=t^\gamma,
\]

choose \(\gamma\) minimizing \(\mathcal M\).

Optionally search over a richer family:

\[
\alpha_a(t)=\mathrm{BetaCDF}(t;p,q),
\]

or a monotone spline.

Keep it simple for the first paper.

---

## 9. Experimental program

Conduct experiments in increasing complexity.

---

### Experiment 1: posterior geometry diagnostics, no learning

Purpose: Show how schedules affect posterior uncertainty and branch averaging.

Setup:

- \(d=2\)
- \(K=3\) or \(4\)
- \(\pi_k=1/K\)
- \(\mu_k\) placed on a circle of radius \(R\)
- \(\sigma=0.3\)
- \(R\in\{1,2,4,8\}\)

Compute by Monte Carlo:

\[
U_y(t)=\mathbb E[\operatorname{Tr}\operatorname{Cov}(Y_1\mid Y_t,A_t)],
\]

\[
U_a(t)=\mathbb E[H(C_1\mid Y_t,A_t)].
\]

Also compute branch ambiguity:

\[
\mathbb E[1-\max_k w_k(t,Y_t,A_t)].
\]

Compute barycentric denoiser error:

\[
D_{\text{branch}}(t)
=
\mathbb E\left[
\min_k
\left\|
\mathbb E[Y_1\mid Y_t,A_t]-\mu_k
\right\|
\right].
\]

Expected figure:

- Curves of \(U_y/U_y(0)\) and \(U_a/U_a(0)\) under each schedule.
- Show shared time may desynchronize the curves.
- Show discrete-ahead can align them better.

---

### Experiment 2: finite-step integration of analytic flow

Purpose: Even with exact velocity, schedules may differ in finite-NFE sampling.

Use exact analytic velocity.

Integrate with Euler or RK4 using:

\[
N\in\{5,10,20,50,100,200\}.
\]

At endpoint, decode category:

\[
\hat C=\arg\max_j A_{1,j}.
\]

Metrics:

\[
\text{branch error}
=
\|Y_1^{\text{gen}}-\mu_{\hat C}\|,
\]

\[
\text{validity}
=
\mathbf 1\{
\|Y_1^{\text{gen}}-\mu_{\hat C}\|\le r
\},
\]

with

\[
r=2\sigma\sqrt d.
\]

Also measure:

\[
\mathrm{KL}(\hat p(\hat C)\|\pi),
\]

and MMD/Wasserstein distance between generated \(Y\) and true mixture.

Expected result:

- With large NFE, all schedules approach correctness.
- With small NFE, schedules differ.
- Aggressive schedules may be stiff.
- Moderate posterior-synchronized schedules may achieve better low-NFE quality.

This experiment tests numerical stiffness and practical solver behavior.

---

### Experiment 3: learned vector field approximation

Purpose: Test whether schedules change the difficulty of learning the flow.

Train the same network for each schedule to predict the analytic ideal velocity.

Input:

\[
(t,Y_t,A_t).
\]

Output:

\[
(\hat b^y,\hat b^a).
\]

Target:

\[
b^\star_t(Y_t,A_t).
\]

Use the same architecture, optimizer, batch size, and training steps across schedules.

Suggested architecture:

- MLP for the toy setting.
- Input dimension \(1+d+K\).
- 4 hidden layers, width 256.
- SiLU activations.
- Output dimension \(d+K\).

Loss:

\[
\mathcal L
=
\mathbb E
\left[
\| \hat b^y-b^{y,\star}\|^2
+
\lambda_a
\| \hat b^a-b^{a,\star}\|^2
\right].
\]

Set \(\lambda_a\) to normalize scales; start with 1, then possibly use inverse target variance.

Evaluate:

- velocity MSE overall,
- velocity MSE by time bin,
- endpoint sample quality after integrating learned flow,
- branch error,
- validity,
- category marginal KL.

Expected result:

Schedules with lower posterior mismatch \(\mathcal M\) should have lower learned velocity error and better endpoint quality at equal training budget.

Important: If all schedules work equally well in \(d=2,K=3\), increase difficulty:

- larger \(K\),
- overlapping modes,
- high-dimensional nuisance coordinates,
- lower network capacity,
- fewer training steps.

---

### Experiment 4: guidance stability under coupled rewards

Purpose: This is closest to molecular generation and MFM guidance.

Use the analytic base flow. Add a reward coupling continuous coordinate and category.

Define category-specific target locations:

\[
\nu_k
\]

possibly different from \(\mu_k\). Reward:

\[
r(y,C)=-\|y-\nu_C\|^2.
\]

For soft category embedding \(a\), use:

\[
r(y,a)
=
-\sum_{k=1}^K \mathrm{softmax}(a/\tau)_k \|y-\nu_k\|^2.
\]

Alternatively compute exact posterior value:

\[
V_t(y,a)
=
\log
\mathbb E[
\exp(\beta r(Y_1,C_1))
\mid Y_t=y,A_t=a
].
\]

Estimate \(\nabla V_t\) by Monte Carlo posterior samples, mimicking MFM lookahead guidance, or derive it analytically for the Gaussian toy if feasible.

Guided dynamics:

\[
\dot x_t
=
b_t^\star(x_t)
+
\lambda(t)\nabla_x V_t(x_t).
\]

Compare schedules.

Metrics:

- target error:

\[
\|Y_1^{\text{gen}}-\nu_{\hat C}\|,
\]

- guided validity:

\[
\hat C=\arg\max a_1,\quad Y_1\text{ near }\nu_{\hat C},
\]

- category-geometry mismatch rate:

\[
\arg\min_k \|Y_1-\nu_k\|\neq \hat C,
\]

- off-distribution rate under the tilted target,
- sensitivity to guidance strength \(\beta\).

Expected result:

Posterior-synchronized or moderate discrete-ahead schedules should be more stable under guidance. Shared or continuous-ahead schedules may average incompatible rewards when category posterior is uncertain. Too aggressive discrete-ahead may prematurely commit and reduce reward optimization.

---

### Experiment 5: molecule-inspired toy geometry

If time permits, build a slightly more molecular synthetic dataset.

Examples:

#### Colored bond system

Two or three particles in \(\mathbb R^2\) or \(\mathbb R^3\), with category representing atom type. Different type pairs imply different preferred bond lengths:

\[
\|y_i-y_j\|\approx \ell_{c_i,c_j}.
\]

Generate data from Gaussian noise around these type-dependent bond lengths.

The endpoint is:

\[
Y_1=(r_1,\dots,r_n)\in\mathbb R^{nd},
\]

\[
A_1=(e_{c_1},\dots,e_{c_n}).
\]

Test whether schedules affect type-geometry consistency.

Metrics:

- bond length error relative to decoded types,
- invalid geometry rate,
- category marginal KL,
- finite-step and learned-flow quality.

This bridges the simple category-conditional Gaussian to molecular generation.

Keep this optional unless the core toy experiments are already strong.

---

## 10. Dimensionality plan

Start with:

\[
d=2
\]

for visualization.

Then perform ablations:

\[
d\in\{2,8,32,128\}.
\]

Dimensionality matters through signal-to-noise and nuisance directions.

Two designs:

### Informative high-dimensional means

Choose \(\mu_k\in\mathbb R^d\) with pairwise separation:

\[
\|\mu_i-\mu_j\|\approx c\sigma\sqrt d.
\]

This keeps classification difficulty roughly stable.

### Low-rank informative means

Only first two dimensions distinguish categories:

\[
\mu_k=(\tilde \mu_k,0,\dots,0).
\]

The remaining coordinates are nuisance. This is more realistic: in molecular systems, only some geometric features may determine type consistency.

Hypothesis:

- If high-dimensional geometry strongly reveals category, schedules matter less.
- If category information is low-rank or ambiguous, schedules matter more.

---

## 11. Notes and research log requirements

Maintain a research log with:

1. Current hypotheses.
2. Experimental setup.
3. Results.
4. Interpretation.
5. Failure cases.
6. Revised hypotheses.

Do not only record successful experiments. Negative results are important.

For every experiment, answer:

- What question does this answer?
- What would support the hypothesis?
- What would falsify the hypothesis?
- Did the result isolate the mechanism, or could it be due to confounding factors?
- What is the next simplest experiment?

---

## 12. Expected paper figures

Aim for these figures.

### Figure 1: conceptual posterior synchronization

Plot normalized uncertainty curves:

\[
U_y(t)/U_y(0),
\qquad
U_a(t)/U_a(0).
\]

Show shared schedule versus posterior-synchronized schedule.

### Figure 2: branch averaging diagram

For \(K=2\) or \(K=3\), show posterior mean trajectories. Illustrate how high-entropy category posterior pulls the continuous denoiser toward an inter-branch barycenter.

### Figure 3: finite-step analytic flow

Endpoint validity versus NFE for schedules.

### Figure 4: learned vector field

Velocity MSE and endpoint validity for learned networks under schedules.

### Figure 5: guidance stability

Guided target success versus guidance strength for schedules.

Optional Figure 6: molecule-inspired bond system.

---

## 13. Expected paper claims

Safe, strong claims:

1. The schedule in multimodal flow matching determines the joint posterior geometry seen during training and sampling.
2. In continuous–categorical data, posterior uncertainty in geometry and category can decay at different rates.
3. This desynchronization can create branch-averaged denoisers and sharp posterior transitions.
4. Even when ideal transport is correct, schedules affect practical performance through approximation error, numerical integration, and guidance stability.
5. Posterior uncertainty curves provide useful diagnostics for choosing modality-specific schedules.
6. Moderate categorical-ahead or posterior-synchronized schedules can improve practical multimodal generation in controlled settings.

Avoid overclaims:

1. Do not say shared schedules are invalid.
2. Do not say discrete-ahead is always optimal.
3. Do not claim molecule SOTA unless actually shown.
4. Do not claim a universal schedule independent of data distribution.
5. Do not imply categories are integers. Always distinguish semantic category \(C\) from one-hot embedding \(A=e_C\).

---

## 14. Draft paper outline

```text
Title: Posterior-Synchronized Scheduling for Multimodal Flow Matching

Abstract
- Mixed continuous-categorical data arise in molecular design.
- Standard multimodal flow matching often uses a shared time schedule.
- We study how schedules affect intermediate posterior geometry.
- Analytical Gaussian-mixture/category model gives exact ideal velocities.
- We show schedules influence approximation difficulty, finite-step integration, and guidance stability.
- Posterior uncertainty synchronization is a useful scheduling principle.

1. Introduction
- Molecular data combine geometry and types.
- Flow matching is attractive but multimodal scheduling is underexplored.
- Shared time implicitly assumes modalities denoise at compatible semantic rates.
- Contributions.

2. Background: Continuous-Categorical Flow Matching
- x=(y,a), C semantic category, a=e_C.
- Modality-specific interpolants.
- Ideal velocity as conditional endpoint denoiser.

3. Analytical Model and Posterior Geometry
- Category-conditional Gaussian model.
- Exact posterior weights and ideal velocity.
- Branch averaging.
- Posterior uncertainty measures U_y and U_a.

4. Scheduling Principles
- Shared, categorical-ahead, categorical-delayed, continuous-ahead.
- Posterior mismatch score.
- Hypothesis: synchronized schedules ease learning/integration/guidance.

5. Experiments
5.1 Posterior geometry diagnostics
5.2 Finite-step analytic sampling
5.3 Learned vector-field approximation
5.4 Coupled reward guidance
5.5 Optional molecule-inspired geometry

6. Discussion
- Implications for molecular flow matching.
- Relation to inference-time guidance and MFM.
- Limitations.
- Future work: real molecular benchmarks and adaptive schedules.