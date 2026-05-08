# Constrained Bayesian Optimisation with Transfer Learning

This document describes the constrained Bayesian-optimisation (BO) method
with two transfer-learning components — a **ranking-weighted GP ensemble
(RGPE)** surrogate and a **voted feasibility classifier** — used in this
project. It is written to be self-contained for use as a thesis methods
section. The notation follows machine-learning conventions; deviations
from the original references are explicitly noted.

> **Notation conventions.** Calligraphic letters $\mathcal{X}, \mathcal{D}$
> denote sets; bold lowercase $\mathbf{x}$ denotes vectors; uppercase
> non-bold $X, Y$ denote stacked matrices/vectors of observations.
> Probability density and probability mass are both written $p(\cdot)$;
> context disambiguates.

---

## 1. Problem statement

Let $\mathcal{X} \subset \mathbb{R}^d$ be a continuous box-bounded search
space, $f : \mathcal{X} \to \mathbb{R}$ a black-box objective accessible
only through expensive (and potentially noisy) evaluations, and
$c : \mathcal{X} \to \{0, 1\}$ a black-box binary feasibility indicator
that is co-evaluated with $f$ at every query.

We consider the **constrained minimisation** problem

$$
\mathbf{x}^\star \;=\; \arg\min_{\mathbf{x}\in\mathcal{X}} f(\mathbf{x})\quad \text{s.t.}\quad c(\mathbf{x}) = 1
$$

under a tight evaluation budget. We additionally have access to $K$
**source tasks** with collected datasets

$$
\mathcal{D}_k \;=\; \{(\mathbf{x}_i^{(k)},\; y_i^{(k)},\; c_i^{(k)})\}_{i=1}^{N_k},\qquad k = 1, \ldots, K
$$

drawn from related but non-identical environments. The transfer-learning
goal is to exploit these source data to accelerate convergence and to
reduce the number of infeasible evaluations on the target task.

In this project, two concrete problems instantiate this setting:

1. **Laser cutting** — $f$ is the burr height after a cut, $c$ is the
   binary "no-miscut" feasibility, $\mathbf{x}=(\text{feedrate},
   \text{gas pressure}, \text{focal position})$. Source tasks differ in
   material, thickness, and machine.
2. **Pressure-chamber PID tuning** — $f$ is the integral-of-time-times-
   absolute-error (ITAE) for a step response, $c$ is the binary
   "stable controller" indicator, $\mathbf{x}=(K_p, K_i, K_d)$. Source
   tasks differ in chamber volume and mass-flow rate.

---

## 2. Background: Bayesian optimisation

Bayesian optimisation maintains a posterior $p(f \mid \mathcal{D})$ over
the objective from data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$
and selects each new query by maximising an **acquisition function**
$\alpha : \mathcal{X} \to \mathbb{R}$:

$$
\mathbf{x}_{n+1} \;=\; \arg\max_{\mathbf{x}\in\mathcal{X}}\; \alpha\!\left(\mathbf{x}\,\big|\,\mathcal{D}_n\right).
$$

### 2.1 Gaussian-process surrogate

We use a zero-mean Gaussian process with kernel $k_\theta$ as surrogate
for $f$:

$$
f \sim \mathcal{GP}(0,\, k_\theta(\cdot, \cdot)),
$$

trained on observations $X = [\mathbf{x}_1, \ldots, \mathbf{x}_n]^\top$,
$\mathbf{y} = [y_1, \ldots, y_n]^\top$. The posterior at a test point
$\mathbf{x}$ is Gaussian with closed-form mean and variance

$$
\mu(\mathbf{x}) \;=\; \mathbf{k}^\top (K + \sigma_n^2 I)^{-1}\,\mathbf{y}, \qquad
\sigma^2(\mathbf{x}) \;=\; k_\theta(\mathbf{x},\mathbf{x}) - \mathbf{k}^\top (K + \sigma_n^2 I)^{-1}\,\mathbf{k},
$$

where $K_{ij} = k_\theta(\mathbf{x}_i, \mathbf{x}_j)$ and
$\mathbf{k}_i = k_\theta(\mathbf{x}_i, \mathbf{x})$. We use a Matérn-5/2
kernel with a learned length-scale per input dimension and a small
WhiteKernel for observation noise. Hyperparameters $\theta$ are fitted
by marginal-likelihood maximisation with a few random restarts.

### 2.2 Expected Improvement

For minimisation with current best feasible value $f^\star = \min_i y_i$
(over feasible points), the **Expected Improvement** acquisition is

$$
\alpha_{\rm EI}(\mathbf{x}) \;=\; \mathbb{E}_{f\sim p(f\mid\mathcal{D})}\!\big[\max(f^\star - f(\mathbf{x}) - \xi,\;0)\big]
$$

with closed form under the Gaussian posterior:

$$
\alpha_{\rm EI}(\mathbf{x}) \;=\; (f^\star - \mu(\mathbf{x}) - \xi)\,\Phi(Z) \;+\; \sigma(\mathbf{x})\,\phi(Z),
\qquad Z \;=\; \frac{f^\star - \mu(\mathbf{x}) - \xi}{\sigma(\mathbf{x})}
$$

where $\Phi$ and $\phi$ are the standard-normal CDF and PDF, and $\xi \ge 0$
is a small exploration parameter (we use $\xi = 0.01$).

### 2.3 Constrained BO via probabilistic feasibility

When evaluations carry a binary feasibility outcome, the standard recipe
[Gardner et al., 2014; Gelbart et al., 2014] is to **gate the EI by the
posterior probability of feasibility**:

$$
\alpha_{\rm cEI}(\mathbf{x}) \;=\; \alpha_{\rm EI}(\mathbf{x}) \,\cdot\, \Pr(c(\mathbf{x}) = 1)
$$

so that points predicted infeasible receive a vanishingly small
acquisition value while feasible regions retain their EI weighting.

We generalise this multiplier with a **constraint-sharpness exponent**
$\beta \ge 0$,

$$
\boxed{\;\alpha_{\beta}(\mathbf{x}) \;=\; \alpha_{\rm EI}(\mathbf{x}) \,\cdot\, \big[\Pr(c(\mathbf{x}) = 1)\big]^\beta\;}
$$

which interpolates between three regimes:

| $\beta$ | regime | interpretation |
|---|---|---|
| $0$ | no constraint | $p^\beta = 1$ everywhere; reduces to plain EI |
| $1$ | standard Gardner cEI | the canonical product form |
| $\beta \to \infty$ | hard mask | only $\Pr(c{=}1) = 1$ candidates survive |

In our laser-cutting study (Section 5) we sweep $\beta \in \{0, 0.25, 0.5,
1, 2, 4, 8\}$ and observe a Pareto-optimal elbow at $\beta = 1$. The PID
study uses $\beta = 1$ throughout.

---

## 3. Surrogate transfer: ranking-weighted GP ensemble (RGPE)

Plain BO ignores all source data. We adopt the **ranking-weighted
Gaussian-process ensemble (RGPE)** of [Feurer et al., 2018] to inject a
learned prior from the source tasks into the target's surrogate.

### 3.1 Per-task GPs

We fit one GP per source task on its feasible-only objective values,

$$
\mathcal{GP}_k\;:\;\big(\mathbf{x}, \mu_k(\mathbf{x}),\,\sigma_k^2(\mathbf{x})\big),\quad k = 1, \ldots, K,
$$

and one GP on the target observations collected so far,
$\mathcal{GP}_t : (\mathbf{x}, \mu_t, \sigma_t^2)$. All GPs share the
same kernel family but are fitted independently (their length-scales
and noise levels differ across tasks).

### 3.2 Pairwise rank loss

The key idea of RGPE is to weight each base model by **how well its
predicted ordering of the target observations agrees with the actual
target observations**. Let $\mathcal{D}_t = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$
be the target data so far. For source GP $k$ we draw $S$ posterior
samples at the target inputs:

$$
\hat{\mathbf{f}}^{(k,s)} \;\sim\; \mathcal{N}\!\big(\boldsymbol{\mu}_k(X),\; \mathrm{diag}\,\boldsymbol{\sigma}_k^2(X)\big),\quad s = 1, \ldots, S,
$$

and define the rank-disagreement loss

$$
\mathcal{L}_k \;=\; \frac{1}{S \,\binom{n}{2}}\sum_{s=1}^{S}\sum_{1 \le a < b \le n}\mathbb{I}\!\left[\,\mathrm{sign}(\hat f_a^{(k,s)} - \hat f_b^{(k,s)}) \;\neq\; \mathrm{sign}(y_a - y_b)\,\right].
$$

$\mathcal{L}_k \in [0, 1]$ is the fraction of observation pairs whose
ranking the source model gets wrong, averaged over Monte-Carlo samples.
A perfectly aligned source has $\mathcal{L}_k = 0$; an anti-correlated
source has $\mathcal{L}_k = 1$; a random source has $\mathcal{L}_k =
\tfrac12$.

### 3.3 RGPE weights

Source weights are obtained by a softmax over the negative losses:

$$
\tilde w_k \;=\; \frac{\exp(-\mathcal{L}_k / \tau)}{\sum_{j=1}^{K}\exp(-\mathcal{L}_j / \tau)},\qquad \tau \;=\; \max(\mathrm{median}_j\,\mathcal{L}_j,\,10^{-3})
$$

(the data-driven temperature $\tau$ keeps weights well-conditioned even
when all losses are similar). The target GP is then assigned a weight
that grows with the amount of target data:

$$
w_t \;=\; \min\!\left(\frac{1}{1 + e^{-(n - n_0)/s}},\;w_t^{\max}\right),
$$

with default $n_0 = 5$, $s = 5$, $w_t^{\max} = 0.5$. Source weights are
renormalised to fill the rest:

$$
w_k \;=\; (1 - w_t)\,\tilde w_k.
$$

This avoids the expensive leave-one-out computation that the original
RGPE paper uses for $w_t$, while preserving the qualitative behaviour:
sources dominate at cold start, the target gradually takes over.

### 3.4 Ensemble predictive distribution

The RGPE surrogate uses a **mixture-of-GPs** prediction at any test point:

$$
\hat\mu(\mathbf{x}) \;=\; \sum_{k=1}^{K} w_k\,\mu_k(\mathbf{x}) \;+\; w_t\,\mu_t(\mathbf{x}),
$$

$$
\hat\sigma^2(\mathbf{x}) \;=\; \sum_{k=1}^{K} w_k\,\big[\sigma_k^2(\mathbf{x}) + (\mu_k(\mathbf{x}) - \hat\mu(\mathbf{x}))^2\big] \;+\; w_t\,\big[\sigma_t^2(\mathbf{x}) + (\mu_t(\mathbf{x}) - \hat\mu(\mathbf{x}))^2\big].
$$

The variance formula is the standard mixture-distribution decomposition
$\mathbb{V}[Y] = \mathbb{E}[\mathbb{V}[Y\mid Z]] + \mathbb{V}[\mathbb{E}[Y\mid Z]]$.

The EI computation in Section 2.2 then uses $(\hat\mu, \hat\sigma)$ in
place of $(\mu, \sigma)$.

---

## 4. Constraint transfer: voted feasibility classifier

Where RGPE transfers knowledge of the **objective**, the
voted-feasibility scheme transfers knowledge of the **feasibility
indicator**. The motivation is identical: source classifiers can answer
"is this $\mathbf{x}$ likely feasible?" before the target has collected
enough data of its own to train a useful classifier.

### 4.1 Per-source classifiers

For each source task $k$ we train a probabilistic feasibility classifier

$$
g_k : \mathcal{X} \to [0, 1], \qquad g_k(\mathbf{x}) \;=\; \widehat{\Pr}(c^{(k)}(\mathbf{x}) = 1).
$$

In this work each $g_k$ is an SVM with RBF kernel and Platt scaling
(`sklearn.svm.SVC(probability=True)`), trained on the source's collected
$(\mathbf{x}_i^{(k)}, c_i^{(k)})$ pairs. Sources with a single observed
class are skipped (no classifier returned).

### 4.2 Equal-weight vote

The simplest combination averages all source classifiers uniformly:

$$
g_{\rm eq}(\mathbf{x}) \;=\; \frac{1}{K'} \sum_{k\,:\,g_k\,\text{exists}} g_k(\mathbf{x}),
$$

where $K' \le K$ counts the sources whose classifiers are available.
This is a Bayes-model-averaging-style combination under a uniform prior
on source-task relevance.

### 4.3 RGPE-weighted vote

When source-target task similarity varies, equal weighting gives equal
voice to a perfectly aligned source and a useless one. We re-use the
**RGPE source weights** $\{w_k\}$ from Section 3.3 (computed on the
target's *objective* observations) as the source-similarity prior for
the *constraint* vote:

$$
\boxed{\; g_{\rm rgpe}(\mathbf{x}) \;=\; \frac{1}{\sum_{k}w_k}\sum_{k\,:\,g_k\,\text{exists}}\, w_k\, g_k(\mathbf{x}) \;}
$$

(the renormalisation accounts for sources whose classifier was skipped).
The implicit assumption is that **a source whose objective ranking
matches the target's well also tends to share the target's feasibility
boundary**. This holds when the underlying physics couples objective
quality and feasibility (true in both case studies: in laser cutting,
miscuts cluster in the same parameter regions across materials; in PID
tuning, the stability boundary co-locates with the basin of low ITAE).

### 4.4 Comparison and robustness

The two voting schemes differ most when one or two source tasks are
**adversarial**: their classifiers confidently mispredict the target's
feasibility. The RGPE-weighted scheme automatically down-weights such
sources because their objective-ranking loss $\mathcal{L}_k$ is also
high; the equal-weight scheme cannot. In the laser-cutting
sweeps of this project the RGPE-weighted vote produces fewer infeasible
evaluations than equal voting on most targets, but the gap is small
when all sources are roughly equally relevant.

A complementary noise study on the source classifiers (flipping a
fraction of source feasibility labels before training) shows that the
RGPE-weighted vote remains close to its clean-data accuracy at noise
levels up to $\approx 30\%$, whereas equal voting degrades roughly
linearly with noise.

---

## 5. Full algorithm

Algorithm 1 lists one BO iteration of the constrained, transfer-learned
optimiser.

```
Algorithm 1: cBO + RGPE + voted feasibility, one iteration

Inputs:
  target dataset D_t = {(x_i, y_i, c_i)}_{i=1..n}
  source GPs   {GP_k}_{k=1..K}      (Section 3.1)
  source clfs  {g_k}_{k=1..K}       (Section 4.1)
  candidate set X_c (random or Sobol over X)
  constraint sharpness beta >= 0

1.  Fit target GP_t on stable target observations.
2.  Compute RGPE source losses L_1, ..., L_K.        (eq. 3.2)
3.  Compute weights {w_k}, w_t.                       (eq. 3.3)
4.  For each candidate x in X_c:
       mu, sigma2  <-  ensemble mean/variance         (eq. 3.4)
       EI(x)       <-  Expected Improvement(mu, sigma)
       g(x)        <-  voted feasibility classifier   (Section 4.2 or 4.3)
       acq(x)      <-  EI(x) * g(x)^beta              (eq. 2.3)
5.  x_{n+1}  <-  argmax_{x in X_c} acq(x)
6.  Evaluate (y_{n+1}, c_{n+1})  =  oracle(x_{n+1})
7.  Append (x_{n+1}, y_{n+1}, c_{n+1}) to D_t
```

Step 1 uses **only stable target points**: when the cumulative number of
stable target observations is below 2, the GP falls back to all target
data. This avoids fitting a GP on a degenerate target set.

Step 4's candidate optimisation is a discrete maximisation over a Sobol
or i.i.d. uniform sample of size $\sim 10^3$. For higher-dimensional
problems a multi-start L-BFGS refinement can be appended.

The complete BO loop runs $N_{\rm iter}$ iterations preceded by an
$N_{\rm init}$-point Sobol initialisation that is shared across schemes
to enable paired comparisons.

---

## 6. Practical considerations

### 6.1 Constraint sharpness $\beta$

The exponent $\beta$ controls how aggressively the acquisition prunes
candidates whose $\Pr(c=1)$ is below 1. A sweep over $\beta \in \{0,
0.25, 0.5, 1, 2, 4, 8\}$ on 21 laser-cutting targets shows:

- $\beta = 0$ leaves the vote unused; if the GP is fit on
  feasible-only data the optimiser still wanders into infeasible
  regions (no penalty in the surrogate), giving the highest miscut
  count among the constrained schemes.
- $\beta = 1$ sits at the elbow of the burr/miscut Pareto front: it
  matches the safety of $\beta \in \{4, 8\}$ (hard-mask regime) at
  noticeably lower burr.
- $\beta > 2$ saturates: the soft mask becomes effectively binary and
  the optimiser is restricted to "fully feasible" candidates, which
  forfeits exploration value with no further safety gain.

We therefore recommend $\beta = 1$ as default and only deviate when the
application has a hard policy on infeasible queries.

### 6.2 Inclusion of infeasible target observations

Two design choices arise when the target evaluates an infeasible point:

1. **Surrogate fit**. We fit the target GP on stable target points
   only. Including infeasibles with a sentinel ($-1$ for laser cutting,
   high-ITAE for PID) is an alternative used by the unconstrained
   schemes in the comparison.
2. **Constraint update**. The target observation contributes to the
   target-data-trained classifier (scheme 3 in the experiments) but
   does *not* update the source classifiers — those remain fixed as
   pretrained models throughout.

### 6.3 Cold start

When fewer than two stable target points have been evaluated, the
constrained branch falls back to drawing the next query at random from
the candidate set. This avoids ill-conditioned GP fits and gives the
optimiser a chance to find at least one stable observation by chance
(with $\sim 17\%$ stability rate in the PID setting, expectation
$5/\,0.17 \approx 30$ random points to find one). The Sobol initial
design is sized to make this rare in practice ($N_{\rm init} = 5$
typically yields at least one stable point per seed).

### 6.4 Computational complexity

Per BO iteration:

- $K$ source-GP predictions on $n$ target inputs: $O(K\,n\,m)$ where $m$
  is the number of source training points (cached at fit time).
- Rank-loss Monte Carlo: $O(K\,S\,n^2)$ for $S$ posterior samples
  (default $S = 30$).
- Target GP fit and prediction: $O(n^3)$ via Cholesky.
- Candidate evaluation: $O(M_c\,(K + 1))$ for $M_c$ candidates.

For our settings ($K \le 23$ sources, $n \le 30$ target observations,
$M_c = 10^3$), an iteration takes $\sim 0.2$–$0.5$ s on a workstation.

### 6.5 Hyperparameter defaults

| symbol | meaning | default |
|---|---|---|
| $\xi$ | EI exploration | $0.01$ |
| $\beta$ | constraint sharpness | $1$ |
| $S$ | RGPE Monte-Carlo samples | $30$ |
| $\tau$ | RGPE weight temperature | $\max(\mathrm{median}\mathcal{L},\,10^{-3})$ |
| $w_t^{\max}$ | target-GP weight cap | $0.5$ |
| $n_0,\,s$ | target-weight schedule midpoint, slope | $5,\,5$ |
| $N_{\rm init}$ | Sobol initialisation size | $5$ |
| $M_c$ | candidate set size | $1\,500$ |

---

## 7. Comparison schemes

The experiments report five BO schemes:

| # | name | surrogate | feasibility constraint |
|---|---|---|---|
| 1 | **Vanilla BO** | target GP only | none (penalty for infeasible) |
| 2 | **RGPE** | RGPE ensemble | none (penalty for infeasible) |
| 3 | **cBO target-clf** | RGPE ensemble | classifier trained on target data |
| 4 | **cBO RGPE-vote** | RGPE ensemble | voted source classifiers, RGPE weights |
| 5 | **cBO equal-vote** | RGPE ensemble | voted source classifiers, uniform weights |

Schemes 1 and 2 use the unconstrained acquisition $\alpha_{\rm EI}$ on a
GP fitted to all target data with infeasible observations replaced by a
high-penalty sentinel. Schemes 3-5 use the constrained acquisition
$\alpha_\beta = \alpha_{\rm EI}\cdot g^\beta$ with the surrogate fitted
on stable target data only.

---

## 8. Empirical takeaways

Across the laser-cutting and PID experiments, three findings are robust:

1. **Source data primarily reduces tail risk, not typical-case
   performance.** Vanilla BO has the lowest median final objective on
   easy targets but the longest right-tail (catastrophic failures
   where it never escapes a degenerate optimum). RGPE and the
   constraint-aware schemes never exhibit such tails; their mean
   objective is dominated by the absence of catastrophic seeds rather
   than by faster convergence on typical seeds.

2. **The voted-classifier constraint reduces infeasible evaluations by
   $\sim 3\times$ at near-zero cost in objective quality.** This is the
   main practical reason to deploy schemes 4 or 5 over plain RGPE.

3. **RGPE-weighted voting is preferable when source-task similarity
   varies.** Under labelling noise on the source classifiers it
   degrades more slowly than equal voting; on benign source pools the
   two schemes are statistically indistinguishable.

A sweet-spot exists for the source-data budget: too little leaves the
classifiers under-trained (they hurt as they help); too much can drive
**negative transfer** when source-target shift is non-trivial — the
classifiers and source GPs become confidently wrong. In the PID study
the optimum is $N_{\rm per\,source} \approx 50$ samples.

---

## 9. References

- E. Brochu, V. M. Cora, N. de Freitas. *A tutorial on Bayesian
  optimization of expensive cost functions, with application to active
  user modeling and hierarchical reinforcement learning.* arXiv:
  1012.2599, 2010.
- J. Snoek, H. Larochelle, R. P. Adams. *Practical Bayesian
  Optimization of Machine Learning Algorithms.* NIPS 2012.
- J. R. Gardner, M. J. Kusner, Z. E. Xu, K. Q. Weinberger,
  J. P. Cunningham. *Bayesian Optimization with Inequality
  Constraints.* ICML 2014.
- M. A. Gelbart, J. Snoek, R. P. Adams. *Bayesian Optimization with
  Unknown Constraints.* UAI 2014.
- M. Feurer, B. Letham, F. Hutter, E. Bakshy. *Practical Transfer
  Learning for Bayesian Optimization.* arXiv: 1802.02219, 2018.
- C. E. Rasmussen, C. K. I. Williams. *Gaussian Processes for Machine
  Learning.* MIT Press, 2006. (Background on GP regression.)
- J. C. Platt. *Probabilistic outputs for support vector machines and
  comparisons to regularized likelihood methods.* Advances in Large
  Margin Classifiers, 1999. (Probability calibration for SVM
  classifiers used in $g_k$.)
