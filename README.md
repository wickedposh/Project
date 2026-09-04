## Project 1 — Lying to a Bayesian: Optimal Randomisation in Repeated Pricing
**DOI:** https://doi.org/10.5281/zenodo.20695310

### Abstract
We apply reinforcement learning to a repeated pricing game between a
Bayesian-updating producer and a strategic consumer with private
willingness-to-pay. We establish existence of a mixed-strategy Nash equilibrium
for the continuous-action game via Reny (1999), then approximate it with three
methods (Monte Carlo, tabular Q-learning, DQN). Across all methods the optimal
mixing satisfies ε* < 1, supporting randomisation as an obfuscation strategy.
Monte Carlo and DQN recover a decreasing ε*(β) — more patient consumers
randomise more — while tabular Q-learning shows no significant trend. We trace
this discrepancy to estimator variance on a value surface that is flat in ε,
rather than to belief-state discretisation.

### Results
- **ε* < 1 across all methods** — randomisation improves consumer surplus over
  deterministic play.
- **ε*(β) decreases** in Monte Carlo and DQN (verified across seeds); tabular
  Q-learning's argmax is too high-variance to resolve the trend.
- The producer's posterior is regularised with a variance floor for numerical
  stability; the bounded-σ behaviour is a consequence of that floor, not an
  emergent property of randomisation.

### Files
- `MC.py` — Monte Carlo simulation
- `QLearning.py` — tabular Q-learning
- `dqn.py`, `dqn_agent.py` — Deep Q-Network
- `eps_star_vs_beta.png` — results plot

---
**DOI:** https://doi.org/10.5281/zenodo.20983755
## Project 2 — Heterogeneous Market: Does Obfuscation Survive a Rational Producer?
*(working title — in progress)*

### Idea
Extends Project 1 to N heterogeneous consumers with private WTP drawn from an
unknown distribution F. The producer is now strategic: it follows a
belief-greedy strategy, learning the acceptance curve online and pricing to
maximise profit. A two-sided Bayesian game.

### Theory
- The producer's optimal price depends on F only through a 2-D local statistic
  (price, effective hazard) — a sufficient-statistic result (DKW consistency
  for large N).
- Existence of a Bayes–Nash equilibrium via Berge's maximum theorem + Kakutani's
  fixed-point theorem; obfuscation enters through a price-insensitive "noise
  floor" in effective demand.

### Key finding
Obfuscation's survival depends on whether consumers are forward-looking:
- **Static / myopic (β = 0):** ε* ≈ 1 — obfuscation unravels to truthful play
  against a best-responding producer.
- **Dynamic / discounted (β > 0):** ε* < 1 — obfuscation survives, because
  corrupting the producer's belief today lowers future prices (a discounted
  intertemporal benefit). ε* decreases with patience.

| β | ε* (mean ± std, 24 seeds) |
|------|------|
| 0.00 | 0.914 ± 0.035 |
| 0.50 | 0.852 ± 0.048 |
| 0.90 | 0.740 ± 0.071 |
| 0.99 | 0.749 ± 0.059 |

Spearman ρ = −0.769, p < 10⁻⁶ (96 raw (β, ε*) pairs). Reproduced across four
independent seed sets. The trend decreases then plateaus (β ≈ 0.9): the
obfuscation incentive saturates once consumers are sufficiently patient.

### Files
- `hetero_dynamic.py` — two-sided dynamic game (producer belief-learning + consumer DQN)

---

## Project 3 — World Models on the Pricing Game
*(in progress)*

### Idea
Does the sufficient-statistic result from Project 2 agree with what a learned
latent representation finds? A world model is trained on the same simulation —
state is the producer's price and its recent history, action is the aggregate
accept rate (the observable realisation of the consumers' obfuscation), target
is the next price. The simulation itself is unmodified: the Bayesian belief
update is intact throughout, so obfuscation retains its effect on future prices.

### Latent dimensionality
Sweeping window length L ∈ {1, 2, 5, 10, 20} against latent dimension
d ∈ {1, 2, 4, 8}, prediction error is flat in d at every L — one latent
dimension carries everything the dynamics need.

| L \ d | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 | 0.332 | 0.331 | 0.332 | 0.332 |
| 2 | 0.274 | 0.268 | 0.267 | 0.266 |
| 5 | 0.262 | 0.256 | 0.254 | 0.257 |
| 10 | 0.276 | 0.269 | 0.270 | 0.273 |
| 20 | 0.258 | 0.253 | 0.246 | 0.247 |

History helps up to L ≈ 2 and then saturates; extra latent capacity does not
help at any L.

VICReg was tried for this and rejected: its variance and covariance terms
actively decorrelate embedding dimensions, so effective dimensionality read off
a VICReg embedding's spectrum is an artifact of the regulariser rather than a
property of the data (a 2-D input gave a participation ratio of ≈ 2.9). The
sweep uses a plain encoder–predictor instead.

### Two hazard definitions
Two producers share the same Bayesian belief update *and the same pricing rule*
— both price by the whole-curve argmax — differing only in how the hazard is
computed at that price: one measures it off the belief curve, the other derives
it from the price via the Lemma, ĥ = 1/(x − c). Holding the pricing mechanism
fixed isolates the hazard definition.

Across 8 seeds the two are statistically indistinguishable: profit 21.17 vs
20.45, 95% CI on the difference [−1.77, +0.34], with obfuscation sustained at
similar levels in both (ε̄ ≈ 0.77). Along the equilibrium path,
corr(x, ĥ) ≈ −0.936 — the Lemma-derived hazard is as good as the measured one.

### Action-conditioning
The consumer's obfuscation choice depends on the producer's price, giving
systematic collinearity between action and state: corr(x, a) = −0.914 across
three seeds. This is the setting in which world models are known to ignore the
action. However ~40% of the action's variation survives regressing on the
state, because the consumer's policy is a **mixed strategy** — obfuscation is
randomisation by construction, so even a converged population produces
stochastic accepts.

An ablation confirms the residual is used rather than ignored (L = 5, d = 1,
3 seeds):

| arm | val MSE |
|---|---|
| action present | 0.270 ± 0.015 |
| action zeroed | 0.288 ± 0.014 |
| action shuffled | 0.318 ± 0.023 |

Both controls are significantly worse (zeroed − present: +0.018, 95% CI
[+0.001, +0.035]; shuffled − present: +0.048, 95% CI [+0.035, +0.061]). That
the shuffled arm is *worse* than the zeroed one — the model is actively harmed
by incorrect action values — is stronger evidence of genuine action-conditioning
than the zeroed comparison alone.

So whether the action-ignoring problem bites depends on how much independent
variation the data-generating policy leaves. In a game whose equilibrium is in
mixed strategies, that variation is structurally preserved.

### Scope
All results are measured along the equilibrium path under belief-greedy
pricing. They support the claim that price and hazard are close to redundant
*at this equilibrium* — not that the 2-D sufficient statistic reduces to 1-D
in general or off-equilibrium.

### Files
- `world_model_dim.py` — L × d sweep for latent dimensionality
- `lemma_producer.py` — measured vs Lemma-derived hazard, pricing rule held fixed
- `action_ablation.py` — action-conditioning ablation (present / zeroed / shuffled)

---
