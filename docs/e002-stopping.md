# E-002 stopping-policy diagnostic

This is a synthetic **fixed-item binary-success** study. It neither observes
a model nor selects a deployment threshold. Drive owns the actual E-002
definition, status, source admission, and interpretation.

## Decision and exact methods

The illustrative decision asks whether an item's conditional per-generation
success probability is above or below threshold θ. Accept, reject, and
inconclusive are distinct outcomes. The development configuration uses
θ=1/2, an indifference zone with endpoints p₀=3/10 and p₁=7/10,
total nominal error budget α=1/10, and looks at n=2,4,8,16.
These numbers test methods; they are **not** a real release criterion.
Schema-v1's historical `holdout_p` grid, alternative θ and half-width grid,
and abstract loss weights were declared in `stopping_records.CONFIG` before
the PR #4 analysis. That grid has now been evaluated and published, so it is
consumed rather than an untouched confirmation set. Historical schema-v1 and
its output values remain unchanged. New diagnostics should call such points
an `evaluation_grid` unless a genuinely untouched holdout exists; for this
finite Bernoulli problem, exact whole-region operating characteristics are
preferred when monotonicity establishes them. A unit trial cost is an
abstract count, not provider pricing.

- Fixed n in {1,2,4,8,16}: compare c/n to θ; equality is inconclusive.
  This is a baseline classifier, not an α-controlled test.
- Naive: at each look compare exact binomial upper/lower tails under θ to
  α/2. This repeats a one-look test without monitoring adjustment and is
  a negative control.
- Group: preallocate α/(2L) to each tail at each of L prespecified looks.
  Bonferroni's union bound controls crossing probability at θ under iid
  Bernoulli trials, though it can be conservative. This is an independently
  specified repeated-exact-test construction, not a claim to implement the
  Lan–DeMets spending function.
- Sequence: allocate α/[2n(n+1)] to each tail at integer time n. The sum
  over n≥1 is α/2 per tail, so the resulting inverted intervals are
  time-uniform by the union bound. This is conservative, not the tighter
  mixture-martingale confidence sequence of Howard et al.
- SPRT-style: likelihood ratio of Bernoulli(p₁) to Bernoulli(p₀);
  accept at LR≥1/α, reject at LR≤α, otherwise continue to the cap.
  Ville's likelihood-ratio bound controls false acceptance at p₀ and
  false rejection at p₁ (and, by Bernoulli monotonicity, the corresponding
  outer composite regions). There is no claimed error guarantee **inside**
  (p₀,p₁). It is a capped test with explicit inconclusive probability,
  not the classical Wald approximate α/β boundary calibration.

The exact dynamic program propagates rational mass over (n,c) and stopping
state. Every reported probability, expected sample count, per-trial cost,
and stopping-time distribution is a rational number, serialized as
`numerator/denominator` in a strict simulation record. Decimal display
below is rounded from exact results. No Monte Carlo uncertainty is involved.
The reference arm uses an independent finite Binomial(n,p) count and the
same fixed threshold classifier. It is not synthetic ground truth; p is.

## Selected exact operating characteristics

For θ=.5, endpoints .3/.7, α=.1 and cap 16:

| Policy | p | Accept | Reject | Cap inconclusive | E[trials] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 16 | .3 | .02567 | .92565 | .04868 | 16 |
| Fixed 16 | .7 | .92565 | .02567 | .04868 | 16 |
| Naive looks | .5 | .05972 | .05972 | .88055 | 15.4375 |
| Group looks | .5 | .01312 | .01312 | .97375 | 15.9375 |
| Time-uniform | .5 | .00002 | .00002 | .99997 | 16 |
| SPRT-style | .3 | .02088 | .84607 | .13305 | 10.29967 |
| SPRT-style | .7 | .84607 | .02088 | .13305 | 10.29967 |

At p=θ=.5 the naive repeated-look rule crosses either tail with probability
.11945, exceeding its nominal total .10, and its stopped-interval coverage
is .88055. The preallocated group rule has crossing probability .02625;
the time-uniform allocation is far more conservative at this cap. Exact
small-path enumeration independently checks the dynamic program. The published
schema-v1 grid includes near-zero, near-one, and near-boundary points, but is
now consumed evidence rather than an untouched holdout. For supported finite
policies, exhaustive path monotonicity plus Bernoulli coupling now checks the
outer composite regions directly at p₀/p₁. The configurable loss table keeps
false acceptance, false rejection, cap, and trial count separate. Any ranking
that gives no penalty to inconclusive outcomes is misleading.

Under an exchangeable beta-mixture with marginal .3 and concentration 2,
SPRT-style acceptance rises to .19655, versus .02088 for iid(.3).
Here the latent **item** probability varies: calling .19655 an item-level
false-accept rate would conflate the marginal and conditional targets.
Under a .3→.7 change after eight attempts it accepts with probability
.18471 and rejects with .58710. Iid guarantees do not transfer to either
scenario. Both stress results are exact probability calculations.

An independent reference of 16, 64, or 256 attempts at true p=.55
incorrectly rejects with probabilities .25589, .17617, and .04765,
respectively (ties are separately inconclusive). Even 256 is not an
error-free reference near the boundary. For a task population equally supported on p=(.2,.5,.8), the between-task
variance component remains .06 as repetitions rise from 1 to 16; the within-task component falls from .19
to .011875. Repeating the same fixed items cannot remove the between-task
component or establish performance over an unsampled question population.

## Evidence and scope

- [Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence
  sequences*](https://arxiv.org/abs/1810.08240) establishes the broader
  optional-stopping-safe confidence-sequence framework. This code uses a
  simpler exact-tail union allocation.
- [Lan and DeMets, alpha spending](https://pubmed.ncbi.nlm.nih.gov/7973215/)
  addresses prespecified/repeated interim looks. The group arm here is a
  conservative Bonferroni allocation, not their optimized boundaries.
- [Wald, *Sequential Analysis*](https://books.google.com/books?id=4FZMAAAAMAAJ)
  gives the simple-hypothesis SPRT lineage. This study uses conservative
  likelihood-ratio thresholds with a finite cap and no forced decision.
- [Pilditch, *Knowing When to Stop*, arXiv:2608.14425v1](https://arxiv.org/abs/2608.14425)
  studies hierarchical Bayesian precision-width stopping across items and
  groups, including benchmark shadow runs. Its stopping criterion is posterior
  precision, not frequentist decision-error calibration. In a moderate-
  heterogeneity known-truth simulation, empirical coverage is about 80% at
  nominal 94% away from exact 0/1 boundaries; the binary/continuous full-run
  comparison is instead a finite full-run-estimate precision check. The
  ordinal 0/15 and 1/15 cells compare a modal-category interval with a mean,
  so they are an estimand mismatch. The paper names `optstop` v0.4.0 for its
  validation. The inspected repository main is now v0.5.0 at commit
  `1fa679ea82081a26c790c9e59f7e1365f8e34c4a`; v0.5.0 changed low-rate interval
  reporting and stopping behavior, so current main must not be treated as
  automatically identical to the paper validation artifact.
- [Balkır et al., *Confident Rankings with Fewer Items*, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.2068/)
  adaptively selects benchmark **items** for continuous-score model ranking;
  it does not set repeated-generation counts on a fixed item. The current
  arXiv v2/proceedings headline reports +0.13 Kendall τ over random sampling
  and 99% accuracy on confident predictions at 2% of items, whereas arXiv v1
  reported +0.12 τ and 95%. The public `trismik/continuous-cat` repository
  has no tags/releases; inspected main is commit
  `4f2b0bdac0b972ee8cef5da6845465db83f576c7` dated 2026-02-02, more than seven
  months before arXiv v2. Its README still cites the arXiv preprint, so
  code-to-proceedings identity is not assumed without a fresh reproduction.
- [*Sequential Bayesian Evaluation of Large Language Model Behavior*,
  arXiv:2511.10661v2](https://arxiv.org/html/2511.10661v2) allocates
  repeated generations across fixed prompts for aggregate properties.
  Its synthetic-judge-score replay exposes a misspecified ordinal-probit
  case where adaptivity worsens nominal 90% interval coverage to 13–23%.
  This is relevant failure evidence, not a direct validation of our binary
  likelihood-ratio rule.

## Reproduce and interpret

```sh
PYTHONPATH=src uv run --python 3.11.13 --no-project python -m unittest discover -s tests -v
PYTHONPATH=src uv run --python 3.11.13 --no-project python -m llm_evaluation.stopping_cli
PYTHONPATH=src uv run --python 3.11.13 --no-project python -m llm_evaluation.stopping_cli --output runs/e002-stopping-v1.json
```

The last command requires a fresh trusted Python process from a clean source
checkout and exclusively finalizes a simulation record. It cannot attest
arbitrary in-memory instrumentation. The record has a distinct type/schema,
explicit rational-number convention, config, exact results, UTC time, Python
version, Git commit, and measured package-source digest. A wheel can preview,
but cannot finalize. CI creates and rereads an ephemeral record.

**Candidate carried forward:** capped SPRT-style testing of a *specified
indifference-zone decision*, alongside fixed-count and conservative
repeated-look controls. It is not a universal choice: the cap leaves
inconclusive cases, behavior near the threshold and under dependence/drift
depends on the target, and real task/model conditions remain unknown.
The present time-uniform allocation is valid under its assumptions but
uninformative at cap 16; it should not be deployed merely because it
controls error. No Bayesian width rule is promoted to a frequentist
decision guarantee. Before model calls, freeze the real threshold, loss
of each decision/cap, subject versions, item/oracle set, source rights,
reference precision, task population, dependence handling, and budget.
