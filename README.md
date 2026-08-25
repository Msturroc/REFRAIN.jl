# REFRAIN.jl

A calibrated rule for choosing between simulator-based models that is
allowed to say **"the data do not support a choice"**.

It adapts the relative-fit form of universal inference of Park,
Balakrishnan and Wasserman (Biometrika 2026) to the likelihood-free
setting, replacing the likelihood ratio their construction needs with a
sample-only integral probability metric that any simulator can supply.

The thesis: it is rarely realistic to assume that a hypothesised model is
either well specified or has a tractable likelihood, and in the
simulation-based sciences both fail at once. When every candidate is
wrong and none has a likelihood, an ABC posterior model probability still
names a winner, often with overwhelming odds, even where no candidate is
meaningfully closer to the data. REFRAIN changes the question from "which
model is true" to "which candidate fits relatively better", and attaches a
significance statement so that it can decline.

The package carries **its own ABC layer** on purpose. "Bring your own
fit" is not usable by someone who came here precisely because they have
no likelihood.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/<user>/REFRAIN.jl")
```

Julia 1.10 or later. Two dependencies, `Distributions` and `StatsBase`.

## Method: one decision, five steps

1. Split the i.i.d. units (**columns**) into a fit half `D1` and a
   held-out test half `D0`.
2. Fit each candidate on `D1` by ABC.
3. Simulate fresh data `P0`, `P1` from the two fits.
4. Relative-fit statistic on the held-out half,
   `T = rho(D0, P0) - rho(D0, P1)`, with `rho` a sample-only integral
   probability metric. `T > 0` favours `M1`.
5. Calibrate `T` against the equidistance null and read the decision off
   the interval.

The rule itself is one line, and it is kept visible in the source rather
than buried:

```julia
p > alpha ? :abstain : (T > 0 ? :M1 : :M0)
```

Abstention is the confidence-set duality of the original construction:
failing to reject leaves both candidates in the set. For the bootstrap it
is literally "the confidence interval for the relative-fit gap contains
zero".

## Two warnings this project paid for

### 1. Abstention is discriminability, not adequacy

Abstaining means **the data cannot resolve which candidate is closer**.
It does not mean neither candidate is any good. Two consequences that
catch people out:

- With the truth outside both candidates, one candidate is generally
  closer and the rule is **meant to commit**. The truth being outside
  both is not grounds to abstain.
- On two identical candidates the rule abstains with probability about
  `1 - alpha`, which is the correct answer for a relative test. It is not
  a 50/50 split between them.

A purely relative test can never return an empty set. The reason is
structural rather than a consequence of there being two candidates: the
set is inverted against a fixed pilot, and no test asks whether the pilot
is worse than itself, so the pilot survives at every `K`. Enlarging the
model class does not buy the missing verdict.

`adequacy_screen` supplies the absolute question separately, on the
held-out half only, and `res.screen` reports it **beside** the decision
and never folded into it. Read both. Committing to `M1` and convicting
`M1` at the same time is a perfectly coherent, and common, report: it
says the indicated action is to widen the model rather than to choose
within it.

### 2. Use the refit bootstrap when the fit is expensive in parameters

`bootstrap_calibrate` resamples the **columns** of `P0` and `P1`, so it
sees the Monte Carlo error of drawing `n_sim` units from a **known**
parameter and nothing else. `theta_hat` was estimated on `D1` and that
error is invisible to it, so the interval is too narrow by whatever that
component is worth.

Measured against the across-replication spread of `T`:

| fitted parameters per candidate | width ratio | verdict |
|---|---|---|
| one  | 0.97 to 1.06 | plug-in is correct |
| two  | 0.83 to 1.15 | plug-in is correct |
| five | 1.16 to 1.46 | plug-in rejects at about twice its nominal rate |

**Raising `n` does not fix it.** The term the resample sees and the term
it misses both scale as `n^{-1/2}`, so their ratio is invariant and the
interval is the same factor too narrow at every sample size.

`calibration = :refit` is the repair: resample the columns of `D1`, refit
**both** candidates on each resample, and use the spread of those refits
as `theta_hat`'s sampling distribution. On the five-parameter comparison
it moves the rejection rate at the tie from 0.115 to 0.050 and from 0.095
to 0.020, and the far-end commitment rate falls only from 0.90 to 0.84.

Three things that are easy to get wrong, all of them measured:

- **Both candidates must be refitted on the same resampled `D1`.** Their
  errors are correlated and `T` is their difference, so a separate
  resample per candidate zeroes the covariance and inflates the interval.
  On a two-parameter geometry that took the ratio to 0.49 and the power
  from 0.54 to 0.06. `refit_deviations` does it correctly and the test
  suite carries the control arm.
- **Do not draw `theta*` from the ABC posterior.** `theta_hat` is a
  posterior mean, whose sampling variability is not the posterior's
  spread. One particle per replicate reports 3.087 against a true 0.851.
- **The `sqrt(S/(S-1))` correction is needed.** Deviations about their own
  mean carry variance `(S-1)/S` times what they estimate.

## Which calibration

There are three, they control different nulls, and the answer is
conditional rather than a ranking.

| calibration | controls | use it |
|---|---|---|
| `:bootstrap` | the equidistance null, large sample | where few parameters are fitted |
| `:refit`     | the same, with the fit uncertainty propagated | where many are |
| `:permutation` | exchangeability of the pooled draws, EXACT in finite samples | only where the two fitted predictives nearly coincide |

The permutation's exactness is real and its scope is narrow. It is exact
where the two predictives coincide, which is precisely the regime in
which there is nothing to decide. Away from it, the permuted pool mixes
the two predictive clusters, the null is too narrow, and the test
over-rejects: measured at 0.140 and 0.349 where the predictives are close
and 0.365 and 0.658 where they are two units apart.

`hoeffding_mmd_test` is the concentration threshold Park, Balakrishnan and
Wasserman themselves use. It is kept because it fails in the
**complementary** regime, safe at every tie whose predictives differ and
committing in 0.676 of replications at the one null where the other two
are exact. Read it as a cross-check, not as a tie rule.

`degeneracy_scaling` reads off an exponent `beta` that is `1/2` where the
bootstrap's asymptotic hypothesis holds and `1` where it has lapsed. Read
it as a **warning**, not as a prediction: measured, `beta` does not order
the level failures, and always run it at a configuration away from the
tie as well, because it measures the candidate PAIR and not the position
of the truth.

## Example 1: does it work at all

Two Gaussians with fixed, mirror-image locations at `-1` and `+1` and a
free scale, against a truth centred at `c`. The truth is outside both
candidates for every `c`, because neither can move its location, and
`c = 0` is a genuine tie by symmetry. Runs in seconds.

```julia
using REFRAIN
include(joinpath(pkgdir(REFRAIN), "examples", "example_models.jl"))

m0 = make_gaussian_fixed_loc(-1.0)      # M0: N(-1, sigma^2)
m1 = make_gaussian_fixed_loc(+1.0)      # M1: N(+1, sigma^2)

# a genuine tie
X = truth_gaussian(0.0, 1.0; n = 400, seed = 1)
res = refrain(X, m0, m1; ipm = :sw, calibration = :bootstrap,
              N = 400, paccmin = 1e-2, max_sims = 25_000)
res.decision       # :abstain
res.p, res.T
res.interval       # contains zero

# and a clear separation
Y = refrain(truth_gaussian(0.8, 1.0; n = 400, seed = 1), m0, m1;
            ipm = :sw, N = 400, paccmin = 1e-2, max_sims = 25_000)
Y.decision         # :M1
Y.screen           # :both_fail or :M0_fails -- read it BESIDE the decision
```

`examples/setting_a.jl` runs this top to bottom and prints a small sweep
in `c`.

If you already have the held-out sample and the two simulated sets, skip
the fitting layer entirely:

```julia
decide(D0, P0, P1; ipm = :sw, calibration = :bootstrap, alpha = 0.05)
```

## Example 2: a realistic comparison

`examples/rainfall.jl` is the pattern that matters once the simulator is
not a one-liner: two cluster point processes for hourly rainfall,
Bartlett-Lewis against Neyman-Scott, neither with a tractable likelihood.
It shows four things worth copying:

- an **in-place** simulator for the hot ABC path that never allocates,
- an **allocating wrapper** for the calibration path, which needs fresh
  matrices,
- a **distance factory** that computes the target summaries once per fit
  and preallocates its buffer there, rather than once per proposal,
- a `hash((seed_base, theta))` simulator seed, which makes the closure
  both reproducible and safe under a driver-level `@threads`.

It also shows the regime where `:refit` is not optional: five parameters
per family.

## The samplers

| name | what it is |
|---|---|
| `rejection_abc` | one tolerance, one pass. Fixed-tolerance or quantile form |
| `abc_smc` | Toni et al. (2009). The default, and what the paper used |
| `apmc` | Lenormand, Jabot and Deffuant (2013). Model comparison is native |
| `abc_smc_model_choice` | the posterior-model-probability baseline |

The contract is deliberately thin: you supply `priors::Vector{<:Distribution}`,
independent **per coordinate**, and `rho_fn(theta)::Float64`. The
simulator is hidden entirely inside that closure. Two constraints follow:
there is no joint prior object, and the perturbation kernel is a
full-covariance MvNormal or MvCauchy, so discrete and simplex-valued
parameters are not supported.

`kernel = :normal` is the default everywhere, so the package reproduces
the paper out of the box. **`kernel = :cauchy` is the recommended
upgrade**: a Cauchy perturbation kernel avoids both the overinflation and
the overconcentration a Gaussian kernel suffers as the tolerance falls.

`apmc` does model comparison natively, so there is no separate
model-choice entry point for it: `models` is a vector of per-model prior
vectors and `rho` a vector of distances, and single-model fitting is the
`length(models) == 1` case. One open question is carried explicitly
rather than decided silently: the author's own later APMC forcibly
switches a Cauchy kernel to Normal under model comparison, for stability,
and that finding is unresolved. Here it is the
`normal_for_comparison` keyword, off by default and warned about.

`abc_smc_model_choice` is here so a reader can reproduce the comparison
the paper argues about rather than take its word. It keeps the
joint-space weight `P(m) / P_{t-1}(m)`, without which the model weights
follow a multiplicative recursion and the reported probability depends on
how many SMC iterations happened to run, and it ships `p_trace`, which
exists to detect exactly that failure as a straight line in the log-odds
against the iteration.

## Reproducibility

`test/identity.jl` digests the IEEE bit patterns of both metrics, all
three resampling calibrations, the concentration threshold and the whole
K-candidate path against the digests stored in the archived repository
the paper's numbers were computed in. It costs seconds. It is the answer
to "is this the code that produced the results", and it is what stops a
port from silently becoming a different method.

Three facts it encodes:

- **The RNG split is deliberate.** The samplers use `Xoshiro`, everything
  above them uses `MersenneTwister`. Do not unify them.
- **Bounds checking changes the last two digits.** `Pkg.test` runs at
  `--check-bounds=yes`, which disables `@inbounds` and with it the SIMD
  vectorisation of the reductions inside the MMD. The measured
  difference is in the sixteenth significant figure and is nonetheless a
  different SHA, so the test re-executes itself in a subprocess at
  `--check-bounds=auto` rather than loosening to a tolerance.
- **The BLAS thread count is part of a published fit.** The weighted
  covariance is a gemm and OpenBLAS accumulates it differently at
  different thread counts. Measured, the checks in the identity file are
  not sensitive to it, but an ABC fit is, so it is pinned there.

## Provenance

- **ABC-SMC**: Toni, Welch, Strelkowa, Ipsen and Stumpf (2009),
  "Approximate Bayesian computation scheme for parameter inference and
  model selection in dynamical systems", J. R. Soc. Interface 6, 187-202.
- **APMC**: Lenormand, Jabot and Deffuant (2013), "Adaptive approximate
  Bayesian computation for complex models", Computational Statistics 28,
  2777-2796.
- **The relative-fit construction**: Park, Balakrishnan and Wasserman
  (2026), "Robust universal inference for misspecified models",
  Biometrika.
- **The three-sample bootstrap of the equidistance null**: after
  Bounliphone, Belilovsky, Blaschko, Antonoglou and Gretton (2016),
  "A test of relative similarity for model selection in generative
  models", ICLR, with the delta-method normal approximation replaced by a
  bootstrap so that it covers any sample-only IPM.
- **The K-candidate elimination**: Hansen, Lunde and Nason (2011), "The
  model confidence set", Econometrica 79, 453-497.
- **The `kernel = :cauchy` option**: Sturrock and Shahrezaei,
  "Overinflation and overconcentration: why Cauchy perturbation kernels
  are the right choice for ABC-SMC".

## Relationship to the archived repository

REFRAIN.jl is the rule, its calibrations and the samplers, packaged for
other problems. The archived `abstention_model_choice` repository is the
record of the particular runs the paper reports: its settings, its
drivers, its stored outputs and the scripts that turn them into every
table and figure. This package is a deliberate **port** of that code, not
a dependency on it, and `test/identity.jl` is the gate that stops the two
drifting.
