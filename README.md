# REFRAIN.jl

A calibrated rule for choosing between simulator-based models that is allowed to report that the data do not support a choice.

REFRAIN.jl adapts the relative-fit form of universal inference of Park, Balakrishnan and Wasserman (Biometrika 2026) to the likelihood-free setting, replacing the likelihood ratio their construction needs with a sample-only integral probability metric that any simulator can supply. It is rarely realistic to assume that a hypothesised model is either well specified or has a tractable likelihood, and in the simulation-based sciences both assumptions fail at once. When every candidate is wrong and none has a likelihood, an ABC posterior model probability still names a winner, often with overwhelming odds, even where no candidate is meaningfully closer to the data. REFRAIN changes the question from which model is true to which candidate fits relatively better, and attaches a significance statement so that the answer can be to decline.

The package carries its own ABC layer rather than asking for a fitted model. Bring your own fit is not a usable contract for someone who came here precisely because there is no likelihood to fit with.

```julia
using Pkg
Pkg.add(url = "https://github.com/Msturroc/REFRAIN.jl")
```

Julia 1.10 or later. Two dependencies, `Distributions` and `StatsBase`.

## The decision rule

The sample of independent units is held as the columns of a matrix and split once into a fit half `D1` and a held-out test half `D0`. Each candidate is fitted on `D1` by ABC, and fresh data `P0` and `P1` are simulated from the two fits. The relative-fit statistic is formed on the held-out half as `T = rho(D0, P0) - rho(D0, P1)`, with `rho` a sample-only integral probability metric, so that a positive `T` favours `M1`. Calibrating `T` against the equidistance null then turns it into one of three decisions.

The rule itself is one line, and it is kept visible in the source rather than buried:

```julia
p > alpha ? :abstain : (T > 0 ? :M1 : :M0)
```

Abstention is the confidence-set duality of the original construction, in which failing to reject leaves both candidates in the set. Under the bootstrap calibration it is literally the statement that the confidence interval for the relative-fit gap contains zero.

```julia
using REFRAIN
include(joinpath(pkgdir(REFRAIN), "examples", "example_models.jl"))

m0 = make_gaussian_fixed_loc(-1.0)      # M0: N(-1, sigma^2)
m1 = make_gaussian_fixed_loc(+1.0)      # M1: N(+1, sigma^2)

X = truth_gaussian(0.0, 1.0; n = 400, seed = 1)      # a genuine tie
res = refrain(X, m0, m1; ipm = :sw, calibration = :bootstrap,
              N = 400, paccmin = 1e-2, max_sims = 25_000)
res.decision       # :abstain
res.interval       # contains zero

Y = refrain(truth_gaussian(0.8, 1.0; n = 400, seed = 1), m0, m1;
            ipm = :sw, N = 400, paccmin = 1e-2, max_sims = 25_000)
Y.decision         # :M1
Y.screen           # read this beside the decision, never folded into it
```

The two candidates here are Gaussians with fixed, mirror-image locations and a free scale, against a truth centred at `c`. The truth lies outside both candidates for every `c`, because neither can move its location, and `c = 0` is a genuine tie by symmetry. It runs in seconds, and `examples/setting_a.jl` runs it top to bottom with a small sweep in `c`. If you already hold the held-out sample and the two simulated sets, the fitting layer can be skipped entirely with `decide(D0, P0, P1; ipm = :sw, calibration = :bootstrap, alpha = 0.05)`.

`examples/rainfall.jl` is the pattern that matters once the simulator is no longer a one-liner. It compares two cluster point processes for hourly rainfall, Bartlett-Lewis against Neyman-Scott, neither of which has a tractable likelihood, and it shows four things worth copying: an in-place simulator for the hot ABC path that never allocates, an allocating wrapper for the calibration path, which needs fresh matrices, a distance factory that computes the target summaries once per fit and preallocates its buffer there rather than once per proposal, and a `hash((seed_base, theta))` simulator seed, which makes the closure both reproducible and safe under a driver-level `@threads`. It is also the regime in which `:refit` is not optional, at five parameters per family.

## What abstention means

Abstaining means that the data cannot resolve which candidate is closer. It does not mean that neither candidate is any good, and two consequences of that distinction catch people out. With the truth outside both candidates one of them is generally still closer, and the rule is meant to commit to it, so the truth lying outside both is not by itself grounds to abstain. On two identical candidates the rule abstains with probability about `1 - alpha`, which is the correct answer for a relative test rather than a fifty-fifty split between them.

A purely relative test can never return an empty set. The reason is structural rather than a consequence of there being two candidates. The set is inverted against a fixed pilot, and no test in the family asks whether the pilot is worse than itself, so the pilot survives at every `K`. Enlarging the model class therefore does not buy the missing verdict.

The absolute question is supplied separately by `adequacy_screen`, computed on the held-out half only, and `res.screen` reports it beside the decision rather than folded into it. Both are worth reading. Committing to `M1` while the screen rejects `M1` is a coherent and common report, and it says that the indicated action is to widen the model rather than to choose within it.

## Choosing a calibration

There are three calibrations, they control different nulls, and which to use is a conditional answer rather than a ranking.

| calibration | null it controls | when to use it |
| --- | --- | --- |
| `:bootstrap` | equidistance, large sample | where few parameters are fitted |
| `:refit` | the same, with fit uncertainty propagated | where many are |
| `:permutation` | exchangeability of the pooled draws, exact in finite samples | only where the two fitted predictives nearly coincide |

The permutation's exactness is real and its scope is narrow. It is exact where the two predictives coincide, which is precisely the regime in which there is nothing to decide. Away from it the permuted pool mixes the two predictive clusters, the null is too narrow and the test over-rejects, measured at 0.140 and 0.349 where the predictives are close and at 0.365 and 0.658 where they are two units apart.

The choice between `:bootstrap` and `:refit` turns on how much is being fitted. `bootstrap_calibrate` resamples the columns of `P0` and `P1`, so it sees the Monte Carlo error of drawing `n_sim` units from a known parameter and nothing else. In fact `theta_hat` was estimated on `D1`, and that error is invisible to a column resample, so the interval is too narrow by whatever the missing component is worth. Measured against the across-replication spread of `T`, the shortfall grows with what is being fitted rather than with the sample size: one fitted parameter per candidate gives a width ratio of 0.97 to 1.06 and two gives 0.83 to 1.15, both correctly scaled, while five gives 1.16 to 1.46, at which the plug-in rejects at about twice its nominal rate. Raising `n` does not repair this, because the term the resample sees and the term it misses both scale as `n^{-1/2}`, so their ratio is invariant and the interval is too narrow by the same factor at every sample size.

`calibration = :refit` is the repair. It resamples the columns of `D1`, refits both candidates on each resample, and uses the spread of those refits as the sampling distribution of `theta_hat`. On the five-parameter comparison it moves the rejection rate at the tie from 0.115 to 0.050 and from 0.095 to 0.020, while the commitment rate at the far end of the sweep falls only from 0.90 to 0.84.

Three things about the repair are easy to get wrong, and all three are measured rather than argued. Both candidates must be refitted on the same resampled `D1`, because their errors are correlated and `T` is their difference, so a separate resample per candidate zeroes the covariance and inflates the interval. On a two-parameter geometry that took the width ratio to 0.49 and the power from 0.54 to 0.06. Drawing `theta*` from the ABC posterior instead of refitting does not work either, since `theta_hat` is a posterior mean whose sampling variability is not the posterior's spread, and one particle per replicate reports 3.087 against a true 0.851. Finally the `sqrt(S/(S-1))` correction is needed, because deviations taken about their own mean carry variance `(S-1)/S` times what they estimate. `refit_deviations` handles all three, and the test suite carries the control arm that catches the first.

`hoeffding_mmd_test` is the concentration threshold Park, Balakrishnan and Wasserman themselves use. It is kept because it fails in the complementary regime, staying safe at every tie whose predictives differ while committing in 0.676 of replications at the one null where the other two are exact. Read it as a cross-check rather than as a tie rule. `degeneracy_scaling` reads off an exponent `beta` that is `1/2` where the bootstrap's asymptotic hypothesis holds and `1` where it has lapsed. Read it as a warning and not as a prediction, since measured values of `beta` do not order the level failures, and always run it at a configuration away from the tie as well, because it measures the candidate pair and not the position of the truth.

## The samplers

Four samplers are provided. `rejection_abc` is one tolerance and one pass, in fixed-tolerance or quantile form. `abc_smc` is Toni et al. (2009), the default and what the paper used. `apmc` is Lenormand, Jabot and Deffuant (2013), in which model comparison is native. `abc_smc_model_choice` is the posterior-model-probability baseline.

The contract is deliberately thin. You supply `priors::Vector{<:Distribution}`, independent per coordinate, and `rho_fn(theta)::Float64`, with the simulator hidden entirely inside that closure. Two constraints follow. There is no joint prior object, and the perturbation kernel is a full-covariance MvNormal or MvCauchy, so discrete and simplex-valued parameters are not supported.

`kernel = :normal` is the default everywhere, so the package reproduces the paper out of the box. A Cauchy perturbation kernel is the recommended upgrade, since it avoids both the overinflation and the overconcentration a Gaussian kernel suffers as the tolerance falls.

`apmc` does model comparison natively, so it has no separate model-choice entry point. Its `models` argument is a vector of per-model prior vectors and `rho` a vector of distances, and single-model fitting is the `length(models) == 1` case. One open question is carried explicitly rather than decided silently: the author's own later APMC forcibly switches a Cauchy kernel to Normal under model comparison, for stability, and that finding is unresolved. Here it is the `normal_for_comparison` keyword, off by default and warned about.

`abc_smc_model_choice` is included so that a reader can reproduce the comparison the paper argues about rather than take its word for it. It keeps the joint-space weight `P(m) / P_{t-1}(m)`, without which the model weights follow a multiplicative recursion and the reported probability comes to depend on how many SMC iterations happened to run. It also ships `p_trace`, which exists to detect exactly that failure, as a straight line in the log-odds against the iteration index.

## Reproducibility

`test/identity.jl` digests the IEEE bit patterns of both metrics, all three resampling calibrations, the concentration threshold and the whole K-candidate path, and checks them against digests stored in the archived repository the paper's numbers were computed in. It costs seconds, it answers the question of whether this is the code that produced the published results, and it is what stops a port from silently becoming a different method.

Three facts about the numerics are encoded there. The RNG split is deliberate, with the samplers using `Xoshiro` and everything above them `MersenneTwister`, and the two should not be unified. Bounds checking changes the last two digits, because `Pkg.test` runs at `--check-bounds=yes`, which disables `@inbounds` and with it the SIMD vectorisation of the reductions inside the MMD. That difference is in the sixteenth significant figure and is nonetheless a different SHA, so the test re-executes itself in a subprocess at `--check-bounds=auto` rather than loosening to a tolerance. The BLAS thread count is likewise part of a published fit, since the weighted covariance is a gemm and OpenBLAS accumulates it differently at different thread counts. Measured, the checks in the identity file are not sensitive to it, but an ABC fit is, so it is pinned there.

## References

Toni, T., Welch, D., Strelkowa, N., Ipsen, A. and Stumpf, M. P. H. (2009). Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. Journal of the Royal Society Interface 6, 187-202.

Lenormand, M., Jabot, F. and Deffuant, G. (2013). Adaptive approximate Bayesian computation for complex models. Computational Statistics 28, 2777-2796.

Park, S., Balakrishnan, S. and Wasserman, L. (2026). Robust universal inference for misspecified models. Biometrika.

Bounliphone, W., Belilovsky, E., Blaschko, M. B., Antonoglou, I. and Gretton, A. (2016). A test of relative similarity for model selection in generative models. ICLR. The three-sample bootstrap here follows this test with the delta-method normal approximation replaced by a bootstrap, so that it covers any sample-only integral probability metric.

Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The model confidence set. Econometrica 79, 453-497. The K-candidate elimination takes this form.

Sturrock, M. and Shahrezaei, V. Overinflation and overconcentration: why Cauchy perturbation kernels are the right choice for ABC-SMC.

REFRAIN.jl is the rule, its calibrations and the samplers, packaged for use on other problems. The archived `abstention_model_choice` repository is the record of the particular runs the paper reports, with its settings, its drivers, its stored outputs and the scripts that turn them into every table and figure. This package is a deliberate port of that code rather than a dependency on it, and `test/identity.jl` is the gate that stops the two from drifting apart.
