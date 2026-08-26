# REFRAIN.jl

A calibrated rule for choosing between simulator-based models that can report that the data do not support a choice.

When every candidate is misspecified and none has a tractable likelihood, an ABC posterior model probability still names a winner, often with overwhelming odds, even where no candidate is meaningfully closer to the data. REFRAIN asks which candidate fits relatively better and attaches a significance statement, so the answer can be to decline. It adapts the relative-fit universal inference of Park, Balakrishnan and Wasserman (Biometrika 2026), replacing the likelihood ratio with a sample-only integral probability metric. It carries its own ABC layer, since bring your own fit is no use to someone who has no likelihood to fit with.

```julia
using Pkg
Pkg.add(url = "https://github.com/Msturroc/REFRAIN.jl")
```

Julia 1.10 or later. Depends on `Distributions` and `StatsBase`.

## Usage

Split the columns once, fit each candidate on one half by ABC, simulate from both fits, and compare them to the held-out half with `T = rho(D0, P0) - rho(D0, P1)`. Calibrating `T` against the equidistance null gives the decision, which is `p > alpha ? :abstain : (T > 0 ? :M1 : :M0)`.

```julia
using REFRAIN
include(joinpath(pkgdir(REFRAIN), "examples", "example_models.jl"))

m0 = make_gaussian_fixed_loc(-1.0)      # N(-1, sigma^2)
m1 = make_gaussian_fixed_loc(+1.0)      # N(+1, sigma^2)

X = truth_gaussian(0.0, 1.0; n = 400, seed = 1)      # a genuine tie
res = refrain(X, m0, m1; ipm = :sw, calibration = :bootstrap,
              N = 400, paccmin = 1e-2, max_sims = 25_000)
res.decision       # :abstain
res.interval       # contains zero
res.screen         # read beside the decision, never folded into it
```

With `D0`, `P0` and `P1` already in hand, `decide(D0, P0, P1; ipm = :sw)` skips the fitting layer. `examples/setting_a.jl` sweeps the above in `c`. `examples/rainfall.jl` is the realistic case, two cluster point processes for hourly rainfall with no tractable likelihood, and shows how to write an allocation-free simulator and a thread-safe seeded closure.

## Two things that catch people out

Abstaining means the data cannot resolve which candidate is closer, not that neither is any good. With the truth outside both candidates one is generally still closer and the rule is meant to commit, and on two identical candidates it abstains with probability about `1 - alpha`, which is correct for a relative test. The absolute question is separate: `adequacy_screen` answers it on the held-out half, and committing to `M1` while the screen rejects `M1` is a coherent report meaning widen the model rather than choose within it.

The plug-in bootstrap holds the fit still, so its interval is too narrow once many parameters are fitted. Measured width ratios are 0.97 to 1.06 at one fitted parameter per candidate and 0.83 to 1.15 at two, both correct, against 1.16 to 1.46 at five, where it rejects at about twice its nominal rate. Raising `n` does not help, since both terms scale as `n^{-1/2}`. Use `calibration = :refit` there, which refits both candidates on each resampled `D1` and restores the level.

## Which calibration

| calibration | null it controls | when to use it |
| --- | --- | --- |
| `:bootstrap` | equidistance, large sample | few parameters fitted |
| `:refit` | the same, with fit uncertainty propagated | many parameters fitted |
| `:permutation` | exchangeability of the pooled draws, exact in finite samples | only where the two fitted predictives nearly coincide |

The permutation is exact only where the predictives coincide, which is the regime in which there is nothing to decide. Away from it it over-rejects, measured at 0.140 and 0.349 where they are close and 0.365 and 0.658 where they are two units apart. `hoeffding_mmd_test` is Park et al.'s concentration threshold, kept as a cross-check because it fails in the complementary regime. `degeneracy_scaling` returns an exponent that warns when the bootstrap's asymptotic hypothesis has lapsed, though it does not order the level failures.

## Samplers and reproducibility

`rejection_abc`, `abc_smc` (Toni et al. 2009, the default), `apmc` (Lenormand et al. 2013, model comparison native) and `abc_smc_model_choice`, the posterior-model-probability baseline. You supply `priors::Vector{<:Distribution}`, independent per coordinate, and `rho_fn(theta)::Float64`, with the simulator inside that closure. Discrete and simplex-valued parameters are not supported. `kernel = :cauchy` is the recommended upgrade on the default Normal.

`test/identity.jl` checks the bit patterns of both metrics, all three calibrations and the K-candidate path against digests from the archived repository the paper's numbers came from. It runs in seconds and is what stops a port from silently becoming a different method.

## References

Park, S., Balakrishnan, S. and Wasserman, L. (2026). Robust universal inference for misspecified models. Biometrika.

Toni, T., Welch, D., Strelkowa, N., Ipsen, A. and Stumpf, M. P. H. (2009). Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. J. R. Soc. Interface 6, 187-202.

Lenormand, M., Jabot, F. and Deffuant, G. (2013). Adaptive approximate Bayesian computation for complex models. Computational Statistics 28, 2777-2796.

Bounliphone, W., Belilovsky, E., Blaschko, M. B., Antonoglou, I. and Gretton, A. (2016). A test of relative similarity for model selection in generative models. ICLR.

Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The model confidence set. Econometrica 79, 453-497.
