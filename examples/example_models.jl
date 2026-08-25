#= ================================================================
   The two worked examples' models.

   These are NOT part of the package: they are what a user writes. They
   live here so the README's examples and the test suite share one
   definition, and so a reader can see what the `RelFitModel` contract
   looks like when it is honoured.

   SETTING A, the "does it work at all" example. Two Gaussians with FIXED
   mirror-image locations and a free scale, against a truth centred at c.
   The truth lies outside both candidates for every c, since neither
   candidate can move its location, and c = 0 is a genuine tie by
   symmetry. One-dimensional, one fitted parameter per candidate, closed
   forms available, seconds to run.

   SETTING B, a second tie of a different kind, used by the refit
   bootstrap's control arm. A Gaussian against a Laplace, both fitting a
   location AND a scale to the same data, so they match the first two
   moments and only tail shape separates them. It is the geometry where a
   separate refit resample per candidate would zero the covariance
   between the two fit errors and inflate the interval, so it is the arm
   that catches that mistake.

   NOTE THE PRIOR MATCHING in setting B. The two scale priors are matched
   ON THE IMPLIED STANDARD DEVIATION, `b_bounds = (0.1/sqrt(2), 6/sqrt(2))`
   against `sigma_bounds = (0.1, 6)`, so the prior boxes coincide exactly
   and tail shape really is the only thing separating the candidates.
   Unmatched boxes are a hidden second knob, and left unmatched they made
   the posterior-model-probability baseline report a stable, confident and
   entirely artefactual preference for the Gaussian.
   ================================================================ =#

using REFRAIN, Random, Statistics, Distributions

# ── truths ───────────────────────────────────────────────────────

"Truth for setting A: `Normal(c, s)` as a 1 x n row matrix."
function truth_gaussian(c::Float64, s::Float64; n::Int, seed::Int=1)
    rng = MersenneTwister(seed)
    return reshape(c .+ s .* randn(rng, n), 1, n)
end

"""
Truth for setting B: a standardised Student-t with `nu` degrees of
freedom, rescaled to unit variance so that both candidates, once moment
matched, differ from it only in TAIL SHAPE.
"""
function truth_student(nu::Float64; n::Int, seed::Int=1)
    @assert nu > 2 "need nu > 2 for finite variance"
    rng = MersenneTwister(seed)
    raw = rand(rng, TDist(nu), n)
    return reshape(raw ./ sqrt(nu / (nu - 2)), 1, n)
end

# ── the ABC distance ─────────────────────────────────────────────

# Scaled Euclidean distance between a candidate's analytic (mean, sd) and
# the fit half's. Deterministic in theta, i.e. no simulation inside it.
function _moment_distance(mean_model, sd_model, m1, sd1)
    sc = max(sd1, 1e-3)
    return hypot((mean_model - m1) / sc, (sd_model - sd1) / sc)
end

# ── candidates ───────────────────────────────────────────────────

"""
    make_gaussian_fixed_loc(mu0; sigma_bounds=(0.2, 5.0)) -> RelFitModel

Setting A candidate: `Normal(mu0, sigma)` with the location FIXED at
`mu0`, a structural choice and not a fitted one, and log-sigma the only
free parameter.

That the location is structural is the whole point of the setting, and it
is what a reader tends to miss: the two candidates cannot simply fit
their way to the middle, because only the scale is free.
"""
function make_gaussian_fixed_loc(mu0::Float64; sigma_bounds::Tuple{Float64,Float64}=(0.2, 5.0))
    priors = [Uniform(log(sigma_bounds[1]), log(sigma_bounds[2]))]
    # The summary is (mean, sd), so the ABC fit and the posterior-model-
    # probability baseline both see the fixed location mu0. That keeps the
    # two decision rules on identical machinery, which is what makes the
    # comparison between them a comparison of DECISION RULES.
    function factory(D1)
        v = vec(D1); m1 = mean(v); sd1 = std(v)
        return theta -> _moment_distance(mu0, exp(theta[1]), m1, sd1)
    end
    function simulator(theta, n; seed::Int=42)
        rng = MersenneTwister(seed)
        return reshape(mu0 .+ exp(theta[1]) .* randn(rng, n), 1, n)
    end
    return RelFitModel(priors, factory, simulator)
end

"""
    make_gaussian_free(; mu_bounds=(-6,6), sigma_bounds=(0.1,6)) -> RelFitModel

Setting B candidate: `Normal(mu, sigma)` with both free.
"""
function make_gaussian_free(; mu_bounds::Tuple{Float64,Float64}=(-6.0, 6.0),
                              sigma_bounds::Tuple{Float64,Float64}=(0.1, 6.0))
    priors = [Uniform(mu_bounds[1], mu_bounds[2]),
              Uniform(log(sigma_bounds[1]), log(sigma_bounds[2]))]
    function factory(D1)
        v = vec(D1); m1 = mean(v); sd1 = std(v)
        return theta -> _moment_distance(theta[1], exp(theta[2]), m1, sd1)
    end
    function simulator(theta, n; seed::Int=42)
        rng = MersenneTwister(seed)
        return reshape(theta[1] .+ exp(theta[2]) .* randn(rng, n), 1, n)
    end
    return RelFitModel(priors, factory, simulator)
end

"""
    make_laplace_free(; mu_bounds=(-6,6), b_bounds=(0.1/sqrt(2), 6/sqrt(2)))
        -> RelFitModel

Setting B candidate: `Laplace(mu, b)`, whose standard deviation is
`b sqrt(2)`.

THE DEFAULT `b_bounds` IS MATCHED ON THE IMPLIED STANDARD DEVIATION to
`make_gaussian_free`'s `sigma_bounds`, so the two prior boxes coincide
exactly. Do not widen one without widening the other: a prior-volume
ratio is a second knob, and it enters the posterior model probability as
an Occam factor that reads exactly like a finding.
"""
function make_laplace_free(; mu_bounds::Tuple{Float64,Float64}=(-6.0, 6.0),
                             b_bounds::Tuple{Float64,Float64}=(0.1 / sqrt(2), 6.0 / sqrt(2)))
    priors = [Uniform(mu_bounds[1], mu_bounds[2]),
              Uniform(log(b_bounds[1]), log(b_bounds[2]))]
    function factory(D1)
        v = vec(D1); m1 = mean(v); sd1 = std(v)
        return theta -> _moment_distance(theta[1], exp(theta[2]) * sqrt(2), m1, sd1)
    end
    function simulator(theta, n; seed::Int=42)
        rng = MersenneTwister(seed)
        b = exp(theta[2])
        return reshape(rand(rng, Laplace(theta[1], b), n), 1, n)
    end
    return RelFitModel(priors, factory, simulator)
end
