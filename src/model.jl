#= ================================================================
   The candidate-model contract, and the split.

   A candidate is three things and no more: a prior, a way of scoring a
   proposed parameter against the fit half, and a simulator. Everything
   else in the package is written against that triple, so a user who can
   supply it can run the rule on any problem.
   ================================================================ =#

"""
    RelFitModel(priors, abc_distance_factory, simulator)

A candidate model for likelihood-free relative fit. Three fields, and the
contracts are worth stating exactly because getting one of them subtly
wrong is silent.

  * `priors::Vector{<:Distribution}` -- one PER COORDINATE, independent.
    There is no joint prior object, so a prior with dependence between
    coordinates cannot be expressed. The perturbation kernel is a
    full-covariance MvNormal or MvCauchy, so discrete and simplex-valued
    parameters are not supported either. Reparametrise to an unbounded or
    box-constrained continuous vector.

  * `abc_distance_factory` -- `D1 -> rho_fn(theta)::Float64`. A FACTORY,
    not a distance: it is called once per fit with the fit half, and the
    closure it returns is called once per ABC proposal. That is where the
    target summaries get computed once instead of a hundred thousand
    times, and where the buffers get preallocated. The simulator is
    hidden entirely inside the closure, so the ABC layer never sees it.

  * `simulator` -- `(theta, n_sim; seed) -> d x n_sim matrix`, one i.i.d.
    unit per COLUMN. It must be a pure function of `(theta, n_sim, seed)`.
    The calibrations call it with seeds they draw themselves, so a
    simulator that reads a global RNG will not reproduce and will not be
    safe under threading.

See the rainfall example in `examples/` for the pattern that matters at
scale: an in-place simulator for the hot ABC path that never allocates,
an allocating wrapper for the calibration path that needs fresh matrices,
a distance factory that caches the target summaries and preallocates its
buffer once per fit, and a `hash((seed_base, theta))` simulator seed that
makes the closure both reproducible and safe under a driver-level
`@threads`.
"""
struct RelFitModel
    priors::Vector{<:Distribution}
    abc_distance_factory
    simulator
end

"""
    split_iid(X; frac_fit=0.5, seed=42) -> (D1, D0)

Split a `d`-rows by `n`-cols data matrix into a fit half `D1` and a
held-out test half `D0` by partitioning COLUMNS. The i.i.d. unit is a
column and rows are features.

The split is what makes the guarantee work: `theta_hat` comes from `D1`,
so conditional on it the held-out `D0` is an independent sample. Fitting
and testing on the same half would inherit the selection effect of the
fit.

An UNEQUAL split is not the free improvement it looks like. Measured on a
seven-parameter comparison, moving `frac_fit` from 0.5 to 0.8 HALVED the
signal-to-noise ratio of the statistic: shrinking `n_0` costs more than
improving `theta_hat` gains.
"""
function split_iid(X::AbstractMatrix; frac_fit::Float64=0.5, seed::Int=42)
    d, n = size(X)
    @assert n >= 2 "split_iid splits COLUMNS (i.i.d. units); need n>=2 columns, got size $(size(X))"
    rng = MersenneTwister(seed)
    perm = randperm(rng, n)
    n_fit = clamp(round(Int, frac_fit * n), 1, n - 1)
    return X[:, perm[1:n_fit]], X[:, perm[(n_fit + 1):end]]
end

"""
    split_contiguous(X; frac_fit=0.5, buffer=0) -> (D1, D0)

Split the columns of `X` into two CONTIGUOUS stretches, the first
`frac_fit` of them for fitting and the remainder for testing, discarding
`buffer` columns between the two.

This is the split to use when the columns are a dependent stationary
sequence rather than independent draws. `split_iid`'s random permutation
interleaves the two halves, so `theta_hat` stays dependent on the half it
is tested against and the guarantee that the pilot is fixed no longer
holds. Two contiguous stretches separated by a buffer are asymptotically
independent as the buffer grows under the usual mixing conditions. There
is no randomness here and so no seed.

Pair it with `bootstrap_calibrate_block`. The split alone is not the
repair, since the resampling of `D0` also assumes independence.
"""
function split_contiguous(X::AbstractMatrix; frac_fit::Float64=0.5,
                          buffer::Int=0)
    d, n = size(X)
    @assert n >= 2 "split_contiguous splits COLUMNS; need n>=2, got size $(size(X))"
    @assert buffer >= 0 "buffer must be non-negative, got $buffer"
    n_fit = clamp(round(Int, frac_fit * n), 1, n - 1)
    lo = min(n_fit + buffer + 1, n)
    return X[:, 1:n_fit], X[:, lo:n]
end

# ── Fitting a candidate on the fit half ──────────────────────────

"""
    abc_posterior(model, D1; N=2000, sampler=:abc_smc, kernel=:normal,
                  paccmin=1e-3, max_sims=100_000_000, seed=42, verbose=false)
        -> (particles, weights)

Fit `model` on the fit half `D1` and return the final weighted particle
population, `particles` being `n_params x N_final`.

`sampler` selects the ABC layer: `:abc_smc` (Toni et al. 2009, the
default and what the paper used), `:rejection` (the simplest thing that
works), or `:apmc` (Lenormand, Jabot and Deffuant 2013).
"""
function abc_posterior(model::RelFitModel, D1::AbstractMatrix;
                       N::Int=2000, sampler::Symbol=:abc_smc,
                       kernel::Symbol=:normal, paccmin::Float64=1e-3,
                       max_sims::Int=100_000_000, seed::Int=42,
                       verbose::Bool=false, kwargs...)
    rho_fn = model.abc_distance_factory(D1)
    if sampler === :abc_smc
        result = abc_smc(N, model.priors, rho_fn;
            perturb=kernel, perturb_weight=kernel, kernel_coeff=2.0,
            paccmin=paccmin, max_sims=max_sims, seed=seed, verbose=verbose,
            kwargs...)
        return result.pts[1][end], result.wts[1][end]
    elseif sampler === :rejection
        result = rejection_abc(N, model.priors, rho_fn;
            max_sims=max_sims, seed=seed, verbose=verbose, kwargs...)
        return result.particles, result.weights
    elseif sampler === :apmc
        result = apmc(N, [model.priors], [rho_fn];
            perturb=kernel, paccmin=paccmin, seed=seed, verbose=verbose,
            kwargs...)
        return result.pts[1][end], result.wts[1][end]
    end
    error("unknown sampler $sampler (use :abc_smc, :rejection or :apmc)")
end

"""
    abc_fit(model, D1; kwargs...) -> theta_hat

Weighted posterior mean of the ABC particles, i.e. a point estimate. This
is the pilot the relative-fit statistic is built on.
"""
function abc_fit(model::RelFitModel, D1::AbstractMatrix; kwargs...)
    particles, weights = abc_posterior(model, D1; kwargs...)
    return particles * (weights ./ sum(weights))
end

"""
    simulate_from_fit(model, theta_hat, n_sim; seed=42) -> d x n_sim matrix

Plug-in simulation: all `n_sim` units from the single point `theta_hat`.
This is what the statistic compares against, and deliberately so. The
target is the PROJECTION, i.e. the closest single member of each
candidate family, so a posterior predictive would be a mixture over the
posterior rather than any one member of the family and would move the
estimand rather than the interval.
"""
function simulate_from_fit(model::RelFitModel, theta_hat::AbstractVector, n_sim::Int;
                           seed::Int=42)
    return model.simulator(theta_hat, n_sim; seed=seed)
end

"""
    simulate_posterior_predictive(model, particles, weights, n_sim; seed=42)
        -> d x n_sim matrix

Posterior-predictive simulation: each of the `n_sim` units is drawn from a
parameter vector resampled from the ABC particles by weight.

PROVIDED FOR COMPLETENESS AND MEASURED TO BE THE WRONG THING for the
relative-fit statistic. See the docstring of `simulate_from_fit` for why,
and note the measurement: substituting it does cut the wrong-direction
rate, but it halves power and the loss is concentrated where the
wider-posterior candidate is the better one, i.e. it penalises whichever
candidate is harder to identify rather than whichever fits worse.
"""
function simulate_posterior_predictive(model::RelFitModel, particles::AbstractMatrix,
                                       weights::AbstractVector, n_sim::Int; seed::Int=42)
    w = weights ./ sum(weights)
    rng = MersenneTwister(seed)
    cols = Vector{Matrix{Float64}}(undef, n_sim)
    for j in 1:n_sim
        idx = sample(rng, 1:length(w), Weights(w))
        cols[j] = model.simulator(particles[:, idx], 1; seed=seed + j)
    end
    return reduce(hcat, cols)
end
