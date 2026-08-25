#= ================================================================
   The decision layer: the three-way rule, the adequacy screen, the
   K-candidate extension, and the two user-facing entry points.

   THE DECISION RULE IS ONE LINE and is kept visible on purpose:

       p > alpha ? :abstain : (T > 0 ? :M1 : :M0)

   Everything else in this package exists to make that line mean
   something. A reader who wants to check what the rule does should be
   able to find it in a second rather than reconstruct it from three
   layers of dispatch.
   ================================================================ =#

"""
    decide(p, T; alpha=0.05) -> Symbol

The three-way rule: `:abstain`, `:M0` or `:M1`. Abstention is a failure
to reject the equidistance null, which is the confidence-set duality:
failing to reject leaves both candidates in the set.
"""
decide(p::Real, T::Real; alpha::Float64=0.05) =
    p > alpha ? :abstain : (T > 0 ? :M1 : :M0)

"""
    decide(D0, P0, P1; ipm=:sw, calibration=:bootstrap, alpha=0.05,
           transform=:none, n_boot=299, seed=42, sw_seed=42,
           n_projections=SW_NPROJ) -> (decision, p, T, T_null)

TEST ONLY, for a reader who already has a held-out sample and two
simulated sets and wants nothing to do with the fitting layer. `D0`,
`P0` and `P1` are `d`-rows by `n`-cols matrices.

`calibration` is `:bootstrap` (the plug-in) or `:permutation`. The refit
bootstrap is NOT available here, because it needs the models and the fit
half in order to refit: reach for `refrain` or call `refit_bootstrap`
directly.
"""
function decide(D0::AbstractMatrix, P0::AbstractMatrix, P1::AbstractMatrix;
                ipm::Symbol=:sw, calibration::Symbol=:bootstrap,
                alpha::Float64=0.05, transform::Symbol=:none,
                n_boot::Int=299, seed::Int=42, sw_seed::Int=42,
                n_projections::Int=SW_NPROJ)
    cal = if calibration === :bootstrap
        bootstrap_calibrate(D0, P0, P1; n_boot=n_boot, ipm=ipm, transform=transform,
                            seed=seed, sw_seed=sw_seed, n_projections=n_projections)
    elseif calibration === :permutation
        r = permutation_calibrate(D0, P0, P1; n_perm=n_boot, ipm=ipm,
                                  transform=transform, seed=seed, sw_seed=sw_seed,
                                  n_projections=n_projections)
        (T_obs=r.T_obs, p=r.p, T_boot=r.T_null)
    elseif calibration === :refit
        error("decide: the refit bootstrap needs the two models and the fit half. " *
              "Use `refrain(X, model0, model1; calibration = :refit)`, or call " *
              "`refit_deviations` and `refit_bootstrap` directly.")
    else
        error("unknown calibration $calibration (use :bootstrap, :permutation or :refit)")
    end
    return (decision=decide(cal.p, cal.T_obs; alpha=alpha),
            p=cal.p, T=cal.T_obs, T_null=cal.T_boot)
end

# ── The absolute adequacy screen ─────────────────────────────────

#= WHAT ABSTENTION DOES NOT MEAN, and this is the single most common
   misreading of the method.

   The test is of RELATIVE fit, so abstaining means "the data cannot
   resolve which candidate is closer", NOT "neither candidate is any
   good". Those come apart exactly where it matters. On a problem whose
   truth lies outside both candidates the rule is MEANT to commit to
   whichever is closer, and does. On two identical candidates it abstains
   with probability about 1 - alpha, which is correct for a relative test
   and is not a 50/50 split.

   A purely relative test can never return an empty set, and the reason
   is structural rather than a consequence of there being two candidates:
   the set is inverted against a fixed pilot, no test asks whether the
   pilot is worse than itself, so the pilot survives at EVERY K. In the
   elimination form of `relfit_compare_K` the same fact appears as the
   loop halting at a singleton. Enlarging the model class therefore does
   not buy the missing verdict, which is an argument FOR the screen and
   not against it.

   The screen supplies the absolute question separately, on the held-out
   half only, and is reported BESIDE the three-way decision and never
   folded into it.

   TWO PROPERTIES TO BE PRECISE ABOUT. It must be computed on the
   HELD-OUT half: theta_hat comes from D1, so conditional on it the
   held-out D0 is independent and the statistics are exchangeable
   whenever D0 follows the fitted law. And its exactness is for the
   SIMPLE null "D0 is distributed as P_theta_hat", not for the composite
   null "the family contains the truth". Under the composite null
   theta_hat differs from theta_0 by O(n_1^{-1/2}) and the screen
   inherits that as OVER-REJECTION, measured at 0.05 with one fitted
   parameter and 0.15 with two against a nominal 0.05. Recalibrating it
   for the composite null is open work. =#

"""
    adequacy_screen(D0, model, theta_hat; ipm=:sw, transform=:none,
                    n_ref=size(D0,2), n_rep=299, alpha=0.05, seed=42,
                    sw_seed=42, n_projections=SW_NPROJ)
        -> (rho_obs, p, rho_null, inadequate)

One-sample goodness-of-fit screen for a SINGLE candidate on the held-out
half. `p` is the one-sided Monte Carlo p-value
`(1 + #{rho* >= rho_obs}) / (1 + n_rep)` and `inadequate` is
`p <= alpha`. Large `rho_obs` means the held-out data sit further from the
fitted predictive than fresh draws from it do, i.e. the candidate fails
in absolute terms.

Read it BESIDE the three-way decision, never folded into it. See the
module comment above for what abstention does and does not mean, and note
that under a well-specified candidate this screen over-rejects by a
measured amount that grows with the number of fitted parameters.
"""
function adequacy_screen(D0::AbstractMatrix, model::RelFitModel,
                         theta_hat::AbstractVector;
                         ipm::Symbol=:sw, transform::Symbol=:none,
                         n_ref::Int=size(D0, 2), n_rep::Int=299,
                         alpha::Float64=0.05, seed::Int=42, sw_seed::Int=42,
                         n_projections::Int=SW_NPROJ)
    D0t = _apply_transform(D0, transform)
    n0 = size(D0t, 2)
    P_ref = _apply_transform(simulate_from_fit(model, theta_hat, n_ref; seed=seed), transform)
    # The nuisance bandwidth is taken from P_ref ALONE. Taking it from a
    # pool that included D0 would make it a function of D0 and destroy the
    # exchangeability between D0 and the simulated replicates.
    bw = ipm === :mmd ? median_bandwidth(P_ref, P_ref) : nothing
    # P_ref is the fixed side here, and it sits SECOND in every call.
    rho_ref = rho_against_fixed(ipm, P_ref; bandwidth=bw, sw_seed=sw_seed,
                                fixed_first=false, n_projections=n_projections)
    rho_obs = rho_ref(D0t)
    rho_null = zeros(n_rep)
    cnt = 0
    for b in 1:n_rep
        # 7919 is prime and larger than any per-replication seed step a
        # driver is likely to use, so replicate streams cannot collide
        # across replications.
        Db = _apply_transform(simulate_from_fit(model, theta_hat, n0; seed=seed + 7919 * b),
                              transform)
        rho_null[b] = rho_ref(Db)
        rho_null[b] >= rho_obs && (cnt += 1)
    end
    p = (1 + cnt) / (1 + n_rep)
    return (rho_obs=rho_obs, p=p, rho_null=rho_null, inadequate=(p <= alpha))
end

"""
    adequacy_label(a0, a1) -> Symbol

Collapse two `adequacy_screen` results into the column reported beside the
three-way decision: `:both_fail`, `:M0_fails`, `:M1_fails` or
`:neither_fails`. `:both_fail` is the case the relative-fit rule cannot
express on its own, and the one that says to go and build a better model.
"""
function adequacy_label(a0, a1)
    a0.inadequate && a1.inadequate && return :both_fail
    a0.inadequate && return :M0_fails
    a1.inadequate && return :M1_fails
    return :neither_fails
end

# ── The full chain, two candidates ───────────────────────────────

"""
    RefrainResult

What `refrain` returns.

  * `decision` -- `:M0`, `:M1` or `:abstain`
  * `p`, `T` -- the p-value and the relative-fit statistic. `T > 0`
    favours `M1`
  * `interval` -- the percentile interval for the population gap whose
    exclusion of zero is exactly `p <= alpha`, or `nothing` under the
    permutation
  * `screen` -- `:both_fail`, `:M0_fails`, `:M1_fails`, `:neither_fails`
    or `:not_run`. READ THIS BESIDE `decision`, never folded into it
  * `screen0`, `screen1` -- the two `adequacy_screen` results in full
  * `theta0`, `theta1` -- the two fitted pilots
  * `D1`, `D0`, `P0`, `P1` -- the split and the two simulated sets
  * `calibration`, `ipm`, `alpha`, `n_sim` -- what was asked for

The source repository's `relfit_compare` returned an 18-field named
tuple, four fields of which existed only to feed a calibration that has
since been measured wrong. `refrain_full` keeps that shape for anyone
porting a driver.
"""
struct RefrainResult
    decision::Symbol
    p::Float64
    T::Float64
    interval::Union{Nothing,Tuple{Float64,Float64}}
    screen::Symbol
    screen0::Any
    screen1::Any
    theta0::Vector{Float64}
    theta1::Vector{Float64}
    D1::Matrix{Float64}
    D0::Matrix{Float64}
    P0::Matrix{Float64}
    P1::Matrix{Float64}
    T_null::Vector{Float64}
    calibration::Symbol
    ipm::Symbol
    alpha::Float64
    n_sim::Int
end

function Base.show(io::IO, r::RefrainResult)
    print(io, "RefrainResult(", r.decision, ", p = ", round(r.p; digits=4),
          ", T = ", round(r.T; digits=6), ", screen = ", r.screen, ")")
end

"""
    refrain(X, model0, model1; kwargs...) -> RefrainResult

END TO END, for a reader who has data and two simulators. `X` is
`d`-rows by `n`-cols with one i.i.d. unit per COLUMN.

    res = refrain(X, model0, model1;
                  ipm = :sw, alpha = 0.05,
                  calibration = :refit,   # :refit | :bootstrap | :permutation
                  sampler = :abc_smc)     # :rejection | :abc_smc | :apmc
    res.decision      # :M0 | :M1 | :abstain
    res.p, res.T
    res.screen        # :both_fail | :M0_fails | :M1_fails | :neither_fails

Five steps: split the columns into a fit half and a held-out half, fit
each candidate on the fit half by ABC, simulate fresh units from each
fit, form `T = rho(D0, P0) - rho(D0, P1)` on the held-out half, and
calibrate.

WHICH CALIBRATION. `:bootstrap` is the plug-in and is right where few
parameters are fitted. `:refit` propagates the fit uncertainty and is
what to use where many are, at a cost of `2S` further fits per decision.
`:permutation` is exact only where the two fitted predictives nearly
coincide, which is the regime in which there is nothing to decide. See
the comment at the top of `calibrate.jl`.

`n_mult` simulates `n_mult * n_0` units per candidate. Nothing requires
the simulated sets to match the held-out half's size: they estimate the
fitted predictives, and a practitioner is free to spend more simulation
on that estimate. It cuts the candidate side's share of the statistic's
noise while staying entirely likelihood-free.
"""
function refrain(X::AbstractMatrix, model0::RelFitModel, model1::RelFitModel;
                 ipm::Symbol=:sw, alpha::Float64=0.05,
                 calibration::Symbol=:bootstrap, sampler::Symbol=:abc_smc,
                 frac_fit::Float64=0.5, split_seed::Int=42,
                 N::Int=2000, kernel::Symbol=:normal, paccmin::Float64=1e-3,
                 max_sims::Int=100_000_000, fit_seed::Int=42, sim_seed::Int=1234,
                 transform::Symbol=:none, n_mult::Int=1,
                 n_boot::Int=299, cal_seed::Int=99, sw_seed::Int=42,
                 n_projections::Int=SW_NPROJ,
                 screen::Bool=true, screen_rep::Int=299,
                 refit_S::Int=5, refit_N::Int=800,
                 verbose::Bool=false)
    calibration in (:bootstrap, :permutation, :refit) ||
        error("unknown calibration $calibration (use :bootstrap, :permutation or :refit)")

    D1, D0 = split_iid(X; frac_fit=frac_fit, seed=split_seed)
    n_sim = n_mult * size(D0, 2)
    fitkw = (; sampler=sampler, N=N, kernel=kernel, paccmin=paccmin,
               max_sims=max_sims, verbose=verbose)
    theta0 = abc_fit(model0, D1; seed=fit_seed, fitkw...)
    theta1 = abc_fit(model1, D1; seed=fit_seed, fitkw...)

    #= P0 and P1 must come from streams that no other replication reuses.
       Drawing them at `sim_seed` and `sim_seed + 1` while a driver steps
       `sim_seed` by one per replication makes replication r's P1 the
       IDENTICAL stream as replication r + 1's P0, so consecutive
       replications are not independent and every reported rate's binomial
       standard error understates the true Monte Carlo error. On a
       location comparison where both candidates are mu + exp(theta)*randn
       on that stream, the coupled draws had correlation 1. =#
    p1_offset = 1_000_000
    P0 = simulate_from_fit(model0, theta0, n_sim; seed=sim_seed)
    P1 = simulate_from_fit(model1, theta1, n_sim; seed=sim_seed + p1_offset)

    T_null = Float64[]
    interval = nothing
    p = NaN; T = NaN
    if calibration === :permutation
        cal = permutation_calibrate(D0, P0, P1; n_perm=n_boot, ipm=ipm,
                                    transform=transform, seed=cal_seed,
                                    sw_seed=sw_seed, n_projections=n_projections)
        p = cal.p; T = cal.T_obs; T_null = cal.T_null
    elseif calibration === :bootstrap
        cal = bootstrap_calibrate(D0, P0, P1; n_boot=n_boot, ipm=ipm,
                                  transform=transform, seed=cal_seed,
                                  sw_seed=sw_seed, n_projections=n_projections)
        p = cal.p; T = cal.T_obs; T_null = cal.T_boot
        interval = percentile_interval(T_null; alpha=alpha)
    else
        dev0, dev1 = refit_deviations(model0, model1, D1, fit_seed;
                                      S=refit_S, n_refit=refit_N,
                                      sampler=sampler, kernel=kernel,
                                      paccmin=paccmin, max_sims=max_sims,
                                      verbose=verbose)
        cal = refit_bootstrap(D0, P0, P1, model0, model1, theta0, theta1, dev0, dev1;
                              n_boot=n_boot, ipm=ipm, transform=transform,
                              seed=cal_seed, n_projections=n_projections)
        p = cal.p; T = cal.T_obs; T_null = cal.T_boot
        interval = percentile_interval(T_null; alpha=alpha)
    end

    scr0 = scr1 = nothing
    label = :not_run
    if screen
        scr0 = adequacy_screen(D0, model0, theta0; ipm=ipm, transform=transform,
                               n_rep=screen_rep, alpha=alpha,
                               seed=sim_seed + 2_000_000, sw_seed=sw_seed,
                               n_projections=n_projections)
        scr1 = adequacy_screen(D0, model1, theta1; ipm=ipm, transform=transform,
                               n_rep=screen_rep, alpha=alpha,
                               seed=sim_seed + 3_000_000, sw_seed=sw_seed,
                               n_projections=n_projections)
        label = adequacy_label(scr0, scr1)
    end

    return RefrainResult(decide(p, T; alpha=alpha), p, T, interval, label,
                         scr0, scr1, theta0, theta1,
                         Matrix{Float64}(D1), Matrix{Float64}(D0),
                         Matrix{Float64}(P0), Matrix{Float64}(P1),
                         T_null, calibration, ipm, alpha, n_sim)
end

"""
    refrain_full(X, model0, model1; kwargs...) -> NamedTuple

The lower-level call, returning the full named tuple the source
repository's `relfit_compare` returned, so a driver written against it
can be ported by changing the function name. It always uses the
permutation calibration and always keeps the ABC particle populations,
which is what that function did.

Prefer `refrain`, which returns a documented struct and lets the
calibration be chosen.
"""
function refrain_full(X::AbstractMatrix, model0::RelFitModel, model1::RelFitModel;
                      frac_fit::Float64=0.5, split_seed::Int=42,
                      N::Int=2000, sampler::Symbol=:abc_smc,
                      kernel::Symbol=:normal, paccmin::Float64=1e-3,
                      max_sims::Int=100_000_000, fit_seed::Int=42, sim_seed::Int=1234,
                      ipm::Symbol=:sw, transform::Symbol=:none, predictive::Bool=false,
                      n_mult::Int=1,
                      n_perm::Int=500, perm_seed::Int=99, sw_seed::Int=42,
                      n_projections::Int=SW_NPROJ,
                      screen::Bool=false, screen_rep::Int=299, alpha::Float64=0.05,
                      verbose::Bool=false)
    D1, D0 = split_iid(X; frac_fit=frac_fit, seed=split_seed)
    n_sim = n_mult * size(D0, 2)
    p1_offset = 1_000_000
    fitkw = (; sampler=sampler, N=N, kernel=kernel, paccmin=paccmin,
               max_sims=max_sims, verbose=verbose)
    pts0, w0 = abc_posterior(model0, D1; seed=fit_seed, fitkw...)
    pts1, w1 = abc_posterior(model1, D1; seed=fit_seed, fitkw...)
    theta_hat_0 = pts0 * (w0 ./ sum(w0))
    theta_hat_1 = pts1 * (w1 ./ sum(w1))
    if predictive
        P0 = simulate_posterior_predictive(model0, pts0, w0, n_sim; seed=sim_seed)
        P1 = simulate_posterior_predictive(model1, pts1, w1, n_sim; seed=sim_seed + p1_offset)
    else
        P0 = simulate_from_fit(model0, theta_hat_0, n_sim; seed=sim_seed)
        P1 = simulate_from_fit(model1, theta_hat_1, n_sim; seed=sim_seed + p1_offset)
    end
    cal = permutation_calibrate(D0, P0, P1; n_perm=n_perm, ipm=ipm,
                                transform=transform, seed=perm_seed, sw_seed=sw_seed,
                                n_projections=n_projections)
    scr0 = scr1 = nothing
    if screen
        scr0 = adequacy_screen(D0, model0, theta_hat_0; ipm=ipm, transform=transform,
                               n_rep=screen_rep, alpha=alpha,
                               seed=sim_seed + 2_000_000, sw_seed=sw_seed,
                               n_projections=n_projections)
        scr1 = adequacy_screen(D0, model1, theta_hat_1; ipm=ipm, transform=transform,
                               n_rep=screen_rep, alpha=alpha,
                               seed=sim_seed + 3_000_000, sw_seed=sw_seed,
                               n_projections=n_projections)
    end
    return (T_obs=cal.T_obs, p=cal.p, T_null=cal.T_null,
            theta_hat_0=theta_hat_0, theta_hat_1=theta_hat_1,
            pts0=pts0, w0=w0, pts1=pts1, w1=w1,
            D1=D1, D0=D0, P0=P0, P1=P1, n_sim=n_sim, ipm=ipm, transform=transform,
            screen0=scr0, screen1=scr1,
            adequacy=(screen ? adequacy_label(scr0, scr1) : :not_run))
end

# ================================================================
#  K candidates
# ================================================================

#= The construction is not two-bound. Park, Balakrishnan and Wasserman
   invert a family of pairwise tests against a common pilot over a whole
   model class, and the confidence set that results is a set of
   candidates, of which the two-model three-way decision is the K = 2
   special case: a singleton is a commitment and a set of size two is
   abstention.

   The naive extension, K(K-1)/2 pairwise tests with a Bonferroni
   correction, is NOT what is done here. It is conservative, and it needs
   a correction only because the pairwise nulls are calibrated
   separately. Instead the elimination is driven by a MAX-TYPE statistic,
   the range of the held-out distances over the candidates still in the
   set, calibrated by the same three-sample bootstrap generalised to
   K + 1 samples. The bootstrap null is joint over the candidates, so it
   already carries the multiplicity and needs no correction. This is the
   model-confidence-set construction of Hansen, Lunde and Nason (2011)
   with their loss differentials replaced by an integral-probability
   contrast and their block bootstrap by the three-sample resample.

   THE MAX-TYPE RULE NEEDS A LARGER HELD-OUT HALF than the scalar one.
   It is the BASIC bootstrap rather than the percentile one, and measured
   at a K = 5 coincidence boundary it runs at 0.0655 with n_0 = 200 and
   at nominal with n_0 = 400.

   `hoeffding_mmd_test` does NOT come along, and that is a fact about the
   statistic: the empirical MMD witness is a single direction in the RKHS
   unit ball and K embeddings do not determine one.

   THE K = 2 REDUCTION IS A CONTRACT, not a coincidence, and
   `test/runtests.jl` asserts it BIT FOR BIT. `relfit_distances_K` must
   difference to `relfit_statistic`, `permutation_calibrate_K` must
   return the p-value of `permutation_calibrate`, and every bootstrap
   replicate of `bootstrap_calibrate_K` must difference to the
   corresponding replicate of `bootstrap_calibrate`. It is the only cheap
   guard against the K path quietly becoming a second, subtly different
   implementation of the published statistic.

   The bootstrap DECISION RULE is the one thing that does not reduce, and
   deliberately: at K = 2 the scalar rule is the percentile-interval dual,
   which counts sign changes of T*, while the max-type rule compares the
   observed range against the range of the CENTRED resamples. Those are
   the percentile and the basic bootstrap of the same quantity. They agree
   asymptotically and differ at finite B. =#

"""
    relfit_distances_K(D0, P; ipm=:sw, transform=:none, bandwidth=nothing,
                       sw_seed=42, n_projections=SW_NPROJ) -> Vector{Float64}

The vector of held-out distances `r_j = rho(D0, P_j)`, `j = 1..K`.
Smaller is closer, so the best candidate is the `argmin`. At `K = 2`,
`r[1] - r[2] == relfit_statistic(D0, P[1], P[2])` exactly.
"""
function relfit_distances_K(D0::AbstractMatrix, P::AbstractVector{<:AbstractMatrix};
                            ipm::Symbol=:sw, transform::Symbol=:none,
                            bandwidth=nothing, sw_seed::Int=42,
                            n_projections::Int=SW_NPROJ)
    D0t = _apply_transform(D0, transform)
    Pt = [_apply_transform(Pj, transform) for Pj in P]
    bw = bandwidth === nothing ? common_bandwidth_K(ipm, D0t, Pt) : bandwidth
    rho = rho_against_fixed(ipm, D0t; bandwidth=bw, sw_seed=sw_seed,
                            n_projections=n_projections)
    return [rho(Pj) for Pj in Pt]
end

# The max-type statistic: the spread of the distances over a candidate
# set. At K = 2 this is |T|.
function _range_over(r::AbstractVector{Float64}, S::AbstractVector{Int})
    hi = -Inf; lo = Inf
    @inbounds for j in S
        r[j] > hi && (hi = r[j])
        r[j] < lo && (lo = r[j])
    end
    return hi - lo
end

"""
    permutation_calibrate_K(D0, P; n_perm=500, ipm=:sw, transform=:none,
                            seed=42, sw_seed=42, bandwidth=nothing,
                            n_projections=SW_NPROJ)
        -> (r_obs, R_obs, p, R_null)

Permutation calibration of the max-type statistic. Pool the `K` simulated
samples, relabel into `K` groups of the original sizes, recompute the
range of the `K` distances with `D0` held fixed, and report
`p = (1 + #{R* >= R_obs}) / (1 + n_perm)`. Exact under exchangeability of
the pooled draws.

At `K = 2` it returns the p-value of `permutation_calibrate` bit for bit,
because the range of two numbers is the absolute value of their
difference and the relabelling draws the same permutation from the same
stream.
"""
function permutation_calibrate_K(D0::AbstractMatrix, P::AbstractVector{<:AbstractMatrix};
                                 n_perm::Int=500, ipm::Symbol=:sw,
                                 transform::Symbol=:none, seed::Int=42,
                                 sw_seed::Int=42, bandwidth=nothing,
                                 n_projections::Int=SW_NPROJ)
    D0t = _apply_transform(D0, transform)
    Pt = [_apply_transform(Pj, transform) for Pj in P]
    K = length(Pt)
    K >= 2 || error("permutation_calibrate_K needs at least two candidates, got $K")
    #= `bandwidth` is supplied by `permutation_set_K` so that every
       elimination step measures in the SAME reproducing-kernel Hilbert
       space, namely the one pooled over all K candidates. Recomputing it
       per surviving subset would make the distances compared at step 2
       incommensurable with the ones that decided step 1. =#
    bw = bandwidth === nothing ? common_bandwidth_K(ipm, D0t, Pt) : bandwidth
    rho = rho_against_fixed(ipm, D0t; bandwidth=bw, sw_seed=sw_seed,
                            n_projections=n_projections)
    r_obs = [rho(Pj) for Pj in Pt]
    R_obs = _range_over(r_obs, collect(1:K))
    pooled = hcat(Pt...)
    sizes = [size(Pj, 2) for Pj in Pt]
    n_total = size(pooled, 2)
    rng = MersenneTwister(seed)
    R_null = zeros(n_perm); cnt = 0
    rb = zeros(K)
    for b in 1:n_perm
        perm = randperm(rng, n_total)
        lo = 1
        for j in 1:K
            hi = lo + sizes[j] - 1
            rb[j] = rho(pooled[:, perm[lo:hi]])
            lo = hi + 1
        end
        Rb = maximum(rb) - minimum(rb)
        R_null[b] = Rb
        Rb >= R_obs && (cnt += 1)
    end
    return (r_obs=r_obs, R_obs=R_obs, p=(1 + cnt) / (1 + n_perm), R_null=R_null)
end

"""
    bootstrap_calibrate_K(D0, P; n_boot=299, ipm=:sw, transform=:none, seed=42,
                          sw_seed=42, alpha=0.05, m=nothing,
                          n_projections=SW_NPROJ)
        -> (r_obs, set, p_mcs, p_steps, eliminated, R_boot)

Three-sample bootstrap generalised to `K + 1` samples, inverted into a
confidence set of candidates by sequential elimination.

Each replicate resamples the columns of `D0` ONCE, shared across all `K`
distance terms, and the columns of each `P_j` independently. The
elimination then runs on those numbers alone: with `S` the surviving set,
compare the observed range over `S` against the range of the CENTRED
resamples over `S`. If `p > alpha` stop and return `S`, otherwise drop
`argmax_j r_obs[j]` and repeat. `p_mcs` is the running maximum of the step
p-values, so the returned set is exactly `{j : p_mcs[j] > alpha}`.

Abstention is `length(set) > 1` and a decisive answer is a singleton. The
set is NEVER EMPTY, because the last surviving candidate is returned once
the loop reaches size one. That is the K-model form of the fact that a
purely relative test cannot convict every candidate, and the reason the
adequacy screen is reported beside it.
"""
function bootstrap_calibrate_K(D0::AbstractMatrix, P::AbstractVector{<:AbstractMatrix};
                               n_boot::Int=299, ipm::Symbol=:sw,
                               transform::Symbol=:none, seed::Int=42,
                               sw_seed::Int=42, alpha::Float64=0.05,
                               m::Union{Nothing,Int}=nothing,
                               n_projections::Int=SW_NPROJ)
    D0t = _apply_transform(D0, transform)
    Pt = [_apply_transform(Pj, transform) for Pj in P]
    K = length(Pt)
    K >= 2 || error("bootstrap_calibrate_K needs at least two candidates, got $K")
    bw = common_bandwidth_K(ipm, D0t, Pt)
    sc = ipm === :sw ? sw_scratch(size(D0t, 1), sw_seed; n_projections=n_projections) : nothing
    rho_obs = rho_against_fixed(ipm, D0t; bandwidth=bw, sw_seed=sw_seed, scratch=sc,
                                n_projections=n_projections)
    r_obs = [rho_obs(Pj) for Pj in Pt]
    n_d = size(D0t, 2); n_j = [size(Pj, 2) for Pj in Pt]
    m === nothing || m >= 2 ||
        error("bootstrap_calibrate_K: m = $m is too small; the unbiased MMD needs m >= 2")
    m_d = m === nothing ? n_d : m
    m_j = m === nothing ? n_j : fill(m, K)
    rng = MersenneTwister(seed)
    R_boot = zeros(n_boot, K)
    #= The draw order is D0* first, then the K simulated samples in index
       order, which is the order `bootstrap_calibrate` uses at K = 2. The
       resampled columns are therefore the same columns off the same
       stream, which is what makes the two agree bit for bit. =#
    Pb = Vector{Matrix{Float64}}(undef, K)
    for b in 1:n_boot
        Db = D0t[:, rand(rng, 1:n_d, m_d)]
        for j in 1:K
            Pb[j] = Pt[j][:, rand(rng, 1:n_j[j], m_j[j])]
        end
        rho_b = rho_against_fixed(ipm, Db; bandwidth=bw, sw_seed=sw_seed, scratch=sc,
                                  n_projections=n_projections)
        for j in 1:K
            R_boot[b, j] = rho_b(Pb[j])
        end
    end
    # m == n is the n-out-of-n bootstrap and must keep its arithmetic
    # untouched rather than being multiplied through by a floating-point 1.0.
    if !(m === nothing || m == n_d)
        s = sqrt(m / n_d)
        @inbounds for j in 1:K, b in 1:n_boot
            R_boot[b, j] = r_obs[j] + s * (R_boot[b, j] - r_obs[j])
        end
    end
    # ── elimination ──
    #= One test per surviving set, the worst candidate dropped on each
       rejection, stopping at the first non-rejection. `p_mcs` is the
       running maximum of the step p-values, which is what makes the
       reported set exactly {j : p_mcs[j] > alpha}: a candidate eliminated
       at an early step cannot be readmitted by a later step's larger
       p-value, so the maximum has to be carried forward. The final
       survivor of a chain of rejections has no test of its own left and
       takes the conventional p-value 1 (Hansen, Lunde and Nason 2011). =#
    S = collect(1:K)
    p_mcs = zeros(K)
    p_steps = Float64[]
    eliminated = Int[]
    running = 0.0
    while length(S) >= 2
        R_S = _range_over(r_obs, S)
        cnt = 0
        @inbounds for b in 1:n_boot
            dhi = -Inf; dlo = Inf
            for j in S
                d = R_boot[b, j] - r_obs[j]
                d > dhi && (dhi = d)
                d < dlo && (dlo = d)
            end
            (dhi - dlo) >= R_S && (cnt += 1)
        end
        p = (1 + cnt) / (1 + n_boot)
        push!(p_steps, p)
        running = max(running, p)
        p > alpha && break
        worst = S[argmax([r_obs[j] for j in S])]
        push!(eliminated, worst)
        p_mcs[worst] = running
        S = filter(!=(worst), S)
    end
    for j in S
        p_mcs[j] = length(S) == 1 ? 1.0 : running
    end
    return (r_obs=r_obs, set=S, p_mcs=p_mcs, p_steps=p_steps,
            eliminated=eliminated, R_boot=R_boot)
end

"""
    permutation_set_K(D0, P; n_perm=299, ipm=:sw, transform=:none, seed=42,
                      sw_seed=42, alpha=0.05, n_projections=SW_NPROJ)
        -> (r_obs, set, p_mcs, p_steps, eliminated)

The same elimination as `bootstrap_calibrate_K` with the permutation null
in place of the bootstrap null. Each step pools only the candidates still
in the set, so the null at every step is the exchangeability null for the
surviving subset. It inherits the permutation's regime: exact when the
surviving predictives are one law, anti-conservative as they separate.

Measured at K = 5, that failure is more legible than at K = 2: at a
two-way tie inside a five-way comparison the permutation returned a
SINGLETON in 0.48 and 0.72 of replications and so dropped the best
candidate in 26% and 36% of them.
"""
function permutation_set_K(D0::AbstractMatrix, P::AbstractVector{<:AbstractMatrix};
                           n_perm::Int=299, ipm::Symbol=:sw, transform::Symbol=:none,
                           seed::Int=42, sw_seed::Int=42, alpha::Float64=0.05,
                           n_projections::Int=SW_NPROJ)
    K = length(P)
    D0t = _apply_transform(D0, transform)
    Pt = [_apply_transform(Pj, transform) for Pj in P]
    bw = common_bandwidth_K(ipm, D0t, Pt)
    r_obs = relfit_distances_K(D0, P; ipm=ipm, transform=transform,
                               bandwidth=bw, sw_seed=sw_seed,
                               n_projections=n_projections)
    S = collect(1:K)
    p_mcs = zeros(K); p_steps = Float64[]; eliminated = Int[]
    running = 0.0; step = 0
    while length(S) >= 2
        step += 1
        # 104729 is prime and far larger than any per-replication seed step
        # a driver is likely to use, so no two elimination steps and no two
        # replications share a relabelling stream.
        p = permutation_calibrate_K(D0, P[S]; n_perm=n_perm, ipm=ipm,
                                    transform=transform, seed=seed + 104_729 * step,
                                    sw_seed=sw_seed, bandwidth=bw,
                                    n_projections=n_projections).p
        push!(p_steps, p)
        running = max(running, p)
        p > alpha && break
        worst = S[argmax([r_obs[j] for j in S])]
        push!(eliminated, worst)
        p_mcs[worst] = running
        S = filter(!=(worst), S)
    end
    for j in S
        p_mcs[j] = length(S) == 1 ? 1.0 : running
    end
    return (r_obs=r_obs, set=S, p_mcs=p_mcs, p_steps=p_steps, eliminated=eliminated)
end

"""
    relfit_compare_K(X, models; kwargs...) -> NamedTuple

The full chain for `K` candidates: split, fit each candidate on the fit
half by ABC, simulate fresh units from each fit, and calibrate the
max-type statistic by both the permutation and the three-sample
bootstrap. Returns the two calibrations' sets alongside the fitted
parameters, the simulated samples and, with `screen=true`, one
`adequacy_screen` per candidate.
"""
function relfit_compare_K(X::AbstractMatrix, models::AbstractVector{RelFitModel};
                          frac_fit::Float64=0.5, split_seed::Int=42,
                          N::Int=2000, sampler::Symbol=:abc_smc,
                          kernel::Symbol=:normal, paccmin::Float64=1e-3,
                          max_sims::Int=100_000_000, fit_seed::Int=42, sim_seed::Int=1234,
                          ipm::Symbol=:sw, transform::Symbol=:none,
                          n_perm::Int=299, n_boot::Int=299, perm_seed::Int=99,
                          boot_seed::Int=199, sw_seed::Int=42, alpha::Float64=0.05,
                          n_projections::Int=SW_NPROJ,
                          screen::Bool=false, screen_rep::Int=299, verbose::Bool=false)
    K = length(models)
    D1, D0 = split_iid(X; frac_fit=frac_fit, seed=split_seed)
    n_sim = size(D0, 2)
    # One stream per candidate, separated by the same 10^6 offset that
    # keeps replications apart at K = 2, so no two candidates and no two
    # replications can share a stream.
    theta_hat = [abc_fit(models[j], D1; N=N, sampler=sampler, kernel=kernel,
                         paccmin=paccmin, max_sims=max_sims, seed=fit_seed,
                         verbose=verbose)
                 for j in 1:K]
    P = [simulate_from_fit(models[j], theta_hat[j], n_sim;
                           seed=sim_seed + 1_000_000 * (j - 1)) for j in 1:K]
    boot = bootstrap_calibrate_K(D0, P; n_boot=n_boot, ipm=ipm, transform=transform,
                                 seed=boot_seed, sw_seed=sw_seed, alpha=alpha,
                                 n_projections=n_projections)
    perm_set = permutation_set_K(D0, P; n_perm=n_perm, ipm=ipm, transform=transform,
                                 seed=perm_seed, sw_seed=sw_seed, alpha=alpha,
                                 n_projections=n_projections)
    scr = nothing
    if screen
        scr = [adequacy_screen(D0, models[j], theta_hat[j]; ipm=ipm, transform=transform,
                               n_rep=screen_rep, alpha=alpha,
                               seed=sim_seed + 2_000_000 + 1_000_000 * (j - 1),
                               sw_seed=sw_seed, n_projections=n_projections)
               for j in 1:K]
    end
    return (r_obs=boot.r_obs, boot_set=boot.set, boot_p=boot.p_mcs,
            boot_steps=boot.p_steps,
            perm_set=perm_set.set, perm_p=perm_set.p_mcs,
            perm_p_full=perm_set.p_steps[1],
            theta_hat=theta_hat, D1=D1, D0=D0, P=P, n_sim=n_sim,
            ipm=ipm, transform=transform, screens=scr)
end
