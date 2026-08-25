#= ================================================================
   The three calibrations, and the repair.

   The statistic is T = rho(D0, P0) - rho(D0, P1) on the held-out half,
   and T > 0 favours M1. What turns it into a three-way decision is a
   null, and there are three of them here with three different regimes.

   WHICH ONE TO USE IS A CONDITIONAL ANSWER, not a ranking, and the
   condition is how much is being fitted.

     `bootstrap_calibrate` where FEW parameters are fitted. It targets
     the equidistance null directly and held its level at all three of
     the one-dimensional ties measured in the paper.

     `refit_bootstrap` where MANY are. The plug-in above resamples the
     COLUMNS of P0 and P1, so it sees the Monte Carlo error of drawing
     n_sim units from a KNOWN parameter and nothing else. theta_hat was
     estimated on D1 and that error is invisible to it. Measured against
     the across-replication spread of T, the plug-in interval is right to
     within a factor of 0.97 to 1.06 with one fitted parameter and 0.83
     to 1.15 with two, and is too NARROW by 1.16 to 1.46 with five, where
     it then rejects at about twice the nominal rate. Raising n does not
     fix it: the term the resample sees and the term it misses both scale
     as n^{-1/2}, so their ratio is invariant.

     `permutation_calibrate` in neither, or rather only in the regime
     where its exactness holds. It is exact for the EXCHANGEABILITY null,
     which coincides with the equidistance null only where the two fitted
     predictives coincide, and that is the regime in which there is
     nothing to decide. Away from it the permuted pool mixes the two
     predictive clusters, the null is too narrow and the test
     over-rejects, measured at 0.140 and 0.349 where the predictives are
     close and 0.365 and 0.658 where they are two units apart.

   `hoeffding_mmd_test` is the concentration threshold of Park,
   Balakrishnan and Wasserman, kept because it fails in the COMPLEMENTARY
   regime and so is a useful cross-check: safe at every tie whose
   predictives differ, and committing in 0.676 of replications at the one
   null where the other two are exact.

   Ported verbatim from `relfit_core.jl`, `refit_bootstrap.jl` and
   `rain_study.jl` of the abstention_model_choice repository.
   ================================================================ =#

"""
    relfit_statistic(D0, P0, P1; ipm=:sw, transform=:none, bandwidth=nothing,
                     sw_seed=42, n_projections=SW_NPROJ) -> Float64

Relative-fit statistic on the held-out half,
`T = rho(D0, P0) - rho(D0, P1)`. `T > 0` favours `M1`, i.e. the candidate
whose simulations are closer to the held-out data.

`bandwidth` and `sw_seed` fix the metric's nuisance choice so that both
terms use the same function class. Pass the value returned by
`common_bandwidth`.
"""
function relfit_statistic(D0::AbstractMatrix, P0::AbstractMatrix, P1::AbstractMatrix;
                          ipm::Symbol=:sw, transform::Symbol=:none,
                          bandwidth=nothing, sw_seed::Int=42,
                          n_projections::Int=SW_NPROJ)
    D0t = _apply_transform(D0, transform)
    P0t = _apply_transform(P0, transform)
    P1t = _apply_transform(P1, transform)
    bw = bandwidth === nothing ? common_bandwidth(ipm, D0t, P0t, P1t) : bandwidth
    rho = rho_against_fixed(ipm, D0t; bandwidth=bw, sw_seed=sw_seed,
                            n_projections=n_projections)
    return rho(P0t) - rho(P1t)
end

# ── Calibration 1: permutation ───────────────────────────────────

"""
    permutation_calibrate(D0, P0, P1; n_perm=500, ipm=:sw, transform=:none,
                          seed=42, sw_seed=42, n_projections=SW_NPROJ)
        -> (T_obs, p, T_null)

Calibrate by permutation. Pool the simulated draws `P0` and `P1`, relabel
into two groups of the original sizes many times, and recompute `T*` with
`D0` held fixed. Two-sided p-value
`p = (1 + #{|T*| >= |T_obs|}) / (1 + n_perm)`.

EXACT for the exchangeability null, which is a strictly stronger
hypothesis than the equidistance null the decision is about, and which
they coincide on only where the two fitted predictives coincide. Read the
module comment before choosing this one.
"""
function permutation_calibrate(D0::AbstractMatrix, P0::AbstractMatrix, P1::AbstractMatrix;
                               n_perm::Int=500, ipm::Symbol=:sw,
                               transform::Symbol=:none, seed::Int=42,
                               sw_seed::Int=42, n_projections::Int=SW_NPROJ)
    D0t = _apply_transform(D0, transform)
    P0t = _apply_transform(P0, transform)
    P1t = _apply_transform(P1, transform)
    # one nuisance choice for the observed statistic and every replicate
    bw = common_bandwidth(ipm, D0t, P0t, P1t)
    # D0 is fixed for the observed statistic and for every replicate, so
    # its projections and sorts are done once rather than 2(1 + n_perm)
    # times.
    rho = rho_against_fixed(ipm, D0t; bandwidth=bw, sw_seed=sw_seed,
                            n_projections=n_projections)
    T(A, B) = rho(A) - rho(B)
    T_obs = T(P0t, P1t)
    pooled = hcat(P0t, P1t)
    n_total = size(pooled, 2); n0 = size(P0t, 2)
    rng = MersenneTwister(seed)
    T_null = zeros(n_perm); count_ge = 0
    for b in 1:n_perm
        perm = randperm(rng, n_total)
        g0 = pooled[:, perm[1:n0]]; g1 = pooled[:, perm[(n0 + 1):end]]
        Tb = T(g0, g1)
        T_null[b] = Tb
        abs(Tb) >= abs(T_obs) && (count_ge += 1)
    end
    return (T_obs=T_obs, p=(1 + count_ge) / (1 + n_perm), T_null=T_null)
end

# ── Calibration 2: three-sample bootstrap of the equidistance null ──

#= The statistic T estimates the population gap
   Delta = rho(P*, F0) - rho(P*, F1), and the equidistance null is
   Delta = 0 with NO constraint that F0 = F1. Resampling the columns of
   D0, P0 and P1 independently with replacement and recomputing T
   approximates the sampling distribution of T about Delta with the two
   predictives held at their realised geometry, so the spread is right
   whether or not the predictives coincide. Sharing one resampled D0*
   across the two rho terms per replicate preserves their correlation,
   which is what makes the difference far less variable than either term.

   The three-way decision is confidence-interval duality: choose a model
   when the percentile interval for Delta excludes zero, abstain when it
   does not. This is the sample-based relative-similarity test of
   Bounliphone et al. (2016) with the delta-method normal approximation
   replaced by a bootstrap, which extends it beyond the MMD to any
   sample-only IPM.

   What is and is not guaranteed: the interval is a large-sample
   percentile interval, not finite-sample exact. At the coincidence
   boundary F0 = F1 the D0-influence of the two terms cancels to first
   order and the bootstrap, which still resamples the simulation noise in
   P0 and P1, is if anything conservative there. So its regime is the
   complement of the permutation's: approximate everywhere, degrading
   nowhere, instead of exact at one boundary and inflating away from it.

   THE m-OUT-OF-n BRANCH. Resampling at m << n is the standard repair for
   a bootstrap in trouble at a boundary (Bickel, Gotze and van Zwet 1997;
   Politis and Romano 1994). Two details make it a drop-in here. The
   resample must be at size m for ALL THREE samples, because the
   statistic's variance has contributions from the simulation noise in P0
   and P1 that an m-out-of-n on D0 alone would leave at their n-scale.
   And the resulting spread is that of T at sample size m, which is
   sqrt(n/m) times too wide for the statistic actually reported, so it is
   rescaled about T_obs by sqrt(m/n) before the interval is read off. The
   rescaling is what stops "m-out-of-n repairs the level" from being the
   trivial observation that a wider interval rejects less often.

   DO NOT REACH FOR IT AS A GENERAL REPAIR. Measured, it moves the width
   the WRONG WAY on the geometries tried: the duplication roughness grows
   as the resample coarsens faster than the sqrt(m/n) rescaling
   compensates. And it cannot address a fit-uncertainty shortfall at all,
   since coarsening a resample changes how the simulated sets are drawn
   and not whether the parameter behind them was estimated.

   `m === nothing` is the default and takes the original branch verbatim,
   so every published number is reproduced bit for bit rather than
   approximately: T_obs + 1.0 * (T_b - T_obs) is not T_b in floating
   point. =#

"""
    bootstrap_calibrate(D0, P0, P1; n_boot=299, ipm=:sw, transform=:none,
                        seed=42, sw_seed=42, m=nothing, n_projections=SW_NPROJ)
        -> (T_obs, p, T_boot)

Calibrate against the equidistance null by a three-sample bootstrap.
Columns of `D0`, `P0` and `P1` are resampled with replacement
independently, `T*` is recomputed each time, and the two-sided p-value is
the percentile-interval dual
`p = min(1, 2 (1 + min(#{T* <= 0}, #{T* >= 0})) / (1 + n_boot))`.
Decision at level `alpha`: abstain if `p > alpha`, otherwise choose the
candidate favoured by the sign of `T_obs`.

The duality is exact, but only for a stated quantile convention. Taking
the equal-tailed percentile interval to run between the `k`-th and
`(n_boot + 1 - k)`-th order statistics of `T_boot` with
`k = floor((n_boot + 1) alpha / 2)`, the interval excludes zero if and
only if `min(#{T* <= 0}, #{T* >= 0}) <= k - 1`, which is precisely
`p <= alpha`. Under a linearly interpolated empirical quantile instead
the two rules part company at `min = k`, so the convention has to be
named: at `n_boot = 299` and `alpha = 0.05`, `k = 7` and the disagreement
is at `min = 7`. `percentile_interval` returns the interval this duality
refers to.

THIS IS THE PLUG-IN. It holds the fitted parameters still, so its
interval is too narrow by whatever the fit uncertainty is worth. Use
`refit_bootstrap` where more than about two parameters per candidate are
fitted, and measure with `width_ratio` if unsure.
"""
function bootstrap_calibrate(D0::AbstractMatrix, P0::AbstractMatrix, P1::AbstractMatrix;
                             n_boot::Int=299, ipm::Symbol=:sw,
                             transform::Symbol=:none, seed::Int=42,
                             sw_seed::Int=42, m::Union{Nothing,Int}=nothing,
                             n_projections::Int=SW_NPROJ)
    D0t = _apply_transform(D0, transform)
    P0t = _apply_transform(P0, transform)
    P1t = _apply_transform(P1, transform)
    # The bandwidth is a nuisance parameter of the statistic, so it is fixed
    # at its observed-sample value and NOT recomputed per resample.
    bw = common_bandwidth(ipm, D0t, P0t, P1t)
    # The resampled D0* is shared by the two terms of a replicate, which
    # is what preserves their correlation; preparing it once per replicate
    # rather than once per term is the same values, computed half as often.
    sc = ipm === :sw ? sw_scratch(size(D0t, 1), sw_seed; n_projections=n_projections) : nothing
    T(D, A, B) = (rho = rho_against_fixed(ipm, D; bandwidth=bw, sw_seed=sw_seed,
                                          scratch=sc, n_projections=n_projections);
                  rho(A) - rho(B))
    T_obs = T(D0t, P0t, P1t)
    n_d = size(D0t, 2); n_0 = size(P0t, 2); n_1 = size(P1t, 2)
    m_d, m_0, m_1 = m === nothing ? (n_d, n_0, n_1) : (m, m, m)
    m === nothing || m >= 2 ||
        error("bootstrap_calibrate: m = $m is too small; the unbiased MMD needs m >= 2")
    rng = MersenneTwister(seed)
    T_boot = zeros(n_boot)
    for b in 1:n_boot
        Db = D0t[:, rand(rng, 1:n_d, m_d)]
        Pb0 = P0t[:, rand(rng, 1:n_0, m_0)]
        Pb1 = P1t[:, rand(rng, 1:n_1, m_1)]
        T_boot[b] = T(Db, Pb0, Pb1)
    end
    # m == n is the n-out-of-n bootstrap and must return its arithmetic
    # unaltered, not multiplied through by a floating-point 1.0.
    if !(m === nothing || m == n_d)
        s = sqrt(m / n_d)
        @inbounds for b in 1:n_boot
            T_boot[b] = T_obs + s * (T_boot[b] - T_obs)
        end
    end
    n_le = count(<=(0.0), T_boot); n_ge = count(>=(0.0), T_boot)
    p = min(1.0, 2 * (1 + min(n_le, n_ge)) / (1 + n_boot))
    return (T_obs=T_obs, p=p, T_boot=T_boot)
end

"""
    percentile_interval(T_boot; alpha=0.05) -> (lo, hi)

The equal-tailed percentile interval whose exclusion of zero is exactly
equivalent to `p <= alpha` from `bootstrap_calibrate`: the `k`-th and
`(B + 1 - k)`-th order statistics with `k = floor((B + 1) alpha / 2)`.
"""
function percentile_interval(T_boot::AbstractVector; alpha::Float64=0.05)
    B = length(T_boot)
    k = floor(Int, (B + 1) * alpha / 2)
    k < 1 && return (-Inf, Inf)          # too few replicates to exclude anything
    s = sort(T_boot)
    return (s[k], s[B + 1 - k])
end

# ── Calibration 2b: the refit bootstrap ──────────────────────────

#= THE RULE THE PAPER RECOMMENDS WHERE THE FIT IS EXPENSIVE IN
   PARAMETERS, and the one piece of the method that existed in the source
   repository only as five hand-copied inline blocks. Promoting it is the
   single most useful thing this package does.

   THE DEFECT IT REPAIRS. `bootstrap_calibrate` resamples the columns of
   P0 and P1, which treats the two simulated samples as fixed
   populations. The only candidate-side variability it sees is the Monte
   Carlo error of drawing n_sim units from a KNOWN parameter. The
   parameter is not known: each P_j is simulated from theta_hat_j, fitted
   on D1, and a column resample cannot see that error.

   THE PROVENANCE OF THE PLUG-IN, so nobody re-derives it as an
   oversight. Park, Balakrishnan and Wasserman take the pilot to be the
   MLE and state that "in the exact setting, the choice of the pilot
   estimate does not affect the validity of the resulting set and only
   affects its size". That freedom is real for THEM because their
   validity rests on a universal-inference bound holding conditional on
   whatever the pilot is. It does not survive the move to a BOOTSTRAP
   calibration, which approximates the sampling distribution of T and so
   must know the pilot was estimated. The inherited justification stops
   applying exactly where the calibration changed.

   THREE THINGS THAT ARE EASY TO GET WRONG HERE, ALL OF THEM MEASURED.

   (1) BOTH CANDIDATES MUST BE REFITTED ON THE SAME RESAMPLED D1. They
   are fitted on one D1, so their errors are correlated, and T is their
   DIFFERENCE with variance Var0 + Var1 - 2Cov. A separate resample per
   candidate zeroes that covariance and inflates the interval by however
   much the two fits co-move. On a setting whose candidates have fixed
   locations and one free scale each this was harmless. On a setting
   where both fit a location and a scale to the same data the width ratio
   went to 0.49, an interval twice too WIDE, and the power at the far end
   of the sweep fell from 0.54 to 0.06. `test/runtests.jl` carries this
   as a control arm, because it is a one-character mistake for a future
   contributor and it is invisible on the setting that motivated the
   repair.

   (2) DO NOT DRAW THETA* FROM THE ABC POSTERIOR. theta_hat is a
   posterior MEAN, whose sampling variability is not the posterior's
   spread, and an ABC posterior carries the acceptance tolerance on top
   of that. Measured, one particle per replicate reports a spread of
   3.087 against a true 0.851, too wide by 3.6 times, and the rule then
   commits in none of forty replications. There is no constant that
   reconciles them: the shrinkage which would is 0.30 at one end of a
   sweep and 0.18 at the other. A particle PER UNIT is worse still, since
   a posterior predictive is a MIXTURE rather than any one member of the
   family, so it moves the estimand instead of widening the interval, and
   moves it asymmetrically towards whichever candidate is easier to
   identify.

   (3) THE sqrt(S/(S-1)) CORRECTION IS NEEDED. Deviations taken about
   their own mean carry variance (S-1)/S times what they estimate.
   Uncorrected at S = 5 the width ratio reads 1.23; corrected it reads
   1.10 to 1.14. It is the standard finite-sample correction and has no
   freedom to fit the target.

   MEASURED AT S = 5, n_refit = 800 on a five-parameter comparison: the
   width ratio moves 1.46 -> 1.26 (sliced Wasserstein) and 1.41 -> 1.15
   (maximum mean discrepancy), and the rejection rate at the tie moves
   0.115 -> 0.050 and 0.095 -> 0.020 against a nominal 0.05. Power is not
   what pays: at the far end of the sweep commitment falls only from 0.90
   to 0.84.

   CAVEATS TO CARRY. The width ratio measures sd agreement, NOT tail
   agreement, and the decision depends on the 2.5% quantiles of T*. And
   S = 5 is a DEFAULT, not a derived choice: the S ladder was started at
   5, 10, 20 and only S = 5 completed.

   The seed offsets 7919, 104_729, 60_000 and 80_000 are load-bearing.
   They are what makes a run reproduce the published numbers, and 7919
   and 104_729 are prime and larger than any per-replication seed step a
   driver is likely to use, so replicate streams cannot collide across
   replications. =#

"""
    refit_deviations(model0, model1, D1, seed; S=5, n_refit=800, kwargs...)
        -> (dev0, dev1)

Estimate the sampling distribution of the two fitted parameter vectors by
resampling the FIT half. Draw `S` bootstrap resamples of the columns of
`D1`, refit BOTH candidates by ABC on each one, centre each candidate's
`S` parameter vectors on their own mean and scale by `sqrt(S/(S-1))`.

Returns two vectors of `S` deviation vectors, aligned by index: `dev0[s]`
and `dev1[s]` come from the SAME resampled `D1`, which is what preserves
the correlation between the two fit errors. Drawing them from separate
resamples zeroes that correlation and inflates the interval.

`kwargs` are passed to `abc_fit`, e.g. `sampler`, `paccmin`, `max_sims`.
"""
function refit_deviations(model0::RelFitModel, model1::RelFitModel,
                          D1::AbstractMatrix, seed::Int;
                          S::Int=5, n_refit::Int=800, kwargs...)
    S >= 2 || error("refit_deviations: S must be at least 2, got $S")
    n1 = size(D1, 2)
    t0 = Vector{Vector{Float64}}(undef, S)
    t1 = Vector{Vector{Float64}}(undef, S)
    for s in 1:S
        base = seed + 7919 * s
        rng = MersenneTwister(base)
        # ONE resample, BOTH candidates. See note (1) above.
        D1b = D1[:, rand(rng, 1:n1, n1)]
        t0[s] = abc_fit(model0, D1b; N=n_refit, seed=base, kwargs...)
        t1[s] = abc_fit(model1, D1b; N=n_refit, seed=base + 104_729, kwargs...)
    end
    sc = sqrt(S / (S - 1))
    mu0 = mean(t0); mu1 = mean(t1)
    return [(t .- mu0) .* sc for t in t0], [(t .- mu1) .* sc for t in t1]
end

"""
    refit_bootstrap(D0, P0, P1, model0, model1, theta0, theta1, dev0, dev1;
                    n_boot=299, ipm=:sw, transform=:none, seed=42,
                    n_projections=SW_NPROJ)
        -> (T_obs, p, T_boot)

The bootstrap of `bootstrap_calibrate` with the fit uncertainty
propagated. `T_obs` is the same plug-in statistic and is unchanged, so
the SIGN, and hence the direction of any commitment, is what the plug-in
would have reported. Only the null widens.

Each replicate resamples the columns of `D0`, picks one index `s` into
the refit deviations, and simulates the two candidate sets at
`theta_j + dev_j[s]`. Drawing both displacements at a COMMON index is
what keeps them from the same alternative history.

Get `dev0` and `dev1` from `refit_deviations`.
"""
function refit_bootstrap(D0::AbstractMatrix, P0::AbstractMatrix, P1::AbstractMatrix,
                         model0::RelFitModel, model1::RelFitModel,
                         theta0::AbstractVector, theta1::AbstractVector,
                         dev0::AbstractVector, dev1::AbstractVector;
                         n_boot::Int=299, ipm::Symbol=:sw,
                         transform::Symbol=:none, seed::Int=42,
                         n_projections::Int=SW_NPROJ)
    length(dev0) == length(dev1) ||
        error("refit_bootstrap: the two deviation sets must be aligned, got " *
              "$(length(dev0)) and $(length(dev1))")
    S = length(dev0)
    #= These two offsets are part of the published arithmetic. The
       projection seed is derived from the replication seed so that
       distinct replications do not share a set of directions, and the
       resampling stream is separated from it by a large offset so the two
       cannot collide. =#
    sw_seed = 60_000 + seed
    D0t = _apply_transform(D0, transform)
    P0t = _apply_transform(P0, transform)
    P1t = _apply_transform(P1, transform)
    bw = common_bandwidth(ipm, D0t, P0t, P1t)
    sc = ipm === :sw ? sw_scratch(size(D0t, 1), sw_seed; n_projections=n_projections) : nothing
    T(D, A, B) = (rho = rho_against_fixed(ipm, D; bandwidth=bw, sw_seed=sw_seed,
                                          scratch=sc, n_projections=n_projections);
                  rho(A) - rho(B))
    T_obs = T(D0t, P0t, P1t)
    n_d = size(D0t, 2); n_0 = size(P0t, 2); n_1 = size(P1t, 2)
    rng = MersenneTwister(80_000 + seed)
    T_boot = zeros(n_boot)
    for b in 1:n_boot
        Db = D0t[:, rand(rng, 1:n_d, n_d)]
        s = rand(rng, 1:S)                       # ONE index for both, see note (1)
        Pb0 = _apply_transform(
            model0.simulator(theta0 .+ dev0[s], n_0; seed=rand(rng, 1:typemax(Int32))),
            transform)
        Pb1 = _apply_transform(
            model1.simulator(theta1 .+ dev1[s], n_1; seed=rand(rng, 1:typemax(Int32))),
            transform)
        T_boot[b] = T(Db, Pb0, Pb1)
    end
    n_le = count(<=(0.0), T_boot); n_ge = count(>=(0.0), T_boot)
    p = min(1.0, 2 * (1 + min(n_le, n_ge)) / (1 + n_boot))
    return (T_obs=T_obs, p=p, T_boot=T_boot)
end

"""
    width_ratio(T_across, T_boot) -> Float64

The instrument that says which bootstrap a problem needs, and it takes no
theory. At a FIXED configuration the population gap is fixed, so the
spread of `T_obs` ACROSS independent replications is exactly the sampling
variability the bootstrap is trying to reproduce. `T_across` is that
vector of observed statistics and `T_boot` the mean bootstrap standard
deviation over the same replications.

One is a correctly scaled interval, above one is too narrow and below one
is too wide. Measured: 0.97 to 1.06 with one fitted parameter per
candidate, 0.83 to 1.15 with two, and 1.16 to 1.46 with five, which is
where `refit_bootstrap` becomes necessary.

It needs replications, so it is a design-time instrument and not
something an analyst holding one data set can run. What that analyst can
run is `refit_bootstrap` itself.
"""
width_ratio(T_across::AbstractVector, T_boot_sd::Real) = std(T_across) / T_boot_sd

# ── Calibration 3: the concentration threshold (Park et al. sec 5.4) ──

#= The permutation calibrates the EXCHANGEABILITY null. Park,
   Balakrishnan and Wasserman instead calibrate their IPM relative-fit
   statistic by concentration: the statistic is a bounded empirical mean,
   so a Hoeffding threshold B sqrt(log(1/alpha) / (2 n0)) gives a
   finite-sample test of the nu-approximate null. The MMD is the natural
   likelihood-free instantiation because its witness function is
   available in closed form from samples,
   f*(x) = (mu_P0(x) - mu_P1(x)) / ||mu_P0 - mu_P1||_H, which lies in the
   unit RKHS ball, so |f*| <= 1 for a kernel bounded by 1 and the
   statistic's terms are bounded by B = 2.

   THAT REPLACEMENT IS NOT FREE and the finite-sample guarantee does NOT
   survive it intact. The Hoeffding bound controls the deviation of
   mean_{D0} f* about its expectation, holding f* and the two simulated
   averages fixed as POPULATION quantities. Here all three are themselves
   plug-ins from finitely many simulated columns: f* is estimated from P0
   and P1, and the term (mean_{P0} f* + mean_{P1} f*)/2 is an average over
   the same finite draws that defined f*. The neglected error is of the
   same order, O(n0^{-1/2}), as the error the threshold controls. What
   remains is a concentration-motivated threshold with the right rate and
   not a finite-sample valid test, and the measured level bears that out:
   it is safe at every tie whose two predictives differ and commits in
   0.676 of replications at the exact-exchangeable null.

   IT DOES NOT GENERALISE BEYOND TWO CANDIDATES, and that is a fact about
   the statistic rather than a gap in the code. The empirical MMD witness
   is a single direction in the RKHS unit ball, and K embeddings do not
   determine one. =#

"""
    hoeffding_mmd_test(D0, P0, P1; alpha=0.05) -> (T, threshold, decision)

Park-style nu-approximate relative-fit test for the MMD, calibrated by a
Hoeffding threshold rather than by resampling. Computes the empirical MMD
witness `f*` between `P0` and `P1`, the split statistic
`T = (mean_{P0} f* + mean_{P1} f*)/2 - mean_{D0} f*`, and the threshold
`t_alpha = 2 sqrt(log(1/alpha) / (2 n0))`.

Decision: `:M1` if `T > t_alpha`, `:M0` if `-T > t_alpha`, else
`:abstain`. Two candidates only.
"""
function hoeffding_mmd_test(D0::AbstractMatrix, P0::AbstractMatrix, P1::AbstractMatrix;
                            alpha::Float64=0.05)
    σ = median_bandwidth(P0, P1)
    γ = 1.0 / (2 * σ^2)
    # biased (V-statistic) MMD norm ||mu_P0 - mu_P1||_H, so it is >= 0
    m = size(P0, 2); n = size(P1, 2)
    K00 = exp.(-γ .* _pairwise_sqdists(P0, P0))
    K11 = exp.(-γ .* _pairwise_sqdists(P1, P1))
    K01 = exp.(-γ .* _pairwise_sqdists(P0, P1))
    norm2 = sum(K00) / m^2 + sum(K11) / n^2 - 2 * sum(K01) / (m * n)
    nrm = sqrt(max(norm2, 0.0))
    n0 = size(D0, 2)
    threshold = 2 * sqrt(log(1 / alpha) / (2 * n0))
    if nrm < 1e-12
        return (T=0.0, threshold=threshold, decision=:abstain)
    end
    # witness evaluated at the columns of A
    witness(A) = vec(mean(exp.(-γ .* _pairwise_sqdists(P0, A)); dims=1) .-
                     mean(exp.(-γ .* _pairwise_sqdists(P1, A)); dims=1)) ./ nrm
    T = (mean(witness(P0)) + mean(witness(P1))) / 2 - mean(witness(D0))
    decision = T > threshold ? :M1 : (-T > threshold ? :M0 : :abstain)
    return (T=T, threshold=threshold, decision=decision)
end

# ── The degeneracy diagnostic ────────────────────────────────────

#= Proposition 2 of the paper assumes sqrt(n0) (T - Delta) has a
   non-degenerate normal limit. The geometry that breaks it is two
   candidates matched on the low-order moments which fit them, so their
   simulated sets sit close together, the two distances react to a small
   change in the data in almost the same way, and their difference loses
   the leading term. Named that way it is untestable by an analyst, who
   does not know the truth. This makes it testable.

   The signature of degeneracy is not the SIZE of the bootstrap spread,
   which carries the units of the metric and so says nothing on its own,
   but the RATE at which that spread shrinks with the resample size. For
   a non-degenerate two-sample U-statistic limit sd(T*) falls like
   m^{-1/2}; under first-order degeneracy the linear terms cancel, the
   leading behaviour is quadratic, and it falls like m^{-1}.

   READ IT AS A WARNING, NOT AS A PREDICTION, and refuse the "if beta >
   0.6 then ..." threshold rule that referees ask for. Measured, beta
   does NOT order the level failures: the deepest departure found, 0.72,
   was at a geometry where the bootstrap UNDER-rejects at 0.035, and a
   shallower 0.66 was at one where it over-rejects at 0.095. It says the
   guarantee has lapsed and the width should be measured. It does not say
   the level fails, in which direction, or by how much.

   ALWAYS RUN THE SEPARATED CONTROL. beta measures the CANDIDATE PAIR and
   not the position of the truth: on every setting tried, a configuration
   far from the tie returned beta, term correlation and omega agreeing
   with the tie to within a few hundredths. That is what it must do, since
   a separation knob moves the truth and leaves both candidates alone, but
   it means a departure at a tie is not evidence about that tie.

   `cor_terms` is the literal form of the diagnostic a referee suggested,
   the bootstrap correlation of the two distance terms, with
   kappa = sd(r0* - r1*) / sqrt(sd(r0*)^2 + sd(r1*)^2), which is 1 when
   the two terms move independently and falls to 0 as they co-move
   exactly. They are reported because they are cheap, not because they
   separate the regimes: the shared resampled D0* correlates the two
   terms by construction in EVERY problem, which is the whole reason the
   statistic is less variable than either distance, so a high correlation
   is the design working and not a warning. Note also that terms can be
   NEGATIVELY correlated, at -0.51 on a geometry whose two predictives
   straddle the data, so strong positive co-movement is not a defect in
   itself. =#

"""
    degeneracy_scaling(D0, P0, P1; ms=nothing, n_boot=299, ipm=:sw,
                       transform=:none, seed=42, sw_seed=42,
                       n_projections=SW_NPROJ)
        -> (ms, sds, beta, r2, cor_terms, kappa)

Estimate how fast the bootstrap spread of the relative-fit statistic
shrinks with the resample size, as a diagnostic for the partially
degenerate regime in which the bootstrap's asymptotic guarantee lapses.

For each `m` in `ms` the three samples are resampled with replacement at
size `m`, WITHOUT the `sqrt(m/n0)` rescaling of `bootstrap_calibrate`,
and `sds[k]` is the standard deviation of the resulting `T*`. `beta` is
minus the least-squares slope of `log sds` on `log ms`:

  * `beta` near `1/2` is the non-degenerate limit the guarantee assumes,
  * `beta` near `1` is first-order degeneracy, where the `n`-out-of-`n`
    bootstrap is not consistent.

`r2` guards against reading a slope off a ladder that is not straight.
`ms` defaults to `n0` divided by 8, 4, 2 and 1.

It draws its own resampling stream, so it estimates the same bootstrap
distribution independently and cannot perturb a reported p-value.
"""
function degeneracy_scaling(D0::AbstractMatrix, P0::AbstractMatrix, P1::AbstractMatrix;
                            ms::Union{Nothing,AbstractVector{Int}}=nothing,
                            n_boot::Int=299, ipm::Symbol=:sw,
                            transform::Symbol=:none, seed::Int=42, sw_seed::Int=42,
                            n_projections::Int=SW_NPROJ)
    D0t = _apply_transform(D0, transform)
    P0t = _apply_transform(P0, transform)
    P1t = _apply_transform(P1, transform)
    # The bandwidth is a nuisance parameter of the statistic and is held at
    # its observed-sample value across the whole ladder, exactly as
    # `bootstrap_calibrate` holds it across replicates. Letting it follow m
    # would confound the rate being measured with a moving statistic.
    bw = common_bandwidth(ipm, D0t, P0t, P1t)
    sc = ipm === :sw ? sw_scratch(size(D0t, 1), sw_seed; n_projections=n_projections) : nothing
    n_d = size(D0t, 2); n_0 = size(P0t, 2); n_1 = size(P1t, 2)
    grid = ms === nothing ? [max(4, n_d ÷ 8), max(4, n_d ÷ 4), max(4, n_d ÷ 2), n_d] :
                            collect(ms)
    all(>=(4), grid) || error("degeneracy_scaling: every m must be at least 4")
    sds = zeros(length(grid))
    cor_terms = NaN; kappa = NaN
    for (k, m) in enumerate(grid)
        # A separate stream per rung, so that a rung can be added or dropped
        # without moving the others.
        rng = MersenneTwister(seed + 1000 * k)
        r0 = zeros(n_boot); r1 = zeros(n_boot)
        for b in 1:n_boot
            Db  = D0t[:, rand(rng, 1:n_d, m)]
            Pb0 = P0t[:, rand(rng, 1:n_0, m)]
            Pb1 = P1t[:, rand(rng, 1:n_1, m)]
            rho = rho_against_fixed(ipm, Db; bandwidth=bw, sw_seed=sw_seed,
                                    scratch=sc, n_projections=n_projections)
            r0[b] = rho(Pb0); r1[b] = rho(Pb1)
        end
        sds[k] = std(r0 .- r1)
        if m == n_d
            s0 = std(r0); s1 = std(r1)
            cor_terms = (s0 > 0 && s1 > 0) ? cor(r0, r1) : NaN
            kappa = (s0 > 0 || s1 > 0) ? sds[k] / sqrt(s0^2 + s1^2) : NaN
        end
    end
    x = log.(float.(grid)); y = log.(sds)
    xb = mean(x); yb = mean(y)
    sxx = sum((x .- xb) .^ 2)
    slope = sxx > 0 ? sum((x .- xb) .* (y .- yb)) / sxx : NaN
    sst = sum((y .- yb) .^ 2)
    r2 = sst > 0 ? 1 - sum((y .- (yb .+ slope .* (x .- xb))) .^ 2) / sst : NaN
    return (ms=grid, sds=sds, beta=-slope, r2=r2, cor_terms=cor_terms, kappa=kappa)
end
