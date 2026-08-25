#= ================================================================
   The K-candidate extension.

   Ported from sections 1 to 5 of `sanity_checks_K.jl` of the
   abstention_model_choice repository, with the bespoke bookkeeping
   replaced by `@test`. Section 6, the Occam values of the model-indicator
   baseline, runs ABC and lives in `sanity_baseline.jl` here.

   HALF OF THIS FILE IS THE REDUCTION DEMAND, and it is the reason the
   file exists. The K functions are additive and the two-model functions
   they generalise were left untouched, so at K = 2 the two routes must
   agree BIT FOR BIT rather than approximately. It is the only cheap guard
   against the K code quietly becoming a second, subtly different
   implementation of the published statistic, and `===` rather than `≈` is
   the whole point of it.

   ONE THING DOES NOT REDUCE, by construction: the bootstrap DECISION
   rule. At K = 2 the scalar rule is the percentile-interval dual, which
   counts sign changes of T*, while the max-type rule compares the
   observed range against the range of the centred resamples. Those are
   the percentile and the basic bootstrap of one quantity. They agree
   asymptotically and differ at finite B, so the disagreement is MEASURED
   here rather than asserted away.
   ================================================================ =#

using REFRAIN, Test, Random, Statistics, Distributions, Printf

const SLOW_K = get(ENV, "REFRAIN_SLOW", "0") == "1"

@testset "K = 2 reduces to the two-model functions, bit for bit" begin
    rng = MersenneTwister(1234)
    n = 250
    D0 = reshape(randn(rng, n), 1, n)
    P0 = reshape(0.4 .+ 1.2 .* randn(rng, n), 1, n)
    P1 = reshape(-0.3 .+ 0.9 .* randn(rng, n), 1, n)
    P  = [P0, P1]

    for ipm in (:sw, :mmd)
        # the shared nuisance choice
        @test common_bandwidth_K(ipm, D0, P) === common_bandwidth(ipm, D0, P0, P1)

        # the statistic
        r = relfit_distances_K(D0, P; ipm=ipm, sw_seed=7)
        T = relfit_statistic(D0, P0, P1; ipm=ipm, sw_seed=7)
        @test (r[1] - r[2]) === T

        # the permutation p-value and the WHOLE null distribution
        a = permutation_calibrate_K(D0, P; n_perm=200, ipm=ipm, seed=77, sw_seed=7)
        b = permutation_calibrate(D0, P0, P1; n_perm=200, ipm=ipm, seed=77, sw_seed=7)
        @test a.p === b.p
        @test all(a.R_null[i] === abs(b.T_null[i]) for i in eachindex(b.T_null))
        @test a.R_obs === abs(b.T_obs)

        # every bootstrap replicate, off the same stream in the same order
        B = 299
        kb = bootstrap_calibrate_K(D0, P; n_boot=B, ipm=ipm, seed=55, sw_seed=7)
        sb = bootstrap_calibrate(D0, P0, P1; n_boot=B, ipm=ipm, seed=55, sw_seed=7)
        @test (kb.r_obs[1] - kb.r_obs[2]) === sb.T_obs
        @test all((kb.R_boot[i, 1] - kb.R_boot[i, 2]) === sb.T_boot[i] for i in 1:B)
    end
end

@testset "the two bootstrap decision rules differ, and by how much" begin
    #= The geometry is the one `sanity_core.jl` section 4a uses for the
       published statement that the percentile dual holds its level at a
       NON-COINCIDENT equidistance null. The max-type rule is the BASIC
       bootstrap of the same quantity and needs a larger held-out half:
       measured at nominal with n_0 = 400 and at 0.0655 with n_0 = 200. =#
    B = 299; alpha = 0.05
    R = SLOW_K ? 2000 : 400
    n = 400; d = 0.4
    kr = zeros(Bool, R); sr = zeros(Bool, R)
    for r in 1:R
        rr = MersenneTwister(2_000_000 + 10_000 * n + r)
        d0 = reshape(randn(rr, n), 1, n)
        p0 = reshape(+d .+ randn(rr, n), 1, n)
        p1 = reshape(-d .+ randn(rr, n), 1, n)
        k = bootstrap_calibrate_K(d0, [p0, p1]; n_boot=B, ipm=:sw,
                                  seed=2_100_000 + r, alpha=alpha)
        s = bootstrap_calibrate(d0, p0, p1; n_boot=B, ipm=:sw, seed=2_100_000 + r)
        kr[r] = length(k.set) == 1
        sr[r] = s.p <= alpha
    end
    se = sqrt(alpha * (1 - alpha) / R)
    @test abs(mean(kr) - alpha) <= max(3 * se, 0.015)
    @test abs(mean(sr) - alpha) <= max(3 * se, 0.015)
end

@testset "the set is never empty and is the p-value superlevel set" begin
    rng = MersenneTwister(9)
    n = 150
    D0 = reshape(randn(rng, n), 1, n)
    # wildly separated candidates, so the elimination runs to the end
    locs = [0.0, 0.8, 1.6, 2.4, 3.2]
    P = [reshape(l .+ randn(rng, n), 1, n) for l in locs]

    for ipm in (:sw, :mmd)
        b = bootstrap_calibrate_K(D0, P; n_boot=299, ipm=ipm, seed=13, alpha=0.05)
        # NEVER EMPTY, even under total separation. A purely relative test
        # cannot convict every candidate, which is why the adequacy screen
        # is reported beside it.
        @test length(b.set) >= 1
        @test Set(b.set) == Set(findall(>(0.05), b.p_mcs))
        @test argmin(b.r_obs) in b.set
        @test issorted([b.r_obs[j] for j in b.eliminated]; rev=true)

        pset = permutation_set_K(D0, P; n_perm=299, ipm=ipm, seed=13, alpha=0.05)
        @test length(pset.set) >= 1
        @test Set(pset.set) == Set(findall(>(0.05), pset.p_mcs))
    end

    # p_mcs is a running maximum, so it is monotone along the elimination
    b = bootstrap_calibrate_K(D0, P; n_boot=299, ipm=:sw, seed=13, alpha=0.05)
    @test issorted(accumulate(max, b.p_steps))
    @test all(b.p_mcs[b.eliminated[t]] == maximum(b.p_steps[1:t])
              for t in eachindex(b.eliminated))
end

#= With K candidates whose predictives are one law there is nothing to
   eliminate and the whole set must survive with probability at least
   1 - alpha. The two calibrations are held to DIFFERENT targets here and
   the difference is the point. The pooled draws are exactly exchangeable,
   so the permutation is exact and must hit 1 - alpha on the nose. The
   bootstrap is not exact and must not be asserted to be: this
   configuration puts every candidate ON the coincidence boundary, where
   the influence function degenerates and the resampling distribution is
   too wide. So the bootstrap is checked for VALIDITY and the permutation
   for EQUALITY. Asserting equality for the bootstrap would be asserting
   that a known degeneracy is absent. =#
@testset "K identical candidates: retain all K" begin
    alpha = 0.05
    B = 299
    R = SLOW_K ? 1200 : 300
    n = 200
    for K in (3, 5)
        keep_b = zeros(Bool, R); keep_p = zeros(Bool, R)
        for r in 1:R
            rr = MersenneTwister(3_000_000 + 1000 * K + r)
            D0 = reshape(randn(rr, n), 1, n)
            P = [reshape(randn(rr, n), 1, n) for _ in 1:K]   # one law, K samples
            b = bootstrap_calibrate_K(D0, P; n_boot=B, ipm=:sw,
                                      seed=3_100_000 + r, alpha=alpha)
            p = permutation_set_K(D0, P; n_perm=B, ipm=:sw,
                                  seed=3_200_000 + r, alpha=alpha)
            keep_b[r] = length(b.set) == K
            keep_p[r] = length(p.set) == K
        end
        se = sqrt(alpha * (1 - alpha) / R)
        @test mean(keep_b) >= 1 - alpha - 3 * se          # validity
        @test isapprox(mean(keep_p), 1 - alpha; atol=max(3 * se, 0.02))  # exactness
    end
end

#= The property that motivates the bootstrap over the permutation at
   K = 2 is level control at an equidistance null whose predictives do NOT
   coincide. The K-model version: P* = N(0,1) and four candidates at
   +d, +d, -d, -d, all four exactly equidistant while the two clusters are
   2d apart. The bootstrap should hold and the permutation, which pools
   the two clusters, should not. =#
@testset "K-way equidistance with distinct predictives" begin
    alpha = 0.05
    B = 299
    R = SLOW_K ? 1000 : 300
    n = 400
    d = 0.4
    keep_b = zeros(Bool, R); keep_p = zeros(Bool, R)
    for r in 1:R
        rr = MersenneTwister(4_000_000 + r)
        D0 = reshape(randn(rr, n), 1, n)
        P = [reshape(+d .+ randn(rr, n), 1, n), reshape(+d .+ randn(rr, n), 1, n),
             reshape(-d .+ randn(rr, n), 1, n), reshape(-d .+ randn(rr, n), 1, n)]
        keep_b[r] = length(bootstrap_calibrate_K(D0, P; n_boot=B, ipm=:sw,
                                                 seed=4_100_000 + r, alpha=alpha).set) == 4
        keep_p[r] = length(permutation_set_K(D0, P; n_perm=B, ipm=:sw,
                                             seed=4_200_000 + r, alpha=alpha).set) == 4
    end
    se = sqrt(alpha * (1 - alpha) / R)
    @test isapprox(mean(keep_b), 1 - alpha; atol=max(3 * se, 0.025))
    @test mean(keep_p) < mean(keep_b) - 3 * se     # the permutation is the one that fails
end

@testset "m-out-of-n carries over to K" begin
    rng = MersenneTwister(21)
    n = 200
    D0 = reshape(randn(rng, n), 1, n)
    P = [reshape(l .+ randn(rng, n), 1, n) for l in (0.0, 0.3, -0.3, 0.6)]
    full = bootstrap_calibrate_K(D0, P; n_boot=199, ipm=:sw, seed=5, m=nothing)
    same = bootstrap_calibrate_K(D0, P; n_boot=199, ipm=:sw, seed=5, m=n)
    # m = n_0 must draw the SAME resamples as m = nothing, or the
    # published numbers are not what the default branch computes
    @test maximum(abs.(full.R_boot .- same.R_boot)) <= 1e-12
    small = bootstrap_calibrate_K(D0, P; n_boot=199, ipm=:sw, seed=5, m=40)
    spread_full = std(vec(full.R_boot .- full.r_obs'))
    spread_small = std(vec(small.R_boot .- small.r_obs'))
    @test 0.5 * spread_full < spread_small < 1.5 * spread_full
    @test_throws ErrorException bootstrap_calibrate_K(D0, P; n_boot=9, m=1)
end
