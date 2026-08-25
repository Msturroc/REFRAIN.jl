#= ================================================================
   The two entry points on a real comparison.

   `refrain` end to end and `decide` on samples a user already has, on
   setting A: two Gaussians with fixed mirror-image locations against a
   truth that is outside both of them at every c, tied by symmetry at
   c = 0 and clearly separated by c = 0.8.

   THE THREE ASSERTIONS THAT MATTER. At the tie the rule must ABSTAIN in
   the great majority of replications, which is the confidence-set duality
   working. Under separation it must COMMIT, and in the right direction.
   And the adequacy screen must fire even where the rule commits happily,
   because the truth is outside both candidates everywhere and a screen
   with power should say so. That last one is the pairing the whole method
   turns on: a commitment is not a certificate of fit.
   ================================================================ =#

using REFRAIN, Test, Random, Statistics, Distributions

include(joinpath(@__DIR__, "..", "examples", "example_models.jl"))

const E2E_KW = (; N=400, paccmin=1e-2, max_sims=25_000, n_boot=299, screen_rep=99)

@testset "refrain abstains at a genuine tie and commits under separation" begin
    m0 = make_gaussian_fixed_loc(-1.0)
    m1 = make_gaussian_fixed_loc(+1.0)
    R = 12

    # c = 0 is a genuine tie by symmetry: both candidates are exactly
    # equidistant from the truth, and the correct answer is "cannot tell".
    tie = [refrain(truth_gaussian(0.0, 1.0; n=400, seed=1000 + r), m0, m1;
                   ipm=:sw, calibration=:bootstrap, split_seed=r, fit_seed=r,
                   sim_seed=20_000 + r, cal_seed=30_000 + r, E2E_KW...)
           for r in 1:R]
    @test mean(t.decision === :abstain for t in tie) >= 0.7

    # c = 0.8 puts the truth close to the +1 candidate
    sep = [refrain(truth_gaussian(0.8, 1.0; n=400, seed=2000 + r), m0, m1;
                   ipm=:sw, calibration=:bootstrap, split_seed=r, fit_seed=r,
                   sim_seed=40_000 + r, cal_seed=50_000 + r, E2E_KW...)
           for r in 1:R]
    @test all(s.decision === :M1 for s in sep)

    #= THE PAIRING, and it is the whole reason the screen is reported
       beside the decision rather than folded into it. The truth is
       N(0.8, 1) and the two candidates are fixed at -1 and +1, so the
       rule commits to M1 in every replication above while the screen
       convicts M0 in every one of them and M1 in a good fraction. A
       commitment is not a certificate of fit, and here the indicated
       action is to widen the model rather than to choose within it. =#
    @test all(s.screen in (:both_fail, :M1_fails, :M0_fails) for s in sep)
    @test all(s.screen0.inadequate for s in sep)          # the rejected candidate
    @test mean(s.screen1.inadequate for s in sep) >= 0.25 # the CHOSEN one, often too

    # the struct is coherent with itself
    s = sep[1]
    @test s.decision === decide(s.p, s.T; alpha=s.alpha)
    @test s.interval !== nothing
    @test (s.interval[1] > 0 || s.interval[2] < 0) == (s.p <= s.alpha)
    @test size(s.D1, 2) + size(s.D0, 2) == 400
    @test size(s.P0, 2) == s.n_sim == size(s.D0, 2)
    @test length(s.T_null) == 299
end

@testset "the three calibrations and the three samplers all run" begin
    m0 = make_gaussian_fixed_loc(-1.0)
    m1 = make_gaussian_fixed_loc(+1.0)
    X = truth_gaussian(0.8, 1.0; n=300, seed=77)

    for cal in (:bootstrap, :permutation, :refit)
        r = refrain(X, m0, m1; calibration=cal, N=300, paccmin=1e-2,
                    max_sims=15_000, n_boot=99, screen=false,
                    refit_S=3, refit_N=150)
        @test r.decision in (:M0, :M1, :abstain)
        @test 0.0 < r.p <= 1.0
        @test r.calibration === cal
        # the truth is at +0.8, so nothing here may choose M0
        @test r.decision !== :M0
    end
    for smp in (:rejection, :abc_smc, :apmc)
        r = refrain(X, m0, m1; sampler=smp, N=300, paccmin=1e-2,
                    max_sims=15_000, n_boot=99, screen=false)
        @test r.decision in (:M0, :M1, :abstain)
    end
    # both metrics
    for ipm in (:sw, :mmd)
        r = refrain(X, m0, m1; ipm=ipm, N=300, paccmin=1e-2,
                    max_sims=15_000, n_boot=99, screen=false)
        @test r.ipm === ipm
        @test r.decision !== :M0
    end
end

@testset "decide on samples the user already has" begin
    rng = MersenneTwister(4242)
    n = 300
    D0 = reshape(randn(rng, n), 1, n)
    P0 = reshape(0.9 .+ randn(rng, n), 1, n)     # far from the data
    P1 = reshape(0.05 .+ randn(rng, n), 1, n)    # close to it
    d = decide(D0, P0, P1; ipm=:sw, calibration=:bootstrap, alpha=0.05)
    @test d.decision === :M1                     # T > 0 favours M1
    @test d.T > 0
    @test d.p <= 0.05
    @test length(d.T_null) == 299

    # the one-line rule, which is the whole decision layer
    @test decide(0.20, +1.0) === :abstain
    @test decide(0.01, +1.0) === :M1
    @test decide(0.01, -1.0) === :M0
    @test decide(0.05, -1.0) === :M0             # p <= alpha commits
end

@testset "refrain_full keeps the old shape for a ported driver" begin
    m0 = make_gaussian_fixed_loc(-1.0)
    m1 = make_gaussian_fixed_loc(+1.0)
    X = truth_gaussian(0.8, 1.0; n=200, seed=8)
    r = refrain_full(X, m0, m1; N=300, paccmin=1e-2, max_sims=15_000,
                     n_perm=99, screen=true, screen_rep=49)
    for k in (:T_obs, :p, :T_null, :theta_hat_0, :theta_hat_1, :pts0, :w0,
              :pts1, :w1, :D1, :D0, :P0, :P1, :n_sim, :ipm, :transform,
              :screen0, :screen1, :adequacy)
        @test haskey(r, k)
    end
    @test r.adequacy in (:both_fail, :M0_fails, :M1_fails, :neither_fails)
end

@testset "K candidates end to end" begin
    #= Five fixed-location Gaussians against a truth between two of them.
       The set must contain the closest candidate and must never be
       empty. =#
    lib = RelFitModel[make_gaussian_fixed_loc(mu) for mu in (-1.0, -0.5, 0.5, 1.0, 1.5)]
    X = truth_gaussian(0.55, 1.0; n=400, seed=13)
    r = relfit_compare_K(X, lib; N=300, paccmin=1e-2, max_sims=15_000,
                         n_perm=99, n_boot=99, screen=false)
    @test length(r.boot_set) >= 1
    @test argmin(r.r_obs) in r.boot_set
    @test length(r.perm_set) >= 1
    @test length(r.theta_hat) == 5
end
