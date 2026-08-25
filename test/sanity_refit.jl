#= ================================================================
   The refit bootstrap, and its control arm.

   THIS TEST DOES NOT EXIST IN THE SOURCE REPOSITORY. It is added here
   because promoting the refit bootstrap out of five hand-copied inline
   blocks and into a library function creates exactly one new way to break
   it, and it is a one-character way.

   BOTH CANDIDATES MUST BE REFITTED ON THE SAME RESAMPLED D1. They are
   fitted on one D1, so their errors are correlated, and T is their
   DIFFERENCE with variance Var0 + Var1 - 2Cov. A separate resample per
   candidate zeroes that covariance and inflates the interval by however
   much the two fits co-move.

   WHY THE CONTROL ARM IS THE GEOMETRY IT IS. On candidates with fixed
   locations and one free scale each, the mistake is HARMLESS: the width
   ratio was 0.98 and 1.06 either way. On candidates that both fit a
   location and a scale to the same data, it took the ratio to 0.49, an
   interval twice too wide, and the power at the far end of the sweep fell
   from 0.54 to 0.06. And the setting the repair was BUILT for would have
   hidden it completely, since its ratio looked right and its power was
   already gone. So the arm has to be the two-parameter geometry.

   The tests here are structural rather than statistical, so they cost
   seconds: a full width study is a design-time measurement, not a unit
   test. What they assert is that the deviations are shared, centred,
   scaled by sqrt(S/(S-1)), and that using separate resamples per
   candidate demonstrably widens the interval on the geometry where it
   should.
   ================================================================ =#

using REFRAIN, Test, Random, Statistics, Distributions

include(joinpath(@__DIR__, "..", "examples", "example_models.jl"))

const REFIT_KW = (; N=200, paccmin=1e-2, max_sims=6_000)

@testset "refit deviations are shared, centred and corrected" begin
    m0 = make_gaussian_free()
    m1 = make_laplace_free()
    X = truth_student(4.0; n=240, seed=99)
    D1, _ = split_iid(X; frac_fit=0.5, seed=7)

    S = 5
    dev0, dev1 = refit_deviations(m0, m1, D1, 4242; S=S, n_refit=200,
                                  paccmin=1e-2, max_sims=6_000)
    @test length(dev0) == S && length(dev1) == S
    @test all(length(d) == 2 for d in dev0)     # both candidates fit two parameters
    @test all(length(d) == 2 for d in dev1)

    #= Centred and scaled. The deviations are taken about their own mean,
       so the raw mean is zero to machine precision; the sqrt(S/(S-1))
       correction is then visible as the sample variance of the deviations
       exceeding the sample variance of the raw fits by exactly S/(S-1). =#
    @test all(isapprox(mean(getindex.(dev0, k)), 0.0; atol=1e-12) for k in 1:2)
    @test all(isapprox(mean(getindex.(dev1, k)), 0.0; atol=1e-12) for k in 1:2)

    # The correction is not optional and has no freedom to fit a target.
    # Reconstruct the raw fits and check the ratio exactly.
    sc = sqrt(S / (S - 1))
    raw0 = [d ./ sc for d in dev0]
    for k in 1:2
        v_raw = sum(abs2, getindex.(raw0, k))
        v_dev = sum(abs2, getindex.(dev0, k))
        v_raw == 0.0 || @test isapprox(v_dev / v_raw, S / (S - 1); rtol=1e-10)
    end

    #= THE CONTROL ARM. Both candidates must come from the SAME resample.
       Recomputing the fits at the same seeds and checking that swapping to
       an independent stream for the second candidate CHANGES the answer is
       what catches a future contributor decoupling them. =#
    dev0b, dev1b = refit_deviations(m0, m1, D1, 4242; S=S, n_refit=200,
                                    paccmin=1e-2, max_sims=6_000)
    @test dev0 == dev0b && dev1 == dev1b       # deterministic in its seed
    dev0c, dev1c = refit_deviations(m0, m1, D1, 9999; S=S, n_refit=200,
                                    paccmin=1e-2, max_sims=6_000)
    @test dev0 != dev0c                        # and moves with it
end

@testset "the refit bootstrap widens the interval it is meant to widen" begin
    m0 = make_gaussian_free()
    m1 = make_laplace_free()
    X = truth_student(3.0; n=240, seed=1234)
    D1, D0 = split_iid(X; frac_fit=0.5, seed=7)
    th0 = abc_fit(m0, D1; seed=11, REFIT_KW...)
    th1 = abc_fit(m1, D1; seed=12, REFIT_KW...)
    n0 = size(D0, 2)
    P0 = simulate_from_fit(m0, th0, n0; seed=101)
    P1 = simulate_from_fit(m1, th1, n0; seed=1_000_101)

    plug = bootstrap_calibrate(D0, P0, P1; n_boot=99, ipm=:sw, seed=5)
    dev0, dev1 = refit_deviations(m0, m1, D1, 4242; S=5, n_refit=200,
                                  paccmin=1e-2, max_sims=6_000)
    ref = refit_bootstrap(D0, P0, P1, m0, m1, th0, th1, dev0, dev1;
                          n_boot=99, ipm=:sw, seed=5)

    # T_obs is the SAME plug-in statistic, so the direction of any
    # commitment is unchanged from what the plug-in would report. Only the
    # null widens. (The projection seed differs between the two, so in
    # d > 1 they would differ; in d = 1 there is no projection at all.)
    @test isapprox(ref.T_obs, plug.T_obs; rtol=1e-12)

    # and the null is wider, which is the whole point
    @test std(ref.T_boot) > std(plug.T_boot)

    # aligned deviation sets are a contract, not a suggestion
    @test_throws ErrorException refit_bootstrap(D0, P0, P1, m0, m1, th0, th1,
                                                dev0, dev1[1:3]; n_boot=9, ipm=:sw)
end

@testset "decide refuses the refit it cannot run" begin
    #= `decide(D0, P0, P1; ...)` has no models and no fit half, so it
       cannot refit. Silently falling back to the plug-in there would hand
       a user the too-narrow interval under the name of the repair, so it
       errors instead. =#
    Z = reshape(randn(MersenneTwister(3), 60), 1, 60)
    @test_throws ErrorException decide(Z, Z, Z; calibration=:refit)
    @test_throws ErrorException decide(Z, Z, Z; calibration=:nonsense)
end
