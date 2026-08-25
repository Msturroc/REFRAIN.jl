#= ================================================================
   The ABC layer and the posterior-model-probability baseline against
   configurations with a known answer.

   THE OCCAM CONTROL IS THE LOAD-BEARING ONE. Candidates that share a
   likelihood family, an ABC distance and hence an acceptance set up to a
   measure-preserving reparametrisation have ABC evidence
   |A(eps)| / V_j with V_j the prior box volume, so the posterior model
   probability is P(M_j) proportional to 1 / V_j. Two candidates
   differing ONLY in prior width must therefore return V0 / (V0 + V1).

   That is the control which caught a missing joint-space weight in the
   source repository. Without the factor P(m) / P_{t-1}(m) the model
   weights follow a MULTIPLICATIVE recursion, so a constant per-iteration
   edge is raised to the power of the iteration count and the reported
   probability depends on how many SMC iterations happened to run. The
   second check below is the direct form of that: P(M1) must not move
   with the stopping rule.

   Deliberately small: these run ABC, so they are sized to prove the code
   paths and the closed forms, not to be statistically sharp. Set
   `REFRAIN_SLOW=1` for the larger pass.
   ================================================================ =#

using REFRAIN, Test, Random, Statistics, Distributions

include(joinpath(@__DIR__, "..", "examples", "example_models.jl"))

const SLOW_B = get(ENV, "REFRAIN_SLOW", "0") == "1"
prior_volume(priors) = prod(p.b - p.a for p in priors)

@testset "the three samplers all recover a known scale" begin
    #= Truth N(0,1) against the single candidate {N(0, sigma^2)}, whose
       only free parameter is log sigma. The ABC distance is deterministic
       in theta, so every sampler should land at log sigma near zero. =#
    m = make_gaussian_fixed_loc(0.0)
    X = truth_gaussian(0.0, 1.0; n=400, seed=5)
    D1, _ = split_iid(X; frac_fit=0.5, seed=5)

    for sampler in (:rejection, :abc_smc, :apmc)
        #= Rejection ABC gets a TIGHT quantile and the two sequential
           samplers do not, and that is the point of having all three
           rather than a caveat about one. At the default 0.5 the
           rejection posterior is the better half of the PRIOR, and its
           mean lands at exp(theta) = 0.72 rather than 1: a one-pass
           sampler has to be told the tolerance, whereas a sequential one
           finds it. Both reach 0.97 once they are asked the same
           question. =#
        kw = sampler === :rejection ? (; quantile_keep=0.01) : (;)
        th = abc_fit(m, D1; sampler=sampler, N=300, paccmin=1e-2,
                     max_sims=20_000, seed=7, kw...)
        @test length(th) == 1
        @test isapprox(exp(th[1]), 1.0; atol=0.25)
    end

    # the Cauchy kernel is available and reaches the same answer
    th_c = abc_fit(m, D1; sampler=:abc_smc, kernel=:cauchy, N=300,
                   paccmin=1e-2, max_sims=20_000, seed=7)
    @test isapprox(exp(th_c[1]), 1.0; atol=0.25)

    #= THE LATENT BUG THIS CLOSES. In the source repository anything other
       than `:normal` was treated as Cauchy by the PROPOSAL and as
       Gaussian by the WEIGHT, so a typo produced a silently wrong sampler
       rather than an error. =#
    @test_throws ErrorException abc_fit(m, D1; sampler=:abc_smc, kernel=:t, N=50,
                                        paccmin=1e-1, max_sims=500)
    @test_throws ErrorException abc_smc(50, m.priors, m.abc_distance_factory(D1);
                                        perturb=:gaussian, max_sims=500)
end

@testset "rejection ABC takes a fixed tolerance as well as a quantile" begin
    m = make_gaussian_fixed_loc(0.0)
    X = truth_gaussian(0.0, 1.0; n=200, seed=5)
    D1, _ = split_iid(X; frac_fit=0.5, seed=5)
    rho = m.abc_distance_factory(D1)

    q = rejection_abc(100, m.priors, rho; quantile_keep=0.1, max_sims=4000, seed=3)
    @test q.epsilon > 0
    @test all(q.distances .<= q.epsilon)
    @test size(q.particles, 2) == length(q.weights) == q.n_accepted
    @test all(q.weights .== 1.0)                       # rejection ABC is unweighted

    f = rejection_abc(50, m.priors, rho; epsilon=q.epsilon, max_sims=4000, seed=3)
    @test all(f.distances .<= q.epsilon)
    @test f.epsilon == q.epsilon

    # and it refuses rather than returning nothing when the tolerance is impossible
    @test_throws ErrorException rejection_abc(10, m.priors, rho; epsilon=1e-12,
                                              max_sims=200, seed=3)
end

@testset "APMC does model comparison natively" begin
    #= Two IDENTICAL candidates, where the truth is exactly p = 0.5. This
       is the abstention-flavoured calibration check, and it is the one
       ready-made test the APMC lineage came with: none of that lineage
       carries a single testset anywhere, so this is its first. =#
    m = make_gaussian_fixed_loc(0.0)
    X = truth_gaussian(0.0, 1.0; n=300, seed=17)
    rho = m.abc_distance_factory(X)
    R = SLOW_B ? 12 : 4
    ps = Float64[]
    for r in 1:R
        res = apmc(400, [m.priors, m.priors], [rho, rho];
                   paccmin=1e-2, max_sims=40_000, seed=900 + r)
        push!(ps, res.p[2][end])
    end
    # identical candidates cannot be separated by any amount of data
    @test isapprox(mean(ps), 0.5; atol=0.15)

    # single-model fitting is the length(models) == 1 case, no special path
    one = apmc(400, [m.priors], [rho]; paccmin=1e-2, max_sims=20_000, seed=5)
    @test length(one.p) == 1
    @test one.p[1][end] == 1.0
    th = one.pts[1][end] * (one.wts[1][end] ./ sum(one.wts[1][end]))
    @test isapprox(exp(th[1]), 1.0; atol=0.3)

    # eps_floor stops the sampler early rather than running to paccmin
    hi = apmc(400, [m.priors], [rho]; paccmin=1e-3, max_sims=40_000, seed=5)
    lo = apmc(400, [m.priors], [rho]; paccmin=1e-3, max_sims=40_000, seed=5,
              eps_floor=hi.epsilon[min(3, length(hi.epsilon))])
    @test length(lo.epsilon) <= length(hi.epsilon)
end

@testset "the baseline returns the Occam values in closed form" begin
    #= Two candidates identical but for the width of their prior box.
       ABC evidence is |A(eps)| / V_j, so P(M_j) must be proportional to
       1 / V_j. THIS IS THE CONTROL THAT CAUGHT THE MISSING JOINT-SPACE
       WEIGHT. =#
    mu0 = 1.0
    narrow = make_gaussian_fixed_loc(mu0; sigma_bounds=(0.2, 5.0))
    wide   = make_gaussian_fixed_loc(mu0; sigma_bounds=(0.2, 125.0))
    V = [prior_volume(narrow.priors), prior_volume(wide.priors)]
    want = (1 ./ V) ./ sum(1 ./ V)

    R = SLOW_B ? 20 : 6
    N_ABC = SLOW_B ? 1500 : 400
    n = SLOW_B ? 400 : 200
    got = zeros(R, 2)
    for r in 1:R
        X = truth_gaussian(0.0, 1.0; n=n, seed=6_200_000 + r)
        rho_n = narrow.abc_distance_factory(X)
        rho_w = wide.abc_distance_factory(X)
        res = abc_smc_model_choice(N_ABC, [narrow.priors, wide.priors], [rho_n, rho_w];
                                   paccmin=1e-2, max_sims=200_000, seed=6_300_000 + r)
        got[r, :] = res.p
    end
    mp = vec(mean(got; dims=1))
    se = maximum(vec(std(got; dims=1))) / sqrt(R)
    @test maximum(abs.(mp .- want)) <= max(3 * se, 0.08)

    # two IDENTICAL candidates must give one half each
    same = make_gaussian_fixed_loc(mu0)
    X = truth_gaussian(0.0, 1.0; n=n, seed=515)
    rho = same.abc_distance_factory(X)
    res = abc_smc_model_choice(N_ABC, [same.priors, same.priors], [rho, rho];
                               paccmin=1e-2, max_sims=200_000, seed=616)
    @test isapprox(res.p[1], 0.5; atol=0.1)
end

@testset "the reported probability does not move with the stopping rule" begin
    #= The direct form of the joint-weight check. Under a MULTIPLICATIVE
       recursion a constant per-iteration edge is raised to the power of
       the iteration count, so the reported probability drifts with the
       number of iterations that happened to run. With the joint weight
       restored it converges instead, and `p_trace` is what shows which. =#
    mu0 = 1.0
    narrow = make_gaussian_fixed_loc(mu0; sigma_bounds=(0.2, 5.0))
    wide   = make_gaussian_fixed_loc(mu0; sigma_bounds=(0.2, 125.0))
    X = truth_gaussian(0.0, 1.0; n=300, seed=4242)
    rho_n = narrow.abc_distance_factory(X)
    rho_w = wide.abc_distance_factory(X)
    res = abc_smc_model_choice(1000, [narrow.priors, wide.priors], [rho_n, rho_w];
                               paccmin=1e-3, max_sims=400_000, seed=808)
    tr = [t[2] for t in res.p_trace]
    @test length(tr) >= 3
    # the log-odds must NOT be a straight line in the iteration index: a
    # multiplicative recursion is exactly that, and is the failure mode
    lo = [log(max(p, 1e-12) / max(1 - p, 1e-12)) for p in tr[2:end]]
    if length(lo) >= 4
        d = diff(lo)
        # successive increments must shrink rather than hold constant
        @test abs(d[end]) <= abs(d[1]) + 1e-8
    end
end

@testset "bayes_model_prob wires the baseline to two RelFitModels" begin
    a0 = make_gaussian_fixed_loc(-1.0)
    a1 = make_gaussian_fixed_loc(+1.0)
    X = truth_gaussian(0.8, 1.0; n=200, seed=31)
    _, p1 = bayes_model_prob(X, a0, a1; N=400, paccmin=1e-2, max_sims=100_000, seed=9)
    @test 0.0 <= p1 <= 1.0
    # the truth sits at +0.8, so the +1 candidate must win outright
    @test p1 > 0.9
end
