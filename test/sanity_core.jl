#= ================================================================
   The primitives against closed forms.

   Ported from `sanity_checks.jl` of the abstention_model_choice
   repository, with its bespoke `check`/`RESULTS` bookkeeping replaced by
   `@test`/`@testset`. Nothing else moved: the geometries, the seeds and
   the tolerances are the ones the paper's own suite uses.

   These are ABC-free and deterministic, so the whole file costs seconds.
   Set `REFRAIN_SLOW=1` for the replication counts the source suite uses
   when it is not in FAST mode, which tightens every Monte Carlo
   tolerance.
   ================================================================ =#

using REFRAIN, Test, Random, Statistics, Distributions, Printf
using REFRAIN: _pairwise_sqdists, _mmd_within, _apply_transform

const SLOW = get(ENV, "REFRAIN_SLOW", "0") == "1"

@testset "1. sliced Wasserstein against closed forms" begin
    rng = MersenneTwister(11)
    n = 500
    a = reshape(randn(rng, n), 1, n)
    b = reshape(randn(rng, n), 1, n)

    # identity of indiscernibles, exact in d = 1
    @test sliced_wasserstein(a, a) == 0.0

    # the primitive sorts internally and must not mutate its arguments
    a_before = copy(a)
    _ = sliced_wasserstein(a, b)
    @test a == a_before

    @test sliced_wasserstein(a, b) == sliced_wasserstein(b, a)

    # shift identity: W1(A, A + delta) = |delta| exactly, at equal sizes
    for delta in (0.7, -2.3)
        @test isapprox(sliced_wasserstein(a, a .+ delta), abs(delta); atol=1e-12)
    end

    # scale identity: W1(A, s A) = |1 - s| mean|A| exactly
    s = 1.6
    @test isapprox(sliced_wasserstein(a, s .* a), abs(1 - s) * mean(abs.(a)); atol=1e-12)

    # d = 1 at equal sizes reduces to the sorted L1 average
    @test isapprox(sliced_wasserstein(a, b),
                   mean(abs.(sort(vec(a)) .- sort(vec(b)))); atol=1e-14)

    # population values
    big = SLOW ? 200_000 : 20_000
    tol_mu = SLOW ? 0.01 : 0.03
    tol_sc = SLOW ? 0.006 : 0.02
    rb = MersenneTwister(12)
    A = reshape(randn(rb, big), 1, big)
    for mu in (0.5, 2.0)                       # W1(N(0,1), N(mu,1)) = |mu|
        B = reshape(mu .+ randn(rb, big), 1, big)
        @test isapprox(sliced_wasserstein(A, B), mu; atol=tol_mu)
    end
    for sc in (1.5, 0.6)                       # W1(N(0,1), N(0,s)) = |1-s| sqrt(2/pi)
        B = reshape(sc .* randn(rb, big), 1, big)
        @test isapprox(sliced_wasserstein(A, B), abs(1 - sc) * sqrt(2 / pi); atol=tol_sc)
    end

    # the projection count is now reachable, and changing it changes the
    # answer in d > 1 and not in d = 1
    M = reshape(randn(MersenneTwister(13), 4 * 60), 4, 60)
    N2 = reshape(randn(MersenneTwister(14), 4 * 60), 4, 60)
    @test sliced_wasserstein(M, N2; n_projections=50) !=
          sliced_wasserstein(M, N2; n_projections=1000)
    @test sliced_wasserstein(a, b; n_projections=50) ==
          sliced_wasserstein(a, b; n_projections=1000)
end

#= Gaussian kernel k(x,y) = exp(-|x-y|^2 / (2 s^2)), P = N(0,1),
   Q = N(D,1). For W ~ N(m, t^2),
   E exp(-W^2/(2 s^2)) = s/sqrt(s^2+t^2) exp(-m^2/(2(s^2+t^2))).
   With s = 1 and t^2 = 2: E k(X,X') = E k(Y,Y') = 1/sqrt(3) and
   E k(X,Y) = exp(-D^2/6)/sqrt(3), so
   MMD^2 = (2/sqrt(3)) (1 - exp(-D^2/6)). =#
mmd2_closed_form(D) = (2 / sqrt(3)) * (1 - exp(-D^2 / 6))

@testset "2. maximum mean discrepancy against closed forms" begin
    #= mmd(X,X) is NOT zero: the unbiased U-statistic drops the diagonal
       from the within terms but not from the cross term. Exact algebra:
       MMD(X,X) = (2/(m-1)) (S/m^2 - 1) with S the full kernel sum. =#
    rng = MersenneTwister(21)
    m = 200
    X = reshape(randn(rng, m), 1, m)
    S = sum(exp.(-0.5 .* _pairwise_sqdists(X, X)))
    @test isapprox(mmd(X, X; bandwidth=1.0), (2 / (m - 1)) * (S / m^2 - 1); atol=1e-12)
    @test mmd(X, X; bandwidth=1.0) < 0     # not an identity of indiscernibles

    Y = reshape(randn(MersenneTwister(22), m), 1, m)
    @test mmd(X, Y; bandwidth=1.0) ≈ mmd(Y, X; bandwidth=1.0)

    # closed form under the alternative, at a FIXED bandwidth of 1
    nrep = SLOW ? 60 : 20
    nsamp = SLOW ? 2000 : 1000
    for D in (0.0, 0.5, 1.5)
        vals = zeros(nrep)
        for r in 1:nrep
            rr = MersenneTwister(1000 * r + round(Int, 100D))
            P = reshape(randn(rr, nsamp), 1, nsamp)
            Q = reshape(D .+ randn(rr, nsamp), 1, nsamp)
            vals[r] = mmd(P, Q; bandwidth=1.0)
        end
        se = std(vals) / sqrt(nrep)
        @test isapprox(mean(vals), mmd2_closed_form(D); atol=max(4 * se, 1e-4))
    end

    # unbiasedness under the null at a FIXED bandwidth
    nrep = SLOW ? 1000 : 200
    nsamp = 400
    vals_fixed = zeros(nrep)
    for r in 1:nrep
        rr = MersenneTwister(7_000 + r)
        P = reshape(randn(rr, nsamp), 1, nsamp)
        Q = reshape(randn(rr, nsamp), 1, nsamp)
        vals_fixed[r] = mmd(P, Q; bandwidth=1.0)
    end
    @test abs(mean(vals_fixed)) <= 3 * std(vals_fixed) / sqrt(nrep)
end

#= Under exchangeability of the pooled simulated draws, i.e. P0 and P1
   from one law with D0 fixed and independent, the permutation p-value is
   exactly uniform on the grid {1/(B+1), ..., (B+1)/(B+1)}. This is the
   only exactness claim in the paper that can be checked directly. =#
@testset "3. permutation exactness under exchangeability" begin
    B = 99
    grid_alpha = 0.05                 # 5/(B+1) sits exactly on the grid
    for (ipm, R, n) in ((:sw,  SLOW ? 4000 : 400, 100),
                        (:mmd, SLOW ? 1500 : 200,  60))
        pv = zeros(R)
        for r in 1:R
            rr = MersenneTwister(300_000 + r)
            D0 = reshape(randn(rr, n), 1, n)          # fixed, arbitrary
            P0 = reshape(randn(rr, n), 1, n)          # same law ...
            P1 = reshape(randn(rr, n), 1, n)          # ... as this one
            pv[r] = permutation_calibrate(D0, P0, P1; n_perm=B, ipm=ipm,
                                          seed=400_000 + r).p
        end
        rate = mean(pv .<= grid_alpha)
        se = sqrt(grid_alpha * (1 - grid_alpha) / R)
        @test isapprox(rate, grid_alpha; atol=3 * se)

        # uniformity over the whole grid, chi-square on B+1 equiprobable cells
        counts = zeros(Int, B + 1)
        for p in pv
            k = clamp(round(Int, p * (B + 1)), 1, B + 1)
            counts[k] += 1
        end
        expct = R / (B + 1)
        chi2 = sum((counts .- expct) .^ 2) / expct
        @test ccdf(Chisq(B), chi2) > 0.001
    end
end

@testset "4. bootstrap of the equidistance null" begin
    B = 299
    alpha = 0.05

    #= 4a. LEVEL at a NON-COINCIDENT equidistance null. P* = N(0,1),
       F0 = N(+d,1), F1 = N(-d,1): the population gap is |d| - |d| = 0
       while the two predictives are far apart, which is exactly the
       geometry the permutation cannot handle. =#
    R = SLOW ? 1000 : 200
    n = 400
    d = 0.4
    rej = zeros(Bool, R)
    for r in 1:R
        rr = MersenneTwister(500_000 + r)
        D0 = reshape(randn(rr, n), 1, n)
        P0 = reshape(+d .+ randn(rr, n), 1, n)
        P1 = reshape(-d .+ randn(rr, n), 1, n)
        rej[r] = bootstrap_calibrate(D0, P0, P1; n_boot=B, ipm=:sw,
                                     seed=600_000 + r).p <= alpha
    end
    @test isapprox(mean(rej), alpha; atol=max(3 * sqrt(alpha * (1 - alpha) / R), 0.015))

    # 4b. COVERAGE of a non-zero population gap, Delta = 0.5 - 0.2 = 0.3
    R = SLOW ? 600 : 200
    n = 2000
    Delta = 0.3
    cov = zeros(Bool, R)
    for r in 1:R
        rr = MersenneTwister(700_000 + r)
        D0 = reshape(randn(rr, n), 1, n)
        P0 = reshape(0.5 .+ randn(rr, n), 1, n)
        P1 = reshape(-0.2 .+ randn(rr, n), 1, n)
        bc = bootstrap_calibrate(D0, P0, P1; n_boot=B, ipm=:sw, seed=800_000 + r)
        lo = quantile(bc.T_boot, alpha / 2); hi = quantile(bc.T_boot, 1 - alpha / 2)
        cov[r] = lo <= Delta <= hi
    end
    @test isapprox(mean(cov), 0.95; atol=0.04)

    #= 4c. the p-value / interval duality, and the reason its quantile
       convention has to be NAMED. p = 2(1+min)/(1+B) <= alpha is
       min <= 6 at B = 299, alpha = 0.05. Under a linearly interpolated
       empirical quantile the two rules part company at exactly one min,
       which is why `percentile_interval` uses order statistics. =#
    function p_of_min(mn)
        T = vcat(fill(-1.0, mn), fill(1.0, B - mn))
        n_le = count(<=(0.0), T); n_ge = count(>=(0.0), T)
        return min(1.0, 2 * (1 + min(n_le, n_ge)) / (1 + B))
    end
    @test p_of_min(6) <= alpha && p_of_min(7) > alpha

    spread(mn) = vcat(collect(range(-1.0, -0.01; length=mn)),
                      collect(range(0.01, 1.0; length=B - mn)))
    n_disagree = 0
    for mn in (5, 6, 7, 8)
        T = spread(mn)
        (quantile(T, alpha / 2) > 0) == (p_of_min(mn) <= alpha) || (n_disagree += 1)
    end
    @test n_disagree == 1

    # 4d. and with the order-statistic convention the duality is EXACT
    ok_all = true; probed = 0
    for BB in (99, 199, 299, 499), al in (0.10, 0.05, 0.02)
        kk = floor(Int, (BB + 1) * al / 2)
        kk < 1 && continue
        ramp(lo, hi, k) = k <= 0 ? Float64[] :
                          k == 1 ? [lo] : collect(range(lo, hi; length=k))
        for mn in 0:min(kk + 2, BB - 1)
            T = vcat(ramp(-1.0, -0.01, mn), ramp(0.01, 1.0, BB - mn))
            length(T) == BB || error("bad construction")
            lo, hi = percentile_interval(T; alpha=al)
            excl = lo > 0 || hi < 0
            pv = min(1.0, 2 * (1 + min(count(<=(0.0), T), count(>=(0.0), T))) / (1 + BB))
            excl == (pv <= al) || (ok_all = false)
            probed += 1
        end
    end
    @test ok_all
    @test probed > 40

    # 4e. degenerate inputs: three coincident samples must abstain
    Z = reshape(zeros(50), 1, 50)
    bz = bootstrap_calibrate(Z, Z, Z; n_boot=99, ipm=:sw, seed=1)
    @test bz.T_obs == 0.0
    @test bz.p > alpha
    @test decide(bz.p, bz.T_obs) === :abstain
end

@testset "5. concentration threshold" begin
    rng = MersenneTwister(31)
    n = 300
    D0 = reshape(randn(rng, n), 1, n)
    P0 = reshape(0.3 .+ randn(rng, n), 1, n)
    P1 = reshape(-0.3 .+ randn(rng, n), 1, n)
    for alpha in (0.05, 0.01)
        h = hoeffding_mmd_test(D0, P0, P1; alpha=alpha)
        @test isapprox(h.threshold, 2 * sqrt(log(1 / alpha) / (2 * n)); atol=1e-12)
    end
    # identical predictives: the witness norm is zero, so it must abstain
    @test hoeffding_mmd_test(D0, P0, P0; alpha=0.05).decision === :abstain
    # the witness is antisymmetric in (P0, P1), so T flips sign
    ha = hoeffding_mmd_test(D0, P0, P1; alpha=0.05)
    hb = hoeffding_mmd_test(D0, P1, P0; alpha=0.05)
    @test isapprox(ha.T, -hb.T; atol=1e-10)
end

@testset "6. split_iid" begin
    X = reshape(collect(1.0:97.0), 1, 97)
    D1, D0 = split_iid(X; frac_fit=0.5, seed=5)
    s1 = Set(vec(D1)); s0 = Set(vec(D0))
    @test isempty(intersect(s1, s0))
    @test union(s1, s0) == Set(vec(X))
    @test size(D1, 2) + size(D0, 2) == 97
    @test split_iid(X; seed=5)[1] == D1
    @test split_iid(X; seed=6)[1] != D1
end

#= 7. THE ADEQUACY SCREEN. The screen is exact for the SIMPLE null that
   D0 follows the fitted predictive, not for the composite null that the
   family contains the truth. This turns that into a quantitative claim:
   the screen fires at the nominal rate against a well-specified
   candidate whose pilot is not estimated, and ABOVE it once the pilot is
   estimated on a finite D1, by more as more parameters are fitted.

   These are NOT the paper's 0.05 and 0.15, which come from ABC-fitted
   pilots. Here the pilot is the closed-form MLE, so this file stays free
   of ABC. What transfers is the MECHANISM and the ORDERING. =#

# A Gaussian candidate carrying only the parameters named. The screen
# touches nothing but the simulator, so the prior and the ABC distance are
# placeholders and are never evaluated.
function _screen_gaussian(free_location::Bool; loc::Float64=0.0)
    sim = function (theta, n; seed::Int=42)
        rng = MersenneTwister(seed)
        mu = free_location ? theta[1] : loc
        sigma = exp(free_location ? theta[2] : theta[1])
        return reshape(mu .+ sigma .* randn(rng, n), 1, n)
    end
    return RelFitModel([Uniform(-1.0, 1.0)], D1 -> (theta -> 0.0), sim)
end

# The screen's own replicate streams step by 7919 within a replication, so
# replications are offset by 10^7 > 7919 * 299 and cannot collide.
function _screen_level(model, pilot; R::Int, n::Int, n_rep::Int)
    fires = 0
    psum = 0.0
    for r in 1:R
        X = reshape(randn(MersenneTwister(770_000 + r), n), 1, n)
        D1, D0 = split_iid(X; frac_fit=0.5, seed=880_000 + r)
        s = adequacy_screen(D0, model, pilot(D1); ipm=:sw, n_rep=n_rep,
                            alpha=0.05, seed=3 + 10_000_000 * r,
                            sw_seed=3 + 10_000_000 * r)
        fires += s.inadequate ? 1 : 0
        psum += s.p
    end
    return (level=fires / R, mean_p=psum / R)
end

@testset "7. adequacy screen under a well-specified candidate" begin
    R = SLOW ? 200 : 40
    n = 400
    n_rep = SLOW ? 299 : 99
    tol = SLOW ? 0.04 : 0.10          # about 2.6 binomial standard errors

    m_fixed = _screen_gaussian(false)  # {N(0, sigma^2)},  theta = [log sigma]
    m_free  = _screen_gaussian(true)   # {N(mu, sigma^2)}, theta = [mu, log sigma]
    # every candidate below CONTAINS the truth N(0,1)
    pilot_oracle(D1) = [0.0]                        # nothing estimated
    pilot_one(D1)    = [log(sqrt(mean(abs2, D1)))]  # one parameter, MLE
    pilot_two(D1)    = [mean(D1), log(std(D1))]     # two parameters, MLE

    a = _screen_level(m_fixed, pilot_oracle; R=R, n=n, n_rep=n_rep)
    b = _screen_level(m_fixed, pilot_one;    R=R, n=n, n_rep=n_rep)
    c = _screen_level(m_free,  pilot_two;    R=R, n=n, n_rep=n_rep)

    # with the pilot at the truth, D0 really is distributed as the fitted
    # law, so the statistics are exchangeable and the p-value is uniform
    @test isapprox(a.level, 0.05; atol=tol)
    @test isapprox(a.mean_p, 0.5; atol=SLOW ? 0.05 : 0.10)

    # the composite null is a different matter, and this is the "because":
    # nothing changes but the pilot being estimated on D1
    @test c.level >= a.level + 0.05
    @test a.level <= b.level <= c.level

    #= DO NOT DELETE THIS LAST ARM. Without it a screen that never fired
       would pass three of the four checks above, which is how the
       original was mutation-tested. The candidate here is {N(1,sigma^2)}
       against a truth at 0, so no scale can rescue it. =#
    d = _screen_level(_screen_gaussian(false; loc=1.0), pilot_one;
                      R=R, n=n, n_rep=n_rep)
    @test d.level == 1.0
end

@testset "8. the nuisance parameter is common to both terms" begin
    rng = MersenneTwister(41)
    n = 600
    tv = rand(rng, TDist(6.0), n) ./ sqrt(6 / 4)
    D0 = reshape(tv, 1, n)
    P0 = reshape(randn(rng, n), 1, n)
    P1 = reshape(rand(rng, Laplace(0.0, 1 / sqrt(2)), n), 1, n)

    # after the fix both calibrations hold ONE bandwidth over the observed
    # statistic and every replicate, so the contrast lives in one RKHS
    bw = common_bandwidth(:mmd, D0, P0, P1)
    @test relfit_statistic(D0, P0, P1; ipm=:mmd) ==
          mmd(D0, P0; bandwidth=bw) - mmd(D0, P1; bandwidth=bw)
    # and it must not depend on the labelling, or the permutation null
    # would not be a null of the reported statistic
    @test common_bandwidth(:mmd, D0, P0, P1) == common_bandwidth(:mmd, D0, P1, P0)

    #= The two data-dependent bandwidths differ enough to matter, which is
       what makes the common one a fix and not a tidy-up. This is the
       measurement, kept as an assertion at a loose bound. =#
    s0 = median_bandwidth(D0, P0)
    s1 = median_bandwidth(D0, P1)
    @test !isapprox(s0, s1; rtol=1e-3)
end

@testset "9. the copula transform is vacuous in one dimension" begin
    #= A rank transform per sample maps every one-dimensional sample to
       the IDENTICAL grid {0.5/n, ..., (n-0.5)/n}, so D0, P0 and P1 become
       the same numbers and T is identically zero. This is the control
       that confirms the mechanism, and the reason `:rank` can never be a
       default. =#
    rng = MersenneTwister(51)
    n = 120
    D0 = reshape(randn(rng, n), 1, n)
    P0 = reshape(1.0 .+ 2.0 .* randn(rng, n), 1, n)
    P1 = reshape(-3.0 .+ 0.4 .* randn(rng, n), 1, n)
    # Every sample ranks to the same MULTISET, though not in the same
    # order, and both metrics are functions of the multiset alone.
    grid = collect((1:n) .- 0.5) ./ n
    @test sort(vec(_apply_transform(D0, :rank))) ≈ grid
    @test sort(vec(_apply_transform(P0, :rank))) ≈ grid
    @test relfit_statistic(D0, P0, P1; ipm=:sw, transform=:rank) == 0.0
    @test isapprox(relfit_statistic(D0, P0, P1; ipm=:mmd, transform=:rank), 0.0; atol=1e-14)
    # and in higher dimensions it is not vacuous
    M0 = reshape(randn(MersenneTwister(52), 3 * n), 3, n)
    M1 = reshape(randn(MersenneTwister(53), 3 * n), 3, n)
    Md = reshape(randn(MersenneTwister(54), 3 * n), 3, n)
    @test relfit_statistic(Md, M0, M1; ipm=:sw, transform=:rank) != 0.0
end
