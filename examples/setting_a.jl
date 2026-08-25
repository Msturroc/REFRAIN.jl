#= ================================================================
   Example 1: does it work at all.

   Two Gaussians with FIXED mirror-image locations at -1 and +1 and a
   free scale, against a truth centred at c. The truth lies outside both
   candidates at every c, because neither candidate can move its
   location, and c = 0 is a genuine tie by symmetry.

   That the locations are STRUCTURAL is the part a reader tends to miss.
   The two candidates cannot fit their way to the middle: only the scale
   is free. If they could, there would be no tie to abstain at.

   Run:  julia --project=. examples/setting_a.jl      (about a minute)
   ================================================================ =#

using REFRAIN, Printf, Statistics

include(joinpath(@__DIR__, "example_models.jl"))

const KW = (; N = 400, paccmin = 1e-2, max_sims = 25_000,
              n_boot = 299, screen_rep = 99)

m0 = make_gaussian_fixed_loc(-1.0)      # M0: N(-1, sigma^2)
m1 = make_gaussian_fixed_loc(+1.0)      # M1: N(+1, sigma^2)

println("REFRAIN on setting A: two fixed-location Gaussians, truth at c\n")

# ── one decision, in full ────────────────────────────────────────

X = truth_gaussian(0.0, 1.0; n = 400, seed = 1)     # the genuine tie
res = refrain(X, m0, m1; ipm = :sw, calibration = :bootstrap, KW...)

println("At the tie c = 0:")
@printf("  decision  %s\n", res.decision)
@printf("  p         %.4f\n", res.p)
@printf("  T         %+.6f   (T > 0 favours M1)\n", res.T)
@printf("  interval  [%+.4f, %+.4f]   contains zero: %s\n",
        res.interval[1], res.interval[2],
        res.interval[1] <= 0 <= res.interval[2])
@printf("  screen    %s\n", res.screen)
@printf("  fits      theta0 = %.4f   theta1 = %.4f  (log sigma)\n\n",
        res.theta0[1], res.theta1[1])

# ── and a sweep, which is where the rule earns its keep ──────────

#= The point of the sweep is not that the rule commits when the gap is
   wide. It is that it ABSTAINS at c = 0 and commits by c = 0.2, i.e.
   that its report distinguishes a tie from a separation. A posterior
   model probability on the same machinery returns the same confident
   answer at both. =#
println("Sweep in c, R = 10 replications each, sliced Wasserstein bootstrap:")
println("     c    abstain   choose M0   choose M1   screen: both fail")
R = 10
for c in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
    out = [refrain(truth_gaussian(c, 1.0; n = 400, seed = 1000 + r), m0, m1;
                   ipm = :sw, calibration = :bootstrap,
                   split_seed = r, fit_seed = r,
                   sim_seed = 20_000 + r, cal_seed = 30_000 + r, KW...)
           for r in 1:R]
    @printf("  %.2f     %.2f        %.2f        %.2f          %.2f\n", c,
            mean(o.decision === :abstain for o in out),
            mean(o.decision === :M0 for o in out),
            mean(o.decision === :M1 for o in out),
            mean(o.screen === :both_fail for o in out))
end

println("""

Read the screen column beside the decision, never folded into it. The
truth is outside BOTH candidates at every c, so a screen with power
should convict widely, and it does. Committing to M1 at c = 0.8 and
convicting M1 in the same replication is a coherent report: it says to
widen the model rather than to choose within it.
""")

# ── the test-only entry point ────────────────────────────────────

#= A reader who already has a held-out sample and two simulated sets
   wants nothing to do with the fitting layer. =#
D1, D0 = split_iid(X; frac_fit = 0.5, seed = 42)
th0 = abc_fit(m0, D1; N = 400, paccmin = 1e-2, max_sims = 25_000, seed = 42)
th1 = abc_fit(m1, D1; N = 400, paccmin = 1e-2, max_sims = 25_000, seed = 42)
P0 = simulate_from_fit(m0, th0, size(D0, 2); seed = 1234)
P1 = simulate_from_fit(m1, th1, size(D0, 2); seed = 1234 + 1_000_000)

d = decide(D0, P0, P1; ipm = :sw, calibration = :bootstrap, alpha = 0.05)
@printf("decide() on samples you already have: %s (p = %.4f, T = %+.6f)\n",
        d.decision, d.p, d.T)
