#= ================================================================
   Example 2: a realistic comparison, and the pattern that matters once
   the simulator is not a one-liner.

   Two cluster point processes for hourly rainfall, weighed against one
   another in hydrology for forty years, NEITHER with a tractable
   likelihood:

     M0  BARTLETT-LEWIS rectangular pulses. Storms arrive as a Poisson
         process at rate lam. Each storm opens an Exp(gam) activity
         WINDOW with one cell at the origin and further cell arrivals at
         Poisson rate bet inside it. Cells are rectangular, Exp(eta) in
         duration and Exp(mux) in intensity.

     M1  NEYMAN-SCOTT rectangular pulses. Storms arrive the same way, but
         each carries 1 + Poisson(nu - 1) cells DISPLACED from the storm
         origin by Exp(bet), with the same rectangular pulses.

   Both fit five parameters. The parametrisation is chosen so the prior
   boxes pair slot for slot, with nu = 1 + bet/g so that g plays exactly
   gam's role, and so neither candidate enjoys an Occam advantage.

   FOUR THINGS HERE ARE WORTH COPYING, and they are the difference
   between a fit that takes a minute and one that takes an hour.

     1. An IN-PLACE simulator, `_simulate_bl!`, for the hot ABC path. It
        is called once per proposal, two hundred thousand times per fit,
        and it never allocates.
     2. An ALLOCATING WRAPPER around it for the `RelFitModel.simulator`
        slot, because `simulate_from_fit` and the calibrations need fresh
        matrices they can keep.
     3. A DISTANCE FACTORY that computes the target summaries ONCE per
        fit and preallocates its buffer there. The closure it returns
        does arithmetic and nothing else.
     4. A `hash((seed_base, theta))` SIMULATOR SEED. It makes the closure
        a pure function of theta, so the fit is reproducible, and it is
        safe under a driver-level `@threads` because no two proposals
        share a stream.

   THIS IS ALSO THE REGIME WHERE `calibration = :refit` IS NOT OPTIONAL.
   Five parameters per family are fitted on the fit half, and a column
   resample of the simulated sets holds every one of them still. Measured
   on this comparison, the plug-in interval is too narrow by a factor of
   1.16 to 1.46 and rejects at about twice its nominal rate.

   Run:  julia --project=. examples/rainfall.jl
   The defaults here are REDUCED so the example runs in a few minutes.
   The production configuration is named at the bottom.
   ================================================================ =#

using REFRAIN, Random, Statistics, Distributions, Printf

const HOURS = 168          # one wet-season week
const SPIN  = 400.0        # burn-in so the field is stationary at hour 0
const D     = 7            # features per week

# ── the hourly accumulator ───────────────────────────────────────

# Spread one rectangular cell across the hourly accumulator EXACTLY: each
# hour receives x times its overlap with [c, c + dur).
@inline function _add_cell!(h::Vector{Float64}, c::Float64, dur::Float64, x::Float64)
    a = max(c, 0.0); b = min(c + dur, Float64(HOURS))
    b > a || return nothing
    i0 = floor(Int, a) + 1
    i1 = min(ceil(Int, b), HOURS)
    @inbounds for i in i0:i1
        h[i] += x * (min(Float64(i), b) - max(Float64(i - 1), a))
    end
    return nothing
end

#= The storm generators ACCUMULATE into `h` without zeroing it, which is
   what lets a truth superpose the two processes into one field.

   THE SKIP RULES are what make the 400 h spin cost-neutral. A
   Bartlett-Lewis storm whose window closes 45 cell-memories before the
   observation window cannot put a cell into it: every cell is born by
   t + L and would need a duration exceeding 45/eta, probability e^{-45}.
   A Neyman-Scott storm born 45 displacement-memories plus 45
   cell-memories early likewise. Both rules only ever skip storms whose
   contribution is below one ulp of any hourly total, and they consume no
   further draws, so a deep-spun dead storm costs two random numbers
   instead of forty. =#

function _add_bl_storms!(h, rng, lam, gam, bet, eta, mux)
    lam > 0.0 || return nothing
    ilam = 1 / lam; igam = 1 / gam; ibet = 1 / bet; ieta = 1 / eta
    dead = -45.0 * ieta
    t = -SPIN
    while true
        t += randexp(rng) * ilam
        t > HOURS && break
        L = randexp(rng) * igam                # the activity window
        t + L < dead && continue
        c = t
        while true
            _add_cell!(h, c, randexp(rng) * ieta, mux * randexp(rng))
            c += randexp(rng) * ibet
            (c > t + L || c > Float64(HOURS)) && break
        end
    end
    return nothing
end

function _add_ns_storms!(h, rng, lam, nu, bet, eta, mux)
    lam > 0.0 || return nothing
    ilam = 1 / lam; ibet = 1 / bet; ieta = 1 / eta
    dead = -45.0 * ibet - 45.0 * ieta
    t = -SPIN
    while true
        t += randexp(rng) * ilam
        t > HOURS && break
        t < dead && continue
        ncell = 1 + rand(rng, Poisson(nu - 1.0))
        for _ in 1:ncell
            c = t + randexp(rng) * ibet       # displaced, not windowed
            _add_cell!(h, c, randexp(rng) * ieta, mux * randexp(rng))
        end
    end
    return nothing
end

# ── the unit: one week's feature vector ──────────────────────────

"""
The `d = 7` feature vector of one week's hourly series: weekly total, max
hourly, dry-hour fraction, lag-1 autocorrelation, max six-hour total,
number of wet spells, mean wet-spell length. A fixed measurable transform
applied identically to data and to simulations.

Two fused passes rather than six separate ones, and the six-hour maximum
by a rolling window. The feature map was 37% of the simulator's cost
before that and the simulator is most of a replication's.
"""
function rain_features!(f::AbstractVector, h::Vector{Float64})
    n = HOURS
    tot = 0.0; mx = 0.0; ndry = 0
    nsp = 0; nwet = 0; inw = false
    w6 = 0.0
    @inbounds for i in 1:6
        w6 += h[i]
    end
    mx6 = w6
    @inbounds for i in 1:n
        v = h[i]
        tot += v
        v > mx && (mx = v)
        if v < 0.1
            ndry += 1
            inw = false
        else
            nwet += 1
            inw || (nsp += 1)
            inw = true
        end
        if i > 6
            w6 += v - h[i - 6]
            w6 > mx6 && (mx6 = w6)
        end
    end
    m = tot / n
    acc = 0.0; ss = 0.0
    @inbounds begin
        d_prev = h[1] - m
        ss += d_prev * d_prev
        for i in 2:n
            d = h[i] - m
            ss += d * d
            acc += d_prev * d
            d_prev = d
        end
    end
    s2 = ss / (n - 1)
    f[1] = tot
    f[2] = mx
    f[3] = ndry / n
    f[4] = s2 > 0 ? acc / ((n - 1) * s2) : 0.0
    f[5] = mx6
    f[6] = nsp
    f[7] = nsp > 0 ? nwet / nsp : 0.0
    return f
end

# ── (1) the in-place simulators, for the hot ABC path ────────────

# theta = [log lam, log mux, log eta, log bet, log gam]
function _simulate_bl!(F::Matrix{Float64}, rng::MersenneTwister, theta, n::Int, seed::Int)
    Random.seed!(rng, seed)
    lam = exp(theta[1]); mux = exp(theta[2]); eta = exp(theta[3])
    bet = exp(theta[4]); gam = exp(theta[5])
    h = Vector{Float64}(undef, HOURS)
    for j in 1:n
        fill!(h, 0.0)
        _add_bl_storms!(h, rng, lam, gam, bet, eta, mux)
        rain_features!(view(F, :, j), h)
    end
    return F
end

# theta = [log lam, log mux, log eta, log bet, log g], nu = 1 + bet/g
function _simulate_ns!(F::Matrix{Float64}, rng::MersenneTwister, theta, n::Int, seed::Int)
    Random.seed!(rng, seed)
    lam = exp(theta[1]); mux = exp(theta[2]); eta = exp(theta[3])
    bet = exp(theta[4]); g = exp(theta[5])
    nu = 1.0 + bet / g
    h = Vector{Float64}(undef, HOURS)
    for j in 1:n
        fill!(h, 0.0)
        _add_ns_storms!(h, rng, lam, nu, bet, eta, mux)
        rain_features!(view(F, :, j), h)
    end
    return F
end

# ── (2) the allocating wrappers, for the calibration path ────────

#= `RelFitModel.simulator` is called by `simulate_from_fit`, by the
   adequacy screen and by the refit bootstrap, all of which KEEP the
   matrix they are handed. Handing them the in-place buffer would alias
   every one of them to the same memory. =#
simulate_bl(theta, n::Int; seed::Int=42) =
    _simulate_bl!(Matrix{Float64}(undef, D, n), MersenneTwister(0), theta, n, seed)
simulate_ns(theta, n::Int; seed::Int=42) =
    _simulate_ns!(Matrix{Float64}(undef, D, n), MersenneTwister(0), theta, n, seed)

# ── (3) and (4) the distance factory ─────────────────────────────

#= The ABC summary is the per-feature MEANS AND STANDARD DEVIATIONS of
   the seven features, fourteen statistics, estimated from a SIMULATED
   sample at each proposal and scaled per statistic by its observed
   value with a floor.

   SECOND MOMENTS ARE NOT DECORATION. On means alone the two candidates
   can be matched exactly, so the summary would be blind to the thing
   being compared. Their weekly-total VARIANCES differ by 27% in closed
   form, so a criterion that includes them sees the difference. It is
   also what the field itself does: the standard references fit the mean,
   variances at several aggregations, the lag-1 autocorrelation and the
   dry probability. Means alone were never the practice.

   NOTE WHERE THE WORK HAPPENS. `mean(D1)`, `std(D1)` and the buffer
   allocation are OUTSIDE the returned closure, so they happen once per
   fit. The closure simulates and reduces, and does nothing else. =#
function rain_distance_factory(simulator!; seed_base::Int=0, n_mult::Int=1)
    return function (D1)
        m1 = vec(mean(D1, dims=2))
        s1 = vec(std(D1, dims=2))
        sc_m = max.(m1, 0.05)
        sc_s = max.(s1, 0.05)
        n_sim = n_mult * size(D1, 2)
        F = Matrix{Float64}(undef, D, n_sim)      # allocated ONCE per fit
        rng = MersenneTwister(0)
        return function (theta)
            #= (4) THE SEED. Hashing (seed_base, theta) makes the distance
               a pure function of theta, so the same proposal always
               scores the same, and it makes distinct proposals use
               distinct streams, so a driver may run replications under
               @threads without two of them sharing draws. =#
            s = Int(hash((seed_base, theta)) % 10_000_000)
            simulator!(F, rng, theta, n_sim, s)
            acc = 0.0
            @inbounds for i in 1:D
                mi = 0.0
                for j in 1:n_sim
                    mi += F[i, j]
                end
                mi /= n_sim
                vi = 0.0
                for j in 1:n_sim
                    vi += (F[i, j] - mi)^2
                end
                si = sqrt(vi / (n_sim - 1))
                acc += ((mi - m1[i]) / sc_m[i])^2 + ((si - s1[i]) / sc_s[i])^2
            end
            return sqrt(acc)
        end
    end
end

# ── the two candidates ───────────────────────────────────────────

#= ONE prior box for BOTH candidates, slot for slot, so the prior volumes
   coincide EXACTLY and neither candidate enjoys an Occam advantage.
   Unmatched boxes are a hidden second knob: they enter the posterior
   model probability as a normalising constant and read exactly like a
   finding. The ranges cover the literature's fitted hourly-gauge values
   with slack. =#
const RAIN_PRIORS = [Uniform(log(0.004), log(0.1)),    # log lam,  storms per hour
                     Uniform(log(0.3),   log(12.0)),   # log mux,  mm per hour
                     Uniform(log(0.15),  log(8.0)),    # log eta,  per hour
                     Uniform(log(0.05),  log(2.0)),    # log bet,  per hour
                     Uniform(log(0.008), log(0.5))]    # log gam or log g

make_bl(; seed_base::Int=0, n_mult::Int=1) =
    RelFitModel(RAIN_PRIORS,
                rain_distance_factory(_simulate_bl!; seed_base=seed_base, n_mult=n_mult),
                simulate_bl)
make_ns(; seed_base::Int=0, n_mult::Int=1) =
    RelFitModel(RAIN_PRIORS,
                rain_distance_factory(_simulate_ns!; seed_base=seed_base + 7, n_mult=n_mult),
                simulate_ns)

# Two members inside the literature's fitted ranges: a long-window
# frontal Bartlett-Lewis against a compact-cluster Neyman-Scott.
const BL_THETA = [log(0.012), log(2.2), log(1.2), log(0.35), log(0.03)]
const NS_THETA = [log(0.0203169), log(1.2994), log(1.2), log(0.25),
                  log(0.25 / (38.0 / 3.0 - 1.0))]

#= THE TRUTH is the superposition of the two members with their storm
   rates scaled by the knob: lam_BL' = (1-w) lam_BL and lam_NS' = w lam_NS.
   A superposition of two Poisson-cluster processes with DIFFERENT cluster
   mechanisms is in neither family for 0 < w < 1, so the truth lies
   outside both candidates along the whole sweep, while w -> 0 and w -> 1
   recover the two families exactly. =#
function truth_rain(w::Float64; n::Int, seed::Int=1)
    rng = MersenneTwister(seed)
    F = Matrix{Float64}(undef, D, n)
    h = Vector{Float64}(undef, HOURS)
    bl = exp.(BL_THETA); ns = exp.(NS_THETA)
    nu = 1.0 + ns[4] / ns[5]
    for j in 1:n
        fill!(h, 0.0)
        _add_bl_storms!(h, rng, (1 - w) * bl[1], bl[5], bl[4], bl[3], bl[2])
        _add_ns_storms!(h, rng, w * ns[1], nu, ns[4], ns[3], ns[2])
        rain_features!(view(F, :, j), h)
    end
    return F
end

# ── run it ───────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    #= REDUCED so the example runs in minutes. The production
       configuration of the paper is n = 1600 weeks, N = 1500 particles,
       max_sims = 200_000, paccmin = 3e-3, n_mult = 4 and n_boot = 299,
       which is about twenty minutes a decision. =#
    n_weeks  = 400
    n_mult   = 2
    fitkw    = (; N = 400, paccmin = 3e-2, max_sims = 12_000)

    m0 = make_bl(; n_mult = n_mult)
    m1 = make_ns(; n_mult = n_mult)

    for w in (0.1, 0.9)
        X = truth_rain(w; n = n_weeks, seed = 4242)
        @printf("\nw = %.2f  (w -> 0 is Bartlett-Lewis, w -> 1 is Neyman-Scott)\n", w)
        for cal in (:bootstrap, :refit)
            t0 = time()
            r = refrain(X, m0, m1; ipm = :sw, calibration = cal,
                        n_mult = n_mult, n_boot = 99, screen = false,
                        refit_S = 3, refit_N = 200,
                        split_seed = 7, fit_seed = 11, sim_seed = 5000,
                        cal_seed = 91, fitkw...)
            @printf("  %-11s decision %-8s p = %.4f  T = %+9.4f   (%.0f s)\n",
                    cal, r.decision, r.p, r.T, time() - t0)
        end
    end

    println("""

    The two calibrations differ in WIDTH, not in direction: T_obs is the
    same plug-in statistic either way, and only the null moves. With five
    parameters per family the plug-in is too narrow by a factor of 1.16
    to 1.46, so where the two disagree it is the refit that should be
    believed.

    At this reduced budget the fits are coarse and the decisions should
    not be read as the paper's. What the example is for is the SHAPE of a
    real model definition.
    """)
end
