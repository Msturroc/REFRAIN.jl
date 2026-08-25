#= ================================================================
   The ABC layer.

   "Bring your own fit" is not usable by someone who came to this method
   precisely because they have no likelihood, so the package carries its
   own samplers rather than asking for a posterior.

   THE SAMPLER CONTRACT IS DELIBERATELY THIN. A user supplies
   `priors::Vector{<:Distribution}`, independent PER COORDINATE, and
   `rho_fn(theta)::Float64`. The simulator is hidden entirely inside that
   closure, so no sampler here ever sees it. Two constraints follow and
   are not negotiable: there is no joint prior object, so priors with
   dependence between coordinates cannot be expressed, and the
   perturbation kernel is a full-covariance MvNormal or MvCauchy, so
   discrete and simplex-valued parameters are not supported.

   Three samplers, plus the model-indicator variant in `modelchoice.jl`:

     `rejection_abc`  the simplest thing that works. One tolerance, one
                      pass. Useful as a control: it is what an ABC-SMC
                      schedule reduces to at its first iteration, so a
                      finding that survives both is not a property of the
                      schedule.

     `abc_smc`        Toni et al. (2009), the sampler the paper used, and
                      the default.

     `apmc`           Lenormand, Jabot and Deffuant (2013). Adaptive
                      population Monte Carlo: keeps the best half of the
                      population rather than regenerating all of it, so
                      it spends fewer simulator calls per effective
                      particle. It does model comparison NATIVELY, so
                      there is no separate model-choice entry point for
                      it: `models` is a vector of per-model prior vectors
                      and `rho` a vector of distances, and single-model
                      fitting is the `length(models) == 1` case.

   THE KERNEL. Gaussian is the default everywhere, so the package
   reproduces the paper out of the box. `kernel = :cauchy` is the
   recommended upgrade and is documented at `abc_smc`.
   ================================================================ =#

# ── Kernel construction, shared by the samplers ──────────────────

#= A LATENT BUG FIXED ON THE WAY ACROSS. In the source repository the
   samplers built the proposal kernel as
   `perturb == :normal ? MvNormal : MvTDist(1, ...)`, i.e. ANYTHING other
   than `:normal` was treated as Cauchy, while `toni_weights!` branched on
   the literal `:cauchy` when computing the importance weight. So
   `perturb = :t`, or any typo, would PROPOSE Cauchy and WEIGHT Gaussian,
   which is a silently wrong sampler rather than an error. Validating the
   symbol here costs nothing and closes it. =#
function _validate_kernel(perturb::Symbol)
    perturb in (:normal, :cauchy) ||
        error("unknown perturbation kernel $perturb (use :normal or :cauchy). " *
              "Anything else used to be silently treated as Cauchy by the " *
              "proposal and as Gaussian by the importance weight.")
    return perturb
end

function _build_kernel(perturb::Symbol, np::Int, Σ_h)
    return perturb === :normal ? MvNormal(zeros(np), Σ_h) :
                                 MvTDist(1, zeros(np), Matrix(Σ_h))
end

# Regularise a covariance to positive definite, exactly as the source
# repository does, so the kernel is the same kernel.
function _regularise(Σ::AbstractMatrix, np::Int)
    Σh = Hermitian(Matrix(Σ))
    ridge = 1e-8 * max(1.0, tr(Σh) / np)
    while !isposdef(Σh + ridge * I(np))
        ridge *= 10
    end
    return Hermitian(Matrix(Σh) + ridge * I(np))
end

"""
    weighted_covariance(particles, weights) -> Symmetric

Weighted empirical covariance of a `np x N` particle matrix.

REPRODUCIBILITY NOTE, and it is not a footnote. The `(C .* w') * C'` in
here is a BLAS `gemm`, and OpenBLAS accumulates it differently at
different thread counts. On a comparison with seven fitted parameters the
whole ABC fit moved when `BLAS.set_num_threads` changed. The BLAS thread
count is therefore part of a reported result, not an implementation
detail. `test/identity.jl` pins it explicitly and says so.
"""
function weighted_covariance(particles::Matrix{Float64}, weights::Vector{Float64})
    np, N = size(particles)
    w = weights ./ sum(weights)
    μ = particles * w
    C = particles .- μ
    Σ = (C .* w') * C'
    return Symmetric(Σ)
end

# ── Importance weights (Toni et al. 2009, equation 5) ────────────

"""
    toni_weights!(weights, new_pts, prev_pts, prev_w, priors, kernel,
                  kernel_type, np)

Importance weights for a new particle population given the previous one
and the perturbation kernel,
`w(theta) = pi(theta) / sum_j wtilde_j K(theta | theta_j)`.

`kernel_type` must be the kernel the proposal actually used. The proposal
and the weight kernel are the SAME kernel in a correct sampler, and the
one investigation this cost in the source project came from their being
allowed to differ.
"""
function toni_weights!(weights::Vector{Float64},
                       new_pts::Matrix{Float64},
                       prev_pts::Matrix{Float64},
                       prev_w::Vector{Float64},
                       priors::Vector{<:Distribution},
                       kernel, kernel_type::Symbol, np::Int)
    N_new = size(new_pts, 2)
    N_prev = size(prev_pts, 2)

    # Normalised log-weights
    w_sum = sum(prev_w)
    log_w_sum = log(w_sum)
    log_w = [w > 0 ? log(w) : -Inf for w in prev_w]

    # Precompute: L_inv, transformed points, kernel log-constant
    L = kernel.Σ.chol.L
    L_inv = inv(L)
    W_prev = L_inv * prev_pts   # np x N_prev
    W_new  = L_inv * new_pts    # np x N_new

    log_const = logpdf(kernel, zeros(np))   # normalising constant
    is_cauchy = kernel_type == :cauchy
    half_df_p = is_cauchy ? (1.0 + np) / 2.0 : 0.0

    lp_buf = Vector{Float64}(undef, N_prev)

    @inbounds for k in 1:N_new
        log_prior = 0.0
        for i in 1:np
            lp_i = logpdf(priors[i], new_pts[i, k])
            if !isfinite(lp_i); log_prior = -Inf; break; end
            log_prior += lp_i
        end
        if !isfinite(log_prior)
            weights[k] = 0.0
            continue
        end

        # Pass 1: log(w_j * K(theta_k | theta_j))
        max_val = -Inf
        for j in 1:N_prev
            sq = 0.0
            for i in 1:np
                v = W_new[i, k] - W_prev[i, j]
                sq += v * v
            end
            lp = if is_cauchy
                log_w[j] + log_const - half_df_p * log1p(sq)
            else
                log_w[j] + log_const - 0.5 * sq
            end
            lp_buf[j] = lp
            if lp > max_val; max_val = lp; end
        end

        # Pass 2: log-sum-exp
        acc = 0.0
        for j in 1:N_prev
            acc += exp(lp_buf[j] - max_val)
        end

        log_denom = max_val + log(acc) - log_w_sum
        weights[k] = exp(log_prior - log_denom)
    end
end

# ── Sampler 1: rejection ABC ─────────────────────────────────────

"""
    RejectionResult

`(particles, weights, distances, epsilon, n_sims, n_accepted)`. The
weights are uniform, which is what rejection ABC returns.
"""
struct RejectionResult
    particles::Matrix{Float64}
    weights::Vector{Float64}
    distances::Vector{Float64}
    epsilon::Float64
    n_sims::Int
    n_accepted::Int
end

"""
    rejection_abc(N, priors, rho_fn; epsilon=nothing, quantile_keep=0.5,
                  max_sims=100_000, seed=42, verbose=false) -> RejectionResult

Rejection ABC. Draw `max_sims` parameter vectors from the prior, score
each one, and keep those within a tolerance. The tolerance is either
supplied outright as `epsilon`, which is the fixed-tolerance form, or set
to the `quantile_keep`-quantile of the realised distances, which is the
quantile form and needs no scale knowledge.

`N` is the target particle count and is used only for the fixed-tolerance
form's early exit: the sweep stops once `N` particles are accepted.

WHY IT IS HERE AND NOT ONLY AS A TOY. Iteration 1 of an ABC-SMC schedule
with the whole budget spent IS rejection ABC, so a tolerance ladder run
through this function is the control that separates a finding about the
posterior from a finding about the schedule. On the location tie of the
paper, the posterior-model-probability baseline commits at every rung of
such a ladder once the tolerance passes the two candidates' distance
floors, which is what establishes that its failure is not an artefact of
the ABC-SMC schedule.
"""
function rejection_abc(N::Int, priors::Vector{<:Distribution}, rho_fn::Function;
                       epsilon::Union{Nothing,Float64}=nothing,
                       quantile_keep::Float64=0.5,
                       max_sims::Int=100_000, seed::Int=42,
                       verbose::Bool=false)
    np = length(priors)
    rng = Xoshiro(seed)
    θ = Matrix{Float64}(undef, np, max_sims)
    d = Vector{Float64}(undef, max_sims)
    n_used = 0
    n_hit = 0
    for i in 1:max_sims
        for j in 1:np
            θ[j, i] = rand(rng, priors[j])
        end
        d[i] = rho_fn(view(θ, :, i))
        n_used = i
        if epsilon !== nothing
            d[i] <= epsilon && (n_hit += 1)
            n_hit >= N && break
        end
    end
    dv = d[1:n_used]
    ε = epsilon === nothing ? quantile(dv, quantile_keep) : epsilon
    keep = findall(<=(ε), dv)
    isempty(keep) && error("rejection_abc: no particle within epsilon = $ε over " *
                           "$n_used simulations. Loosen the tolerance or raise max_sims.")
    if verbose
        @printf("  rejection ABC: eps=%.6g  kept=%d/%d  sims=%d\n",
                ε, length(keep), n_used, n_used)
        flush(stdout)
    end
    return RejectionResult(θ[:, keep], ones(length(keep)), dv[keep], ε, n_used, length(keep))
end

# ── Sampler 2: ABC-SMC (Toni et al. 2009) ────────────────────────

"""
    ToniResult

`(pts, wts, its, proposals)`. `pts[1][t]` is the particle matrix after
iteration `t` and `wts[1][t]` its weights, so `pts[1][end]` is the final
population. The outer vector exists so the shape matches `APMCResult`,
which is indexed by model.
"""
struct ToniResult
    pts::Vector{Vector{Matrix{Float64}}}
    wts::Vector{Vector{Vector{Float64}}}
    its::Vector{Int}
    proposals::Vector{Int}   # total proposals (including prior rejections) per iteration
end

"""
    abc_smc(N, priors, rho_fn; perturb=:normal, perturb_weight=perturb,
            kernel_coeff=2.0, prop=0.5, paccmin=0.02, max_sims=1_000_000,
            max_proposals=100_000_000, seed=42, use_psis=false,
            verbose=false) -> ToniResult

ABC-SMC after Toni et al. (2009), with adaptive quantile thresholds.
Iteration 1 samples from the prior and keeps the best `prop` fraction.
Each later iteration regenerates all `N` particles by resample, perturb
and accept against a strictly decreasing tolerance, then reweights.

THE KERNEL. `perturb = :normal` is the default, so the package reproduces
the published numbers out of the box. `perturb = :cauchy` is the
RECOMMENDED UPGRADE: a Cauchy perturbation kernel avoids both the
overinflation and the overconcentration that a Gaussian kernel suffers as
the tolerance falls (Sturrock and Shahrezaei). It costs nothing here,
since the branch is a `MvTDist(1, ...)` in the proposal and a matching
`-((1+np)/2) log1p(sq)` in the weight.

`perturb_weight` exists because the proposal and weight kernels MUST
match, and the source project records an investigation lost to their
being allowed to drift apart. It defaults to `perturb` and there is no
good reason to set it otherwise.

`use_psis` applies Pareto-smoothed importance sampling to the weights.
Off by default, since it changes the reported numbers.
"""
function abc_smc(N::Int, priors::Vector{<:Distribution}, rho_fn::Function;
                 perturb::Symbol    = :normal,
                 perturb_weight::Symbol = perturb,
                 kernel_coeff::Float64 = 2.0,
                 prop::Float64      = 0.5,
                 paccmin::Float64   = 0.02,
                 max_sims::Int      = 1_000_000,
                 max_proposals::Int = 100_000_000,
                 seed::Int          = 42,
                 use_psis::Bool     = false,
                 verbose::Bool      = false)

    _validate_kernel(perturb)
    _validate_kernel(perturb_weight)
    np = length(priors)
    rng = Xoshiro(seed)
    total_sims = 0
    total_proposals = 0

    pts_history = Matrix{Float64}[]
    wts_history = Vector{Float64}[]
    its_history = Int[]
    proposals_history = Int[]

    # ── Iteration 1: sample from prior ───────────────────────────

    init_particles = zeros(np, N)
    init_distances = zeros(N)

    for i in 1:N
        for j in 1:np
            init_particles[j, i] = rand(rng, priors[j])
        end
        init_distances[i] = rho_fn(init_particles[:, i])
        total_sims += 1
    end

    # Adaptive threshold: prop-quantile
    ε = quantile(init_distances, prop)
    keep_mask = init_distances .<= ε
    n_keep = sum(keep_mask)

    particles = init_particles[:, keep_mask]
    distances = init_distances[keep_mask]
    weights = ones(n_keep)

    push!(pts_history, copy(particles))
    push!(wts_history, copy(weights))
    push!(its_history, N)
    push!(proposals_history, N)

    if verbose
        @printf("  t=1: eps=%.4f  kept=%d/%d  sims=%d\n", ε, n_keep, N, total_sims)
        flush(stdout)
    end

    # ── Subsequent iterations ────────────────────────────────────

    iteration = 1
    while total_sims < max_sims && total_proposals < max_proposals
        iteration += 1
        N_prev = size(particles, 2)
        N_prev == 0 && break

        Σ_w = weighted_covariance(particles, weights)
        Σ_h = _regularise(kernel_coeff * Matrix(Σ_w), np)

        kernel = _build_kernel(perturb, np, Σ_h)
        # Separate kernel object for the importance weight. It is the same
        # covariance; only the shape may differ, and it should not.
        wt_kernel = perturb_weight == perturb ? kernel :
                    _build_kernel(perturb_weight, np, Σ_h)

        w_norm = weights ./ sum(weights)

        ε_new = quantile(distances, prop)
        ε_new = min(ε_new, ε * 0.999)   # ensure strict decrease
        ε = ε_new

        # ── Generate N new particles by rejection sampling ───────

        new_particles = zeros(np, N)
        new_distances = zeros(N)
        n_accepted = 0
        iter_sims = 0
        iter_proposals_start = total_proposals

        for i in 1:N
            (total_sims >= max_sims || total_proposals >= max_proposals) && break

            accepted = false
            while !accepted && total_sims < max_sims && total_proposals < max_proposals
                total_proposals += 1

                parent_idx = sample(rng, 1:N_prev, Weights(w_norm))
                parent = particles[:, parent_idx]

                θ_new = parent .+ rand(rng, kernel)

                # Prior check BEFORE the simulator, the standard saving
                in_prior = true
                for j in 1:np
                    if !insupport(priors[j], θ_new[j])
                        in_prior = false
                        break
                    end
                end
                if !in_prior
                    continue   # prior rejection, no simulator call
                end

                ρ = rho_fn(θ_new)
                iter_sims += 1
                total_sims += 1

                if ρ <= ε
                    n_accepted += 1
                    new_particles[:, n_accepted] = θ_new
                    new_distances[n_accepted] = ρ
                    accepted = true
                end
            end
        end

        push!(its_history, iter_sims)
        push!(proposals_history, total_proposals - iter_proposals_start)

        if n_accepted == 0
            verbose && println("  t=$iteration: no particles accepted, stopping")
            break
        end

        new_particles = new_particles[:, 1:n_accepted]
        new_distances = new_distances[1:n_accepted]

        # ── Importance weights ───────────────────────────────────

        new_weights = zeros(n_accepted)
        toni_weights!(new_weights, new_particles, particles, weights,
                      priors, wt_kernel, perturb_weight, np)

        for i in 1:n_accepted
            if !isfinite(new_weights[i]) || new_weights[i] < 0
                new_weights[i] = 0.0
            end
        end
        if all(new_weights .== 0)
            new_weights .= 1.0   # fallback to uniform
        end

        if use_psis
            lw = log.(max.(new_weights, 1e-300))
            psis_smooth!(lw)
            new_weights = exp.(lw .- maximum(lw))
        end

        particles = new_particles
        distances = new_distances
        weights = new_weights

        push!(pts_history, copy(particles))
        push!(wts_history, copy(weights))

        # Acceptance rate = accepted / simulator calls
        pacc = n_accepted / max(iter_sims, 1)

        if verbose
            @printf("  t=%d: eps=%.4f  pacc=%.4f  N=%d  sims=%d (total=%d)\n",
                    iteration, ε, pacc, n_accepted, iter_sims, total_sims)
            flush(stdout)
        end

        pacc <= paccmin && break
    end

    return ToniResult([pts_history], [wts_history], its_history, proposals_history)
end

# ── PSIS (Vehtari et al. 2024), used only when asked for ─────────

function _fit_gpd_pwm(exceedances::AbstractVector{Float64})
    n = length(exceedances)
    n < 2 && return (0.0, max(mean(exceedances), 1e-10))
    sorted = sort(exceedances)
    a0 = 0.0; a1 = 0.0
    for i in 1:n
        a0 += sorted[i]
        a1 += sorted[i] * (i - 1) / (n - 1)
    end
    a0 /= n; a1 /= n
    denom = a0 - 2 * a1
    abs(denom) < 1e-10 && return (0.0, max(a0, 1e-10))
    k_hat = a0 / denom - 2.0
    sigma_hat = 2.0 * a0 * a1 / denom
    k_hat = clamp(k_hat, -0.5, 2.0)
    sigma_hat = max(sigma_hat, 1e-10)
    return (k_hat, sigma_hat)
end

"""
    psis_smooth!(log_weights) -> k_hat

Pareto-smoothed importance sampling: replace the upper tail of the
log-weights by fitted generalised-Pareto quantiles. Returns the fitted
shape, which is the usual diagnostic (above 0.7 the estimate is
unreliable). Modifies `log_weights` in place.
"""
function psis_smooth!(log_weights::AbstractVector{Float64})
    n = length(log_weights)
    n < 10 && return 0.0
    M = min(ceil(Int, n / 5), ceil(Int, 3 * sqrt(n)))
    M = max(M, 5); M = min(M, n - 1)
    sorted_idx = sortperm(log_weights)
    cutoff_val = log_weights[sorted_idx[n - M]]
    tail_indices = sorted_idx[(n - M + 1):n]
    tail_vals = log_weights[tail_indices] .- cutoff_val
    k_hat, sigma_hat = _fit_gpd_pwm(tail_vals)
    for (rank, idx) in enumerate(tail_indices)
        p = (rank - 0.5) / M
        q = abs(k_hat) < 1e-6 ? -sigma_hat * log(1.0 - p) :
            sigma_hat / k_hat * ((1.0 - p)^(-k_hat) - 1.0)
        log_weights[idx] = cutoff_val + q
    end
    return k_hat
end

# ── Sampler 3: APMC (Lenormand, Jabot and Deffuant 2013) ─────────

#= PORTED FROM `apmc_v2_threads.jl` of the author's optimisation
   repository, which is the last version in that chain that is both
   standalone and readable, and which already fixes the original's
   `pacc[j,i] == 0` comparison-not-assignment bug, takes `perturb` as a
   Symbol, drops an `eval(Meta.parse(...))` of the covariance estimator,
   and carries per-thread RNGs, systematic resampling and ESS.

   THREE THINGS DELIBERATELY NOT CARRIED ACROSS.

   The ORIGINAL `updated_abc_model_comparison_threads.jl` commits type
   piracy, importing `Distributions.rand` and `pdf` and overloading them
   on plain `Vector`, which will fight anything else a user loads. It
   also `eval`s its covariance estimator from a string, and the `pacc`
   bug is still live in it.

   V3 AND LATER drag in RealNVP, SINF and copula machinery, and v6
   `include`s v5, so neither is standalone.

   THE GPU FORK hard-codes four toy problems in its `GPUProblemData`.

   TWO CHOICES MADE ON THE WAY IN. The original takes its kernel
   covariance from a Ledoit-Wolf SHRINKAGE estimator, which would mean a
   dependency on CovarianceEstimation for a default nobody asked for.
   Here `covar` is a keyword defaulting to the plain weighted covariance
   that `abc_smc` uses, so the two samplers agree by default and
   shrinkage is opt-in: pass any `(params_matrix) -> covariance` you like.
   And the original's hard-coded early stop,
   `i > 11 && abs(eps[i] - eps[i-5]) < 1e-3`, is now the `eps_tol` and
   `eps_lag` keywords.

   THE CAUCHY-UNDER-MODEL-COMPARISON QUESTION IS LEFT OPEN, ON PURPOSE.
   `apmc_v6_threads.jl` FORCIBLY switches a Cauchy kernel to Normal
   whenever `length(models) > 1`, commented "auto-switching to :normal
   kernel for model comparison stability". That is an unresolved finding
   rather than a disagreement between authors, since the Cauchy-kernel
   argument is the same author's. It is carried here as
   `normal_for_comparison`, defaulting to FALSE with a warning when it
   would have fired, so the behaviour is visible and chosen rather than
   silent. Note that the abstention repository's own model-indicator
   sampler passes the SAME symbol to the proposal and to the weight, so
   whatever the instability is, it is not the proposal-weight mismatch
   that the separate `perturb_weight` of `abc_smc` exists to prevent. =#

"""
    APMCResult

`(pts, wts, p, epsilon, its, pacc, ess, dists)`, each indexed by model
then by iteration, except `epsilon` and `its` which are per iteration.
`p[j][end]` is model `j`'s final posterior probability.
"""
struct APMCResult
    pts::Vector{Vector{Matrix{Float64}}}
    wts::Vector{Vector{Vector{Float64}}}
    p::Vector{Vector{Float64}}
    epsilon::Vector{Float64}
    its::Vector{Int}
    pacc::Vector{Vector{Float64}}
    ess::Vector{Vector{Float64}}
    dists::Vector{Vector{Float64}}
end

"""Effective sample size from unnormalised weights."""
function effective_sample_size(w::AbstractVector{<:Real})
    s = sum(w)
    s <= 0 && return 0.0
    wn = w ./ s
    return 1.0 / sum(wn .^ 2)
end

"""Systematic resampling: O(1/N) variance against O(1) for multinomial."""
function systematic_resample(weights::AbstractVector{<:Real}, n::Int;
                             rng::AbstractRNG = Random.default_rng())
    N = length(weights)
    N == 0 && return Int[]
    total = sum(weights)
    if total <= 0
        return rand(rng, 1:N, n)
    end
    w = weights ./ total
    cw = cumsum(w)
    cw[end] = 1.0
    u0 = rand(rng) / n
    u = u0 .+ collect(0:n-1) ./ n
    indices = Vector{Int}(undef, n)
    j = 1
    for i in 1:n
        while j < N && cw[j] < u[i]
            j += 1
        end
        indices[i] = j
    end
    return indices
end

# The default covariance estimator: the plain weighted covariance
# `abc_smc` uses, so the two samplers agree unless told otherwise.
_apmc_default_covar(params::AbstractMatrix) = Matrix(cov(params; dims=1))

"""
    apmc(N, models, rho; kwargs...) -> APMCResult

Adaptive population Monte Carlo ABC (Lenormand, Jabot and Deffuant 2013),
with model comparison NATIVE rather than bolted on.

`models` is a vector of per-model prior vectors and `rho` a vector of
distance functions, one per model. Single-model fitting is the
`length(models) == 1` case and needs no separate entry point, which is
the whole reason there is no `apmc_model_choice`.

Unlike `abc_smc`, which regenerates all `N` particles each iteration,
APMC keeps the best `prop * N` and tops up, so it spends fewer simulator
calls per effective particle at the cost of a population that is no
longer i.i.d. across iterations.

Keywords worth knowing:

  * `perturb` -- `:normal` (default) or `:cauchy`
  * `normal_for_comparison` -- force `:normal` when comparing more than
    one model. FALSE by default, with a warning where it would fire. See
    the module comment: this is an unresolved finding carried as an
    explicit switch rather than applied silently.
  * `covar` -- `(params_matrix) -> covariance`. Defaults to the plain
    weighted covariance, so APMC and `abc_smc` agree. Pass a Ledoit-Wolf
    shrinkage estimator here if you want the original's behaviour.
  * `eps_floor` -- stop once the tolerance reaches this value. Zero, the
    default, disables it. IT IS THE ABSTENTION-ADJACENT KNOB: below the
    level at which the data can distinguish the candidates, a shrinking
    tolerance starts selecting on noise and favours whichever candidate
    is most flexible. Setting the floor at a noise estimate stops the
    sampler before it starts answering a question the data cannot.
  * `eps_tol`, `eps_lag` -- the convergence test, replacing the
    original's hard-coded `i > 11 && abs(eps[i] - eps[i-5]) < 1e-3`.
"""
function apmc(N::Int, models::AbstractVector, rho::AbstractVector;
              prop::Float64    = 0.5,
              paccmin::Float64 = 0.01,
              kernel_coeff::Float64 = 2.0,
              perturb::Symbol  = :normal,
              normal_for_comparison::Bool = false,
              covar            = _apmc_default_covar,
              eps_floor::Float64 = 0.0,
              eps_tol::Float64 = 1e-3,
              eps_lag::Int     = 10,
              max_iters::Int   = 2000,
              max_sims::Int    = 100_000_000,
              seed::Int        = 42,
              verbose::Bool    = false)

    _validate_kernel(perturb)
    lm = length(models)
    length(rho) == lm ||
        error("apmc: got $lm model prior vectors and $(length(rho)) distances")

    if lm > 1 && perturb === :cauchy && normal_for_comparison
        perturb = :normal
        verbose && println("  apmc: normal_for_comparison forced the kernel to :normal")
    elseif lm > 1 && perturb === :cauchy
        @warn("apmc: running a Cauchy kernel under model comparison. The author's " *
              "v6 sampler switches this to :normal for stability, and that finding " *
              "is unresolved. Pass normal_for_comparison = true to follow it.",
              maxlog = 1)
    end

    np = [length(models[j]) for j in 1:lm]
    s  = round(Int, N * prop)
    s >= 1 || error("apmc: prop * N rounds to zero")
    rng = Xoshiro(seed)
    total_sims = 0

    # Live population, stored as parallel vectors rather than the source's
    # packed matrix, which is what made the original hard to read.
    pop_m = Int[]                      # model index per particle
    pop_θ = Vector{Vector{Float64}}()  # parameters
    pop_d = Float64[]                  # distance

    pts = [Matrix{Float64}[] for _ in 1:lm]
    wts = [Vector{Float64}[] for _ in 1:lm]
    p_hist = [Float64[] for _ in 1:lm]
    pacc_hist = [Float64[] for _ in 1:lm]
    ess_hist = [Float64[] for _ in 1:lm]
    sig = [Matrix{Float64}(I, np[j], np[j]) for j in 1:lm]
    epsilon = Float64[]
    its = Int[]
    dists_hist = Vector{Float64}[]

    # ── Iteration 1: the joint prior ─────────────────────────────
    for _ in 1:N
        m = rand(rng, 1:lm)
        θ = [rand(rng, models[m][j]) for j in 1:np[m]]
        push!(pop_m, m); push!(pop_θ, θ); push!(pop_d, rho[m](θ))
        total_sims += 1
    end
    push!(its, N)
    push!(epsilon, quantile(pop_d, prop))
    keep = sortperm(pop_d)[1:min(s, length(pop_d))]
    pop_m = pop_m[keep]; pop_θ = pop_θ[keep]; pop_d = pop_d[keep]
    push!(dists_hist, copy(pop_d))

    function _harvest!()
        for j in 1:lm
            idx = findall(==(j), pop_m)
            M = isempty(idx) ? zeros(np[j], 0) : reduce(hcat, pop_θ[idx])
            push!(pts[j], M)
        end
    end
    _harvest!()
    for j in 1:lm
        push!(wts[j], ones(size(pts[j][end], 2)))
        push!(pacc_hist[j], 1.0)
        push!(ess_hist[j], effective_sample_size(wts[j][end]))
    end
    let e = [sum(wts[j][end]) for j in 1:lm], tot = sum(e)
        for j in 1:lm; push!(p_hist[j], tot > 0 ? e[j] / tot : 1.0 / lm); end
    end
    for j in 1:lm
        nj = size(pts[j][end], 2)
        if nj > np[j]
            idx = systematic_resample(wts[j][end], N; rng=rng)
            sig[j] = Matrix(covar(pts[j][end][:, idx]'))
        end
    end

    verbose && @printf("  apmc t=1: eps=%.6g  kept=%d  sims=%d\n",
                       epsilon[end], length(pop_d), total_sims)

    # ── Main loop ────────────────────────────────────────────────
    iter = 1
    while iter < max_iters && total_sims < max_sims
        iter += 1
        if length(epsilon) > 1
            maximum(pacc_hist[j][end] for j in 1:lm) <= paccmin && break
        end
        if eps_floor > 0.0 && epsilon[end] <= eps_floor
            verbose && println("  apmc: tolerance reached eps_floor, stopping")
            break
        end

        ker = Vector{Any}(undef, lm)
        for j in 1:lm
            ker[j] = _build_kernel(perturb, np[j], _regularise(kernel_coeff * sig[j], np[j]))
        end

        prev_pts = [pts[j][end] for j in 1:lm]
        prev_wts = [wts[j][end] for j in 1:lm]
        alive = [j for j in 1:lm if size(prev_pts[j], 2) > 0]
        isempty(alive) && break

        n_new = N - length(pop_d)
        n_new <= 0 && (n_new = N - s)
        new_m = Int[]; new_θ = Vector{Vector{Float64}}(); new_d = Float64[]
        proposed = zeros(Int, lm)
        for _ in 1:n_new
            total_sims >= max_sims && break
            local m, θn
            while true
                m = alive[rand(rng, 1:length(alive))]
                idx = sample(rng, 1:size(prev_pts[m], 2), Weights(prev_wts[m]))
                θn = prev_pts[m][:, idx] .+ rand(rng, ker[m])
                all(insupport(models[m][j], θn[j]) for j in 1:np[m]) && break
            end
            proposed[m] += 1
            push!(new_m, m); push!(new_θ, θn); push!(new_d, rho[m](θn))
            total_sims += 1
        end
        push!(its, length(new_d))
        isempty(new_d) && break

        # Combine and keep the best s
        all_m = vcat(pop_m, new_m)
        all_θ = vcat(pop_θ, new_θ)
        all_d = vcat(pop_d, new_d)
        n_old = length(pop_d)
        order = sortperm(all_d)[1:min(s, length(all_d))]
        # Acceptance rate per model: how many of that model's PROPOSALS survived
        survived = zeros(Int, lm)
        for k in order
            k > n_old && (survived[all_m[k]] += 1)
        end
        for j in 1:lm
            push!(pacc_hist[j], proposed[j] > 0 ? survived[j] / proposed[j] : 0.0)
        end

        kept_new = Set(k for k in order if k > n_old)
        pop_m = all_m[order]; pop_θ = all_θ[order]; pop_d = all_d[order]
        is_new = [k in kept_new for k in order]
        push!(epsilon, pop_d[end])
        push!(dists_hist, copy(pop_d))

        _harvest!()
        # Importance weights: carried-forward particles keep their weight,
        # freshly accepted ones get the Toni weight against the previous
        # population.
        for j in 1:lm
            idx = findall(==(j), pop_m)
            nj = length(idx)
            if nj == 0
                push!(wts[j], Float64[]); continue
            end
            w = zeros(nj)
            fresh = [k for (a, k) in enumerate(idx) if is_new[k]]
            old_order = [k for k in idx if !is_new[k]]
            if !isempty(fresh) && size(prev_pts[j], 2) > 0
                Mfresh = reduce(hcat, pop_θ[fresh])
                wf = zeros(size(Mfresh, 2))
                toni_weights!(wf, Mfresh, prev_pts[j], prev_wts[j],
                              models[j], ker[j], perturb, np[j])
                pos = 1
                for (a, k) in enumerate(idx)
                    if is_new[k]
                        w[a] = wf[pos]; pos += 1
                    end
                end
            elseif !isempty(fresh)
                for (a, k) in enumerate(idx); is_new[k] && (w[a] = 1.0); end
            end
            # Carried-forward particles: their previous weights, in order
            if !isempty(old_order)
                prevw = prev_wts[j]
                pos = 1
                for (a, k) in enumerate(idx)
                    if !is_new[k]
                        w[a] = pos <= length(prevw) ? prevw[pos] : 1.0
                        pos += 1
                    end
                end
            end
            for a in eachindex(w)
                (isfinite(w[a]) && w[a] >= 0) || (w[a] = 0.0)
            end
            all(w .== 0) && (w .= 1.0)
            push!(wts[j], w)
        end

        let e = [isempty(wts[j][end]) ? 0.0 : sum(wts[j][end]) for j in 1:lm],
            tot = sum(e)
            for j in 1:lm; push!(p_hist[j], tot > 0 ? e[j] / tot : 0.0); end
        end
        for j in 1:lm
            push!(ess_hist[j], effective_sample_size(wts[j][end]))
            nj = size(pts[j][end], 2)
            if nj > np[j]
                idx = systematic_resample(wts[j][end], N; rng=rng)
                newsig = Matrix(covar(pts[j][end][:, idx]'))
                isposdef(_regularise(newsig, np[j])) && (sig[j] = newsig)
            end
        end

        if verbose
            @printf("  apmc t=%d: eps=%.6g  p=%s  sims=%d\n", iter, epsilon[end],
                    string(round.([p_hist[j][end] for j in 1:lm]; digits=3)), total_sims)
            flush(stdout)
        end

        if length(epsilon) > eps_lag + 1
            abs(epsilon[end] - epsilon[end - eps_lag]) < eps_tol && break
        end
    end

    return APMCResult(pts, wts, p_hist, epsilon, its, pacc_hist, ess_hist, dists_hist)
end
