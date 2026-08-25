#= ================================================================
   The posterior-model-probability baseline.

   This is here so a reader can reproduce the comparison the paper argues
   about rather than take its word for it. It is the SAME ABC-SMC as
   `abc_smc`, extended in the canonical way: the population lives over
   the joint space (model index m, parameters theta_m), all models share
   one adaptive tolerance schedule, and models that cannot reach low
   distances lose particles as the tolerance shrinks.

   P(M_m | data) is the sum of the importance weights of model-m
   particles over the total weight.

   WHAT THIS BASELINE DOES AT A TIE, measured rather than asserted, and
   the reason the rest of the package exists. On a location tie between
   two mirror-image candidates it commits in EVERY replication, with the
   direction a coin flip decided by the sign of the sample mean. The
   mechanism is not evidence-integral amplification but tolerance-driven
   ELIMINATION: the losing candidate's summary distance has a floor that
   the shared tolerance falls below, which empties its acceptance region.
   The certainty is therefore not Monte Carlo error and does not shrink
   with more particles. Verified over a 64-fold particle sweep: with the
   data held fixed the standard deviation of P(M1) across sampler seeds
   is 0.0000 at N = 1500, and P(M1) is exactly 0 or 1 at N = 1500, 6000,
   24000 and 96000 alike. The estimator is exact at modest N and the
   ESTIMAND is degenerate, which is why the fix has to be a different
   decision rule and not more compute.

   Nor is it a threshold artefact. Read as a three-action Bayes rule under
   0-1 loss with abstention cost `ell`, the rule commits when
   max_j P(M_j) >= 1 - ell, so the usual 0.95/0.05 cuts ARE that rule at
   ell = 0.05. Sweeping `ell` over its whole range changes nothing at
   such a tie, because the realised max_j P(M_j) is 1.000000 in every
   replication and no loss function can make a one look uncertain.
   ================================================================ =#

"""
    ModelChoiceResult

`(p, n_per_model, n_iters, total_sims, epsilon, p_trace)`.

`p_trace` is the model-probability vector after EVERY iteration and is
not decoration. A correctly weighted joint sampler converges to the
marginal-likelihood ratio, whereas a multiplicative recursion drifts
without limit, which shows up as a straight line in the log-odds against
the iteration index. It is the cheapest available check that the joint
weight below has not been broken by an unrelated edit.
"""
struct ModelChoiceResult
    p::Vector{Float64}          # final posterior model probabilities
    n_per_model::Vector{Int}    # particle count per model in final population
    n_iters::Int
    total_sims::Int
    epsilon::Float64
    p_trace::Vector{Vector{Float64}}
end

# Build a perturbation kernel from a model's weighted covariance,
# regularised to positive-definite, mirroring `abc_smc`.
function _mc_build_kernel(pts::Matrix{Float64}, wts::Vector{Float64},
                          np::Int, kernel_coeff::Float64, perturb::Symbol)
    Σ = size(pts, 2) <= np ? Matrix{Float64}(I, np, np) :  # too few particles
                             kernel_coeff * Matrix(weighted_covariance(pts, wts))
    return _build_kernel(perturb, np, _regularise(Σ, np))
end

"""
    abc_smc_model_choice(N, model_priors, rho_fns; kwargs...) -> ModelChoiceResult

ABC-SMC model selection over candidate models, each a prior vector plus a
distance function. Defaults match `abc_smc`: Gaussian kernel,
`kernel_coeff = 2`, `prop = 0.5`, `paccmin = 0.01`.

The perturbation kernel is used for BOTH the proposal and the importance
weight, so the two cannot drift apart.
"""
function abc_smc_model_choice(N::Int, model_priors::Vector, rho_fns::Vector;
        perturb::Symbol    = :normal,
        kernel_coeff::Float64 = 2.0,
        prop::Float64      = 0.5,
        paccmin::Float64   = 0.01,
        max_sims::Int      = 5_000_000,
        max_proposals::Int = 200_000_000,
        model_prior        = nothing,
        seed::Int          = 42,
        verbose::Bool      = false)

    _validate_kernel(perturb)
    lm  = length(model_priors)
    nps = [length(p) for p in model_priors]
    mpr = model_prior === nothing ? fill(1.0 / lm, lm) : model_prior ./ sum(model_prior)
    rng = Xoshiro(seed)
    total_sims = 0
    total_proposals = 0

    # ── Iteration 1: sample (model, theta) from the joint prior ──
    init_m = zeros(Int, N)
    init_d = zeros(N)
    init_θ = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        m = sample(rng, 1:lm, Weights(mpr))
        θ = [rand(rng, model_priors[m][j]) for j in 1:nps[m]]
        init_m[i] = m
        init_θ[i] = θ
        init_d[i] = rho_fns[m](θ)
        total_sims += 1
    end

    ε = quantile(init_d, prop)
    keep = init_d .<= ε

    pts = [reduce(hcat, [init_θ[i] for i in 1:N if keep[i] && init_m[i] == m];
                  init = zeros(nps[m], 0)) for m in 1:lm]
    wts = [ones(size(pts[m], 2)) for m in 1:lm]
    cur_dists = init_d[keep]          # pooled kept distances for the next eps

    p_trace = Vector{Vector{Float64}}()
    let e = [sum(wts[m]) for m in 1:lm]
        push!(p_trace, sum(e) > 0 ? e ./ sum(e) : fill(1.0 / lm, lm))
    end

    if verbose
        @printf("  t=1: eps=%.4f  per-model kept=%s  sims=%d\n",
                ε, string([size(pts[m], 2) for m in 1:lm]), total_sims)
        flush(stdout)
    end

    iteration = 1
    while total_sims < max_sims && total_proposals < max_proposals
        iteration += 1

        prev_pts = [copy(pts[m]) for m in 1:lm]
        prev_wts = [copy(wts[m]) for m in 1:lm]

        ker = [_mc_build_kernel(pts[m], wts[m], nps[m], kernel_coeff, perturb)
               for m in 1:lm]

        # Shared tolerance for all models: the prop-quantile of the pooled
        # current distances, forced to decrease strictly. This is what
        # makes the models compete on equal footing.
        ε_new = quantile(cur_dists, prop)
        ε = min(ε_new, ε * 0.999)

        flat_w = vcat(wts...)
        flat_m = vcat([fill(m, size(pts[m], 2)) for m in 1:lm]...)
        flat_c = vcat([collect(1:size(pts[m], 2)) for m in 1:lm]...)
        total_now = sum(flat_w)
        total_now <= 0 && break
        w_norm = flat_w ./ total_now
        n_flat = length(flat_w)

        new_θ = [Vector{Vector{Float64}}() for _ in 1:lm]
        new_d = [Vector{Float64}() for _ in 1:lm]
        iter_sims = 0
        n_accepted = 0

        for _ in 1:N
            (total_sims >= max_sims || total_proposals >= max_proposals) && break
            accepted = false
            while !accepted && total_sims < max_sims && total_proposals < max_proposals
                total_proposals += 1
                pf = sample(rng, 1:n_flat, Weights(w_norm))
                m  = flat_m[pf]
                parent = pts[m][:, flat_c[pf]]
                θ_new = parent .+ rand(rng, ker[m])

                in_prior = true
                for j in 1:nps[m]
                    if !insupport(model_priors[m][j], θ_new[j]); in_prior = false; break; end
                end
                in_prior || continue

                ρ = rho_fns[m](θ_new)
                iter_sims += 1
                total_sims += 1
                if ρ <= ε
                    push!(new_θ[m], θ_new)
                    push!(new_d[m], ρ)
                    n_accepted += 1
                    accepted = true
                end
            end
        end

        if n_accepted == 0
            verbose && println("  t=$iteration: none accepted, stopping")
            break
        end

        new_pts = [reduce(hcat, new_θ[m]; init = zeros(nps[m], 0)) for m in 1:lm]
        new_wts = [zeros(size(new_pts[m], 2)) for m in 1:lm]
        for m in 1:lm
            nm = size(new_pts[m], 2)
            nm == 0 && continue
            if size(prev_pts[m], 2) == 0
                new_wts[m] .= 1.0                  # no same-model parents: uniform
            else
                toni_weights!(new_wts[m], new_pts[m], prev_pts[m], prev_wts[m],
                              model_priors[m], ker[m], perturb, nps[m])
            end
            for k in 1:nm
                (isfinite(new_wts[m][k]) && new_wts[m][k] >= 0) || (new_wts[m][k] = 0.0)
            end
            all(new_wts[m] .== 0) && (new_wts[m] .= 1.0)
        end

        #= THE JOINT-SPACE CORRECTION ON THE MODEL INDEX. Do not remove
           this. It is the difference between a posterior model
           probability and a number that depends on how many SMC
           iterations happened to run.

           `toni_weights!` returns pi_m(theta) / q_m(theta), where q_m is
           the within-model kernel mixture normalised by that model's OWN
           weight sum. That is the correct weight CONDITIONAL on the model
           index, but the proposal also draws the index, from the previous
           population's model marginal P_{t-1}(m), since the resampling
           step allocates proposals in proportion to each model's total
           weight. The joint proposal density is therefore
           P_{t-1}(m) K_M(m|m') q_m(theta), and the weight needs the extra
           factor P(m) / [P_{t-1}(m) K_M(m|m')]. With no model jumps K_M
           is the identity and the missing factor is P(m) / P_{t-1}(m).

           Without it the acceptance rate cancels and the model weights
           follow W_m^t proportional to W_m^{t-1} pi_m(A^t), a
           MULTIPLICATIVE recursion. A constant per-iteration edge, such
           as the one a wider prior box confers through its normalising
           constant, is then raised to the power of the iteration count
           instead of converging to the acceptance-probability ratio.

           `test/runtests.jl` carries the closed-form control: two
           candidates differing ONLY in prior box volume must return
           V0/(V0+V1), and P(M1) must not move with the stopping rule. =#
        prev_tot = sum(sum(prev_wts[m]) for m in 1:lm)
        if prev_tot > 0
            for m in 1:lm
                size(new_pts[m], 2) == 0 && continue
                pm_prev = sum(prev_wts[m]) / prev_tot
                pm_prev > 0 && (new_wts[m] .*= mpr[m] / pm_prev)
            end
        end

        pts = new_pts
        wts = new_wts
        cur_dists = vcat(new_d...)

        let e = [sum(wts[m]) for m in 1:lm]
            push!(p_trace, sum(e) > 0 ? e ./ sum(e) : fill(1.0 / lm, lm))
        end

        pacc = n_accepted / max(iter_sims, 1)
        if verbose
            probs = [sum(wts[m]) for m in 1:lm]; probs ./= sum(probs)
            @printf("  t=%d: eps=%.4f pacc=%.4f per-model=%s p=%s sims=%d\n",
                    iteration, ε, pacc, string([size(pts[m],2) for m in 1:lm]),
                    string(round.(probs, digits=3)), total_sims)
            flush(stdout)
        end
        pacc <= paccmin && break
    end

    evid = [sum(wts[m]) for m in 1:lm]
    p = sum(evid) > 0 ? evid ./ sum(evid) : fill(1.0 / lm, lm)
    return ModelChoiceResult(p, [size(pts[m], 2) for m in 1:lm], iteration, total_sims, ε,
                             p_trace)
end

"""
    bayes_model_prob(X, model0, model1; N=2000, kwargs...) -> (result, p1)

The baseline in the form the comparison needs: run
`abc_smc_model_choice` on two `RelFitModel`s against the WHOLE data set
and return `P(M1 | X)` beside the full result.

Note the asymmetry with `refrain`, and it is deliberate rather than an
oversight: the baseline sees all `n` units where the relative-fit rule
sees only the held-out half. That is the comparison in the baseline's
favour, and it still commits at a tie.
"""
function bayes_model_prob(X::AbstractMatrix, model0::RelFitModel, model1::RelFitModel;
                          N::Int=2000, kwargs...)
    rho0 = model0.abc_distance_factory(X)
    rho1 = model1.abc_distance_factory(X)
    res = abc_smc_model_choice(N, [model0.priors, model1.priors], [rho0, rho1]; kwargs...)
    return res, res.p[2]
end
