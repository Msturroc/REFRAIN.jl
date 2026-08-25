#= ================================================================
   Sample-only integral probability metrics.

   Two of them, and the choice is not a detail. Only an IPM computed from
   samples alone is available to a likelihood-free comparison: the
   Kullback-Leibler, density-power and Hellinger variants of Park,
   Balakrishnan and Wasserman (2026) all need pointwise densities, which
   is exactly what a simulator does not give you.

   The sliced Wasserstein-1 distance carries the units of the data and is
   exactly scale-equivariant. The maximum mean discrepancy under the
   median-bandwidth heuristic is exactly scale-INVARIANT, because the
   bandwidth rescales with the data. Consequence for reporting, measured
   rather than argued: multiply a data set by ten and the sliced
   Wasserstein gap multiplies by ten while the maximum mean discrepancy
   gap does not move in the sixth decimal. So gaps from different problems
   may be compared on the maximum mean discrepancy axis and may not be
   compared on the sliced Wasserstein one.

   Ported verbatim from `relfit_core.jl` of the abstention_model_choice
   repository, which is where the paper's numbers were computed.
   `test/identity.jl` digests the IEEE bit patterns against that
   repository's stored digests, so this file may be reorganised but not
   rearithmeticked.
   ================================================================ =#

# ── IPM primitive 1: sliced Wasserstein-1 ────────────────────────

"""
    sliced_wasserstein(A, B; n_projections=1000, seed=42) -> Float64

Sliced Wasserstein-1 distance between two empirical distributions stored
as `d`-rows by `n`-cols matrices, i.e. one i.i.d. unit per COLUMN.
Projects both onto random one-dimensional directions, computes the
one-dimensional Wasserstein-1 distance per projection, and averages.
Lower is closer and zero is identical.

In one dimension no projection is used at all, since the average over
directions of a one-dimensional quantile contrast is the exact
one-dimensional Wasserstein-1 distance.
"""
#= The distance splits into a per-sample part (project onto each
   direction, sort) and a per-pair part (the quantile contrast).  Keeping
   them apart matters because every caller in this package holds one of
   the two samples FIXED across hundreds of evaluations: the permutation
   null re-labels P0 and P1 with the held-out D0 fixed, the bootstrap
   shares one resampled D0* between the two terms of each replicate, and
   the adequacy screen compares a fixed reference sample against n_rep
   fresh ones.  Computing the fixed side once instead of 2(1 + n_perm)
   times is the same arithmetic on the same values, so it is exact rather
   than approximate: no result moves by so much as an ulp.

   A `SWSample` is a sample already projected and sorted; `Theta` is the
   shared set of directions, `nothing` in one dimension. =#

struct SWSample
    S::Matrix{Float64}     # n x n_projections, each column sorted
    n::Int
end

# The directions, in the order the projection loop drew them.
function _sw_directions(d::Int, n_projections::Int, seed::Int)
    rng = MersenneTwister(seed)
    Θ = Matrix{Float64}(undef, d, n_projections)
    for k in 1:n_projections
        θ = randn(rng, d); θ ./= norm(θ)
        @inbounds for j in 1:d
            Θ[j, k] = θ[j]
        end
    end
    return Θ
end

function sw_prepare(X::AbstractMatrix, Θ::Union{Nothing,Matrix{Float64}})
    n = size(X, 2)
    K = Θ === nothing ? 1 : size(Θ, 2)
    return sw_prepare!(Matrix{Float64}(undef, n, K), X, Θ)
end

#= The in-place form exists because the table of sorted projections is
   1.6 MB at n = 200 and 1000 directions, and a calibration builds one
   per replicate: allocating it afresh 299 times was the single largest
   source of garbage in a four-dimensional replication, and garbage
   inside a @threads loop is a stop-the-world cost paid by every thread. =#
function sw_prepare!(S::Matrix{Float64}, X::AbstractMatrix,
                     Θ::Union{Nothing,Matrix{Float64}})
    n = size(X, 2)
    if Θ === nothing
        # In one dimension every unit "projection" is +1 or -1, and the
        # quantile contrast is invariant to the sign (the evaluation grid
        # is symmetric), so the average over random projections equals a
        # single evaluation of the 1D Wasserstein-1 distance.
        copyto!(S, X)
        sort!(vec(S))
        return SWSample(S, n)
    end
    # read-only below, so an already-Float64 matrix needs no copy
    samples = X isa Matrix{Float64} ? X : Matrix{Float64}(X)
    #= `theta' * samples` is implemented as `(samples' * theta)'`, so
       projecting straight into the output column with `mul!` is the same
       BLAS call on the same arguments, with the direction vector and the
       projected vector no longer allocated 1000 times per sample.
       Sorting a contiguous view sorts the same values into the same
       unique order. =#
    for k in 1:size(Θ, 2)
        col = view(S, :, k)
        mul!(col, adjoint(samples), view(Θ, :, k))
        sort!(col)
    end
    return SWSample(S, n)
end

#= The evaluation grid depends only on the two sample sizes, never on the
   projection, so it is built once here instead of 1000 times inside the
   loop. =#
function sw_distance(a::SWSample, b::SWSample)
    na = a.n; nb = b.n
    K = size(a.S, 2)
    @assert size(b.S, 2) == K "sliced Wasserstein: projection count mismatch"
    n_eval = max(na, nb)
    ia = Vector{Int}(undef, n_eval); ib = Vector{Int}(undef, n_eval)
    @inbounds for i in 1:n_eval
        t = (i - 0.5) / n_eval
        ia[i] = clamp(ceil(Int, t * na), 1, na)
        ib[i] = clamp(ceil(Int, t * nb), 1, nb)
    end
    total = 0.0
    @inbounds for k in 1:K
        w1 = 0.0
        for i in 1:n_eval
            w1 += abs(a.S[ia[i], k] - b.S[ib[i], k])
        end
        total += w1 / n_eval
    end
    return total / K
end

"""
    SW_NPROJ

The default number of random projection directions, 1000. In the source
repository this was a hardcoded constant that `relfit_compare` could not
reach. Here it is only the default of the `n_projections` keyword, which
every entry point threads through. Changing it changes the reported
numbers, so a run that departs from 1000 is a different run and not a
tuning of the same one.
"""
const SW_NPROJ = 1000

function sliced_wasserstein(A::AbstractMatrix, B::AbstractMatrix;
                            n_projections::Int=SW_NPROJ, seed::Int=42)
    d = size(A, 1)
    @assert size(B, 1) == d "Dimension mismatch"
    Θ = d == 1 ? nothing : _sw_directions(d, n_projections, seed)
    return sw_distance(sw_prepare(A, Θ), sw_prepare(B, Θ))
end

# ── IPM primitive 2: maximum mean discrepancy (Gaussian kernel) ──

# Pairwise squared Euclidean distances between columns of A (d x m) and
# B (d x n): returns an m x n matrix. Uses |x-y|^2 = |x|^2+|y|^2-2 x.y.
function _pairwise_sqdists(A::AbstractMatrix, B::AbstractMatrix)
    sa = vec(sum(abs2, A; dims=1)); sb = vec(sum(abs2, B; dims=1))
    return max.(sa .+ sb' .- 2 .* (A' * B), 0.0)
end

"""
    median_bandwidth(P, Q) -> Float64

Median-heuristic bandwidth: the median pairwise distance over the pooled
samples. Exported because the refit bootstrap and any hand-rolled
calibration need to fix the bandwidth once and reuse it, which is not
possible from outside if this is private.
"""
function median_bandwidth(P::AbstractMatrix, Q::AbstractMatrix)
    pooled = hcat(P, Q); n = size(pooled, 2)
    D2 = _pairwise_sqdists(pooled, pooled)
    vals = Float64[]
    @inbounds for j in 2:n, i in 1:(j - 1)
        push!(vals, sqrt(D2[i, j]))
    end
    med = isempty(vals) ? 1.0 : median(vals)
    return med <= 0 ? 1.0 : med
end

"""
    mmd(P, Q; bandwidth=:median) -> Float64

Unbiased squared maximum mean discrepancy between two empirical
distributions under the Gaussian kernel
`k(x,y) = exp(-|x-y|^2 / (2 sigma^2))`. `P` and `Q` are `d`-rows by
`n`-cols matrices. Lower is closer and zero is identical. The unbiased
U-statistic can go slightly negative, which is harmless here because
relative fit only ever uses the DIFFERENCE of two of them.
"""
#= `pp` and `qq` let a caller supply a within-sample term it has already
   computed. Every calibration in this package holds one of the two
   samples and the bandwidth fixed across hundreds of evaluations, which
   makes that sample's within-term the same number every time;
   substituting the stored value is the identical expression with a
   200 x 200 kernel block not rebuilt. =#
function mmd(P::AbstractMatrix, Q::AbstractMatrix; bandwidth=:median,
             pp=nothing, qq=nothing)
    @assert size(P, 1) == size(Q, 1) "mmd: row (feature) dimension mismatch"
    m = size(P, 2); n = size(Q, 2)
    @assert m >= 2 && n >= 2 "mmd unbiased U-statistic needs >=2 samples per set"
    σ = bandwidth === :median ? median_bandwidth(P, Q) : Float64(bandwidth)
    γ = 1.0 / (2 * σ^2)
    term_pp = pp === nothing ? _mmd_within(P, γ) : pp
    term_qq = qq === nothing ? _mmd_within(Q, γ) : qq
    Kpq = exp.(-γ .* _pairwise_sqdists(P, Q))
    term_pq = sum(Kpq) / (m * n)
    return term_pp + term_qq - 2 * term_pq
end

# The within-sample term of the unbiased U-statistic.
function _mmd_within(P::AbstractMatrix, γ::Float64)
    m = size(P, 2)
    K = exp.(-γ .* _pairwise_sqdists(P, P))
    return (sum(K) - m) / (m * (m - 1))
end

# ── Transforms ───────────────────────────────────────────────────

#= `:rank` is the copula transform: each ROW is replaced by its own tied
   ranks mapped into (0, 1), so a sample keeps its dependence structure
   and loses its marginals entirely. It is applied per sample, which is
   what makes it a copula rather than a common rescaling.

   IT IS OFFERED AND NOT RECOMMENDED, and the reason is a measured
   failure rather than a caveat. A rank transform is marginal-free only
   for CONTINUOUS variables. On count data with many ties, and zeros in
   particular, the tie PATTERN encodes the marginal distribution, so the
   transform does not remove the marginals but re-encodes them in a form
   that can favour one candidate. Measured on a count setting whose
   candidates were matched on their marginals by construction, the
   rejection rate at a genuine tie went to 0.83 against a nominal 0.05,
   with every rejection in the same direction. Do not reach for it on
   discrete data, and do not use it at all without first establishing
   that the tie structure is common to the truth and to every candidate.

   In one dimension it is not merely unhelpful but vacuous: every sample
   ranks to the identical grid, so D0, P0 and P1 become the same numbers
   and the statistic is identically zero. =#
function _apply_transform(X::AbstractMatrix, transform::Symbol)
    transform === :none  && return X
    transform === :log1p && return log1p.(max.(X, 0.0))
    if transform === :rank
        d, n = size(X)
        Z = Matrix{Float64}(undef, d, n)
        @inbounds for i in 1:d
            Z[i, :] = (tiedrank(vec(collect(view(X, i, :)))) .- 0.5) ./ n
        end
        return Z
    end
    error("unknown transform $transform (use :none, :log1p or :rank)")
end

# ── The metric dispatcher and its fixed-sample form ──────────────

#= Both metrics carry a nuisance choice that must be held FIXED across
   the two terms of the contrast and across every calibration replicate,
   or the statistic is not a contrast within one function class and the
   resimulated null is not the null of the statistic actually reported.

   For the MMD that choice is the kernel bandwidth. Taking the median
   heuristic separately for (D0,P0) and for (D0,P1) gives a difference of
   squared distances in two DIFFERENT reproducing-kernel Hilbert spaces.
   It still vanishes when the two predictives coincide, so the null
   survives, but its magnitude and possibly its sign away from
   coincidence are not a clean contrast. Measured on a Gaussian-against-
   Laplace comparison, the two bandwidths were 0.930 and 0.841 and the
   statistic differed from its common-bandwidth value by 17%. A single
   pooled bandwidth also makes the O(1/m) inflation that resampling with
   replacement induces in the within-sample kernel terms common to both
   terms, so it cancels in the difference instead of leaking into the
   bootstrap null.

   For the sliced Wasserstein it is the set of random projections. In one
   dimension the distance is computed exactly and the choice is vacuous,
   but in higher dimensions a hardcoded projection seed shared by every
   replication is a common systematic offset rather than an averaging
   error. One seed per test, reused within the test, is what is wanted. =#

"""
    ipm_distance(ipm, A, B; bandwidth=nothing, sw_seed=42, n_projections=SW_NPROJ)

Dispatch to `sliced_wasserstein` (`ipm = :sw`) or `mmd` (`ipm = :mmd`).
"""
function ipm_distance(ipm::Symbol, A::AbstractMatrix, B::AbstractMatrix;
                      bandwidth=nothing, sw_seed::Int=42,
                      n_projections::Int=SW_NPROJ)
    ipm === :sw  && return sliced_wasserstein(A, B; seed=sw_seed,
                                              n_projections=n_projections)
    ipm === :mmd && return mmd(A, B; bandwidth = bandwidth === nothing ? :median : bandwidth)
    error("unknown ipm $ipm (use :sw or :mmd)")
end

"""
    SWScratch

Everything a sliced-Wasserstein contrast can reuse across replicates: the
directions, which depend only on the dimension and the seed, and the two
tables of sorted projections. The bootstrap changes its fixed sample on
every replicate, so without somewhere to keep these it would redraw 1000
directions and reallocate 1.6 MB tables 299 times over.

Exported because `refit_bootstrap` and any user-written calibration need
it, and because a user who cannot construct one cannot write the
calibration this package recommends.
"""
mutable struct SWScratch
    Θ::Union{Nothing,Matrix{Float64}}
    K::Int
    Sfix::Matrix{Float64}
    Svar::Matrix{Float64}
end

"""
    sw_scratch(d, sw_seed; n_projections=SW_NPROJ) -> SWScratch

Allocate the reusable sliced-Wasserstein workspace for `d`-dimensional
data at projection seed `sw_seed`.
"""
function sw_scratch(d::Int, sw_seed::Int; n_projections::Int=SW_NPROJ)
    Θ = d == 1 ? nothing : _sw_directions(d, n_projections, sw_seed)
    K = Θ === nothing ? 1 : n_projections
    return SWScratch(Θ, K, Matrix{Float64}(undef, 0, K), Matrix{Float64}(undef, 0, K))
end

function _sw_fill!(sc::SWScratch, X::AbstractMatrix, which::Symbol)
    n = size(X, 2)
    S = which === :fix ? sc.Sfix : sc.Svar
    if size(S, 1) != n
        S = Matrix{Float64}(undef, n, sc.K)
        which === :fix ? (sc.Sfix = S) : (sc.Svar = S)
    end
    return sw_prepare!(S, X, sc.Θ)
end

"""
    rho_against_fixed(ipm, fixed; bandwidth=nothing, sw_seed=42,
                      fixed_first=true, scratch=nothing,
                      n_projections=SW_NPROJ) -> Function

Return the closure `X -> rho(fixed, X)` with the fixed side's projections
and sorts done once. `fixed_first` says which argument the fixed sample
occupies, so the call reproduces the original argument order exactly
rather than relying on the metrics being symmetric. For the MMD there is
nothing to precompute beyond the fixed sample's within-term.

Exported for the same reason as `SWScratch`: every calibration in this
package is written against it, including the one it recommends, so a user
extending the rule needs it.
"""
function rho_against_fixed(ipm::Symbol, fixed::AbstractMatrix;
                           bandwidth=nothing, sw_seed::Int=42,
                           fixed_first::Bool=true, scratch=nothing,
                           n_projections::Int=SW_NPROJ)
    if ipm === :sw
        sc = scratch === nothing ?
             sw_scratch(size(fixed, 1), sw_seed; n_projections=n_projections) : scratch
        f = _sw_fill!(sc, fixed, :fix)
        return fixed_first ? (X -> sw_distance(f, _sw_fill!(sc, X, :var))) :
                             (X -> sw_distance(_sw_fill!(sc, X, :var), f))
    end
    if ipm === :mmd && bandwidth !== nothing
        # the fixed sample's within-term, computed once
        w = _mmd_within(fixed, 1.0 / (2 * Float64(bandwidth)^2))
        return fixed_first ? (X -> mmd(fixed, X; bandwidth=bandwidth, pp=w)) :
                             (X -> mmd(X, fixed; bandwidth=bandwidth, qq=w))
    end
    return fixed_first ?
        (X -> ipm_distance(ipm, fixed, X; bandwidth=bandwidth, sw_seed=sw_seed,
                           n_projections=n_projections)) :
        (X -> ipm_distance(ipm, X, fixed; bandwidth=bandwidth, sw_seed=sw_seed,
                           n_projections=n_projections))
end

"""
    common_bandwidth(ipm, D0, P0, P1) -> Union{Nothing,Float64}

The single nuisance bandwidth used by both terms of the relative-fit
contrast and by every calibration replicate: the median heuristic over
the pooled held-out data and both simulated samples. Returns `nothing`
for the sliced Wasserstein, which has no such parameter.
"""
function common_bandwidth(ipm::Symbol, D0::AbstractMatrix,
                          P0::AbstractMatrix, P1::AbstractMatrix)
    ipm === :mmd || return nothing
    return median_bandwidth(D0, hcat(P0, P1))
end

"""
    common_bandwidth_K(ipm, D0, P) -> Union{Nothing,Float64}

The `K`-candidate form: the median heuristic over the held-out data
pooled with all `K` simulated samples. At `K = 2` this is
`common_bandwidth(ipm, D0, P[1], P[2])`.
"""
function common_bandwidth_K(ipm::Symbol, D0::AbstractMatrix,
                            P::AbstractVector{<:AbstractMatrix})
    ipm === :mmd || return nothing
    return median_bandwidth(D0, hcat(P...))
end
