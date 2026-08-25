#= ================================================================
   Bit-identity against the digests stored in the repository the paper's
   numbers were computed in.

   This is the answer to "is this the code that produced the results".
   It costs seconds where re-running a study costs an hour and a half,
   and it is what keeps a PORT from silently becoming a different method.

   HOW IT WORKS. Every Float64 goes through `reinterpret`, so the SHA-256
   sees the IEEE bit pattern and no decimal rounding sits between the
   computation and the check. The reference strings below are copied
   verbatim from `perf_identity.jl` of the abstention_model_choice
   repository. They are not regenerated here and must not be: a
   regeneration to make a failing gate pass defeats the whole file.

   THREE FACTS IT ENCODES, all load-bearing and none obvious.

   1. THE RNG SPLIT IS DELIBERATE. The samplers use `Xoshiro`, everything
      above them uses `MersenneTwister`. Do not unify them. There is no
      statistical reason to prefer one and every reason not to touch a
      stream a stored digest depends on.

   2. BOUNDS CHECKING CHANGES THE LAST TWO DIGITS, and this file found it
      rather than assuming it. `Pkg.test` runs with `--check-bounds=yes`,
      which disables every `@inbounds` and with it the SIMD vectorisation
      of the reductions inside `_pairwise_sqdists` and `_mmd_within`. The
      maximum mean discrepancy in four dimensions then reads
      0.025417205363256423 against the published 0.025417205363256645, a
      difference in the sixteenth significant figure that is nonetheless
      a different SHA. The sliced Wasserstein is unaffected, since its
      per-projection loop is a plain scalar accumulation.

      So bit-identity is only meaningful at the same bounds-checking
      setting, and this file RE-EXECUTES ITSELF in a subprocess at
      `--check-bounds=auto` when it detects otherwise. That is the honest
      fix. Loosening the comparison to a tolerance would turn a gate into
      a formality.

   3. THE BLAS THREAD COUNT IS PART OF A PUBLISHED FIT. The
      `(C .* w') * C'` in `weighted_covariance` is a gemm, and OpenBLAS
      accumulates it differently at different thread counts: on a
      seven-parameter fit the whole posterior moved when it changed.
      MEASURED, the checks in this file are NOT sensitive to it, since
      none of them runs the sampler and the gemm in `_pairwise_sqdists`
      reproduces at 1, 8 and 16 threads alike. The pin below is therefore
      belt and braces, set so that a check added later inherits a pinned
      count rather than the machine's, and any added check that fits by
      ABC must keep it.

   The inputs are frozen bit patterns in `identity_data.jl` rather than
   simulator calls, because REFRAIN.jl ships no gene-expression or
   rainfall model. See that file's header.
   ================================================================ =#

using REFRAIN, Test, SHA, LinearAlgebra, Random

#= See note 2. `check_bounds` is 0 for auto, 1 for yes, 2 for no. Only 1
   changes the arithmetic, and only that case needs the subprocess. =#
if Base.JLOptions().check_bounds == 1
    @info "identity.jl: re-running at --check-bounds=auto, see note 2 in this file"
    _proj = dirname(@__DIR__)
    _cmd = `$(Base.julia_cmd()) --check-bounds=auto --project=$(_proj) $(@__FILE__)`
    @testset "bit-identity (delegated to a bounds-checking-clean subprocess)" begin
        @test success(pipeline(_cmd; stdout=stdout, stderr=stderr))
    end
else

include("identity_data.jl")

# See note 3 above.
const _BLAS_PIN = 1
BLAS.set_num_threads(_BLAS_PIN)

# ── exact serialisation, copied from perf_identity.jl ────────────

_ser(x::Float64)       = string(reinterpret(UInt64, x); base=16, pad=16)
_ser(x::Float32)       = string(reinterpret(UInt32, x); base=16, pad=8)
_ser(x::Integer)       = string(x)
_ser(x::Bool)          = x ? "T" : "F"
_ser(x::Symbol)        = ":" * String(x)
_ser(x::AbstractString) = String(x)
_ser(::Nothing)        = "nothing"
_ser(x::AbstractArray) = string(size(x), "[", join((_ser(v) for v in x), ","), "]")
_ser(x::Tuple)         = "(" * join((_ser(v) for v in x), ";") * ")"
_ser(x::NamedTuple)    = "(" * join((string(k, "=", _ser(getfield(x, k))) for k in keys(x)), ";") * ")"

digest(x) = bytes2hex(sha256(_ser(x)))

#= Reference digests, verbatim from `perf_identity.jl`. Only the entries
   whose inputs are frozen here appear: the setting-specific simulator and
   ABC-fit digests need model code this package does not ship, and the
   source repository remains their home. =#
const REFERENCE = Dict{String,String}(
    "sw_d1"                   => "4b0ac6d7ea4443e5997c7dac581678a642abaf898fcfca2988f7acc0b9a63423",
    "sw_d4"                   => "2316525df28b229e135ea575ff47773ee005fdda814a61fa644c7af52ae0d987",
    "sw_d4_seed"              => "e78151ee63fc0c6564e4cc3c2d0874271d8e69b2f1a7306366d373cc2c65dbe8",
    "sw_d4_ragged"            => "971273ab104402f7744e6b580d8c45e41be70747945e10aa851c5f6c2e5683a9",
    "mmd_d1"                  => "eeeeba9a8167fe1c0c319f96e5faf31e830d9bb04de1ef3799e89eca173fc470",
    "mmd_d4"                  => "a1719a44e28cc8520facb2dc5d256754a45a2995a6b85d22f7f1e59a4b1217e7",
    "median_bandwidth_d4"     => "35249069511faf8be36f38f639eb47edd677fc3a1588bd10a50c50367a7aa2db",
    "perm_sw_d4"              => "f1af6a35fd7c1f1e86bbd3b771f11c7719efb4ec53850e244dad4f2932ef8795",
    "perm_mmd_d4"             => "633bdde2bd051d97aef9387fe615cd8704c14a83c4a2e960ecfe0943bfe7ba99",
    "boot_sw_d4"              => "9af51424513f53fff45aa771bda73a2dd1be08e8ea23b3e581f9db02ef0ca512",
    "boot_mmd_d4"             => "62555758902cbc35dc3d0eb2aa90c742085900744c00e3c616d8cfd1011366e0",
    "mofn_sw_d4"              => "335c83a325e4df795050ee05d77ba21850c1cdc668bddcc8cee9d44dc8b5c87b",
    "mofn_mmd_d4"             => "76a8e5516fa2247bf4351ff20ea3a7281043b4ce9b6a464b2e2e068526ce9e6c",
    "hoeffding_d4"            => "49af1e331df17a0d7221dec993ceca76e43a194d7fe31a0a5cf2e98f08632a26",
    "relfit_distances_K2_sw"  => "52df8d7aa94a6db586a0239a74ca411320a0f4c351b2ddb7a4ccf94c2e3cb6e4",
    "relfit_distances_K2_mmd" => "8b25e06e11ff73bb5e41be859d69c40b574dddb3b67e2239585c8c407657aa8f",
    "relfit_distances_K5_sw"  => "d3e8bcbfd2fd71099564643fddaf5d052d27aa219b6a92f6e1098d9de424b4c4",
    "relfit_distances_K5_mmd" => "3a4ea044ade88b2ac5ca7b495e62e8d639d9b84a8522522d13fc77a01c112883",
    "common_bandwidth_K5"     => "89e17c9e8d3b1d4f1fdd31f8c88f7c1b3f4eb622fad7e518bb7015643cff3f51",
    "perm_K5_sw"              => "aa0d6a2559f28c01f83ce211361ffbe8525dac8ae1d1dad6bb84a1f5f6645a18",
    "perm_K5_mmd"             => "f869858261a7651a197156bca8c512acb9971a0c1d5ba5f6b5b139d62777622a",
    "boot_K5_sw"              => "253d894cfb9a3d32b0c1d8974565ac3b7fdfe65e85531f0ebaf80d02fd7ce359",
    "boot_K5_mmd"             => "53ce3e8375c5180220020ef5a2747a3f566e114b2f18553df9c96025b964135f",
    "boot_K5_sw_mofn"         => "9d6b0ae003088842b0864f575d534a3f719c536ea6997036b749bb31cc8c188c",
    "perm_set_K5_sw"          => "32dc27ee2b1f4b488d212b6592e8e6bf44332ef65b42a7e3285a6ecaa8b10acf",
)

const DC0 = DC0_DATA
const DD0 = DD0_DATA
const PD  = PD_ALL

# The checks, in the order and at the arguments perf_identity.jl uses.
const CHECKS = Pair{String,Function}[
    # 4. both IPMs, in d = 1 (the exact branch) and d = 4 (the projection
    #    branch, which is where a single matmul would differ from 1000
    #    matvecs if it differed at all)
    "sw_d1"  => () -> sliced_wasserstein(P1a, P1b),
    "sw_d4"  => () -> sliced_wasserstein(P4a, P4b),
    "sw_d4_seed"  => () -> [sliced_wasserstein(P4a, P4b; seed=s) for s in 1:5],
    "sw_d4_ragged" => () -> sliced_wasserstein(P4a[:, 1:150], P4b),
    "mmd_d1" => () -> mmd(P1a, P1b),
    "mmd_d4" => () -> mmd(P4a, P4b),
    "median_bandwidth_d4" => () -> median_bandwidth(P4a, P4b),

    # 5. the two resampling calibrations and the concentration threshold
    "perm_sw_d4" => () ->
        permutation_calibrate(DC0, P4a, P4b; n_perm=49, ipm=:sw, seed=31, sw_seed=61),
    "perm_mmd_d4" => () ->
        permutation_calibrate(DC0, P4a, P4b; n_perm=49, ipm=:mmd, seed=31, sw_seed=61),
    "boot_sw_d4" => () ->
        bootstrap_calibrate(DC0, P4a, P4b; n_boot=49, ipm=:sw, seed=51, sw_seed=61),
    "boot_mmd_d4" => () ->
        bootstrap_calibrate(DC0, P4a, P4b; n_boot=49, ipm=:mmd, seed=51, sw_seed=61),
    #= The m-out-of-n branch at the primary m = n0^(2/3). Its purpose here
       is to fail if the rescaling or the resample sizes are ever touched.
       The branch it does NOT exercise, m = n, is covered by the two
       checks above, which is the point of keeping them separate. =#
    "mofn_sw_d4" => () ->
        bootstrap_calibrate(DC0, P4a, P4b; n_boot=49, ipm=:sw, seed=51, sw_seed=61, m=34),
    "mofn_mmd_d4" => () ->
        bootstrap_calibrate(DC0, P4a, P4b; n_boot=49, ipm=:mmd, seed=51, sw_seed=61, m=34),
    "hoeffding_d4" => () -> hoeffding_mmd_test(DC0, P4a, P4b; alpha=0.05),

    #= 6. the K-candidate extension. The first two entries are the
       load-bearing ones: `relfit_distances_K` at K = 2 must DIFFERENCE to
       `relfit_statistic`, and every bootstrap replicate at K = 2 must
       difference to the corresponding replicate of `bootstrap_calibrate`.
       `runtests.jl` asserts both as equalities, and digesting the K = 2
       outputs here means this gate fails if either SIDE of that equality
       moves, not only if they move apart. =#
    "relfit_distances_K2_sw" => () ->
        relfit_distances_K(DD0, PD[1:2]; ipm=:sw, sw_seed=7),
    "relfit_distances_K2_mmd" => () ->
        relfit_distances_K(DD0, PD[1:2]; ipm=:mmd, sw_seed=7),
    "relfit_distances_K5_sw"  => () ->
        relfit_distances_K(DD0, PD; ipm=:sw, sw_seed=7),
    "relfit_distances_K5_mmd" => () ->
        relfit_distances_K(DD0, PD; ipm=:mmd, sw_seed=7),
    "common_bandwidth_K5" => () -> common_bandwidth_K(:mmd, DD0, PD),
    "perm_K5_sw" => () ->
        permutation_calibrate_K(DD0, PD; n_perm=49, ipm=:sw, seed=5, sw_seed=7),
    "perm_K5_mmd" => () ->
        permutation_calibrate_K(DD0, PD; n_perm=49, ipm=:mmd, seed=5, sw_seed=7),
    "boot_K5_sw" => () -> (b = bootstrap_calibrate_K(DD0, PD; n_boot=99, ipm=:sw,
                                                     seed=5, sw_seed=7, m=nothing);
                           (b.r_obs, b.set, b.p_mcs, b.p_steps, b.eliminated, b.R_boot)),
    "boot_K5_mmd" => () -> (b = bootstrap_calibrate_K(DD0, PD; n_boot=99, ipm=:mmd,
                                                      seed=5, sw_seed=7, m=nothing);
                            (b.r_obs, b.set, b.p_mcs, b.p_steps, b.eliminated, b.R_boot)),
    "boot_K5_sw_mofn" => () -> (b = bootstrap_calibrate_K(DD0, PD; n_boot=99, ipm=:sw,
                                                          seed=5, sw_seed=7, m=60);
                                (b.r_obs, b.set, b.p_mcs, b.p_steps, b.eliminated, b.R_boot)),
    "perm_set_K5_sw"  => () ->
        permutation_set_K(DD0, PD; n_perm=49, ipm=:sw, seed=5, sw_seed=7, alpha=0.05),
]

@testset "bit-identity against the published digests" begin
    @test length(CHECKS) == length(REFERENCE)
    for (name, f) in CHECKS
        got = digest(f())
        want = REFERENCE[name]
        #= A failure here is never "the test is wrong": either the
           arithmetic moved or the inputs did. The name is carried into
           the comparison so the report says which check, not just that
           two hex strings differ. =#
        @test (name, got) == (name, want)
    end
end

end # the check-bounds branch opened near the top of this file
