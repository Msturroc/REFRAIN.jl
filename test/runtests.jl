#= ================================================================
   REFRAIN.jl test suite.

     identity.jl        bit-identity against the digests stored in the
                        repository the paper's numbers were computed in.
                        This is the one that answers "is this the code
                        that produced the results".
     sanity_core.jl     the primitives against closed forms
     sanity_k.jl        the K-candidate extension, half of it the K = 2
                        bit-for-bit reduction demand
     sanity_refit.jl    the refit bootstrap and its control arm
     sanity_baseline.jl the ABC layer and the model-indicator baseline
                        against configurations with a known answer
     end_to_end.jl      the two entry points on a real comparison

   `REFRAIN_SLOW=1` runs the Monte Carlo sections at the replication
   counts the source suites use outside their FAST mode, which tightens
   every tolerance. The default is sized to run in a minute or two.
   ================================================================ =#

using Test

@testset "REFRAIN" begin
    @testset "identity" begin include("identity.jl") end
    @testset "primitives" begin include("sanity_core.jl") end
    @testset "K candidates" begin include("sanity_k.jl") end
    @testset "refit bootstrap" begin include("sanity_refit.jl") end
    @testset "ABC and the baseline" begin include("sanity_baseline.jl") end
    @testset "end to end" begin include("end_to_end.jl") end
end
