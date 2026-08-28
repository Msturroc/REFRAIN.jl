"""
    REFRAIN

Likelihood-free model comparison that is allowed to say "I cannot tell".

Instead of asking "is model M true?", which is the Bayes-factor question
and which destabilises when every candidate is wrong, REFRAIN inverts a
split-sample test of RELATIVE fit: "which candidate is closer to the data
under a chosen divergence?". The answer is allowed to be "cannot tell
which", i.e. the rule can ABSTAIN.

Five steps, and the package is a thin layer over each:

  1. split the i.i.d. units (columns) into a fit half `D1` and a held-out
     half `D0`
  2. fit each candidate on `D1` by ABC
  3. simulate fresh data `P0`, `P1` from the two fits
  4. `T = rho(D0, P0) - rho(D0, P1)` on the held-out half, `rho` a
     sample-only integral probability metric
  5. calibrate `T` against the equidistance null and read the three-way
     decision off the interval

TWO WARNINGS THE PROJECT PAID FOR, both stated at length in the README.

ABSTENTION IS DISCRIMINABILITY, NOT ADEQUACY. It means the data cannot
resolve which candidate is closer, not that neither is any good. Read
`res.screen` beside `res.decision` and never fold one into the other.

USE THE REFIT BOOTSTRAP WHEN THE FIT IS EXPENSIVE IN PARAMETERS. The
plug-in interval holds `theta_hat` fixed and so is too narrow by whatever
the fit uncertainty is worth: measured at 0.97 to 1.06 with one fitted
parameter, 0.83 to 1.15 with two, and 1.16 to 1.46 with five, where it
then rejects at about twice its nominal rate. Raising `n` does not fix
it, because both terms scale as `n^{-1/2}` and the ratio is invariant.

Entry points: `refrain` end to end, `decide` for a reader who already has
the samples.
"""
module REFRAIN

using Distributions, LinearAlgebra, Statistics, Random, StatsBase, Printf

# metrics
export sliced_wasserstein, mmd, median_bandwidth, common_bandwidth,
       common_bandwidth_K, ipm_distance, rho_against_fixed,
       SWScratch, sw_scratch, sw_prepare, sw_prepare!, sw_distance, SWSample,
       SW_NPROJ

# statistic, calibrations and diagnostics
export relfit_statistic, permutation_calibrate, bootstrap_calibrate,
       bootstrap_calibrate_block, block_indices,
       percentile_interval, refit_deviations, refit_bootstrap, width_ratio,
       hoeffding_mmd_test, degeneracy_scaling

# decision layer
export refrain, refrain_full, decide, RefrainResult,
       adequacy_screen, adequacy_label, split_iid, split_contiguous,
       relfit_distances_K, permutation_calibrate_K, bootstrap_calibrate_K,
       permutation_set_K, relfit_compare_K

# models and fitting
export RelFitModel, abc_posterior, abc_fit, simulate_from_fit,
       simulate_posterior_predictive

# ABC layer
export rejection_abc, abc_smc, apmc, abc_smc_model_choice, bayes_model_prob,
       RejectionResult, ToniResult, APMCResult, ModelChoiceResult,
       weighted_covariance, toni_weights!, psis_smooth!,
       effective_sample_size, systematic_resample

#= LOAD ORDER MATTERS AND IS NOT ALPHABETICAL. `metrics` depends on
   nothing. `abc` depends on nothing above it. `model` needs the samplers
   for `abc_posterior`. `calibrate` needs `RelFitModel` and `abc_fit` for
   the refit bootstrap. `decide` needs all of them.

   In the source repository this dependency was expressed in PROSE rather
   than in code, and the file it named did not exist. Making it real is
   most of what packaging buys. =#
include("metrics.jl")
include("abc.jl")
include("model.jl")
include("calibrate.jl")
include("decide.jl")
include("modelchoice.jl")

#= THE RNG SPLIT IS DELIBERATE AND MUST NOT BE UNIFIED. The samplers use
   `Xoshiro`, everything above them uses `MersenneTwister`. It is what the
   published numbers were computed with, and `test/identity.jl` fails on
   any change to it. There is no statistical reason to prefer one, and
   every reason not to touch a stream that a stored digest depends on. =#

end # module
