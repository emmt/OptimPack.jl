"""

Package `OptimPack` provides numerical optimization methods.

* Large scale optimization:
  * `vmlmb`
  * `spg2`
  * `conjgrad`

* Nelder & Mead *Simplex* method.

* Derivative free Powell's methods:
  * `newuoa`
  * `bobyqa`
  * `cobyla`

* Univariate functions:

  * Brent's methods:

    * `fmax` for maximizing a univariate function in an given open interval.

    * `fmin` for minimizing a univariate function in an given open interval.

    * `fzero` for finding the root of a univariate a function in a given interval.

  * Global optimization:

    * `BraDi.minimize` and `BraDi.maximize` for finding the global minimum or maximum of a
      univariate function by the *"Bracket then Dig"* algorithm.

    * `Step.minimize` and `Step.maximize` for finding the global minimum or maximum of a
      univariate function by the *"Select The Easiest Point"* algorithm.

"""
module OptimPack

export
    # Bracket then Dig algorithm.
    BraDi,

    # S.T.E.P. algorithm.
    Step,

    # Brent methods.
    fmax,
    fmaxbrkt,
    fmin,
    fminbrkt,
    fzero,

    # Nelder & Mead "Simplex" method.
    Simplex,
    simplex,

    # Spectral Projection Gradient.
    SPG,
    spg,
    spg!,

    #nlcg,
    #vmlmb,
    #
    ## Powell methods.
    Cobyla, cobyla, cobyla!,
    Bobyqa, bobyqa, bobyqa!,
    Newuoa, newuoa, newuoa!,

    # Re-export from `LinearAlgebra`.
    issuccess

# Public but not exported API.
using TypeUtils: @public
@public @dispatch_on_multiplier,
        BoundedSet,
        ConvexSet,
        Problems,
        adapt_multiplier_precision,
        axpby!,
        configure!,
        has_contraints,
        inner,
        iterate!,
        line_search_limits,
        line_search_step_max,
        mult!,
        one_norm,
        project_direction!,
        project_variables!,
        recode!,
        recode,
        restart!,
        scale!,
        solve!,
        sup_norm,
        two_norm,
        unblocked_variables!,
        xpby!

using LinearAlgebra
using Neutrals
using Printf
using TypeUtils
using Unitful
using Unitful: AbstractQuantity

using Base:
    @propagate_inbounds,
    OneTo,
    axes1,
    elsize,
    tail

import LinearAlgebra: issuccess

if !isdefined(Base, :get_extension)
    using Requires
end

# TODO LoopStyles could be an independent package.
include("LoopStyles.jl")
using .LoopStyles
@public LoopStyle,
        LoopStyles,
        LoopStyleDot,
        LoopStyleFor,
        LoopStyleGPU,
        LoopStyleInBounds,
        LoopStyleMap,
        LoopStyleSIMD,
        LoopStyleTurbo

include("macros.jl")
include("types.jl")
include("API.jl")
include("common.jl")
include("vectops.jl")

include("BoundedSets.jl")
import .BoundedSets: BoundedSet

include("Brent.jl")
import .Brent: fmax, fmaxbrkt, fmin, fminbrkt, fzero

include("BraDi.jl")
include("Step.jl")

include("SPG.jl")
import .SPG: spg, spg!

include("Simplex.jl")
import .Simplex: simplex

include("Powell.jl")
import .Powell:
    Cobyla, cobyla, cobyla!,
    Bobyqa, bobyqa, bobyqa!,
    Newuoa, newuoa, newuoa!

include("Problems.jl")

function __init__()
    @static if !isdefined(Base, :get_extension)
        # Extend methods when other packages are loaded.
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" =  include(
            "../ext/OptimPackCUDAExt.jl")
        @require CUTEst = "1b53aba6-35b6-5f92-a507-53c67d53f819" include(
            "../ext/OptimPackCUTEstExt.jl")
        @require LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890" include(
            "../ext/OptimPackLoopVectorizationExt.jl")
        @require OptimPack_jll = "8115cc2e-fb29-5d71-b5cb-a4fb1c5dcd4c" include(
            "../ext/OptimPackOptimPack_jllExt.jl")
    end
end

end # module
