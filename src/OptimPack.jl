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
@public Problems configure! solve! restart! iterate!

using LinearAlgebra, Printf

using Base:
    @propagate_inbounds,
    OneTo,
    axes1,
    elsize,
    tail,
    throw_boundserror

import LinearAlgebra: issuccess

if !isdefined(Base, :get_extension)
    using Requires
end

include("common.jl")

include("Brent.jl")
import .Brent: fmax, fmaxbrkt, fmin, fminbrkt, fzero

include("BraDi.jl")
include("Step.jl")

include("SPG.jl")
import .SPG: spg, spg!

include("Powell.jl")
import .Powell:
    Cobyla, cobyla, cobyla!,
    Bobyqa, bobyqa, bobyqa!,
    Newuoa, newuoa, newuoa!

include("Problems.jl")

function __init__()
    @static if !isdefined(Base, :get_extension)
        # Extend methods when other packages are loaded.
        @require OptimPack_jll = "8115cc2e-fb29-5d71-b5cb-a4fb1c5dcd4c" include(
            "../ext/OptimPackOptimPack_jll.jl")
    end
end

end # module
