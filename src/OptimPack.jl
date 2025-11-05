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

    #nlcg,
    #spg2,
    #vmlmb,
    #
    ## Powell methods.
    #Cobyla, cobyla, cobyla!,
    #Bobyqa, bobyqa, bobyqa!,
    #Newuoa, newuoa, newuoa!,
    # Re-export from `LinearAlgebra`.
    issuccess


using LinearAlgebra, Printf

using Base:
    @propagate_inbounds,
    OneTo,
    axes1,
    elsize,
    tail,
    throw_boundserror

import LinearAlgebra: issuccess

# FIXME _path_to_deps_jl = joinpath(@__DIR__, "..", "deps", "deps.jl")
# FIXME isfile(_path_to_deps_jl) ||
# FIXME     error("OptimPack not properly installed.  Please run Pkg.build(\"OptimPack\")")
# FIXME include(_path_to_deps_jl)

# FIXME # Load pieces of code.
# FIXME include("bindings.jl")

include("Brent.jl")
import .Brent: fmax, fmaxbrkt, fmin, fminbrkt, fzero

include("BraDi.jl")
include("Step.jl")

# FIXME include("spg2.jl")
# FIXME import .SPG: spg2


#include("powell.jl")
#import .Powell:
#    Cobyla, cobyla, cobyla!,
#    Bobyqa, bobyqa, bobyqa!,
#    Newuoa, newuoa, newuoa!

end # module
