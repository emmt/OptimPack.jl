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

* Brent's methods:
  * `fmin` for minimizing a function of one variable.
  * `fzero` for finding the root of a function of one variable.

* `bradi` for finding the global minimum of a function of one variable.

"""
module OptimPack

export
    fmin,
    fzero,
    # FIXME nlcg,
    # FIXME spg2,
    # FIXME vmlmb,

    # Powell methods.
    Cobyla, cobyla, cobyla!,
    Bobyqa, bobyqa, bobyqa!,
    Newuoa, newuoa, newuoa!,

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

import Base:
    ENV,
    checkbounds,
    copy,
    copyto!,
    eltype,
    fill!,
    getindex,
    length,
    ndims,
    setindex!,
    size

import LinearAlgebra: issuccess

# FIXME _path_to_deps_jl = joinpath(@__DIR__, "..", "deps", "deps.jl")
# FIXME isfile(_path_to_deps_jl) ||
# FIXME     error("OptimPack not properly installed.  Please run Pkg.build(\"OptimPack\")")
# FIXME include(_path_to_deps_jl)

# FIXME # Load pieces of code.
# FIXME include("bindings.jl")

include("brent.jl")
import .Brent: fzero, fmin

# FIXME include("spg2.jl")
# FIXME import .SPG: spg2

include("bradi.jl")
@deprecate fmin_global BraDi.minimize

include("powell.jl")
import .Powell:
    Cobyla, cobyla, cobyla!,
    Bobyqa, bobyqa, bobyqa!,
    Newuoa, newuoa, newuoa!

end # module
