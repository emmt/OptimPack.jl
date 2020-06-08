#
# OptimPack.jl --
#
# Julia wrapper for OptimPack.
#
#-------------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2014-2020, Éric Thiébaut.
#

module OptimPack

export
    fmin,
    fzero,
    nlcg,
    spg2,
    vmlmb

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

import LinearAlgebra: dot

_path_to_deps_jl = joinpath(@__DIR__, "..", "deps", "deps.jl")
isfile(_path_to_deps_jl) ||
    error("OptimPack not properly installed.  Please run Pkg.build(\"OptimPack\")")
include(_path_to_deps_jl)

# Load pieces of code.
include("bindings.jl")
include("brent.jl")
include("bradi.jl")
include("powell.jl")
include("spg2.jl")
import .SPG: spg2
import .Brent: fzero, fmin
@deprecate fmin_global BraDi.minimize

end # module
