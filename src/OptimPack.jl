#
# OptimPack.jl --
#
# Julia wrapper for OptimPack.
#
# ----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT "Expat"
# License:
#
# Copyright (C) 2014-2019, Éric Thiébaut.
#
# ----------------------------------------------------------------------------

module OptimPack

export
    fmin,
    fmin_global,
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

isfile(joinpath(@__DIR__,"..","deps","deps.jl")) ||
    error("OptimPack not properly installed.  Please run Pkg.build(\"OptimPack\")")
include("../deps/deps.jl")

# Load pieces of code.
include("bindings.jl")
include("Brent.jl")
include("powell.jl")
include("spg2.jl")

end # module
