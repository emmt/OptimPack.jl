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

# Functions must be imported to be extended with new methods.
import Base: ENV, size, length, eltype, ndims, copy, copyto!, fill!
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
