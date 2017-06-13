#
# OptimPack.jl --
#
# A package of optimization methods for Julia.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT "Expat"
# License:
#
# Copyright (C) 2014-2017, Éric Thiébaut.
#

isdefined(Base, :__precompile__) && __precompile__(true)

module OptimPack

export nlcg, vmlmb, spg2

export fzero, fmin, fmin_global

# Load other components.
include("clib.jl")
include("brent.jl")
include("bradi.jl")
include("powell.jl")
include("spg.jl")
include("deprecations.jl")

# Provide some aliases for popular algorithms.
import .Brent: fmin, fzero
import .CLib: Float, vmlmb, nlcg

end # module
