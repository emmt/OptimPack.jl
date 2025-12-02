const StaticMultiplier{v} = Union{Neutral{v},AbstractQuantity{Neutral{v}}}

if !isdefined(@__MODULE__, :Memory)
    const Memory{T} = Vector{T}
end

const PlusMinus = Union{typeof(+),typeof(-)}

"""
    OptimPack.ConvexSet

Abstract type inherited by types representing convex sets of feasible variables.

"""
abstract type ConvexSet end
