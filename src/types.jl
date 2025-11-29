const StaticMultiplier{v} = Union{Neutral{v},AbstractQuantity{Neutral{v}}}

if !isdefined(@__MODULE__, :Memory)
    const Memory{T} = Vector{T}
end
