#
# benchmarks.jl --
#
# Perform a few benchmark tests on OptimPack arrays.
#
# ----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT "Expat"
# License:
#
# Copyright (C) 2014-2019, Éric Thiébaut.
#
# ----------------------------------------------------------------------------


module OptimPackBenchmarks

using OptimPack
using BenchmarkTools

function slowvnorm2(x::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    res = zero(T)
    for i in eachindex(x)
        res += x[i]^2
    end
    return sqrt(res)
end

function slowvdot(x::AbstractArray{T,N},
                  y::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    @assert axes(x) == axes(y)
    res = zero(T)
    for i in eachindex(x, y)
        res += x[i]*y[i]
    end
    return res
end

function slowvcopy!(dst::AbstractArray{T,N},
                    src::AbstractArray{T,N}) where {T,N}
    @assert axes(dst) == axes(src)
    for i in eachindex(dst, src)
        dst[i] = src[i]
    end
    return dst
end

function vnorm2(x::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    res = zero(T)
    @inbounds @simd for i in eachindex(x)
        res += x[i]^2
    end
    return sqrt(res)
end

function vdot(x::AbstractArray{T,N},
              y::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    @assert axes(x) == axes(y)
    res = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        res += x[i]*y[i]
    end
    return res
end

function vcopy!(dst::AbstractArray{T,N},
                src::AbstractArray{T,N}) where {T,N}
    @assert axes(dst) == axes(src)
    @inbounds @simd for i in eachindex(dst, src)
        dst[i] = src[i]
    end
    return dst
end

T = Float64
dims = (20,30,40)
len = prod(dims)

x = randn(T, dims)
y = randn(T, dims)
z = randn(T, dims)

E = OptimPack.DenseVariableSpace(T, dims)
vx = vcopy!(OptimPack.create(E), x)
vy = vcopy!(OptimPack.create(E), y)
vz = vcopy!(OptimPack.create(E), z)

@assert vnorm2(vz) == vnorm2(z)
@assert vnorm2(vz) ≈ OptimPack.norm2(vz)
@assert vdot(vx,vy) == vdot(x,y)
@assert vdot(vx,vy) ≈ OptimPack.dot(vx,vy)

println("\nSlow versions (with bound checking):")
println("  VNORM2 of $len elements:")
print("    Julia arrays ------------->")
@btime slowvnorm2($z)
print("    DenseVariable arrays ----->")
@btime slowvnorm2($vz)
println("  VDOT of $len elements:")
print("    Julia arrays ------------->")
@btime slowvdot($x,$y)
print("    DenseVariable arrays ----->")
@btime slowvdot($vx,$vy)
print("    mixed type arrays -------->")
@btime slowvdot($x,$vy)
print("    mixed type arrays -------->")
@btime slowvdot($vx,$y)

println("\nFast versions (no bound checking and SIMD):")
println("  VNORM2 of $len elements:")
print("    Julia arrays ------------->")
@btime vnorm2($z)
print("    DenseVariable arrays ----->")
@btime vnorm2($vz)
println("  VDOT of $len elements:")
print("    Julia arrays ------------->")
@btime vdot($x,$y)
print("    DenseVariable arrays ----->")
@btime vdot($vx,$vy)
print("    mixed type arrays -------->")
@btime vdot($x,$vy)
print("    mixed type arrays -------->")
@btime vdot($vx,$y)

println("\nVersions provided by the C library:")
print("  VNORM2 of $len elements ---->")
@btime OptimPack.norm2($vz)
print("  VDOT of $len elements ------>")
@btime OptimPack.dot($vx,$vy)

end
