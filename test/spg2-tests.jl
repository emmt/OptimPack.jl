module SpectralProjectioGradientTests

using OptimPack
using Test

# The Rosenbrock function writes:
#
#    f(x) = sum((1 .- x[1:2:end]).^2) + sum(c10*(x[2:2:end] - x[1:2:end].^2).^2)
#
# The following method returns the cost function f(x) and overwrites
# the values of gx by the gradient ∇f(x).
function rosenbrock_fg!(x::AbstractVector{T},
                        gx::AbstractVector{T}) where {T<:AbstractFloat}
    # Check standard indexing and compatibility of dimensions.
    @assert !Base.has_offset_axes(x, gx)
    @assert size(x) == size(gx)

    # A few constants.
    c1 = T(1)
    c2 = T(2)
    c10 = T(10)
    c200 = T(200)

    # Compute f(x) and store ∇f(x) in gx.
    x1 = x[1:2:end]
    x2 = x[2:2:end]
    t1 = c1 .- x1
    t2 = c10*(x2 - x1.*x1)
    g2 = c200*(x2 - x1.*x1)
    gx[1:2:end] = -c2*(x1 .* g2 + t1)
    gx[2:2:end] = g2
    return sum(t1.*t1) + sum(t2.*t2)
end

function rosenbrock_init!(x0::AbstractVector{T}) where {T<:AbstractFloat}
  x0[1:2:end] .= -1.2
  x0[2:2:end] .=  1.0
  return x0
end

# The following function overwrites `dst` with `src` projected on the positive
# hyperorthant.
function nonnegative_projector!(dst::AbstractArray{T,N},
                                src::AbstractArray{T,N}) where {T,N}
    @assert axes(dst) == axes(src)
    @inbounds for i in eachindex(dst, src)
        dst[i] = max(src[i], zero(T))
    end
    return dst
end

T = Float32
n = 20
x0 = Vector{T}(undef, n)
xexact = ones(T, n) # exact solution


res = spg2(rosenbrock_fg!, copyto!, rosenbrock_init!(x0), 10;
           maxit=500, maxfc=1000, printer=nothing, verb=true)
x1 = res.x
println("Result without bounds: ", sqrt(sum((x1 - xexact).^2)/sum(xexact.^2)))

println()
res = spg2(rosenbrock_fg!,  nonnegative_projector!, rosenbrock_init!(x0), 10;
           maxit=500, maxfc=1000, printer=nothing, verb=true)
x2 = res.x

println("Result without bounds: ", sqrt(sum((x2 - xexact).^2)/sum(xexact.^2)))

end # module
