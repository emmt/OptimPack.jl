"""

Module `OptimPack.Problems` provides a collection of problems for testing optimization
methods.

"""
module Problems

using TypeUtils: @public
@public Rosenbrock ARWHEAD CHROSEN VARDIM TRIGSSQS SPHRPTS xinit xbest fbest

using Neutrals

"""
    OptimPack.Problems.xinit(f) -> x0

Return the initial point for testing the optimization of `f`.

"""
function xinit end

"""
    OptimPack.Problems.xbest(f) -> xbest

Return the solution of the optimization of `f`.

"""
function xbest end

"""
    OptimPack.Problems.fbest(f) -> fbest

Return the value of `f(xbest)` at the solution `xbest` of the optimization of `f`.

"""
fbest(f) = f(xbest(f))

randu(r::Tuple{Real,Real}, dims::Integer...) = randu(r, dims)
randu(r::Tuple{Real,Real}, dims::Tuple{Vararg{Integer}}) = randu(promote(r...), dims)
randu(r::Tuple{T,T}, dims::Tuple{Vararg{Integer}}) where {T<:Real} =
    randu(convert(Tuple{Float64,Float64}, r), dims)
function randu((a,b)::Tuple{T,T}, dims::Tuple{Vararg{Integer}}) where {T<:AbstractFloat}
    x = Array{T}(undef, dims)
    w = abs(b - a)
    for i in eachindex(x)
        x[i] = a + w*rand(T)
    end
    return x
end

#------------------------------------------------------------------------------ Rosenbrock -

struct Rosenbrock{T<:AbstractFloat}
    n::Int
    a::T
    b::T
    function Rosenbrock{T}(; n::Integer = 2, a::Real=1, b::Real=100) where {T<:AbstractFloat}
        @assert iseven(n)
        return  new{T}(n, a, b)
    end
end
Rosenbrock(args...; kwds...) = Rosenbrock{Float64}(args...; kwds...)

function (f::Rosenbrock{R})(x::AbstractArray{<:Real}) where {R}
    m = f.n÷2
    @assert eachindex(x) == 𝟙:2m
    T = promote_type(R, eltype(x))
    a = convert(T, f.a)
    b = convert(T, f.b)
    if m == 1
        x1 = x[1]
        x2 = x[2]
        return convert(T, (a - x1)^2 + b*(x2 - x1^2)^2)
    else
        s = zero(T)
        @inbounds @simd for k in 𝟙:m
            x1 = x[2k-1]
            x2 = x[2k]
            s += convert(T, (a - x1)^2 + b*(x2 - x1^2)^2)
        end
        return s
    end
end

Base.summary(f::Rosenbrock) = "Rosenbrock's test function (n=$(f.n), a=$(f.a), b=$(f.b))"

function xinit(f::Rosenbrock{T}) where {T}
    m = f.n÷2
    x = Vector{T}(undef, 2m)
    x1 = convert(T, -3)
    x2 = convert(T,  3)
    for k in 𝟙:m
        x[2k-1] = x1
        x[2k] = x2
    end
    return x
end

function xbest(f::Rosenbrock{T}) where {T}
    m = f.n÷2
    x = Vector{T}(undef, 2m)
    x1 = f.a
    x2 = x1^2
    for k in 𝟙:m
        x[2k-1] = x1
        x[2k] = x2
    end
    return x
end

fbest(f::Rosenbrock{T}) where {T} = zero(T)

#------------------------------------------------------------------------------ Himmelblau -

"""
    OptimPack.Problems.Himmelblau(x, y)
    OptimPack.Problems.Himmelblau([x, y])

Return the value of the Himmelblau's function, a multi-modal function used to test the
performances of optimization algorithms. The function is defined by:

    f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

The Himmelblau's function has one local maximum:

    f(-0.270845, -0.923039) = 181.617,

and four identical local minima:

    f( 3.0,       2.0)      = 0.0,
    f(-2.805118,  3.131312) = 0.0,
    f(-3.779310, -3.283186) = 0.0,
    f( 3.584428, -1.848126) = 0.0.

See also: http://en.wikipedia.org/wiki/Himmelblau%27s_function

"""
Himmelblau(x::T, y::T) where {T<:AbstractFloat} = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
Himmelblau(x, y) = Himmelblau(promote(x, y)...)
Himmelblau(x::T, y::T) where {T<:Integer} = Himmelblau(float(x), float(y))
function Himmelblau(x::AbstractVector)
    @assert length(x) == 2
    i = firstindex(x)
    return @inbounds Himmelblau(x[i], x[i+1])
end

Base.summary(f::typeof(Himmelblau)) = "Himmelblau's test function"

xinit(f::typeof(Himmelblau)) = [4, 2]
xbest(f::typeof(Himmelblau)) = [3, 2]
fbest(f::typeof(Himmelblau)) = 𝟘

#------------------------------------------------------------------------------ Beale -

"""
    OptimPack.Problems.Beale(x, y)
    OptimPack.Problems.Beale([x, y])

Return the value of the Beale's function, a function used to test the performances of
optimization algorithms. The function is defined by:

    f(x, y) = (3//2 - x + x*y)^2 + (9//4 - x +x*y^2)^2 + (21//8 - x + x*y^3)^2

The Beale's function has one local minimum on `-4.5 ≤ x ≤ 4.5`, `-4.5 ≤ y ≤ 4.5`:

    f(3, 0.5) = 0

See also: https://en.wikipedia.org/wiki/Test_functions_for_optimization

"""
Beale(x::T, y::T) where {T<:AbstractFloat} =
    (3//2 - x + x*y)^2 + (9//4 - x +x*y^2)^2 + (21//8 - x + x*y^3)^2
Beale(x, y) = Beale(promote(x, y)...)
Beale(x::T, y::T) where {T<:Integer} = Beale(float(x), float(y))
function Beale(x::AbstractVector)
    @assert length(x) == 2
    i = firstindex(x)
    return @inbounds Beale(x[i], x[i+1])
end

Base.summary(f::typeof(Beale)) = "Beale's test function"

xinit(f::typeof(Beale)) = [1, 2]
xbest(f::typeof(Beale)) = [3, 1//2]
fbest(f::typeof(Beale)) = 𝟘

#---------------------------------------------------------------------------------- VARDIM -

"""
    f = OptimPack.Problems.VARDIM{T=Float64}(n)

Build a callable object implementing the `VARDIM` test problem on page 98 of A.G. Buckley,
*"Test functions for unconstrained minimization"*, Technical Report 1989 CS-3, Dalhousie
University, Canada (1989).

Typical usage with `NEWUOA`:

```julia
using OptimPack, OptimPack_jll, Test
using OptimPack: Problems
n = 20
f = Problems.VARDIM(n)
x0 = Problems.xinit(f)
status, x, fx, nf = newuoa(f, x0; rhobeg=1/2n, rhoend=1e-6, maxevals=100_000);
@test x ≈ Problems.xbest(f) atol=1e-5
@test fx ≈ Problems.fbest(f) atol=1e-10
```

"""
struct VARDIM{T<:AbstractFloat}
    n::Int
end
VARDIM(n::Integer) = VARDIM{Float64}(n)

function (f::VARDIM)(x::AbstractArray)
    @assert length(x) == f.n
    u = oneunit(eltype(x))
    q = 1*(u - u)^2
    r = 1*(u - u)
    @inbounds for i in eachindex(x)
        y = x[i] - u
        q += y*y
        r += i*y
    end
    return q + r^2 + r^4
end

function xinit(f::VARDIM{T}) where {T<:AbstractFloat}
    n = f.n
    x0 = Vector{T}(undef, n)
    for k in 1:n
        x0[k] = T(n - k)/n
    end
    return x0
end

xbest(f::VARDIM{T}) where {T<:AbstractFloat} = ones(T, f.n)
fbest(f::VARDIM{T}) where {T<:AbstractFloat} = zero(T)

Base.summary(f::VARDIM) = "VARDIM test function (n=$(f.n))"

#-------------------------------------------------------------------------------- TRIGSSQS -
# Trigonometric sum of squares.

"""
    f = OptimPack.Problems.TRIGSSQS{T=Float64}(n)

Build a callable object implementing the `TRIGSSQS` test problem defined in M.J.D. Powell,
*"The NEWUOA software for unconstrained optimization without derivatives"*, Nonconvex
Optimization and Its Applications, G. Di Pillo & M. Roma (Eds.), Springer Science p. 255-297
(2006).

Typical usage with `NEWUOA`:

```julia
using OptimPack, OptimPack_jll, Test
using OptimPack: Problems
n = 20
npts = (2n+1, round(Int, sqrt((n + 1//2)*(n + 1)*(n + 2))), (n+1)*(n+2)÷2)
f = Problems.TRIGSSQS(n)
x0 = Problems.xinit(f)
status, x, fx, nf = newuoa(f, x0; rhobeg=0.1, rhoend=1e-6, npt=npts[1], maxevals=10_000);
@test x ≈ Problems.xbest(f) rtol=1e-5
@test fx ≈ Problems.fbest(f) atol=2e-3
```

"""
struct TRIGSSQS{T<:AbstractFloat}
    θ::Vector{T}
    b::Vector{T}
    St::Matrix{T} # the transpose of S
    Ct::Matrix{T} # the transpose of C
    xr::Vector{T} # random
    yr::Vector{T} # random noise
    x0::Vector{T} # initial point
    xbest::Vector{T} # solution
end

Base.summary(f::TRIGSSQS) = "TRIGSSQS test function (m=$(length(f.b)), n=$(length(f.θ))))"

xinit(f::TRIGSSQS) = f.x0
xbest(f::TRIGSSQS) = f.xbest

TRIGSSQS(n::Int) = TRIGSSQS{Float64}(n)
TRIGSSQS(m::Int, n::Int) = TRIGSSQS{Float64}(m, n)
TRIGSSQS{T}(n::Int) where {T<:AbstractFloat} = TRIGSSQS{T}(2n, n)
function TRIGSSQS{T}(m::Int, n::Int) where {T<:AbstractFloat}
    St = T.(rand(-100:100, (n, m)))
    Ct = T.(rand(-100:100, (n, m)))
    θ = T(10).^(randu(T.((0.1,1.0)), n))
    pi = T(π)
    xr = randu((-pi, pi), n) # uniform random in [-π,π]
    yr = randu((-pi, pi), n) # uniform random in [-π-,π]
    b = Vector{T}(undef, m)
    xbest = @. xr/θ
    x0 = @. (xr + yr/10)/θ
    for i in 1:m
        s = zero(T)
        c = zero(T)
        for j in 1:n
            sj, cj = sincos(θ[j]*xbest[j])
            s += St[j,i]*sj
            c += Ct[j,i]*cj
        end
        b[i] = s + c
    end
    return TRIGSSQS{T}(θ, b, St, Ct, xr, yr, x0, xbest)
end

function (f::TRIGSSQS{T})(x::AbstractArray) where {T}
    J, I = axes(f.St)
    @assert axes(f.Ct) == (J, I)
    @assert eachindex(f.b) == I
    @assert eachindex(f.θ) == J
    @assert eachindex(x) == J
    E = promote_type(T, eltype(x))
    r = zero(T)
    @inbounds for i in I
        s = zero(E)
        c = zero(E)
        for j in J
            sj, cj = sincos(f.θ[j]*x[j])
            s += f.St[j,i]*sj
            c += f.Ct[j,i]*cj
        end
        r += (f.b[i] - (s + c))^2
    end
    return r
end

#--------------------------------------------------------------------------------- SPHRPTS -

"""
    f = OptimPack.Problems.SPHRPTS{T=Float64}(n)

Build a callable object implementing the `SPHRPTS` test problem defined in M.J.D. Powell,
*"The NEWUOA software for unconstrained optimization without derivatives"*, Nonconvex
Optimization and Its Applications, G. Di Pillo & M. Roma (Eds.), Springer Science p. 255-297
(2006).

Typical usage with `NEWUOA`:

```julia
using OptimPack, OptimPack_jll, Test
using OptimPack: Problems
n = 20
npts = (2n+1, round(Int, sqrt((n + 1//2)*(n + 1)*(n + 2))), (n+1)*(n+2)÷2)
f = Problems.SPHRPTS(n)
x0 = Problems.xinit(f)
status, x, fx, nf = newuoa(f, x0; rhobeg=1/n, rhoend=1e-6, npt=npts[1], maxevals=50_000);
@test x ≈ Problems.xbest(f) rtol=1e-5
@test fx ≈ Problems.fbest(f) atol=2e-3
```

"""
struct SPHRPTS{T<:AbstractFloat,P<:AbstractMatrix{T}}
    p::P
end
SPHRPTS(n::Integer) = SPHRPTS{Float64}(n)
function SPHRPTS{T}(n::Integer) where {T<:AbstractFloat}
    @assert iseven(n)
    return SPHRPTS(Array{T}(undef, 3, n÷2))
end

Base.summary(f::SPHRPTS) = "SPHRPTS test function (n=$(2*size(f.p, 2)))"

function (f::SPHRPTS{R})(x::AbstractVector{S}) where {R<:AbstractFloat,S<:Real}
    T = promote_type(R, S)
    p = f.p
    t, m = size(p)
    @assert t == 3
    @assert eachindex(x) == 1:2m
    @assert axes(p) == (1:3, 1:m)
    @inbounds for k in 1:m
        sa, ca = sincos(x[2k-1])
        sb, cb = sincos(x[2k])
        p[1,k] = ca*cb
        p[2,k] = sa*cb
        p[3,k] = sb
    end
    r = zero(T)
    for j in 1:m
        s = zero(T)
        for k in 1:m
            s += (p[1,j] - p[1,k])^2 + (p[2,j] - p[2,k])^2 + (p[3,j] - p[3,k])^2
        end
        r += s
    end
    return inv(r)
end

function xinit(f::SPHRPTS{T}) where {T}
    m = size(f.p, 2)
    n = 2m
    x = Array{T}(undef, n)
    a = 4T(π)/n
    for k in 1:m
        x[2k-1] = a*k
        x[2k] = 0
    end
    return x
end

#--------------------------------------------------------------------------------- ARWHEAD -

struct ARWHEAD{T<:AbstractFloat}
    n::Int
end
ARWHEAD(n::Integer) = ARWHEAD{Float64}(n)

Base.summary(f::ARWHEAD) = "ARWHEAD test function (n=$(f.n))"

function (f::ARWHEAD)(x::AbstractVector)
    n = f.n
    @assert eachindex(x) == 𝟙:n
    s = zero(float(eltype(x)))
    for i in 1:n-1
        s += (x[i] + x[n]^2)^2 - 4x[i] + 3
    end
    return s
end

xinit(f::ARWHEAD{T}) where {T} = ones(T, n)

#--------------------------------------------------------------------------------- CHROSEN -

struct CHROSEN{T<:AbstractFloat}
    n::Int
end
CHROSEN(n::Integer) = CHROSEN{Float64}(n)

Base.summary(f::CHROSEN) = "CHROSEN test function (n=$(f.n))"

function (f::CHROSEN)(x::AbstractVector)
    n = f.n
    @assert eachindex(x) == 𝟙:n
    s = zero(float(eltype(x)))
    for i in 1:n-1
        s += 4*(x[i] - x[i+1]^2)^2 + (1 - x[i+1])^2
    end
    return s
end

xinit(f::CHROSEN{T}) where {T} = -ones(T, n)

end # module
