"""
    FzeroTests

Tests for Brent's `fzero`. Typical usage:

    FzeroTests.runtest(T::Type...; verb=false)

"""
module FzeroTests

using Test, Printf, Unitful, TypeUtils
using OptimPack.Brent

# Counter of function evaluations.
const nevals = Ref{Int}(0)

struct TestFunc{F,Ta,Tb,Tx}
    name::String
    func::F
    a::Ta
    b::Tb
    x0::Tx
    function TestFunc(name::AbstractString, f, a, b, x0)
        return new{typeof(f),typeof(a),typeof(b),typeof(x0)}(name, f, a, b, x0)
    end
end

function (p::TestFunc)(x)
    nevals[] += 1
    return p.func(x)
end

params(p::TestFunc) = (p.a, p.b, p.x0)

macro TestFunc(name::Symbol, expr::Expr, a, b, x0)
    code = :(const $name = TestFunc($(string(name)), $expr, $a, $b, $x0))
    return esc(code)
end

@TestFunc(fzero_test_1, x -> 1/(x - 3) - 6,
          3, 4, 19//6) # NOTE purposely f(a) = Inf

@TestFunc(fzero_test_2, x -> exp(-x) - cos(x),
          -4.7, 1.1, 0.0)

@TestFunc(fzero_test_3, x -> log(x^2 + x + 2) - x + 1,
          -1.8, 5.7, 4.152590736757158)

@TestFunc(fzero_test_4, x -> sin(x)^2 - x^2 + 1,
          -1.3, 7.2, 1.4044916482153411)

@TestFunc(fzero_test_5, x -> exp(-x^2 + x + 2) - cos(x + 1) + x^3 + 1,
          -2.1, 2.7, -1.0)

@TestFunc(fzero_test_6, x -> x^11 + x + 1,
          -2.7, 3.2, -0.844397528792023)

# Like fzero_test_6 but with units.
@TestFunc(fzero_test_6_unit, x -> x^11 + x*u"cm^10" + 1u"cm^11",
          -2.7u"cm", 52u"mm", -0.844397528792023u"cm")

@TestFunc(fzero_test_7, x -> (x - 2)*(x^10 + x + 1)*exp(x - 1),
          -2.7, 5.2, 2.0)

# Test functions for root finding from Vakkalagadda Satya Sai Prakash, in "Implementation of
# Brent-Dekker and A Better Root Finding Method and Brent-Dekker Method's Parallelization".

@TestFunc(fzero_prakash_1, x -> exp(x)/2 - 5x + 2,
          1, 6, 3.401795803857807)

@TestFunc(fzero_prakash_2, x -> -2x^4 + 2x^3 - 16x^2 - 60x + 100,
          -2, 4, 1.240787113746981)

@TestFunc(fzero_prakash_3, x -> exp(x)*cos(x) - x*sin(x),
          2, 6, 4.668600322499089)

@TestFunc(fzero_prakash_4, x -> x^5 - 5x + 3,
          -1.6, 1.2, 0.6180339887498948)

@TestFunc(fzero_prakash_5, x -> x^3 - 0.926*x^2 + 0.0371*x + 0.043,
          -0.1, 0.8, 0.2910955026957230)

@TestFunc(fzero_prakash_6, x -> -9 + sqrt(99 + 2x - x^2) + cos(2x),
          1.7, 5.5, 4.178182868653094)

@TestFunc(fzero_prakash_7, x -> sin(cosh(x)),
          -1.0, 2.5, 1.811526272460853)

@TestFunc(fzero_prakash_8, x -> exp(-exp(-x)) - x,
          0, 1, 0.5671432904097839)

# Test functions from the GNU Scientific Library (GSL).

@TestFunc(fzero_gsl_1, x -> x^20 - 1, 0.1, 2.0, 1.0)

@TestFunc(fzero_gsl_2, x -> sqrt(abs(x))*sign(x),
          -1.0/3.0, 1.0, 0.0)

@TestFunc(fzero_gsl_3, x -> x^2 - 1e-8, 0, 1, 1e-4)

@TestFunc(fzero_gsl_4, x -> x*exp(-x),
          -1.0, 5.0, 0.0) # (-1.0/3.0, 2.0, 0.0)

@TestFunc(fzero_gsl_5, x -> (x - oneunit(x))^7,
          0, 3, 1) # (0.9995, 1.0002, 1)

const test_funcs = (
    fzero_test_1, fzero_test_2, fzero_test_3, fzero_test_4,
    fzero_test_5, fzero_test_6, fzero_test_7,
    fzero_prakash_1, fzero_prakash_2, fzero_prakash_3, fzero_prakash_4,
    fzero_prakash_5, fzero_prakash_6, fzero_prakash_7, fzero_prakash_8,
    fzero_gsl_1, fzero_gsl_2, fzero_gsl_3, fzero_gsl_4, fzero_gsl_5,
    fzero_test_6_unit)

runtests(; kwds...) = runtests(Float32, Float64, BigFloat; kwds...)

function runtests(Ts::Type{<:AbstractFloat}...; verb::Bool=false)
    if verb
        println("Test function                   x               f(x)   ncalls Type    ")
        println("-------------------- ---------------------- ---------- ------ --------")
    end
    @testset "Zero of univariate function" begin
        @testset "f=$(f.name), T=$T" for f in test_funcs, T in Ts

            # Search bounds and solution.
            (a, b, x0) = params(f)
            # NOTE: When the solution is 0, setting atol to a sensible value is critical.

            # Tolerances for the solution. Taking care that `x0` may be given with finite precision.
            prec = get_precision(x0)
            ϵ = if prec == AbstractFloat
                # x0 given with arbitrary precision
                eps(T)
            else
                # x0 given with finite precision
                max(eps(T), eps(prec)) |> T
            end
            xtol = (3*abs(x0) + abs(a - b))*ϵ
            ftol = max(abs(f(a)), abs(f(b)))*sqrt(eps(T))

            nevals[] = 0 # reset counter
            (xm, fm, lo, hi, nf) = if T == Float64
                # Use default precision and tolerances.
                @inferred fzero(f, a, b)
            else
                # Explicitly specify precision and tolerances (the same as the default ones).
                @inferred fzero(T, f, a, b; rtol = eps(T), atol = eps(T)*abs(a - b))
            end
            @test nf == nevals[]
            @test get_precision(xm, fm, lo, hi) == T
            @test xm ≈ x0       atol=xtol rtol=0
            @test fm ≈ f(xm)    atol=ftol rtol=0
            @test fm ≈ zero(fm) atol=ftol rtol=0
            if verb
                @printf("%-20s %22.15g %10.3e %6d %-8s\n", f.name, xm, fm, nf, repr(T))
            end
        end
    end
end

end # module

if isinteractive()
    FzeroTests.runtests(; verb=true)
    nothing
end
