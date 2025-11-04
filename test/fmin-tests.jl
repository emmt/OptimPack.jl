"""
    FmnTests

Tests for local minimization by Brent's `fmin`, for local maximization by `fmax`, and for
global optimization by `BraDi` and `Step` methods. Typical usage:

    FzeroTests.runtest(T::Type...; verb=false)

"""
#
# fmin-tests.jl -
#
# Test univariate optimization.
#
module FminTests

using Test, Printf, Unitful, TypeUtils
using OptimPack.Brent
#using OptimPack: BraDi, Step

# Number of function evaluations.
const nevals = Ref{Int}(0)

# This structure defines a given extremum of a given function.
struct Extremum{Tx,Tf,Ta,Tb}
    type::Symbol # one of: :local_min, :local_max, :global_min, :global_max
    x::Tx        # correct solution
    fx::Tf       # f(x)
    a::Ta        # interval of search is [a,b]
    b::Tb
    n::Int       # number of samples for BraDi
end

function Extremum(type::Symbol; x::Number, fx::Number, a::Number, b::Number, n::Integer=0)
    return Extremum(type, x, fx, a, b, n)
end

local_min( ; kwds...) = Extremum(:local_min; kwds...)
local_max( ; kwds...) = Extremum(:local_max; kwds...)
global_min(; kwds...) = Extremum(:global_min; kwds...)
global_max(; kwds...) = Extremum(:global_max; kwds...)

Base.range(p::Extremum) = range(start = p.a, stop = p.b, length = p.n)

"""
    obj = TestFunc(name, func, args...; periodic=false)

yields an callable object that wraps objective function `func` to be tested for
univariate optimization. Arguments `args...` are any number of instances of
`Extremum`.

The wrapper is intended to bundle the settings for testing the optimization of
the function and may be called as a function to evaluate the objective
function: `obj(x)` yields `func(x)`.

To improve type-stability and correctly represent the behavior of the objective
function at a given numerical precision, `func(x)` must be written so as to
perform computations with the same floating-point type as `x`. This is tested
to some extend. Also when the objective function is called via the wrapper, the
global variable storing the number of calls is automatically incremented.

"""
struct TestFunc{F}
    name::String
    func::F
    periodic::Bool
    list::Vector{<:Extremum}
end

function (p::TestFunc)(x)
    nevals[] += 1
    return p.func(x)
end

function TestFunc(name::AbstractString, func, args::Extremum...;
                  periodic::Bool=false)
    return TestFunc(name, func, periodic, collect(args))
end

is_keyword(x::Any) = false
is_keyword(x::Expr) = x.head === :(=) && length(x.args) == 2 && x.args[1] isa Symbol

macro TestFunc(name::Symbol, args...)
    kwds = filter(is_keyword, args) # extract keywords
    if !isempty(kwds) # if any keywords, remove them
        args = filter(!is_keyword, args)
    end
    code = :(const $name = TestFunc($(string(name)), $(args...); $(kwds...)))
    return esc(code)
end

#for func in (:fmin, :fmax)
#
#    @eval Brent.$func(p::TestFunc{F,Tx,Tf}, a::Number, b::Number; kwds...) where {F,Tx,Tf} =
#        $func(float(promote_type(real_type(Tx), real_type(Tf))), p, a, b; kwds...)
#
#end

# Brent's test functions (see Brent's book p. 104).  Brent's 2nd test function
# is a simple parabola whose minimum is at xm=0, this is good to check for
# excessive number of iterations due to excessive precision when xm=0.
@TestFunc(
    brent_2,
    x -> x^2,
    local_min( x = 0, fx = 0, a = -1, b = 2))

@TestFunc(
    brent_3,
    x -> (x + 1)*x^2,
    local_min( x = 0, fx = 0, a = -0.6, b = 2),
    local_max( x = -2//3, fx = 4//27, a = -2, b = 7//25))

@TestFunc(
    brent_3_unit,
    x -> (x + oneunit(x))*x^2,
    local_min( x = 0u"mm", fx = 0u"mm^3", a = -0.6u"mm", b = 2u"mm"),
    local_max( x = (-2//3)u"mm", fx = (4//27)u"mm^3", a = -2u"mm", b = (7//25)u"mm"))

@TestFunc(
    brent_4,
    x -> (x + sin(x))*exp(-x^2),
    local_min( x = -0.6795786600198815, fx = -0.8242393984760767, a = -6,   b = 0.6),
    global_min(x = -0.6795786600198815, fx = -0.8242393984760767, a = -10,  b = 10, n = 3),
    local_max( x = +0.6795786600198815, fx = +0.8242393984760767, a = -0.6, b = 6),
    global_max(x = +0.6795786600198815, fx = +0.8242393984760767, a = -10,  b = 10, n = 3))

@TestFunc(
    brent_5,
    x -> (x - sin(x))*exp(-x^2),
    local_min( x = -1.195136641756661, fx = -0.06349052893643988, a = -6,  b = 1.1),
    global_min(x = -1.195136641756661, fx = -0.06349052893643988, a = -10, b = 10, n = 3),
    local_max( x = +1.195136641756661, fx = +0.06349052893643988, a = -1.1,  b = 6),
    global_max(x = +1.195136641756661, fx = +0.06349052893643988, a = -10, b = 10, n = 3))

# Michalewicz's functions.
@TestFunc(
    michalewicz_1,
    x -> x*sin(10x),
    local_min( x = 1.733637792398336, fx = -1.730760860785851, a = 1.5, b = 1.9),
    global_min(x = 1.733637792398336, fx = -1.730760860785851, a = -1,  b = 1.9, n = 12),
    local_max( x = 1.420743672519119, fx =  1.417237411377428, a = 1.2, b = 1.7),
    global_max(x = 1.420743672519119, fx =  1.417237411377428, a = -1,  b = 1.9, n = 12))

@TestFunc(
    michalewicz_2,
    x -> begin
        a = sin(x)
        b = x^2/π
        s = zero(a)
        for i in 1:10
            s = oftype(s, s + sin(b*i)^20)
        end
        return a*s
    end,
    local_min( x = 2.567092475376642, fx = 0.2105521936702232, a = 2.53, b = 2.61),
    global_min(x = 2.567092475376642, fx = 0.2105521936702232, a = 0.7,  b = 3, n = 22),
    local_max( x = 2.220865159657191, fx = 3.979338598164439,  a = 2.15, b = 2.29),
    global_max(x = 2.220865159657191, fx = 3.979338598164439,  a = 0.7,  b = 3, n = 22))

# Problems in AMPGO (http://infinity77.net/global_optimization/test_functions_1d.html).
@TestFunc(
    ampgo_2,
    x -> sin(x) + sin((10//3)*x),
    local_min( x = 5.145735290256128, fx = -1.899599349152113, a = 4.3, b = 6.2),
    global_min(x = 5.145735290256128, fx = -1.899599349152113, a = 2.5, b = 7.5, n = 7))

@TestFunc(
    ampgo_3,
    x -> begin
        s = float(zero(x))
        # NOTE: The formula (in AMPGO web page) is for `k ∈ 1:6` but the figure and the
        #       given minimum is for `k ∈ 1:5`.
        for k in 1:5
            s = oftype(s, s - k*sin((k + 1)*x + k))
        end
        return s
    end,
    periodic = true,
    local_min( x = -0.4913908362593146, fx = -12.03124944216714, a = -1, b = 0),
    global_min(x = -0.4913908362593146, fx = -12.03124944216714, a = -π,  b = π, n = 13))

@TestFunc(
    ampgo_4,
    x -> -(16x^2 - 24x + 5)*exp(-x),
    local_min( x = 2.868033988749895, fx = -3.850450708800219, a = 0.7, b = 8.0),
    global_min(x = 2.868033988749895, fx = -3.850450708800219, a = 0.1, b = 8.0, n = 9))

@TestFunc(
    ampgo_5,
    x -> (3x - 1.4)*sin(18x),
    local_min( x = 0.9660858038268510, fx = -1.489072538689604, a = 0.8, b = 1.1),
    global_min(x = 0.9660858038268510, fx = -1.489072538689604, a = 0.0, b = 1.2, n = 10),
    local_max( x = 1.139043919974300,  fx =  2.010281351381081, a = 1.0, b = 1.2),
    global_max(x = 1.139043919974300,  fx =  2.010281351381081, a = 0.0, b = 1.2, n = 10))

@TestFunc(
    ampgo_6,
    x -> -(x + sin(x))*exp(-x^2),
    local_min( x = 0.6795786601089, fx = -0.8242393984761, a = -0.6, b = 10),
    global_min(x = 0.6795786601089, fx = -0.8242393984761, a = -10, b = 10))

@TestFunc(
    ampgo_7,
    x -> sin(x) + sin(10x/3) + log(x) - 0.84*x + 3,
    local_min( x = 5.1997783686004, fx = -1.6013075464944, a = 4.2, b = 6.1),
    global_min(x = 5.1997783686004, fx = -1.6013075464944, a = 2.7, b = 7.5))

@TestFunc(
    ampgo_8,
    x -> begin
        s = float(zero(x))
        # NOTE: The formula (in AMPGO web page) is for `k ∈ 1:6` but the figure and the
        #       given minimum is for `k ∈ 1:5`.
        for k in 1:5
            s = oftype(s, s - k*cos((k + 1)*x + k))
        end
        return s
    end,
    periodic = true,
    local_min( x = -0.8003211004719731, fx = -14.50800792719503, a = -1.4, b = -0.2),
    global_min(x = -0.8003211004719731, fx = -14.50800792719503, a = -pi, b = pi, n = 13))

@TestFunc(
    ampgo_9,
    x -> sin(x) + sin(2x/3),
    local_min( x = 17.0391989476448, fx = -1.9059611187158, a = 13.6, b = 20.4),
    global_min(x = 17.0391989476448, fx = -1.9059611187158, a = 3.1, b = 20.4))

@TestFunc(
    ampgo_10,
    x -> -x*sin(x),
    local_min( x = 7.9786657125325, fx = -7.9167273715878, a = 5, b = 10),
    global_min(x = 7.9786657125325, fx = -7.9167273715878, a = 0, b = 10))

@TestFunc(
    ampgo_11,
    x -> 2cos(x) + cos(2x),
    local_min( x = 2.0943950957161, fx = -1.5, a = 0, b = 3),
    global_min(x = 2.0943950957161, fx = -1.5, a = -π/2, b = 2π))

@TestFunc(
    ampgo_12,
    x -> sin(x)^3 + cos(x)^3,
    local_min( x = π, fx = -1, a = 1.6, b = 3.9,),
    global_min(x = π, fx = -1, a = 0,   b = 2π))

@TestFunc(
    ampgo_13,
    x -> -x^(2//3) - (1 - x^2)^(1//3),
    local_min( x = 1/sqrt(2), fx = -1.5874010519682, a = 0.001, b = 0.99))

@TestFunc(
    ampgo_14,
    x -> -exp(-x)*sin(2π*x),
    local_min( x = 0.2248803858915620, fx = -0.7886853874086725, a = 0,    b = 0.7),
    global_min(x = 0.2248803858915620, fx = -0.7886853874086725, a = 0,    b = 4),
    local_max( x = 0.7248803858915620, fx =  0.4783618683306960, a = 0.23, b = 1.2),
    global_max(x = 0.7248803858915620, fx =  0.4783618683306960, a = 0,    b = 4))

@TestFunc(
    ampgo_15,
    x -> (x^2 - 5x + 6)/(x^2 + 1),
    local_min( x =  2.414213562373095, fx = -0.03553390593273762, a = -0.4, b = 5),
    global_min(x =  2.414213562373095, fx = -0.03553390593273762, a = -5,   b = 5),
    local_max( x = -0.414213562373095, fx =  7.035533905932738,   a = -5,   b = 2),
    global_max(x = -0.414213562373095, fx =  7.035533905932738,   a = -5,   b = 5))

@TestFunc(
    ampgo_18,
    x -> x ≤ 3 ? (x - 2)^2 : 2log(x - 2) + 1,
    local_min( x = 2, fx = 0, a = 0, b = 6))

@TestFunc(
    ampgo_20,
    x -> (sin(x) - x)*exp(-x^2),
    local_min( x =  1.195136641756661, fx = -0.06349052893643988, a = -1.1, b = 6.2),
    global_min(x =  1.195136641756661, fx = -0.06349052893643988, a = -10,  b = 10, n = 3),
    local_max( x = -1.195136641756661, fx =  0.06349052893643988, a = -5.0, b = 1.1),
    global_max(x = -1.195136641756661, fx =  0.06349052893643988, a = -10, b = 10, n = 3))

@TestFunc(
    ampgo_21,
    x -> (sin(x) + cos(2x))*x,
    local_min( x = 4.7954086865801, fx = -9.5083504406331, a = 3.1, b = 6.5),
    global_min(x = 4.7954086865801, fx = -9.5083504406331, a = 0, b = 10))

@TestFunc(
    ampgo_22,
    x -> exp(-3x) - sin(x)^3,
    local_min( x = 14.1371669411515, fx = -1.0, a = 11.1, b = 17.2),
    global_min(x = 14.1371669411515, fx = -1.0, a = 0, b = 20))

# Test functions from GSL (GNU Scientific Library).
@TestFunc(
    gsl_fmin_1,
    x -> x^4 - 1,
    local_min( x = 0, fx = -1, a = -3, b = 17))

@TestFunc(
    gsl_fmin_2,
    x -> sqrt(abs(x)),
    local_min( x = 0, fx = 0, a = -2.0, b = 1.5))

@TestFunc(
    gsl_fmin_3 ,
    # NOTE: This function is discontinuous at the location of the minimum.
    x -> x < 1 ? float(one(x)) : -exp(-x),
    local_min( x = 1.0, fx = -0.3678794411714423, a = -2.0, b = 4.0))

@TestFunc(
    gsl_fmin_4,
    x -> x - 30/(1 + 100_000*(x - 0.8)^2),
    local_min( x = 0.7999998333333325, fx = -29.20000008333333, a = 0.72, b = 2.0),
    global_min(x = 0.7999998333333325, fx = -29.20000008333333, a = -1,   b = 2),
    local_max( x = 0.7157358311799512, fx = 0.6735444095680056, a = 0.45, b = 0.79),
    global_max(x = 0.7157358311799512, fx = 0.6735444095680056, a = -1,   b = 2))

const test_funcs = (
    brent_2, brent_3, brent_4, brent_5,
    michalewicz_1, michalewicz_2,
    ampgo_2, ampgo_3, ampgo_4, ampgo_5, ampgo_6, ampgo_7,
    ampgo_8, ampgo_9, ampgo_10, ampgo_11, ampgo_12, ampgo_13,
    ampgo_14, ampgo_15, ampgo_18, ampgo_20, ampgo_21, ampgo_22,
    gsl_fmin_1, gsl_fmin_2, gsl_fmin_3, gsl_fmin_4,
    brent_3_unit)

round_value(val; sigdigits=3, base=10) = round(unitless(val); sigdigits, base)*unit(val)

runtests(; kwds...) = runtests(Float32, Float64, BigFloat; kwds...)

relative_precision(x₁::Any, x₂::Any) =
    relative_precision(get_precision(x₁), get_precision(x₂))
function relative_precision(::Type{T₁}, ::Type{T₂}) where {T₁<:AbstractFloat, T₂<:AbstractFloat}
    if isconcretetype(T₁) && isconcretetype(T₂)
        return max(eps(T₁), eps(T₂))
    elseif isconcretetype(T₁)
        return eps(T₁)
    elseif isconcretetype(T₂)
        return eps(T₂)
    else
        return eps(Float64)
    end
end

function runtests(Ts::Type{<:AbstractFloat}...; verb::Bool=false)
    algs = (:fmin, :fmax)
    if verb
        println("Test function                   x                   f(x)    ncalls Type    ")
        println("-------------- --------------------------------- ---------- ------ --------")
    end
    @testset "Extremum of univariate function" begin
        @testset "f=$(f.name), T=$T" for f in test_funcs, T in Ts
            for p in f.list

                # Bounds and solution for this extremum.
                a, b, xm, fm = p.a, p.b, p.x, p.fx

                # Default precision and tolerances for fmin.
                prec = get_precision(a, b)
                rtol = sqrt(eps(T))
                atol = eps(T)*abs(b - a)

                # Set expected precision according to doc. of fmin.
                xeps = relative_precision(xm, T)
                feps = relative_precision(fm, T)
                xtol = 3*(sqrt(xeps)*abs(xm) + xeps*abs(b - a))
                ftol = 2*(iszero(fm) ? oneunit(fm) : abs(fm))*sqrt(feps)
                if f === ampgo_15
                    # NOTE: Require a bit more relative precision, this is needed for
                    #       maximizing AMPGO 15-th test function.
                    rtol /= 2
                end
                if f === gsl_fmin_1
                    # gsl_fmin_1(x) = x^4 - 1 which results in rapid loss of significant digits.
                    # Set xtol such that f(xtol) == f(xm) with xm the solution.
                    xtol = (T === Float32 ? 8e-4 : T === BigFloat ? 1e-20 : 2e-6)
                end

                nevals[] = 0
                if p.type === :local_min && :fmin ∈ algs
                    x, fx, lo, hi, nf = if precision === T
                        @inferred fmin(f, a, b)
                    else
                        @inferred fmin(T, f, a, b; rtol, atol)
                    end
                elseif p.type === :local_max && :fmax ∈ algs
                    x, fx, lo, hi, nf = if precision === T
                        @inferred fmax(f, a, b)
                    else
                        @inferred fmax(T, f, a, b; rtol, atol)
                    end
                elseif p.type === :global_min && (:bradi ∈ algs || :step ∈ algs)
                    if :bradi ∈ algs
                        r = range(p)
                        isempty(r) && continue
                        periodic = f.periodic
                        x, fx, lo, hi, nf = if precision === T
                            @inferred BraDi.minimize(f, r; periodic=periodic)
                        else
                            @inferred BraDi.minimize(T, f, r; periodic=periodic,
                                                     atol=atol, rtol=rtol)
                        end
                    else
                        continue
                    end
                elseif p.type === :global_max && (:bradi ∈ algs || :step ∈ algs)
                    if :bradi ∈ algs
                        r = range(p)
                        isempty(r) && continue
                        periodic = f.periodic
                        x, fx, lo, hi, nf = if precision === T
                            @inferred BraDi.maximize(f, r; periodic=periodic)
                        else
                            @inferred BraDi.maximize(T, f, r; periodic=periodic,
                                                     atol=atol, rtol=rtol)
                        end
                    else
                        continue
                    end
                else
                    continue
                end
                if verb
                    @printf("%-14s %22.15g ± %8.3g %10.3e %6d %-8s\n",
                            f.name, adapt_precision(Float64, x),
                            round_value(adapt_precision(Float64, abs(x - xm))),
                            adapt_precision(Float64, fx), nf, repr(T))
                end
                @test nf == nevals[]
                @test get_precision(x) === T
                @test get_precision(fx) === T
                fx1 = adapt_precision(T, @inferred f(x))
                @test get_precision(fx1) === T
                @test fx == fx1
                @test fx ≈ fm atol=ftol rtol=0
                @test  x ≈ xm atol=xtol rtol=0
                @test lo ≤ x ≤ hi
                if p.type === :local_min || p.type === :global_min
                    @test fx ≤ min(f(lo), f(hi))
                else
                    @test fx ≥ max(f(lo), f(hi))
                end
            end
        end
    end
end

end # module

if isinteractive()
    FminTests.runtests(; verb=true)
    nothing
end
