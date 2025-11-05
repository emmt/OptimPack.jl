"""
    OptimPack.Brent

This module provides methods to find a local root or a local minimum of an univariate
function by Brent's methods described in:

* Richard Brent, *"Algorithms for minimization without derivatives,"* Prentice-Hall, inc."
  (1973).

Exported symbols:

* `fzero` to find a local root of an univariate function in a given interval.

* `fmax` to find a local maximum of an univariate function in a given interval.

* `fmaxbrkt` to find a local maximum of an univariate function in a bracketing interval.

* `fmin` to find a local minimum of an univariate function in a given interval.

* `fminbrkt` to find a local minimum of an univariate function in a bracketing interval.

Non-exported public symbols:

* `Brent.maximize` is an alias to `fmax`.

* `Brent.minimize` is an alias to `fmin`.

"""
module Brent

export
    fmax,
    fmaxbrkt,
    fmin,
    fminbrkt,
    fzero

using TypeUtils: @public
@public maximize, minimize, tolerances

using Neutrals
using TypeUtils
using Base: @pure
import Base.MathConstants: φ

# goldstep = 1/φ^2 = 2 - φ ≈ 0.3812
@pure goldstep(::Type{T}) where {T<:AbstractFloat} = inv(convert(T, φ)^2)

# Type for undefined argument/option.
const Undef = typeof(undef)

# Simple structure to change the sign of the result returned by a callable object.
struct Negate{F}
    func::F
end
Negate(f::Negate) = f.func
@inline (obj::Negate)(args...; kwds...) = -obj.func(args...; kwds...)

"""
    fzero([T,] f, a, b) -> (x, fx, lo, hi, nf)
    fzero([T,] f, a, fa=undef, b, fb=undef) -> (x, fx, lo, hi, nf)

Apply Van Wijngaarden–Dekker–Brent method to find a zero of the function `f(x)` in the
interval `[a,b]`.

`f(a)` and `f(b)` must have opposite signs (an exception is thrown if this does not hold).
If any of these function values is known, optional arguments `fa = f(a)` and/or `fb = f(b)`
may be specified to save computations; otherwise, `undef` means that the corresponding
function value is not yet known.

Optional argument `T` is to specify the floating-point type used for computations. By
default, `T` is inferred from the numeric type of `a`, `fa`, `b`, and `fb`.

The method returns a 5-tuple: a zero `x` in the given interval `[a,b]` to within the
tolerance `rtol*abs(x) + atol`, the corresponding function value `fx = f(x)`, lower and
upper bounds `lo` and `hi` for the solution, and the number `nf` of calls to `f`. Absolute
and relative tolerances, `atol` and `rtol`, may be specified by keywords. If `rtol =
eps(T)`, the machine relative precision, the error is approximately bounded by:

    abs(x - z) ≤ 3*rtol*abs(z) + 2*atol

with `z` the exact solution. To avoid that rounding errors prevent convergence, `rtol ≥
eps(T)` and `atol > zero(atol)` should hold. By default, `rtol = eps(T)` and `atol` is
`eps(T)*abs(b - a)`.

!!! note
    The variable `x` and the function value `f(x)` may be quantities with units as provided
    by the `Unitful` package.

The `fzero` method is based on Richard Brent's F77 code ZEROIN which itself is a slightly
modified translation of the Algol 60 procedure ZERO given in:

* Richard Brent, *"Algorithms for minimization without derivatives"*, Prentice-Hall, inc.
  (1973).

"""
fzero(f, a::Number, b::Number; kwds...) = fzero(f, a, undef, b, undef; kwds...)

function fzero(f,
               a::Number, fa::Union{Number,Undef},
               b::Number, fb::Union{Number,Undef} = undef; kwds...)
    return fzero(concrete_precision(a, fa, b, fb), f, a, fa, b, fb; kwds...)
end

fzero(::Type{T}, f, a::Number, b::Number; kwds...) where {T<:AbstractFloat} =
    fzero(T, f, a, undef, b, undef; kwds...)

function fzero(::Type{T}, f,
               a::Number, fa::Union{Number,Undef},
               b::Number, fb::Union{Number,Undef} = undef;
               kwds...) where {T<:AbstractFloat}
    # Convert bounds and get tolerances, then initialize search.
    Tx = convert_real_type(T, promote_typeof(a, b))
    a = convert(Tx, a)
    b = convert(Tx, b)
    atol, rtol = tolerances(fzero, a, b; kwds...)
    return fzero_init(f, a, fa, b, fb, atol, rtol, 0)
end

# Private function `fzero_init` initializes the search with 0, 1 or 2 function values,
# bounds, `a` and `b`, and tolerances, `atol` and `rtol`, set with the numerical precision
# `T` to use for computations.
function fzero_init(f, a::Tx, fa::Undef, b::Tx, fb::Undef, atol::Tx, rtol::T,
                    eval::Int) where {Tx<:Number,T<:AbstractFloat}
    return fzero_init(f, a, f(a), b, undef, atol, rtol, eval + 1)
end

function fzero_init(f, a::Tx, fa::Undef, b::Tx, fb::Number, atol::Tx, rtol::T,
                    eval::Int) where {Tx<:Number,T<:AbstractFloat}
    return fzero_init(f, b, fb, a, fa, atol, rtol, eval)
end

function fzero_init(f, a::Tx, fa::Number, b::Tx, fb::Undef, atol::Tx, rtol::T,
                    eval::Int) where {Tx<:Number,T<:AbstractFloat}
    Tf = convert_real_type(T, typeof(fa))
    fa = convert(Tf, fa)
    iszero(fa) && return (a, fa, a, a, eval)
    fb = convert(Tf, f(b))
    eval += 1
    iszero(fb) && return (b, fb, b, b, eval)
    return fzero_search(f, a, fa, b, fb, atol, rtol, eval)
end

function fzero_init(f, a::Tx, fa::Number, b::Tx, fb::Number, atol::Tx, rtol::T,
                    eval::Int) where {Tx<:Number,T<:AbstractFloat}
    Tf = convert_real_type(T, promote_typeof(fa, fb))
    fa = convert(Tf, fa)
    iszero(fa) && return (a, fa, a, a, eval)
    fb = convert(Tf, fb)
    iszero(fb) && return (b, fb, b, b, eval)
    return fzero_search(f, a, fa, b, fb, atol, rtol, eval)
end

# Private function `fzero_search` is Brent's `fzero` method when f(a) and f(b) have been
# checked for early termination and when all parameters have the same numerical precision.
function fzero_search(f, a::Tx, fa::Tf, b::Tx, fb::Tf, atol::Tx, rtol::T,
                      eval::Int) where {T<:AbstractFloat,Tx<:Number,Tf<:Number}
    # Check the assumptions about function values.
    (fa > zero(fa)) == (fb > zero(fb)) && throw_bad_argument(
        "f(a) and f(b) must have different signs, got ", fa, " and ", fb)

    # Initialize.
    c, fc = a, fa
    e = d = b - a

    # Loop to improve the interval bracketing the root.
    while true
        # Make sure B is the best point so far.
        if abs(fc) < abs(fb)
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa
        end

        # Compute tolerance.
        tol = rtol*abs(b) + atol
        # NOTE: In Brent's book, the tolerance is denoted `δ` and is given by:
        #
        #     δ = 2⋅ϵ⋅abs(x) + t
        #
        # with `ϵ = eps(T)/2` the relative machine precision halved for rounded arithmetic
        # for computations done with floating-point type `T` (see Eq. (2.9) p. 51 in Brent's
        # book) and `t > 0` chosen by the caller. This corresponds to `atol = t` and `rtol =
        # 2⋅ϵ`. The value of `ϵ` should not be decreased below `eps(T)/2`, for then rounding
        # errors might prevent convergence. Hence `rtol ≥ eps(T)` is recommended. If `ϵ =
        # eps(T)/2`, that is `rtol = eps(T)`, the error is approximately bounded by:
        #
        #     abs(x - z) ≤ 6⋅ϵ⋅abs(z) + 2⋅t = 3*rtol*abs(z) + 2*atol
        #
        # with `z` the exact solution (see Eq. (2.18) p. 52 in Brent's book).

        # Check for convergence.
        m = (c - b)/2
        abs(m) ≤ tol && return (b, fb, minmax(b, c)..., eval)

        # See if a bisection is forced.
        if abs(e) < tol || abs(fa) ≤ abs(fb)
            # Bounds decreasing too slowly, use bisection.
            d = e = m
        else
            # Attempt quadratic interpolation if possible, linear interpolation otherwise.
            #
            # NOTE: Below, `p` has the units of `a` and `b`, while `q`, `r`, and `s` are
            # dimensionless.
            s = fb/fa
            if a == c
                # Linear interpolation.
                p = 2*m*s
                q = 𝟙 - s
            else
                # Inverse quadratic interpolation.
                q = fa/fc
                r = fb/fc
                p = (2*m*q*(q - r) - (b - a)*(r - 𝟙))*s
                q = (q - 𝟙)*(r - 𝟙)*(s - 𝟙)
            end
            if p > zero(p)
                q = -q
            else
                p = -p
            end
            if 2*p < min(3*m*q - tol*abs(q), abs(e*q))
                # Take the interpolation point.
                e = d
                d = p/q
            else
                # Force a bisection.
                d = e = m
            end
        end
        a, fa = b, fb
        if abs(d) > tol
            b += d
        elseif m > zero(m)
            b += tol
        else
            b -= tol
        end
        fb = convert(Tf, f(b))
        eval += 1
        iszero(fb) && return (b, fb, b, b, eval)
        if (fb > zero(fb)) == (fc > zero(fc))
            # Drop point C (make it coincident with point A) and adjust bounds of interval.
            c, fc = a, fa
            e = d = b - a
        end
    end
end

"""
    Brent.minimize([T,] f, a, b, args...; kwds...) -> (xm, fm, lo, hi, nf)
    fmin([T,] f, a, b, args...; kwds...) -> (xm, fm, lo, hi, nf)

Apply Brent's algorithm to find a local minimum of the function `f(x)` in the interval
`[a,b]`.

The result is the 5-tuple `(xm, fm, lo, hi, nf)` with `xm` the estimated value for which `f`
attains a local minimum value in `[a,b]`, `fm = f(xm)` the function value at `xm`, `lo` and
`hi` the bounds for the position of the local minimum, and `nf` the number of function
calls.

Optional argument `T` is the floating-point type to use for computations. If `T` is
unspecified, it is inferred from the numeric type of the arguments `a`, `b`, and `args...`.

To save computations, `args...` can consist in up to 3 points `x`, `w`, and `w` in the
interval `[a,b]` along with their function values:

    fmin([T,] f, a, b, x, f(x), [w, f(w), [v, f(v)]]; kwds...)

The given points need not be distinct nor ordered, they will be taken into account to
initialize the search.

The method used is a combination of golden section search and successive parabolic
interpolation. Convergence is never much slower than that for a Fibonacci search. If `f` has
a continuous second derivative which is positive at the minimum (which is not at `a` or
`b`), then convergence is superlinear, and usually of the order of about `1.3247`.

Keywords `rtol` and `atol` can be used to specify a tolerance:

    tol = rtol*abs(x) + atol

for the solution. The function `f` is never evaluated at two points closer than `tol`. If
`f` is `δ`-unimodal on `(a,b)` for some `δ < tol`, then `x` approximates the global minimum
on the interval with an error less than `3*tol`. Otherwise, `x` may approximate a local, but
non-global, minimum to the same accuracy. The relative tolerance `rtol` should be no smaller
than twice the relative machine precision `ϵ = eps(T)`, and preferably not much less than
`√ϵ`, the square root of the relative machine precision. The default values for the absolute
and relative tolerances are `atol = eps(T)*abs(b - a)` and `rtol = sqrt(eps(T))`.

This function is based on Richard Brent's FORTRAN 77 code FMIN which itself is a slightly
modified translation of the Algol 60 procedure LOCALMIN given in:

* Richard Brent, *"Algorithms for minimization without derivatives,"* Prentice-Hall, inc.
  (1973).

"""
 function fmin end

const minimize = fmin

function fmin(f, a::Number, b::Number, args::Number...; kwds...)
    return fmin(concrete_precision(a, b, args...), f, a, b, args...; kwds...)
end

function fmin(::Type{T}, f, a::Number, b::Number;
              kwds...) where {T<:AbstractFloat}
    # Determine suitable type for the variables in the computations.
    Tx = convert_real_type(T, promote_typeof(a, b))

    # Convert input values.
    a = convert(Tx, a)
    b = convert(Tx, b)

    # Get tolerances.
    atol, rtol = tolerances(fmin, a, b; kwds...)

    # Order end points of search interval.
    if a > b
        a, b = b, a
    end

    # Initialize the search with a point in the interval.
    x = a + goldstep(T)*(b - a)
    fx = convert_real_type(T, f(x))

    # Run Brent's algorithm.
    return fmin_search(f, a, b, x, fx, x, fx, x, fx, atol, rtol, 1)
end

function fmin(::Type{T}, f, a::Number, b::Number,
              x::Number, fx::Number;
              kwds...) where {T<:AbstractFloat}
    # Determine suitable types for the variables and for the function values in the
    # computations.
    Tx = convert_real_type(T, promote_typeof(a, b, x))
    Tf = convert_real_type(T, promote_typeof(fx))

    # Convert input values.
    a = convert(Tx, a)
    b = convert(Tx, b)
    x = convert(Tx, x)
    fx = convert(Tf, fx)

    # Get tolerances.
    atol, rtol = tolerances(fmin, a, b; kwds...)

    # Order end points of search interval and check initial x.
    if a > b
        a, b = b, a
    end
    (a ≤ x ≤ b) || throw_bad_argument(
        "given point `x` is not inside the search interval `[a,b]`")

    # Run Brent's algorithm.
    return fmin_search(f, a, b, x, fx, x, fx, x, fx, atol, rtol)
end

function fmin(::Type{T}, f, a::Number, b::Number,
              x::Number, fx::Number,
              w::Number, fw::Number; kwds...) where {T<:AbstractFloat}
    # Determine suitable types for the variables and for the function values in
    # the computations.
    Tx = convert_real_type(T, promote_typeof(a, b, x, w))
    Tf = convert_real_type(T, promote_typeof(fx, fw))

    # Convert input values.
    a = convert(Tx, a)
    b = convert(Tx, b)
    x = convert(Tx, x)
    w = convert(Tx, w)
    fx = convert(Tf, fx)
    fw = convert(Tf, fw)

    # Get tolerances.
    atol, rtol = tolerances(fmin, a, b; kwds...)

    # Order end points of search interval and check initial x and w.
    if a > b
        a, b = b, a
    end
    (a ≤ x ≤ b) || throw_bad_argument(
        "given point `x` is not inside the search interval `[a,b]`")
    (a ≤ w ≤ b) || throw_bad_argument(
        "given point `w` is not inside the search interval `[a,b]`")

    # Reorder the points as assumed by Brent's algorithm before running it.
    if fw < fx
        x, fx, w, fw = w, fw, x, fx
    end
    return fmin_search(f, a, b, x, fx, w, fw, w, fw, atol, rtol)
end

function fmin(::Type{T}, f, a::Number, b::Number,
              x::Number, fx::Number,
              w::Number, fw::Number,
              v::Number, fv::Number;
              kwds...) where {T<:AbstractFloat}
    # Determine suitable types for the variables and for the function values in the
    # computations.
    Tx = convert_real_type(T, promote_typeof(a, b, x, w, v))
    Tf = convert_real_type(T, promote_typeof(fx, fw, fv))

    # Convert input values.
    a = convert(Tx, a)
    b = convert(Tx, b)
    x = convert(Tx, x)
    w = convert(Tx, w)
    v = convert(Tx, v)
    fx = convert(Tf, fx)
    fw = convert(Tf, fw)
    fv = convert(Tf, fv)

    # Get tolerances.
    atol, rtol = tolerances(fmin, a, b; kwds...)

    # Order end points of search interval and check initial x, w, and v.
    if a > b
        a, b = b, a
    end
    (a ≤ x ≤ b) || throw_bad_argument(
        "given point `x` is not inside the search interval `[a,b]`")
    (a ≤ w ≤ b) || throw_bad_argument(
        "given point `w` is not inside the search interval `[a,b]`")
    (a ≤ v ≤ b) || throw_bad_argument(
        "given point `v` is not inside the search interval `[a,b]`")

    # Reorder the points as assumed by Brent's algorithm before running it.
    if fw < fx
        x, fx, w, fw = w, fw, x, fx
    end
    if fv < fx
        x, fx, v, fv = v, fv, x, fx
    end
    if abs(x - v) < abs(x - w)
        v, fv, w, fw = w, fw, v, fv
    end
    return fmin_search(f, a, b, x, fx, w, fw, v, fv, atol, rtol)
end

"""
    fmin_search(f, a, b, x, fx, w, fw, v, fv, atol, rtol, eval=0)

Run the main loop of Brent's algorithm assuming that all parameters are properly set (as
explained below). Tolerances `atol` and `rtol` may be (both) specified as keywords.

Original Brent's algorithm assumes that the minimum is in the open interval `(a,b)` with `a
< b` and keeps track of the following variables:

- `x`, `fx = f(x)`: position and least function value found so far;
- `w`, `fw = f(w)`: previous values of `x` and `fx`;
- `v`, `fv = f(v)`: previous values of `w` and `fw`;
- `d`: computed step (new try is: `u = x + d`, unless `d` too small);
- `e`: the previous value of `d`, if a parabolic step is taken; the difference between the
  most distant current endpoint and `x`, if a golden step is taken.

Other variables need not be saved, notably:

- `u`, `fu = f(u)`: the next point to try and its function value.

Thus the main loop of Brent's algorithm can be entered with any `x`, `w`, `v` (not
necessarily distinct) which are in `[a,b]` and such that:

    fx = f(x)  ≤  fw = f(w)
    |x - w| ≤ |x - v|           (to avoid a tie)

other internal variables are:

    d = x - w
    e = w - v

"""
function fmin_search(f, a::Tx, b::Tx,
                     x::Tx, fx::Tf,
                     w::Tx, fw::Tf,
                     v::Tx, fv::Tf,
                     atol::Tx, rtol::T,
                     eval::Int = 0) where {T<:AbstractFloat,Tx<:Number,Tf<:Number}
    # Compiler will optimize out these assertions if they hold.
    @assert get_precision(Tx) === T
    @assert get_precision(Tf) === T

    # Constant for golden step.
    c = goldstep(T)

    # Initialize.
    d = x - w
    e = w - v

    while true

        # Compute mid-point and check the stopping criterion.
        m = (a + b)/2
        tol = rtol*abs(x) + atol
        tol2 = 2*tol
        if abs(x - m) ≤ tol2 - (b - a)/2
            return (x, fx, a, b, eval)
        end

        # Determine next step to take.
        take_golden_step = true
        if abs(e) > tol
            # Fit a parabola (make sure final Q ≥ 0).
            r = (x - w)*(fx - fv)
            q = (x - v)*(fx - fw)
            if q > r
                p = (x - w)*r - (x - v)*q
                q = 2*(q - r)
            else
                p = (x - v)*q - (x - w)*r
                q = 2*(r - q)
            end
            if 2*abs(p) < q*abs(e) && q*(a - x) < p < q*(b - x)
                # Take the parabolic interpolation step.
                take_golden_step = false
                e = d
                d = p/q
                s = x + d
                # F must not be evaluated too close to A or B.
                if s - a < tol2 || b - s < tol2
                    d = (x < m ? tol : -tol)
                end
            end
        end
        if take_golden_step
            # Take a golden-section step.
            e = (x < m ? b : a) - x
            d = c*e
        end

        # F must not be evaluated too close to X.
        if abs(d) ≥ tol
            u = x + d
        elseif d > zero(d)
            u = x + tol
        else
            u = x - tol
        end
        fu = convert(Tf, f(u))
        eval += 1

        # Update A, B, V, W, and X.
        if fu ≤ fx
            if u < x
                b = x
            else
                a = x
            end
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else
            if u < x
                a = u
            else
                b = u
            end
            if fu ≤ fw || w == x
                v, fv = w, fw
                w, fw = u, fu
            elseif fu ≤ fv || v == x || v == w
                v, fv = u, fu
            end
        end
    end
end

"""
    Brent.maximize([T,] f, a, b, args...; kwds...) -> (xm, fm, lo, hi, nf)
    fmax([T,] f, a, b, args...; kwds...) -> (xm, fm, lo, hi, nf)

Apply Brent's algorithm to find a local maximum of the function `f(x)` in the interval
`[a,b]`. See [`fmin`](@ref) for details.

""" function fmax end

const maximize = fmax

function fmax(f, a::Number, b::Number, args::Number...; kwds...)
    return fmax(concrete_precision(a, b, args...), f, a, b, args...; kwds...)
end

function fmax(::Type{T}, f, a::Number, b::Number; kwds...) where {T<:AbstractFloat}
    return fmax_result(fmin(T, Negate(f), a, b; kwds...)...)
end

fmax_result(xm, fm, lo, hi, nf) = (xm, -fm, lo, hi, nf)

# When function values are specified, convert them first to the required precision so that
# negating these values is correct (in case function returns an unsigned value).

function fmax(::Type{T}, f, a::Number, b::Number,
              x::Number, fx::Number; kwds...) where {T<:AbstractFloat}
    return fmax_result(fmin(T, Negate(f), a, b,
                            x, -convert_real_type(T, fx); kwds...)...)
end

function fmax(::Type{T}, f, a::Number, b::Number,
              x::Number, fx::Number,
              w::Number, fw::Number; kwds...) where {T<:AbstractFloat}
    return fmax_result(fmin(T, Negate(f), a, b,
                            x, -convert_real_type(T, fx),
                            w, -convert_real_type(T, fw); kwds...)...)
end

function fmax(::Type{T}, f, a::Number, b::Number,
              x::Number, fx::Number,
              w::Number, fw::Number,
              v::Number, fv::Number; kwds...) where {T<:AbstractFloat}
    return fmax_result(fmin(T, Negate(f), a, b,
                            x, -convert_real_type(T, fx),
                            w, -convert_real_type(T, fw),
                            v, -convert_real_type(T, fv); kwds...)...)
end

"""
    fminbrkt(f, x, fx, w, fw, v, fv; atol=..., rtol=...)

Run Brent's algorithm to minimize function `f` given 3 points `x`, `w`, and `v` bracketing
the minimum and the corresponding function values `fx = f(x)`, `fw = f(w)`, and `fv = f(v)`
and such that `x ∈ [v,w]` and `fx ≤ min(fv, fw)` hold.

"""
function fminbrkt(f, x::Number, fx::Number, w::Number, fw::Number, v::Number, fv::Number;
                  kwds...)
    return exec(fminbrkt, f, x, fx, w, fw, v, fv; kwds...)
end

"""
    fmaxbrkt(f, x, fx, w, fw, v, fv; atol=..., rtol=...)

Run Brent's algorithm to maximize function `f` given 3 points `x`, `w`, and `v` bracketing
the maximum and the corresponding function values `fx = f(x)`, `fw = f(w)`, and `fv = f(v)`
and such that `x ∈ [v,w]` and `fx ≥ max(fv, fw)` hold.

"""
function fmaxbrkt(f, x::Number, fx::Number, w::Number, fw::Number, v::Number, fv::Number;
                  kwds...)
    return exec(fmaxbrkt, f, x, fx, w, fw, v, fv; kwds...)
end

function exec(alg::Union{typeof(fminbrkt),typeof(fmaxbrkt)},
              f, x::Number, fx::Number, w::Number, fw::Number, v::Number, fv::Number;
              kwds...)
    # Determine types for computations.
    T = floating_point_type(x, fx, w, fw, v, fv)
    Tx = promote_typeof(x, w, v)
    Tf = promote_typeof(fx, fw, fv)

    # Convert values.
    x, fx = convert(Tx, x), convert(Tf, fx)
    w, fw = convert(Tx, w), convert(Tf, fw)
    v, fv = convert(Tx, v), convert(Tf, fv)

    # Check that bracketing conditions hold.
    a, b = minmax(w, v) # search interval
    a ≤ x ≤ b || throw_bad_argument("point `x` is not in interval `[v,w]`")
    if alg === fminbrkt
        fx ≤ min(fv, fw) || throw_bad_argument("`f(x) ≤ min(f(v), f(w))` does not hold")
    else
        fx ≥ max(fv, fw) || throw_bad_argument("`f(x) ≥ max(f(v), f(w))` does not hold")
    end

    # Get tolerances.
    atol, rtol = tolerances(fmin, a, b; kwds...)

    # Order v and w and call Brent's algorithm.
    if abs(x - v) < abs(x - w)
        v, fv, w, fw = w, fw, v, fv
    end
    if alg === fminbrkt
        return fmin_search(f, a, b, x, fx, w, fw, v, fv, atol, rtol)
    else
        return fmax_result(fmin_search(Negate(f), a, b, x, -fx, w, -fw, v, -fv, atol, rtol))
    end
end

"""
    Brent.concrete_precision(args...) -> T

Return the precision of arguments `args...` or `Float64` if none of `args...` is
floating-point.

"""
@inline function concrete_precision(args...)
    T = get_precision(args...)
    return isconcretetype(T) ? T : Float64
end

"""
    Brent.tolerances(alg, a, b; atol=undef, rtol=undef) -> (atol, rtol)

Return the absolute and relative tolerances for the solution sought by Brent's algorithm
`alg` applied to interval `[a,b]`. Arguments `a` and `b` must have the same units and the
same floating-point type as the solution sought by the algorithm.

"""
function tolerances(alg::F, a::Tx, b::Tx;
                    atol=undef, rtol=undef) where {Tx,F<:Union{typeof(fmin),typeof(fzero)}}
    T = get_precision(a, b)
    (T <: AbstractFloat && isconcretetype(T)) || throw_bad_argument(
        "search bounds `a` and `b` must be floating-point numbers")
    if atol === undef
        atol = (eps(T)*abs(b - a))::Tx
    else
        atol ≥ zero(atol) || throw_bad_argument(
            "absolute tolerance `atol` must be non-negative, got ", atol)
        atol = convert(Tx, atol)::Tx
    end
    if rtol === undef
        rtol = if F <: typeof(fmin)
            sqrt(eps(T))::T # default rtol for fmin or fmax
        else
            eps(T)::T # default rtol for fzero
        end
    else
        zero(rtol) ≤ rtol < oneunit(rtol) || throw_bad_argument(
            "relative tolerance `rtol` must be ≥ 0 and < 1, got ", rtol)
        rtol = convert(T, rtol)::T
    end
    iszero(atol) && iszero(rtol) && throw_bad_argument(
        "absolute and relative tolerances `atol` and `rtol` cannot be both zero")
    return atol, rtol
end

"""
    Brent.promote_typeof(args...) -> T

yields the promoted type of the types of arguments `args...`.

"""
@inline promote_typeof(args...) = promote_type(map(typeof, args)...)

@noinline throw_bad_argument(msg::AbstractString) = throw(ArgumentError(msg))
@noinline throw_bad_argument(args...) = throw_bad_argument(string(args...))

end # module
