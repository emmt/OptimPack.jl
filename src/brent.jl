#
# brent.jl --
#
# Find a local root or a local minimum of an univariate function by Brent's
# methods described in:
#
# [1] Richard Brent, "Algorithms for minimization without derivatives,"
#     Prentice-Hall, inc. (1973).
#
#-------------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 1973, Richard Brent.
# Copyright (C) 2015-2020, Éric Thiébaut.
#

module Brent

export
    fzero,
    fmin

"""
    fzero_atol(T)
    fzero_rtol(T)

yields default absolute and relative tolerances for Brent's `fzero` method.
Argument `T` is the floating-point type used for the computations.

"""
fzero_atol(::Type{T}) where {T<:AbstractFloat} = floatmin(T)
fzero_rtol(::Type{T}) where {T<:AbstractFloat} = eps(T)

"""
    fmin_atol(T)
    fmin_rtol(T)

yields default absolute and relative tolerances for Brent's `fmin` method.
Argument `T` is the floating-point type used for the computations.

"""
fmin_atol(::Type{T}) where {T<:AbstractFloat} = floatmin(T)
fmin_rtol(::Type{T}) where {T<:AbstractFloat} = sqrt(eps(T))

# goldstep = 1/φ^2 = 2 - φ ≈ 0.3812
import Base.MathConstants: φ
goldstep(::Type{T}) where {T<:AbstractFloat} = T(2 - φ)

"""
# Van Wijngaarden–Dekker–Brent method for finding a zero of a function

    fzero([T=Float64,] f, a, b; atol=floatmin(T), rtol=eps(T)) -> (x, fx)

seeks a local root of the function `f(x)` in the interval `[a,b]`.

It is assumed that `f(a)` and `f(b)` have opposite signs (an error is raised if
this does not hold).  `fzero` returns a zero `x` in the given interval `[a,b]`
to within a tolerance: `rtol*abs(x) + atol`.

If the function value at the endpoints `a` and `b` of the search interval are
known, the method can be called as:

    fzero([T=Float64,] f, a, fa, b, fb; atol=floatmin(T), rtol=eps(T))

to save some computations.

This function has been derived from Richard Brent's F77 code ZEROIN which
itself is a slightly modified translation of the Algol 60 procedure ZERO
given in:

> Richard Brent, "Algorithms for minimization without derivatives",
> Prentice-Hall, inc. (1973).


## Arguments

* `T` is the floating-point data type to use for computations.  Defaulting to
  `Float64`.  The function `f(x)` must return a result of type `T` or which can
  be automatically promoted to type `T`.

* `f` - The user-supplied function whose local root is being sought.  Called as
  `f(x)` to evaluate the function at any `x` in the interval `[a,b]`.

* `a`, `b` - The endpoints of the initial search interval.

* `fa`, `fb` - The function values at the endpoints of the initial search
  interval.


## Keywords

* `atol` is the absolute tolerance for the solution.  The default value
  for `atol` is `floatmin(T)`.

* `rtol` is the relative tolerance for the solution.  The recommended (and
  default) value for `rtol` is `eps(T)` where `eps(T)` is the relative
  machine precision defined as the smallest representable number such that `1 +
  eps > 1'.


## Tolerance

In original Brent's method, the tolerance (denoted `tol` in the code) is
defined as:

    δ = 2⋅ϵ⋅|b| + t

where `ϵ = eps(T)/2` is the relative machine precision halved for rounded
arithmetic (see Eq. (2.9) p. 51 in Brent's book), `b` (initially one of the
bounds) is the estimated solution and `t > 0` is chosen by the caller.

The value of `ϵ` should not be decreased below `eps(T)/2`, the relative machine
precision halved for rounded arithmetic, for then rounding errors might prevent
convergence.  If `ϵ = eps(T)/2`, the error is approximately bounded by:

    abs(b - x) ≤ 6⋅ϵ⋅abs(x) + 2⋅t

with `x` and `b` the exact and estimated solutions (see Eq. (2.18) p. 52 in
Brent's book).

In this version of the code, the tolerance is computed as:

    tol = δ = rtol*abs(b) + atol

where the absolute and relative tolerances are related to the parameters `ϵ`
and `t` of Brent's method by:

    atol ≡ t
    rtol ≡ 2⋅ϵ ≥ eps(T)


## Result

The returned value is the 2-tuple `(x, fx)` with `x` the estimated value of an
abscissa for which `f` is approximately zero in `[a,b]`; `fx = f(x)` is the
function value at `x`.

"""
fzero(f, a::Real, b::Real; kwds...) = fzero(Float64, f, a, b; kwds...)

fzero(f, a::Real, fa::Real, b::Real, fb::Real; kwds...) =
    fzero(Float64, f, a, fa, b, fb; kwds...)

function fzero(::Type{T}, f, a::Real, b::Real;
               atol::Real = fzero_atol(T),
               rtol::Real = fzero_rtol(T)) where {T<:AbstractFloat}
    _fzero1(f, T(a), T(b), T(atol), T(rtol))
end

function fzero(::Type{T}, f, a::Real, fa::Real, b::Real, fb::Real;
               atol::Real = fzero_atol(T),
               rtol::Real = fzero_rtol(T)) where {T<:AbstractFloat}
    _fzero1(f, T(a), T(fa), T(b), F(fb), T(atol), T(rtol))
end

# 2 helpers are used by fzero: _fzero1 for early return and _fzero2 to refine
# the interval bt Brent's method.  In these 2 functions, the type of the
# floting-point variables should be stable and equal to `T`.  Type stability is
# one of the reasons for this splitting of the code.

function _fzero1(f, a::T, b::T,
                 atol::T, rtol::T) where {T<:AbstractFloat}
    # Compute the function value at the endpoints.  Return as early as
    # possible.
    fa = T(f(a))
    fa == 0 && return (a, fa)
    fb = T(f(b))
    fb == 0 && return (b, fb)

    # Iterate to reduce the interval.
    return _fzero2(f, a, fa, b, fb, atol, rtol)
end

function _fzero1(f, a::T, fa::T, b::T, fb::T,
                 atol::T, rtol::T) where {T<:AbstractFloat}
    # Return immediately if possible.
    fa == 0 && return (a, fa)
    fb == 0 && return (b, fb)

    # Iterate to reduce the interval.
    return _fzero2(f, a, fa, b, fb, atol, rtol)
end

function _fzero2(f, a::T, fa::T, b::T, fb::T,
                 atol::T, rtol::T) where {T<:AbstractFloat}

    # Check the assumptions and the tolerance parameters.
    (fa > 0) == (fb > 0) && error("f(a) and f(b) must have different signs")
    @assert atol > 0
    @assert 0 < rtol < 1

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

        # Check for convergence.
        m = (c - b)/2
        (abs(m) ≤ tol || fb == 0) && break

        # See if a bisection is forced.
        if abs(e) < tol || abs(fa) ≤ abs(fb)
            # Bounds decreasing too slowly, use bisection.
            d = e = m
        else
            s = fb/fa
            if a == c
                # Linear interpolation.
                p = 2*m*s
                q = 1 - s
            else
                # Inverse quadratic interpolation.
                q = fa/fc
                r = fb/fc
                p = (2*m*q*(q - r) - (b - a)*(r - 1))*s
                q = (q - 1)*(r - 1)*(s - 1)
            end
            if p > 0
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
        elseif m > 0
            b += tol
        else
            b -= tol
        end
        fb = T(f(b))
        fb == 0 && break
        if (fb > 0) == (fc > 0)
            # Drop point C (make it coincident with point A) and adjust bounds
            # of interval.
            c, fc = a, fa
            e = d = b - a
        end
    end
    return (b, fb)
end

"""
# Brent's method for finding a minimum of a function

    fmin([T=Float64,] f, a, b; atol=..., rtol=...) -> (x, fx, xlo, xhi)

seeks a local minimum of a function `f(x)` in the interval `[a,b]`.

The method used is a combination of golden section search and successive
parabolic interpolation.  Convergence is never much slower than that for a
Fibonacci search.  If `f` has a continuous second derivative which is positive
at the minimum (which is not at `a` or `b`), then convergence is superlinear,
and usually of the order of about 1.3247.

The keywords `rtol` and `atol` specify a tolerance `tol = rtol*abs(x) + atol`.
The function `f` is never evaluated at two points closer than `tol`.

If `f` is a unimodal function and the computed values of `f` are always
unimodal when separated by at least `sqrt(eps)*abs(x) + (atol/3)`, then `fmin`
approximates the abscissa of the global minimum of `f` on the interval `[a,b]`
with an error less than `3*sqrt(eps)*abs(fmin) + atol`.

If `f` is not unimodal, then `fmin` may approximate a local, but perhaps
non-global, minimum to the same accuracy.

This function has been derived from Richard Brent's FORTRAN 77 code FMIN which
itself is a slightly modified translation of the Algol 60 procedure LOCALMIN
given in:

> Richard Brent, "Algorithms for minimization without derivatives,"
> Prentice-Hall, inc. (1973).

## Arguments

* `f` - The user-supplied function whose local minimum is being sought.  Called
  as `f(x)` to evaluates the function at `x`.

* `a`, `b` - The endpoints of the interval.

## Keywords

* `atol` - A positive absolute error tolerance.

* `rtol` - A positive relative error tolerance.  `rtol` should be no smaller
  than twice the relative machine precision, and preferably not much less than
  the square root of the relative machine precision.

* `T` - The floating-point data type to use for computations.  Defaulting to
  `Float64`.  The function `f(x)` must return a result of type `T` or which can
  be automatically promoted to type `T`.


## Result

The returned value is the 4-tuple `(x, fx, xlo, xhi)` where `x` is the
estimated value of an abscissa for which `f` attains a local minimum value in
`[a,b]`, `fx` is the function value at `x`, `xlo` and `xhi` are the bounds for
the position of the local minimum.

"""
fmin(f, a::Real, b::Real; kwds...) = fmin(Float64, f, a, b; kwds...)

function fmin(::Type{T}, f, a::Real, b::Real;
              atol::Real = fmin_atol(T),
              rtol::Real = fmin_rtol(T)) where {T<:AbstractFloat}
    fmin0(f, T(a), T(b), T(atol), T(rtol))
end

"""
    fmin0(f, a, b, atol, rtol)

runs Brent's algorithm given a search interval `[a,b]` and tolerances.
Floating-point type is guessed from the type of the arguments.

"""
function fmin0(f, a::T, b::T,
               atol::T, rtol::T) where {T<:AbstractFloat}
    # Make sure A and B are properly ordered.  Initialize the search and call
    # the real worker.
    if a > b
        a, b = b, a
    end
    x = a + goldstep(T)*(b - a)
    fx = T(f(x))
    _fmin(f, a, b, x, fx, x, fx, x, fx, atol, rtol)
end

"""
    fmin1(f, a, b, x, fx, atol, rtol)

runs Brent's algorithm with a single given initial point, `x`, inside the
search interval `[a,b]` and with one known function value: `fx = f(x)`.
Floating-point type is guessed from the type of the arguments.

"""
function fmin1(f, a::T, b::T,
               x::T, fx::T,
               atol::T, rtol::T) where {T<:AbstractFloat}
    # Make sure A and B are properly ordered and check that given point is in
    # the interval.  Then call main loop of Brent's algorithm.
    if a > b
        a, b = b, a
    end
    a ≤ x ≤ b || error("given point outside search interval")
    _fmin(f, a, b, x, fx, x, fx, x, fx, atol, rtol)
end

"""
    fmin2(f, a, b, x, fx, w, fw, atol, rtol)

runs Brent's algorithm with two given initial points `x` and `w` (not
necessarily distinct) inside the search interval `[a,b]` and with known
function values `fx = f(x)` and `fw = f(w)`.  Floating-point type is guessed
from the type of the arguments.

"""
function fmin2(f, a::T, b::T,
               x::T, fx::T,
               w::T, fw::T,
               atol::T, rtol::T) where {T<:AbstractFloat}
    # Make sure A and B are properly ordered and check that given points are in
    # the interval.
    if a > b
        a, b = b, a
    end
    (a ≤ x ≤ b && a ≤ w ≤ b) ||
        error("given point(s) outside search interval")

    # Reorder the points as assumed by Brent's algorithm.
    if fw < fx
        x, fx, w, fw = w, fw, x, fx
    end
    _fmin(f, a, b, x, fx, w, fw, w, fw, atol, rtol)
end

"""
    fmin3(f, a, b, x, fx, w, fw, v, fv, atol, rtol)

runs Brent's algorithm with three given initial points `x`, `w` and `v` (not
necessarily distinct) inside the search interval `[a,b]` and with known
function values `fx = f(x)`, `fw = f(w)` and `fv = f(v)`.  Floating-point type
is guessed from the type of the arguments.

"""
function fmin3(f, a::T, b::T,
               x::T, fx::T,
               w::T, fw::T,
               v::T, fv::T,
               atol::T, rtol::T) where {T<:AbstractFloat}
    # Make sure A and B are properly ordered and check that given points are in
    # the interval.
    if a > b
        a, b = b, a
    end
    (a ≤ x ≤ b && a ≤ w ≤ b && a ≤ v ≤ b) ||
        error("given point(s) outside search interval")

    # Reorder the points as assumed by Brent's algorithm.
    if fw < fx
        x, fx, w, fw = w, fw, x, fx
    end
    if fv < fx
        x, fx, v, fv = v, fv, x, fx
    end
    if fv < fw
        w, fw, v, fv = v, fv, w, fw
    end
    _fmin(f, a, b, x, fx, w, fw, v, fv, atol, rtol)
end

"""
    _fmin(f, a, b, x, fx, w, fw, v, fv, atol, rtol)

performs the main loop of Brent's algorithm assuming that all parameters are
properly set (as explained below).

Original Brent's algorithm assumes that the minimum is in the open interval
`(a,b)` with `a ≤ b` and keeps track of the following variables:

- `x`, `fx = f(x)`: position and least function value found so far;
- `w`, `fw = f(w)`: previous values of `x` and `fx`;
- `v`, `fv = f(v)`: previous values of `w` and fw`;
- `d`: computed step (new try is: u = x + d, unless d too small);
- `e`: the previous value of `d`, if a parabolic step is taken; the
  difference between the most distant current endpoint and `x`, if a
  golden step is taken.

Other variables need not be saved, notably:

- `u`, `fu = f(u)` the next point to try and its function value.

Thus the main loop of Brent's algorithm can be entered with any `x`, `w`, `v`
(not necessarily distinct) which are in `[a,b]` and such that:

    fx = f(x)  ≤  fw = f(w)  ≤  fv = f(v)

other internal variables are:

    d = x - w
    e = w - v

"""
function _fmin(f, a::T, b::T,
               x::T, fx::T,
               w::T, fw::T,
               v::T, fv::T,
               atol::T, rtol::T) where {T<:AbstractFloat}
    # Constant for golden step.
    c = goldstep(T)

    # Check tolerances. (Other arguments are assumed to be checked by the
    # caller.)
    @assert atol ≥ 0
    @assert 0 < rtol < 1

    # Initialize.
    d = x - w
    e = w - v

    while true

        # Compute mid-point and check the stopping criterion.
        m = (a + b)/2
        tol = rtol*abs(x) + atol
        tol2 = 2*tol
        if abs(x - m) ≤ tol2 - (b - a)/2
            return (x, fx, a, b)
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
        elseif d > 0
            u = x + tol
        else
            u = x - tol
        end
        fu = T(f(u))

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
    fminbrkt(f, x, fx, w, fw, v, fv, atol, rtol)

runs Brent's algorithm to minimize function `f` with a bracket of the minimum
defined by 3 points (`x`, `w` and `v`) with known function values (`fx`, `fw`
and `fv`) and such that the least function value is at `x` which is inside the
interval `[v,w]`.

"""
function fminbrkt(f, x::T, fx::T, w::T, fw::T,
                  v::T, fv::T, atol::T, rtol::T) where {T<:AbstractFloat}
    # Reorder the points as assumed by Brent's algorithm.
    if fv < fw
        v, fv, w, fw = w, fw, v, fv
    end
    if v < w
        a, b = v, w
    else
        a, b = w, v
    end
    (a ≤ x ≤ b && fx ≤ fw) || error("illegal bracket")
    _fmin(f, a, b, x, fx, w, fw, v, fv, atol, rtol)
end

end # module
