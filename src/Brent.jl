#
# Brent.jl --
#
# Find a local root or a local minimum of an univariate function by Brent's
# methods described in:
#
# [1] Richard Brent, "Algorithms for minimization without derivatives,"
#     Prentice-Hall, inc. (1973).
#
# Global minimizer `fmin_global` is described in:
#
# [2] Ferréol Soulez, Éric Thiébaut, Michel Tallon, Isabelle Tallon-Bosc
#     and Paulo Garcia, "Optimal a posteriori fringe tracking in optical
#     interferometry", Proc. SPIE 9146, Optical and Infrared Interferometry
#     IV, 91462Y (July 24, 2014); doi:10.1117/12.2056590
#
#-----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 1973, Richard Brent.
# Copyright (C) 2015, Éric Thiébaut.
#
#-----------------------------------------------------------------------------


# FIXME: make high level drivers fmin1, fmin2, ... which
#        check their arguments and set default tolerances
#        while low level functions do not do that?

# The following functions compute the default and minimal tolerances
# for the tolearances of the Brent's methods.
fmin_min_rtol(T) = convert(T,2*eps(T))
fmin_def_rtol(T) = convert(T,sqrt(eps(T)))
fmin_min_atol(T) = zero(T)
fmin_def_atol(T) = convert(T,3*realmin(T))
fzero_min_rtol(T) = zero(T)
fzero_def_rtol(T) = convert(T,4*eps(T))
fzero_min_atol(T) = zero(T)
fzero_def_atol(T) = convert(T,2*realmin(T))

# Declare specialized versions of these functions for the common
# floating-point types and manage to make them return a precomputed
# constant value.
# `goldstp` is the square of the inverse of the golden ratio.
let goldstp = (3.0 - sqrt(5.0))/2.0
    for T in (Float16, Float32, Float64)
        for func in (:fmin_min_rtol, :fmin_min_atol,
                     :fmin_def_rtol, :fmin_def_atol,
                     :fzero_min_rtol, :fzero_min_atol,
                     :fzero_def_rtol, :fzero_def_atol)
            @eval ($func)(::Type{$T}) = $(eval(func)(T))
        end
        @eval fmin_goldstp(::Type{$T}) = $(convert(T, goldstp))
    end
end

fmin_get_atol(T, x) = (x == nothing ? fmin_def_atol(T) : convert(T, x))
fmin_get_rtol(T, x) = (x == nothing ? fmin_def_rtol(T) : convert(T, x))
fzero_get_atol(T, x, a, b) = convert(T, (x == nothing ? eps(T)*abs(a - b) : x))
fzero_get_rtol(T, x) = (x == nothing ? fzero_def_rtol(T) : convert(T, x))

#
# `fzero()` seeks a local root of a function F(X) in an interval [A,B].
#
# It is assumed that F(A) and F(B) have opposite signs.  This is checked,
# and an error is raised if this is not satisfied.  FZERO returns a zero X
# in the given interval [A,B] to within a tolerance: RTOL*abs(X) + ATOL.
#
# This function has been derived from Richard Brent's F77 code ZEROIN which
# itself is a slightly modified translation of the Algol 60 procedure ZERO
# given in:
#
# [1] Richard Brent, "Algorithms for minimization without derivatives,"
#     Prentice-Hall, inc. (1973).
#
# Arguments:
#
#   f    - The user-supplied function whose local root is being sought.
#          Called as F(X) to evaluates the function at any X in the
#          interval [A,B].
#
#   a, b - The endpoints of the initial search interval.
#
# Keywords:
#
#   atol - The absolute tolerance for the solution.
#
#   rtol - The relative tolerance for the solution.  The recommended (and
#          default) value for RTOL is 4*EPSILON where EPSILON is the
#          relative machine precision defined as the smallest representable
#          number such that 1 + EPSILON > 1.
#
#   T    - The floating-point data type to use for computations.  By
#          default, T=Float64.  The function f(x) must return a result
#          of type T or which can be automatically promoted to type T.
#
# Result:
#
#    The returned value is a tuple of 2 values:
#
#    (X, FX) - Where X is the estimated value of an abscissa for which
#           F is approximately zero in [A,B]; FX is the function value
#           at X.
#
function fzero(f::Function, a::Real, b::Real;
               T::Type=Float64, atol=nothing, rtol=nothing)

    # Explicitly declare types of variables to prevent accidental changes
    # of type during the execution of the code.  We also re-assign the
    # value of the passed arguments to force conversion (FIXME: there should
    # be a macro for that).
    a::T = a
    b::T = b
    local fa::T, fb::T, c::T, fc::T, tol::T, m::T, e::T, d::T

    # Some constants.
    const ZERO::T = zero(T)
    const ONE::T = one(T)
    const TWO::T = ONE + ONE
    const HALF::T = ONE/TWO
    const THREE::T = ONE + TWO

    # Get tolerance parameters.
    atol::T = fzero_get_atol(T, atol, a, b)
    rtol::T = fzero_get_rtol(T, rtol)

    # Compute the function value at the endpoints and check the
    # assumptions.
    fa = f(a)
    fa == ZERO && return (a, fa)
    fb = f(b)
    fb == ZERO && return (b, fb)
    if (fa > ZERO) == (fb > ZERO)
        error("f(a) and f(b) must have different signs")
    end
    fc = fb # to trigger bound update below
    while true
        if (fb > ZERO) == (fc > ZERO)
            # Drop point C (make it coincident with point A)
            # and adjust bounds of interval.
            c, fc = a, fa
            e = d = b - a
        end

        # Make sure B is the best point so far.
        if abs(fc) < abs(fb)
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa
        end

        # Compute tolerance.  In original Brent's method, the precision and
        # the computed tolerance are given by:
        #    PREC = 4*EPS*abs(X) + 2*T
        #    TOL = 2*EPS*abs(b) + T = PREC/2
        # and we want:
        #    PREC = RTOL*abs(X) + ATOL
        # thus the expression of the tolerance parameter becomes:
        tol = HALF*(rtol*abs(b) + atol)

        # Check for convergence.
        m = (c - b)*HALF
        if abs(m) <= tol || fb == ZERO
	    return (b, fb)
        end

        # See if a bisection is forced.
        if abs(e) < tol || abs(fa) <= abs(fb)
            # Bisection.
            d = e = m
        else
            local p::T, q::T, r::T, s::T = fb/fa, two_p::T
            if a == c
                # Linear interpolation.
                p = TWO*m*s
                q = ONE - s
            else
                # Inverse quadratic interpolation.
                q = fa/fc
                r = fb/fc
                p = (TWO*m*q*(q - r) - (b - a)*(r - ONE))*s
                q = (q - ONE)*(r - ONE)*(s - ONE)
            end
            if p > ZERO
                q = -q
            else
                p = -p
            end
            two_p = p + p
            if two_p < THREE*m*q - tol*abs(q) && two_p < abs(e*q)
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
        elseif m > ZERO
            b += tol
        else
            b -= tol
        end
        fb = f(b)
        fb == ZERO && return (b, fb)
    end
end

# FIXME: AUTOMATIC BRACKETING
#
# If the interval to consider is not bounded or only left/right bounded,
# the idea is to find a suitable interval (A,B) where at least one
# minimum must exists (if the function is continue) and start Brent's
# algorithm with correct values for X, FX, ... (in order to save some
# function evaluations).

#
# `fmin()` seeks a local minimum of a function F(X) in an interval [A,B].
#
# The method used is a combination of golden section search and successive
# parabolic interpolation.  Convergence is never much slower than that for
# a Fibonacci search.  If F has a continuous second derivative which is
# positive at the minimum (which is not at A or B), then convergence is
# superlinear, and usually of the order of about 1.324....
#
# The values RTOL and ATOL define a tolerance TOL = RTOL*abs(X) + ATOL.  F
# is never evaluated at two points closer than TOL.
#
# If F is a unimodal function and the computed values of F are always
# unimodal when separated by at least SQEPS*abs(X) + (ATOL/3), then FMIN
# approximates the abscissa of the global minimum of F on the interval
# [A,B] with an error less than 3*SQEPS*abs(FMIN) + ATOL.
#
# If F is not unimodal, then FMIN may approximate a local, but perhaps
# non-global, minimum to the same accuracy.
#
# This function has been derived from Richard Brent's FORTRAN77 code FMIN
# which itself is a slightly modified translation of the Algol 60 procedure
# LOCALMIN given in:
#
# [1] Richard Brent, "Algorithms for minimization without derivatives,"
#     Prentice-Hall, inc. (1973).
#
# Arguments:
#
#   a, b - The endpoints of the interval.
#
#   f    - The user-supplied function whose local minimum is being
#          sought.  Called as F(X) to evaluates the function at X.
#
# Keywords:
#
#   atol - A positive absolute error tolerance.
#
#   rtol - A positive relative error tolerance.  RTOL should be no smaller
#          than twice the relative machine precision, and preferably not
#          much less than the square root of the relative machine
#          precision.
#
#   T    - The floating-point data type to use for computations.  By
#          default, T=Float64.  The function f(x) must return a result
#          of type T or which can be automatically promoted to type T.
#
# Result:
#
#   The returned value is a tuple of 4 values:
#
#   (X, FX, XLO, XHI) - X is the estimated value of an abscissa for which F
#          attains a local minimum value in [A,B]; FX is the function value
#          at X; XLO and XHI are the bounds for the position of the local
#          minimum.
#
function fmin(f::Function, a::Real, b::Real;
              T::Type=Float64, atol=nothing, rtol=nothing)

    # Enforce floating point type.
    a::T = a
    b::T = b
    atol::T = fmin_get_atol(T, atol)
    rtol::T = fmin_get_rtol(T, rtol)

    # Make sure A and B are properly ordered.
    if a > b
        a, b = b, a
    end

    # Initialize the search.
    const c::T = fmin_goldstp(T) # square of the inverse of the golden ratio
    x::T = a + c*(b - a)
    fx::T = f(x)
    _fmin(f, a, b, x, fx, x, fx, x, fx, atol, rtol)
end

# This function is called to start Brent's algorithm with a single given
# point, x, inside the search interval [a,b] and with known function value
# fx = f(x).
function _fmin1{T<:Real}(f::Function, a::T, b::T, x::T, fx::T,
                         atol::T, rtol::T)
    # Make sure A and B are properly ordered.
    if a > b
        a, b = b, a
    end
    if x < a || x > b
        error("given point outside search interval")
    end
    _fmin(f, a, b, x, fx, x, fx, x, fx, atol, rtol)
end

# This function is called to start Brent's algorithm with two given points,
# (x and w not necessarily distinct) inside the search interval [a,b] and
# with known function values (fx and fw).
function _fmin2{T}(f::Function, a::T, b::T, x::T, fx::T, w::T, fw::T,
                   atol::T, rtol::T)
    # Make sure A and B are properly ordered and check that given
    # points are in the interval.
    if a > b
        a, b = b, a
    end
    if w < a || w > b || x < a || x > b
        error("given point(s) outside search interval")
    end

    # Reorder the points as assumed by Brent's algorithm.
    if fw < fx
        x, fx, w, fw = w, fw, x, fx
    end
    _fmin(f, a, b, x, fx, w, fw, w, fw, atol, rtol)
end

# This function is called to start Brent's algorithm with 3 given
# points (x, w, and v not necessarily distinct) inside the search
# interval [a,b] and with known function values (fx, fw and fv).
function _fmin3{T<:Real}(f::Function, a::T, b::T, x::T, fx::T,
                         w::T, fw::T, v::T, fv::T, atol::T, rtol::T)
    # Make sure A and B are properly ordered.
    if a > b
        a, b = b, a
    end
    if v < a || v > b || w < a || w > b || x < a || x > b
        error("given point(s) outside search interval")
    end

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

#
# Original Brent's algorithm assumes that the minimum is in (a,b) with
# a <= b and keeps track of the following variables:
#
#   x, fx = f(x) - least function value found so far;
#   w, fw = f(w) - previous value of x, fx;
#   v, fv = f(v) - previous value of w, fw;
#   d - computed step (new try is: u = x + d, unless d too small);
#   e - the previous value of d, if a parabolic step is taken; the
#       difference between the most distant current endpoint and x,
#       if a golden step is taken.
#
# Other variables need not be saved.
#   u, fu = f(u) - next point to try and its function value.
#
# Thus the main loop of Brent's algorithm can be entered with any x, w, v
# (not necessarily distinct) which are in [a,b] and such that:
#
#    fx = f(x)  <=  fw = f(w)  <=  fv = f(v)
#
# with:
#
#    d = x - w
#    e = w - v
#
# The following function is the main loop of Brent's algorithm.  It assumes
# that all parameters are properly set (as explained above).
#
function _fmin{T<:Real}(f::Function, a::T, b::T, x::T, fx::T,
                        w::T, fw::T, v::T, fv::T, atol::T, rtol::T)
    # Explicitly declare types of variables to prevent accidental changes
    # of type during the execution of the code.  We also re-assign the
    # value of the passed arguments to force conversion (FIXME: there should
    # be a macro for that).
    a::T = a; b::T = b
    x::T = x; fx::T = fx
    w::T = w; fw::T = fw
    v::T = v; fv::T = fv
    atol::T = atol; rtol::T = rtol
    const ZERO::T = zero(T)
    const ONE::T = one(T)
    const TWO::T = ONE + ONE
    const HALF::T = ONE/TWO
    const c::T = fmin_goldstp(T)
    local d::T = x - w
    local e::T = w - v
    local m::T, tol::T, t2::T, u::T, fu::T
    local take_golden_step::Bool

    while true

        # Compute mid-point and check the stopping criterion.
        m = HALF*(a + b)
        tol = rtol*abs(x) + atol
        t2 = tol + tol
        if abs(x - m) <= t2 - HALF*(b - a)
            return (x, fx, a, b)
        end

        # Determine next step to take.
        take_golden_step = true
        if abs(e) > tol
            # Fit a parabola (make sure final Q >= 0).
            local p::T, q::T, r::T
            r = (x - w)*(fx - fv)
            q = (x - v)*(fx - fw)
            if q > r
                p = (x - w)*r - (x - v)*q
                q = (q - r)*TWO
            else
                p = (x - v)*q - (x - w)*r
                q = (r - q)*TWO
            end
            if abs(p) < HALF*q*abs(e) && q*(a - x) < p < q*(b - x)
                # Take the parabolic interpolation step.
                take_golden_step = false
                e = d
                d = p/q
                u = x + d
                # F must not be evaluated too close to A or B.
                if u - a < t2 || b - u < t2
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
        if abs(d) >= tol
            u = x + d
        elseif d > ZERO
            u = x + tol
        else
            u = x - tol
        end
        fu = f(u)

        # Update A, B, V, W, and X.
        if fu <= fx
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
            if fu <= fw || w == x
                v, fv = w, fw
	        w, fw = u, fu
            elseif fu <= fv || v == x || v == w
                v, fv = u, fu
            end
        end
    end
end

# This function is called to start Brent's algorithm with a bracket of the
# minimum defined by 3 points (x, w and v) with known function values (fx,
# fw and fv) and such that the least function value is at x which is inside
# the interval [v,w].
function _fminbrkt{T<:Real}(f::Function, x::T, fx::T, w::T, fw::T,
                            v::T, fv::T, atol::T, rtol::T)
    # Reorder the points as assumed by Brent's algorithm.
    if fv < fw
        v, fv, w, fw = w, fw, v, fv
    end
    if v < w
        a, b = v, w
    else
        a, b = w, v
    end
    if x < a || x > b || fx > fw
        error("illegal bracket")
    end
    _fmin(f, a, b, x, fx, w, fw, v, fv, atol, rtol)
end


#
# `fmin_global()` finds the global minimum of an univariate function F.
# The argument X is a vector of coordinates in monotonic order; X[1] and
# X[end] are the endpoints of the global search interval and the other
# values of X are such that no more than a single local minimum lies in any
# subinterval [X(i),X(i+2)].
#
# X = linspace(A,B,N) if these arguments are supplied instead; i.e. the
# global search is performed in the (included) interval [A,B] which is cut
# in N pieces of equal lengths.
#
# If specified, keywords `atol` and `rtol` set the absolute and relative
# tolerances for the precision.
#
# This function implements the BraDi ("Bracket" then "Dig") algorithm
# described in [1].
#
# [1] Ferréol Soulez, Éric Thiébaut, Michel Tallon, Isabelle Tallon-Bosc
#     and Paulo Garcia, "Optimal a posteriori fringe tracking in optical
#     interferometry", Proc. SPIE 9146, Optical and Infrared Interferometry
#     IV, 91462Y (July 24, 2014); doi:10.1117/12.2056590
#
# SEE ALSO: fmin.
function fmin_global(f::Function, a::Real, b::Real, n::Integer;
                     T::Type=Float64, atol=nothing, rtol=nothing)
    fmin_global(f, linspace(convert(T,a), convert(T,b), n),
                atol=atol, rtol=rtol)
end

function fmin_global{T<:Real}(f::Function, x::Vector{T};
                              atol=nothing, rtol=nothing)
    atol::T = fmin_get_atol(T, atol)
    rtol::T = fmin_get_rtol(T, rtol)

    xbest = xa = xb = xc = x[1]
    fbest = fa = fb = fc = f(xc)
    n = length(x)
    for j in 2 : n + 1
        xa = xb
        fa = fb
        xb = xc
        fb = fc
        if j <= n
            xc = x[j]
            fc = f(xc)
            if fc < fbest
                xbest = xc
                fbest = fc
            end
        end
        if fa >= fb <= fc
            # A minimum has been bracketed in [XA,XC].
            xm, fm = _fminbrkt(f, xb, fb, xa, fa, xc, fc, atol, rtol)
            if fm < fbest
                xbest = xm
                fbest = fm
            end
        end
    end
    return (xbest, fbest)
end
