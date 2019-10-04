#
# spg2.jl --
#
# Implements Spectral Projected Gradient Method (Version 2: "continuous
# projected gradient direction") to find the local minimizers of a given
# function with convex constraints, described in:
#
# [1] E. G. Birgin, J. M. Martinez, and M. Raydan, "Nonmonotone spectral
#     projected gradient methods on convex sets", SIAM Journal on Optimization
#     10, pp. 1196-1211 (2000).
#
# [2] E. G. Birgin, J. M. Martinez, and M. Raydan, "SPG: software for
#     convex-constrained optimization", ACM Transactions on Mathematical
#     Software (TOMS) 27, pp. 340-349 (2001).
#
# ----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 2014-2019: Éric Thiébaut.
#
# ----------------------------------------------------------------------------

struct SpgResult{T,N}
    x::DenseArray{T,N}
    xbest::DenseArray{T,N}
    f::Float64
    fbest::Float64
    pgtwon::Float64
    pginfn::Float64
    fcnt::Int
    pcnt::Int
    iter::Int
    status::Symbol
    #(::Type{SpgResult(x::DenseArray{T,N}}){T,N}, xbest::DenseArray{T,N}) =
    #    new{T,N}(x, xbest, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, :WORK_IN_PROGRESS)
end
SpgResult(x::DenseArray{T,N}, xbest::DenseArray{T,N}) where {T,N} =
    SpgResult{T,N}(x, xbest, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, :WORK_IN_PROGRESS)

function spg2(fg!::Function, prj!::Function,
              x0::DenseArray{T,N}, m::Integer;
              maxit::Integer=-1, maxfc::Integer=-1,
              eps1::Real=1e-6, eps2::Real=1e-6,
              eta::Real=1.0, ftol::Real=1e-4,
              lmin::Real=1e-30, lmax::Real=1e+30,
              amin::Real=0.1, amax::Real=0.9,
              printer=nothing, verb::Bool=false) where {T,N}

    # Allocate workspace.
    x0 = copy(x0)
    dims = size(x0)
    space = DenseVectorSpace(T, dims)
    d = Array{T}(undef, dims)
    g = Array{T}(undef, dims)
    x = Array{T}(undef, dims)
    g = Array{T}(undef, dims)
    g0 = Array{T}(undef, dims)
    xbest = Array{T}(undef, dims)
    _x = wrap(space, x)
    _g = wrap(space, g)
    _d = wrap(space, d)
    _x0 = wrap(space, x0)
    _g0 = wrap(space, g0)
    _xbest = wrap(space, xbest)

    # Initialization.
    ws = SpgResult(x, xbest)
    local sty, sts, f0, f
    if m > 1
        lastfv = Array{T}(undef, m)
        fill!(lastfv, inf(T))
    else
        lastfv = nothing
    end

    # Project initial guess.
    prj!(x, x0)
    ws.pcnt += 1

    # Evaluate function and gradient.
    f = fg!(x, g)
    ws.fcnt += 1

    # Initialize best solution and best function value.
    ws.fbest = f
    copyto!(_xbest, _x)

    # Main loop.
    while true

        # Compute continuous projected gradient (and its norms).
        axpby!(_d, 1.0, _x, eta, _g)
        prj!(_d, _d)
        axpby!(_d, 1.0/eta, _x, -1.0/eta, _d)
        ws.pcnt += 1
        ws.pgtwon = norm2(_d)
        ws.pginfn = norminf(_d)

        # Print iteration information
        if printer <: Function
            printer(ws)
        end
        if verb
            @printf("ITER = %-5d  EVAL = %-5d  PROJ = %-5d  F(%s) =%24.17e  ||PG||oo = %17.10e\n",
                    ws.iter, ws.fcnt, ws.pcnt, (f <= ws.fbest ? "+" : "-"), f, ws.pginfn)
        end

        # Test stopping criteria.
        if ws.pginfn <= eps1
            # Gradient infinite-norm stopping criterion satisfied, stop.
            ws.status = :CONVERGENCE_WITH_INFNORM
            return ws
        end
        if ws.pgtwon <= eps2
            # Gradient 2-norm stopping criterion satisfied, stop.
            ws.status = :CONVERGENCE_WITH_TWONORM
            return ws
        end
        if maxit >= 0 && ws.iter >= maxit
            # Maximum number of iterations exceeded, stop.
            ws.status = :TOO_MANY_ITERATIONS
            return ws
        end
        if maxfc >= 0 && ws.fcnt >= maxfc
            # Maximum number of function evaluations exceeded, stop.
            ws.status = :TOO_MANY_EVALUATIONS
            return ws
        end

        # Store function value for the nonmonotone line search and
        # find maximum function value since m last calls.
        if m > 1
            lastfv[(ws.iter%m) + 1] = f
            fmax = maximum(lastfv)
        else
            fmax = f
        end

        # Compute spectral steplength.
        if iter == 0
            # Initial steplength.
            lambda = min(lmax, max(lmin, 1.0/ws.pginfn))
        else
            _s = _x0 # alias
            _y = _g0 # alias
            axpby!(_s, 1.0, _x, -1.0, _x0)
            axpby!(_y, 1.0, _g, -1.0, _g0)
            sty = dot(_s, _y)
            if sty > 0.0
                # Safeguarded Barzilai & Borwein spectral steplength.
                sts = dot(_s, _s)
                lambda = min(lmax, max(lmin, sts/sty))
            else
                lambda = lmax
            end
        end

        # Save current point.
        copyto!(_x0, _x)
        copyto!(_g0, _g)
        f0 = f

        # Compute the spectral projected gradient direction and <G,D>
        axpby!(_x, 1.0, _x, -lambda, _g)
        prj(_x, _x)
        ws.pcnt += 1
        axpby!(_d, 1.0, _x, -1.0, _x0)
        delta = dot(_g0, _d)

        # Nonmonotone line search.
        stp = 1.0 # Step length for first trial.
        while true
            # Evaluate function and gradient at trial point.
            f = fg!(x, g)
            ws.fcnt += 1

            # Compare the new function value against the best function
            # value and, if smaller, update the best function value and the
            # corresponding best point.
            if f < ws.fbest
                ws.fbest = f
                copyto!(_xbest, _x)
            end

            # Test stopping criteria.
            if f <= fmax + stp*ftol*delta
                # Nonmonotone Armijo-like stopping criterion satisfied, stop.
                break
            end
            if maxfc >= 0 && ws.fcnt >= maxfc
                # Maximum number of function evaluations exceeded, stop.
                ws.status = :TOO_MANY_EVALUATIONS
                return ws
            end

            # Safeguarded quadratic interpolation.
            q = -delta*(stp*stp)
            r = (f - f0 - stp*delta)*2.0
            if r > 0.0 && amin*r <= q <= amax*stp*r
                stp = q/r
            else
                stp /= 2.0
            end

            # Compute trial point.
            axpby!(_x, 1.0, _x0, stp, _d)
        end

        if ws.status != :WORK_IN_PROGRESS
            # The number of function evaluations was exceeded inside the line
            # search.
            return ws
        end

        # Proceed with next iteration.
        ws.iter += 1

    end
end
