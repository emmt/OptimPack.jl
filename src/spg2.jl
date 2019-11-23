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

mutable struct SpgResult{T,N}
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
    #    new{T,N}(x, xbest, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, :SEARCHING)
end
SpgResult(x::DenseArray{T,N}, xbest::DenseArray{T,N}) where {T,N} =
    SpgResult{T,N}(x, xbest, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, :SEARCHING)

function spg2(fg!::Function,
              prj!::Function,
              x0::DenseArray{T,N},
              m::Integer = 10;
              maxit::Integer = typemax(Int),
              maxfc::Integer = typemax(Int),
              eps1::Real = 1e-6,
              eps2::Real = 1e-6,
              eta::Real = 1.0,
              ftol::Real = 1e-4,
              lmin::Real = 1e-30,
              lmax::Real = 1e+30,
              amin::Real = 0.1,
              amax::Real = 0.9,
              printer = nothing,
              verb::Bool = false) where {T<:AbstractFloat,N}

    # Allocate workspace.
    x0 = copy(x0)
    dims = size(x0)
    space = DenseVariableSpace(T, dims)
    d = Array{T}(undef, dims)
    g = Array{T}(undef, dims)
    x = Array{T}(undef, dims)
    g = Array{T}(undef, dims)
    g0 = Array{T}(undef, dims)
    xbest = Array{T}(undef, dims)

    # Wrap OptimPack variables over Julia workspace arrays.
    Vx = wrap(space, x)
    Vg = wrap(space, g)
    Vd = wrap(space, d)
    Vx0 = wrap(space, x0)
    Vg0 = wrap(space, g0)

    # Initialization.
    ws = SpgResult(x, xbest)
    local sty, sts, f0, f
    if m > 1
        lastfv = Array{T}(undef, m)
        fill!(lastfv, T(Inf))
    else
        lastfv = nothing
    end

    # Project initial guess.
    prj!(x0, x0)
    copyto!(x, x0)
    ws.pcnt += 1

    # Evaluate function and gradient.
    f = fg!(x, g)
    ws.fcnt += 1

    # Initialize best solution and best function value.
    ws.fbest = f
    copyto!(xbest, x)

    # Main loop.
    while true

        # Compute continuous projected gradient (and its norms)
        # as: `pg = (x - prj(x - eta*g))/eta`.
        combine!(Vd, 1, Vx, -eta, Vg)
        prj!(d, d)
        combine!(Vd, 1/eta, Vx, -1/eta, Vd)
        ws.pcnt += 1
        ws.pgtwon = norm2(Vd)
        ws.pginfn = norminf(Vd)

        # Print iteration information
        if printer !== nothing
            printer(ws)
        end
        if verb
            @printf("ITER = %-5d  EVAL = %-5d  PROJ = %-5d  F(%s) =%24.17e  ||PG||oo = %17.10e\n",
                    ws.iter, ws.fcnt, ws.pcnt, (f ≤ ws.fbest ? "+" : "-"), f, ws.pginfn)
        end

        # Test stopping criteria.
        if ws.pginfn ≤ eps1
            # Gradient infinite-norm stopping criterion satisfied, stop.
            ws.status = :INFNORM_CONVERGENCE
            return ws
        end
        if ws.pgtwon ≤ eps2
            # Gradient 2-norm stopping criterion satisfied, stop.
            ws.status = :TWONORM_CONVERGENCE
            return ws
        end
        if ws.iter ≥ maxit
            # Maximum number of iterations exceeded, stop.
            ws.status = :TOO_MANY_ITERATIONS
            return ws
        end
        if ws.fcnt ≥ maxfc
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
        if ws.iter == 0
            # Initial steplength. (FIXME: check type stability)
            lambda = min(lmax, max(lmin, 1/ws.pginfn))
        else
            Vs = Vx0 # alias
            Vy = Vg0 # alias
            combine!(Vs, 1, Vx, -1, Vx0)
            combine!(Vy, 1, Vg, -1, Vg0)
            sty = dot(Vs, Vy)
            if sty > 0
                # Safeguarded Barzilai & Borwein spectral steplength.
                sts = dot(Vs, Vs)
                lambda = min(lmax, max(lmin, sts/sty))
            else
                lambda = lmax
            end
        end

        # Save current point.
        copyto!(x0, x)
        copyto!(g0, g)
        f0 = f

        # Compute the spectral projected gradient direction and <G,D>
        combine!(Vx, 1, Vx, -lambda, Vg)
        prj!(x, x)
        ws.pcnt += 1
        combine!(Vd, 1, Vx, -1, Vx0) # d = x - x0
        delta = dot(Vg0, Vd)

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
                copyto!(xbest, x)
            end

            # Test stopping criteria.
            if f ≤ fmax + stp*ftol*delta
                # Nonmonotone Armijo-like stopping criterion satisfied, stop.
                break
            end
            if ws.fcnt ≥ maxfc
                # Maximum number of function evaluations exceeded, stop.
                ws.status = :TOO_MANY_EVALUATIONS
                return ws
            end

            # Safeguarded quadratic interpolation.
            q = -delta*(stp*stp)
            r = (f - f0 - stp*delta)*2
            if r > 0 && amin*r ≤ q ≤ amax*stp*r
                stp = q/r
            else
                stp /= 2
            end

            # Compute trial point.
            combine!(Vx, 1, Vx0, stp, Vd) # x = x0 + stp*d
        end

        if ws.status != :SEARCHING
            # The number of function evaluations was exceeded inside the line
            # search.
            return ws
        end

        # Proceed with next iteration.
        ws.iter += 1

    end
end
