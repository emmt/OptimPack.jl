#
# Powell.jl --
#
# Mike Powell's derivative free optimization methods for Julia.
#
# ----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 2015, Éric Thiébaut.
#
# ----------------------------------------------------------------------------

# Possible status values returned by COBYLA.
const COBYLA_INITIAL_ITERATE       = convert(Cint,  2)
const COBYLA_ITERATE               = convert(Cint,  1)
const COBYLA_SUCCESS               = convert(Cint,  0)
const COBYLA_ROUNDING_ERRORS       = convert(Cint, -1)
const COBYLA_TOO_MANY_EVALUATIONS  = convert(Cint, -2)
const COBYLA_BAD_ADDRESS           = convert(Cint, -3)
const COBYLA_CORRUPTED             = convert(Cint, -4)

# Get a textual explanation of the status returned by COBYLA.
function cobyla_reason(status::Integer)
    ptr = ccall((:cobyla_reason, opklib), Ptr{Uint8}, (Cint,), status)
    if ptr == C_NULL
        error("unknown COBYLA status: ", status)
    end
    bytestring(ptr)
end

# Wrapper for the objective function in COBYLA, the actual objective
# function is provided by the client data.
function cobyla_objfun(n::Cptrdiff_t, m::Cptrdiff_t, x_::Ptr{Cdouble},
                       c_::Ptr{Cdouble}, f_::Ptr{Any})
    x = pointer_to_array(x_, n)
    f = unsafe_pointer_to_objref(f_)
    fx::Cdouble = (m > 0 ? f(x, pointer_to_array(c_, m)) : f(x))
    return fx
end
const cobyla_objfun_c = cfunction(cobyla_objfun, Cdouble,
                                  (Cptrdiff_t, Cptrdiff_t, Ptr{Cdouble},
                                   Ptr{Cdouble}, Ptr{Any}))

function cobyla_check(n::Integer, m::Integer, rhobeg::Real, rhoend::Real)
    if n < 2
        "bad number of variables"
    elseif m < 0
        "bad number of constraints"
    elseif rhoend < 0 || rhoend > rhobeg
        "bad trust region radius settings"
    end
end

function cobyla!(f::Function, x::Vector{Cdouble},
                 m::Integer, rhobeg::Real, rhoend::Real;
                 verbose::Integer=0, maxeval::Integer=500)
    n = length(x)
    reason = cobyla_check(n, m, rhobeg, rhoend)
    reason == nothing || error(reason)
    w = Array(Cdouble, n*(3*n + 2*m + 11) + 4*m + 6)
    iact = Array(Cptrdiff_t, m + 1)
    maxfun = Cptrdiff_t[maxeval]
    status = ccall((:cobyla, opklib), Cint,
                   (Cptrdiff_t, Cptrdiff_t, Ptr{Void}, Ptr{Any},
                    Ptr{Cdouble}, Cdouble, Cdouble, Cptrdiff_t,
                    Ptr{Cptrdiff_t}, Ptr{Cdouble}, Ptr{Cptrdiff_t}),
                   n, m, cobyla_objfun_c, pointer_from_objref(f),
                   x, rhobeg, rhoend, verbose, maxfun, w, iact)
    if status != COBYLA_SUCCESS
        error(cobyla_reason(status))
    end
    #return w[1]
end

# Context for reverse communication variant of COBYLA.
type CobylaContext
    ptr::Ptr{Void}
    n::Int
    m::Int
    rhobeg::Cdouble
    rhoend::Cdouble
    verbose::Int
    maxeval::Int
end

# Create a new reverse communication workspace for COBYLA algorithm.
# A typical usage is:
# ```
# x = Array(Cdouble, n)
# c = Array(Cdouble, m)
# x[...] = ... # initial solution
# ctx = cobyla_create(n, m, rhobeg, rhoend, verbose=1, maxeval=500)
# status = cobyla_get_status(ctx)
# while status == COBYLA_ITERATE
#     fx = ...       # compute function value at X
#     c[...] = ...   # compute constraints at X
#     status = cobyla_iterate(ctx, fx, x, c)
# end
# if status != COBYLA_SUCCESS
#     println("Something wrong occured in COBYLA: ", cobyla_reason(status))
# end
# ```
function cobyla_create(n::Integer, m::Integer,
                       rhobeg::Real, rhoend::Real;
                       verbose::Integer=0, maxeval::Integer=500)
    reason = cobyla_check(n, m, rhobeg, rhoend)
    reason == nothing || error(reason)
    ptr = ccall((:cobyla_create, opklib), Ptr{Void},
                (Cptrdiff_t, Cptrdiff_t, Cdouble, Cdouble,
                 Cptrdiff_t, Cptrdiff_t),
                n, m, rhobeg, rhoend, verbose, maxeval)
    if ptr == C_NULL
        reason = (errno() == Base.Errno.ENOMEM
                  ? "insufficient memory"
                  : "unexpected error")
        error(reason)
    end
    ctx = CobylaContext(ptr, n, m, rhobeg, rhoend, verbose, maxeval)
    finalizer(ctx, ctx -> ccall((:cobyla_delete, opklib), Void,
                                (Ptr{Void},), ctx.ptr))
    return ctx
end

# Perform the next iteration of the reverse communication variant of the
# COBYLA algorithm.  On entry, the workspace status must be
# `COBYLA_ITERATE`, `f` and `c` are the function value and the constraints
# at `x`.  On exit, the returned value (the new workspace status) is:
# `COBYLA_ITERATE` if a new trial point has been stored in `x` and if user
# is requested to compute the function value and the constraints on the new
# point; `COBYLA_SUCCESS` if algorithm has converged and `x` has been set
# with the variables at the solution (the corresponding function value can
# be retrieved with `cobyla_get_last_f`); anything else indicates an error
# (see `cobyla_reason` for an explanatory message).
function cobyla_iterate(ctx::CobylaContext, f::Real, x::Vector{Cdouble},
                        c::Vector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    length(c) == ctx.m || error("bad number of constraints")
    ccall((:cobyla_iterate, opklib), Cint,
          (Ptr{Void}, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}),
          ctx.ptr, f, x, c)
end
# The same but without constraints.
function cobyla_iterate(ctx::CobylaContext, f::Real, x::Vector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    ctx.m == 0 || error("bad number of constraints")
    ccall((:cobyla_iterate, opklib), Cint,
          (Ptr{Void}, Cdouble, Ptr{Cdouble}, Ptr{Void}),
          ctx.ptr, f, x, C_NULL)
end

# Restart COBYLA algorithm using the same parameters.  The return value is
# the new status of the algorithm, see `cobyla_get_status` for details.
function cobyla_restart(ctx::CobylaContext)
    ccall((:cobyla_restart, opklib), Cint, (Ptr{Void},), ctx.ptr)
end

# Get the current status of the algorithm.  Result is: `COBYLA_ITERATE` if
# user is requested to compute F(X) and C(X); `COBYLA_SUCCESS` if algorithm
# has converged; anything else indicates an error (see `cobyla_reason` for
# an explanatory message).
function cobyla_get_status(ctx::CobylaContext)
    ccall((:cobyla_get_status, opklib), Cint, (Ptr{Void},), ctx.ptr)
end

# Get the current number of function evaluations.  Result is -1 if
# something is wrong (e.g. CTX is NULL), nonnegative otherwise.
function cobyla_get_nevals(ctx::CobylaContext)
    ccall((:cobyla_get_nevals, opklib), Cptrdiff_t, (Ptr{Void},), ctx.ptr)
end

# Get the current size of the trust region.  Result is 0 if algorithm not
# yet started (before first iteration), -1 if something is wrong (e.g. CTX
# is NULL), strictly positive otherwise.
function cobyla_get_rho(ctx::CobylaContext)
    ccall((:cobyla_get_rho, opklib), Cdouble, (Ptr{Void},), ctx.ptr)
end

# Get the last function value.  Upon convergence of `cobyla_iterate`
# (i.e. return with status `COBYLA_SUCCESS`), this value corresponds to the
# function at the solution; otherwise, this value corresponds to the
# previous set of variables.
function cobyla_get_last_f(ctx::CobylaContext)
    ccall((:cobyla_get_last_f, opklib), Cdouble, (Ptr{Void},), ctx.ptr)
end

# Get a textual explanation of the current status.
cobyla_get_reason(ctx::CobylaContext) = cobyla_reason(cobyla_get_status(ctx))

function cobyla_test(revcom::Bool=false)
    # Beware that order of operations may affect the result (whithin
    # rounding errors).  I have tried to keep the same ordering as F2C
    # which takes care of that, in particular when converting expressions
    # involving powers.
    prt(s) = println("\n       "*s)
    for nprob in 1:10
        if nprob == 1
            # Minimization of a simple quadratic function of two variables.
            prt("Output from test problem 1 (Simple quadratic)")
            n = 2
            m = 0
            xopt = Array(Cdouble, n)
            xopt[1] = -1.0
            xopt[2] = 0.0
            function ftest(x::Vector{Cdouble})
                r1 = x[1] + 1.0
                r2 = x[2]
                fc = 10.0*(r1*r1) + (r2*r2)
                return fc
            end
        elseif nprob == 2
            # Easy two dimensional minimization in unit circle.
            prt("Output from test problem 2 (2D unit circle calculation)")
            n = 2
            m = 1
            xopt = Array(Cdouble, n)
            xopt[1] = sqrt(0.5)
            xopt[2] = -xopt[1]
            function ftest(x::Vector{Cdouble}, con::Vector{Cdouble})
                fc = x[1]*x[2]
                con[1] = 1.0 - x[1]*x[1] - x[2]*x[2]
                return fc
            end
        elseif nprob == 3
            # Easy three dimensional minimization in ellipsoid.
            prt("Output from test problem 3 (3D ellipsoid calculation)")
            n = 3
            m = 1
            xopt = Array(Cdouble, n)
            xopt[1] = 1.0/sqrt(3.0)
            xopt[2] = 1.0/sqrt(6.0)
            xopt[3] = -0.33333333333333331
            function ftest(x::Vector{Cdouble}, con::Vector{Cdouble})
                fc = x[1]*x[2]*x[3]
                con[1] = 1.0 - (x[1]*x[1]) - 2.0*(x[2]*x[2]) - 3.0*(x[3]*x[3])
                return fc
            end
        elseif nprob == 4
            # Weak version of Rosenbrock's problem.
            prt("Output from test problem 4 (Weak Rosenbrock)")
            n = 2
            m = 0
            xopt = Array(Cdouble, n)
            xopt[1] = -1.0
            xopt[2] = 1.0
            function ftest(x::Vector{Cdouble})
                r2 = x[1]
                r1 = r2*r2 - x[2]
                r3 = x[1] + 1.0
                fc = r1*r1 + r3*r3
                return fc
            end
        elseif nprob == 5
            # Intermediate version of Rosenbrock's problem.
            prt("Output from test problem 5 (Intermediate Rosenbrock)")
            n = 2
            m = 0
            xopt = Array(Cdouble, n)
            xopt[1] = -1.0
            xopt[2] = 1.0
            function ftest(x::Vector{Cdouble})
                r2 = x[1]
                r1 = r2*r2 - x[2]
                r3 = x[1] + 1.0
                fc = r1*r1*10.0 + r3*r3
                return fc
            end
        elseif nprob == 6
            # This problem is taken from Fletcher's book Practical Methods
            # of Optimization and has the equation number (9.1.15).
            prt("Output from test problem 6 (Equation (9.1.15) in Fletcher)")
            n = 2
            m = 2
            xopt = Array(Cdouble, n)
            xopt[1] = sqrt(0.5)
            xopt[2] = xopt[1]
            function ftest(x::Vector{Cdouble}, con::Vector{Cdouble})
                fc = -x[1] - x[2]
                r1 = x[1]
                con[1] = x[2] - r1*r1
                r1 = x[1]
                r2 = x[2]
                con[2] = 1.0 - r1*r1 - r2*r2
                return fc
            end
        elseif nprob == 7
            # This problem is taken from Fletcher's book Practical Methods
            # of Optimization and has the equation number (14.4.2).
            prt("Output from test problem 7 (Equation (14.4.2) in Fletcher)")
            n = 3
            m = 3
            xopt = Array(Cdouble, n)
            xopt[1] = 0.0
            xopt[2] = -3.0
            xopt[3] = -3.0
            function ftest(x::Vector{Cdouble}, con::Vector{Cdouble})
                fc = x[3]
                con[1] = x[1]*5.0 - x[2] + x[3]
                r1 = x[1]
                r2 = x[2]
                con[2] = x[3] - r1*r1 - r2*r2 - x[2]*4.0
                con[3] = x[3] - x[1]*5.0 - x[2]
                return fc
            end
        elseif nprob == 8
            # This problem is taken from page 66 of Hock and Schittkowski's
            # book Test Examples for Nonlinear Programming Codes. It is
            # their test problem Number 43, and has the name Rosen-Suzuki.
            prt("Output from test problem 8 (Rosen-Suzuki)")
            n = 4
            m = 3
            xopt = Array(Cdouble, n)
            xopt[1] = 0.0
            xopt[2] = 1.0
            xopt[3] = 2.0
            xopt[4] = -1.0
            function ftest(x::Vector{Cdouble}, con::Vector{Cdouble})
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                fc = (r1*r1 + r2*r2 + r3*r3*2.0 + r4*r4 - x[1]*5.0
                      - x[2]*5.0 - x[3]*21.0 + x[4]*7.0)
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                con[1] = (8.0 - r1*r1 - r2*r2 - r3*r3 - r4*r4 - x[1]
                          + x[2] - x[3] + x[4])
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                con[2] = (10.0 - r1*r1 - r2*r2*2.0 - r3*r3 - r4*r4*2.0
                          + x[1] + x[4])
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                con[3] = (5.0 - r1*r1*2.0 - r2*r2 - r3*r3 - x[1]*2.0
                          + x[2] + x[4])
                return fc
            end
        elseif nprob == 9
            # This problem is taken from page 111 of Hock and
            # Schittkowski's book Test Examples for Nonlinear Programming
            # Codes. It is their test problem Number 100.
            prt("Output from test problem 9 (Hock and Schittkowski 100)")
            n = 7
            m = 4
            xopt = Array(Cdouble, n)
            xopt[1] =  2.330499
            xopt[2] =  1.951372
            xopt[3] = -0.4775414
            xopt[4] =  4.365726
            xopt[5] = -0.624487
            xopt[6] =  1.038131
            xopt[7] =  1.594227
            function ftest(x::Vector{Cdouble}, con::Vector{Cdouble})
                r1 = x[1] - 10.0
                r2 = x[2] - 12.0
                r3 = x[3]
                r3 *= r3
                r4 = x[4] - 11.0
                r5 = x[5]
                r5 *= r5
                r6 = x[6]
                r7 = x[7]
                r7 *= r7
                fc = (r1*r1 + r2*r2*5.0 + r3*r3 + r4*r4*3.0
                      + r5*(r5*r5)*10.0 + r6*r6*7.0 + r7*r7
                      - x[6]*4.0*x[7] - x[6]*10.0 - x[7]*8.0)
                r1 = x[1]
                r2 = x[2]
                r2 *= r2
                r3 = x[4]
                con[1] = (127.0 - r1*r1*2.0 - r2*r2*3.0 - x[3]
                          - r3*r3*4.0 - x[5]*5.0)
                r1 = x[3]
                con[2] = (282.0 - x[1]*7.0 - x[2]*3.0 - r1*r1*10.0
                          - x[4] + x[5])
                r1 = x[2]
                r2 = x[6]
                con[3] = (196.0 - x[1]*23.0 - r1*r1 - r2*r2*6.0
                          + x[7]*8.0)
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                con[4] = (r1*r1*-4.0 - r2*r2 + x[1]*3.0*x[2]
                          - r3*r3*2.0 - x[6]*5.0 + x[7]*11.0)
                return fc
            end
        elseif nprob == 10
            # This problem is taken from page 415 of Luenberger's book
            # Applied Nonlinear Programming. It is to maximize the area of
            # a hexagon of unit diameter.
            prt("Output from test problem 10 (Hexagon area)")
            n = 9
            m = 14
            xopt = Array(Cdouble, n)
            function ftest(x::Vector{Cdouble}, con::Vector{Cdouble})
                fc = -0.5*(x[1]*x[4] - x[2]*x[3] + x[3]*x[9] - x[5]*x[9]
                           + x[5]*x[8] - x[6]*x[7])
                r1 = x[3]
                r2 = x[4]
                con[1] = 1.0 - r1*r1 - r2*r2
                r1 = x[9]
                con[2] = 1.0 - r1*r1
                r1 = x[5]
                r2 = x[6]
                con[3] = 1.0 - r1*r1 - r2*r2
                r1 = x[1]
                r2 = x[2] - x[9]
                con[4] = 1.0 - r1*r1 - r2*r2
                r1 = x[1] - x[5]
                r2 = x[2] - x[6]
                con[5] = 1.0 - r1*r1 - r2*r2
                r1 = x[1] - x[7]
                r2 = x[2] - x[8]
                con[6] = 1.0 - r1*r1 - r2*r2
                r1 = x[3] - x[5]
                r2 = x[4] - x[6]
                con[7] = 1.0 - r1*r1 - r2*r2
                r1 = x[3] - x[7]
                r2 = x[4] - x[8]
                con[8] = 1.0 - r1*r1 - r2*r2
                r1 = x[7]
                r2 = x[8] - x[9]
                con[9] = 1.0 - r1*r1 - r2*r2
                con[10] = x[1]*x[4] - x[2]*x[3]
                con[11] = x[3]*x[9]
                con[12] = -x[5]*x[9]
                con[13] = x[5]*x[8] - x[6]*x[7]
                con[14] = x[9]
                return fc
            end
        else
            error("bad problem number ($nprob)")
        end

        x = Array(Cdouble, n)
        for icase in 1:2
            for i in 1:n
                x[i] = 1.0
            end
            rhobeg = 0.5
            rhoend = (icase == 2 ? 1e-4 : 0.001)
            if revcom
                # Test the reverse communication variant.
                c = (m > 0 ? Array(Cdouble, m) : nothing)
                ctx = cobyla_create(n, m, rhobeg, rhoend, verbose=1, maxeval=2000)
                status = cobyla_get_status(ctx)
                while status == COBYLA_ITERATE
                    if m > 0
                        # Some constraints.
                        fx = ftest(x, c)
                        status = cobyla_iterate(ctx, fx, x, c)
                    else
                        # No constraints.
                        fx = ftest(x)
                        status = cobyla_iterate(ctx, fx, x)
                    end
                end
                if status != COBYLA_SUCCESS
                    println("Something wrong occured in COBYLA: ", cobyla_reason(status))
                end
            else
                cobyla!(ftest, x, m, rhobeg, rhoend, verbose=1, maxeval=2000)
            end
            if nprob == 10
                tempa = x[1] + x[3] + x[5] + x[7]
                tempb = x[2] + x[4] + x[6] + x[8]
                tempc = 0.5/sqrt(tempa*tempa + tempb*tempb)
                tempd = tempc*sqrt(3.0)
                xopt[1] = tempd*tempa + tempc*tempb
                xopt[2] = tempd*tempb - tempc*tempa
                xopt[3] = tempd*tempa - tempc*tempb
                xopt[4] = tempd*tempb + tempc*tempa
                for i in 1:4
                    xopt[i + 4] = xopt[i]
                end
            end
            temp = 0.0
            for i in 1:n
                r1 = x[i] - xopt[i]
                temp += r1*r1
            end
            @printf("\n     Least squares error in variables =%16.6E\n", sqrt(temp))
        end
        @printf("  ------------------------------------------------------------------\n")
    end
end

# ----------------------------------------------------------------------------

# Possible status values returned by NEWUOA.
const NEWUOA_INITIAL_ITERATE      = convert(Cint,  2)
const NEWUOA_ITERATE              = convert(Cint,  1)
const NEWUOA_SUCCESS              = convert(Cint,  0)
const NEWUOA_BAD_NPT              = convert(Cint, -1)
const NEWUOA_ROUNDING_ERRORS      = convert(Cint, -2)
const NEWUOA_TOO_MANY_EVALUATIONS = convert(Cint, -3)
const NEWUOA_STEP_FAILED          = convert(Cint, -4)
const NEWUOA_BAD_ADDRESS          = convert(Cint, -5)
const NEWUOA_CORRUPTED            = convert(Cint, -6)

# Get a textual explanation of the status returned by NEWUOA.
function newuoa_reason(status::Integer)
    ptr = ccall((:newuoa_reason, opklib), Ptr{Uint8}, (Cint,), status)
    if ptr == C_NULL
        error("unknown NEWUOA status: ", status)
    end
    bytestring(ptr)
end

# Wrapper for the objective function in NEWUOA, the actual objective
# function is provided by the client data.
function newuoa_objfun(n::Cptrdiff_t, x_::Ptr{Cdouble}, f_::Ptr{Any})
    x = pointer_to_array(x_, n)
    f = unsafe_pointer_to_objref(f_)
    fx::Cdouble = f(x)
    return fx
end
const newuoa_objfun_c = cfunction(newuoa_objfun, Cdouble,
                                  (Cptrdiff_t, Ptr{Cdouble}, Ptr{Any}))

function newuoa_check(n::Integer, npt::Integer, rhobeg::Real, rhoend::Real)
    if n < 2
        "bad number of variables"
    elseif npt < n + 2 || npt > div((n + 2)*(n + 1),2)
        "NPT is not in the required interval"
    elseif rhoend < 0 || rhoend > rhobeg
        "bad trust region radius settings"
    end
end

function newuoa!(f::Function, x::Vector{Cdouble},
                 npt::Integer, rhobeg::Real, rhoend::Real;
                 verbose::Integer=0, maxeval::Integer=500)
    n = length(x)
    reason = newuoa_check(n, npt, rhobeg, rhoend)
    reason == nothing || error(reason)
    w = Array(Cdouble, (npt + 13)*(npt + n) + div(3*n*(n + 3),2))
    status = ccall((:newuoa, opklib), Cint,
                   (Cptrdiff_t, Cptrdiff_t, Ptr{Void}, Ptr{Any},
                    Ptr{Cdouble}, Cdouble, Cdouble, Cptrdiff_t,
                    Cptrdiff_t, Ptr{Cdouble}),
                   n, npt, newuoa_objfun_c, pointer_from_objref(f),
                   x, rhobeg, rhoend, verbose, maxeval, w)
    if status != NEWUOA_SUCCESS
        error(newuoa_reason(status))
    end
    return nothing
end

# Context for reverse communication variant of NEWUOA.
type NewuoaContext
    ptr::Ptr{Void}
    n::Int
    npt::Int
    rhobeg::Cdouble
    rhoend::Cdouble
    verbose::Int
    maxeval::Int
end

# Create a new reverse communication workspace for NEWUOA algorithm.
# A typical usage is:
# ```
# x = Array(Cdouble, n)
# x[...] = ... # initial solution
# ctx = newuoa_create(n, npt, rhobeg, rhoend, verbose=0, maxeval=500)
# status = newuoa_get_status(ctx)
# while status == NEWUOA_ITERATE
#   fx = ... # compute function value at X
#   status = newuoa_iterate(ctx, fx, x)
# end
# if status != NEWUOA_SUCCESS
#   println("Something wrong occured in NEWUOA: ", newuoa_reason(status))
# end
# ```
function newuoa_create(n::Integer, npt::Integer,
                       rhobeg::Real, rhoend::Real;
                       verbose::Integer=0, maxeval::Integer=500)
    reason = newuoa_check(n, npt, rhobeg, rhoend)
    reason == nothing || error(reason)
    ptr = ccall((:newuoa_create, opklib), Ptr{Void},
                (Cptrdiff_t, Cptrdiff_t, Cdouble, Cdouble,
                 Cptrdiff_t, Cptrdiff_t),
                n, npt, rhobeg, rhoend, verbose, maxeval)
    if ptr == C_NULL
        reason = (errno() == Base.Errno.ENOMEM
                  ? "insufficient memory"
                  : "unexpected error")
        error(reason)
    end
    ctx = NewuoaContext(ptr, n, npt, rhobeg, rhoend, verbose, maxeval)
    finalizer(ctx, ctx -> ccall((:newuoa_delete, opklib), Void,
                                (Ptr{Void},), ctx.ptr))
    return ctx
end

# Perform the next iteration of the reverse communication version of the
# NEWUOA algorithm.  On entry, the wokspace status must be
# `NEWUOA_ITERATE`, `f` is the function value at `x`.  On exit, the
# returned value (the new wokspace status) is: `NEWUOA_ITERATE` if a new
# trial point has been stored in `x` and if user is requested to compute
# the function value for the new point; `NEWUOA_SUCCESS` if algorithm has
# converged; anything else indicates an error (see `newuoa_reason` for an
# explanatory message).
function newuoa_iterate(ctx::NewuoaContext, f::Real, x::Vector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    ccall((:newuoa_iterate, opklib), Cint,
          (Ptr{Void}, Cdouble, Ptr{Cdouble}),
          ctx.ptr, f, x)
end

# Restart NEWUOA algorithm using the same parameters.  The return value is the
# new status of the algorithm, see `newuoa_get_status` for details.
function newuoa_restart(ctx::NewuoaContext)
    ccall((:newuoa_restart, opklib), Cint, (Ptr{Void},), ctx.ptr)
end

# Get the current status of the algorithm.  Result is: `NEWUOA_ITERATE` if
# user is requested to compute F(X); `NEWUOA_SUCCESS` if algorithm has
# converged; anything else indicates an error (see `newuoa_reason` for an
# explanatory message).
function newuoa_get_status(ctx::NewuoaContext)
    ccall((:newuoa_get_status, opklib), Cint, (Ptr{Void},), ctx.ptr)
end

# Get the current number of function evaluations.  Result is -1 if
# something is wrong (e.g. CTX is NULL), nonnegative otherwise.
function newuoa_get_nevals(ctx::NewuoaContext)
    ccall((:newuoa_get_nevals, opklib), Cptrdiff_t, (Ptr{Void},), ctx.ptr)
end

# Get the current size of the trust region.  Result is 0 if algorithm not
# yet started (before first iteration), -1 if something is wrong (e.g. CTX
# is NULL), strictly positive otherwise.
function newuoa_get_rho(ctx::NewuoaContext)
    ccall((:newuoa_get_rho, opklib), Cdouble, (Ptr{Void},), ctx.ptr)
end

function newuoa_test(revcom::Bool=false)
    # The Chebyquad test problem (Fletcher, 1965) for N = 2,4,6 and 8, with
    # NPT = 2N+1.
    function ftest(x::Vector{Cdouble})
        n = length(x)
        np = n + 1
        y = Array(Cdouble, np, n)
        for j in 1:n
            y[1,j] = 1.0
            y[2,j] = x[j]*2.0 - 1.0
        end
        for i in 2:n
            for j in 1:n
                y[i+1,j] = y[2,j]*2.0*y[i,j] - y[i-1,j]
            end
        end
        f = 0.0
        iw = 1
        for i in 1:np
            sum = 0.0
            for j in 1:n
                sum += y[i,j]
            end
            sum /= n
            if iw > 0
                sum += 1.0/(i*i - 2*i)
            end
            iw = -iw
            f += sum*sum
        end
        return f
    end

    # Run the tests.
    rhoend = 1e-6
    for n = 2:2:8
        npt = 2*n + 1
        x = Array(Cdouble, n)
        for i in 1:n
            x[i] = i/(n + 1)
        end
        rhobeg = x[1]*0.2
        @printf("\n\n    Results with N =%2d and NPT =%3d\n", n, npt)
        if revcom
            # Test the reverse communication variant.
            ctx = newuoa_create(n, npt, rhobeg, rhoend, verbose=2, maxeval=5000)
            status = newuoa_get_status(ctx)
            while status == NEWUOA_ITERATE
                fx = ftest(x)
                status = newuoa_iterate(ctx, fx, x)
            end
            if status != NEWUOA_SUCCESS
                println("Something wrong occured in NEWUOA: ", newuoa_reason(status))
            end
        else
            newuoa!(ftest, x, npt, rhobeg, rhoend, verbose=2, maxeval=5000)
        end
    end
end

# ----------------------------------------------------------------------------

const BOBYQA_SUCCESS              = convert(Cint,  0)
const BOBYQA_BAD_NPT              = convert(Cint, -1)
const BOBYQA_TOO_CLOSE            = convert(Cint, -2)
const BOBYQA_ROUNDING_ERRORS      = convert(Cint, -3)
const BOBYQA_TOO_MANY_EVALUATIONS = convert(Cint, -4)
const BOBYQA_STEP_FAILED          = convert(Cint, -5)

function bobyqa_reason(status::Integer)
    if status == BOBYQA_SUCCESS
        return "algorithm converged"
    elseif status == BOBYQA_BAD_NPT
        return "NPT is not in the required interval"
    elseif status == BOBYQA_TOO_CLOSE
        return "insufficient space between the bounds"
    elseif status == BOBYQA_ROUNDING_ERRORS
        return "too much cancellation in a denominator"
    elseif status == BOBYQA_TOO_MANY_EVALUATIONS
        return "maximum number of function evaluations exceeded"
    elseif status == BOBYQA_STEP_FAILED
        return "a trust region step has failed to reduce quadratic approximation"
    else
        return "unknown BOBYQA status"
    end
end

# Wrapper for the objective function in BOBYQA, the actual objective
# function is provided by the client data.
function bobyqa_objfun(n::Cptrdiff_t, x_::Ptr{Cdouble}, f_::Ptr{Any})
    x = pointer_to_array(x_, n)
    f = unsafe_pointer_to_objref(f_)
    fx::Cdouble = f(x)
    return fx
end
const bobyqa_objfun_c = cfunction(bobyqa_objfun, Cdouble,
                                  (Cptrdiff_t, Ptr{Cdouble}, Ptr{Any}))

function bobyqa!(f::Function, x::Vector{Cdouble},
                 xl::Vector{Cdouble}, xu::Vector{Cdouble},
                 rhobeg::Real, rhoend::Real;
                 npt::Union{Integer,Void}=nothing,
                 verbose::Integer=0, maxeval::Integer=500)
    n = length(x)
    length(xl) == n || error("bad length for inferior bound")
    length(xu) == n || error("bad length for superior bound")
    if npt == nothing
        npt = 2*n + 1
    end
    w = Array(Cdouble, (npt + 5)*(npt + n) + div(3*n*(n + 5),2))
    status = ccall((:bobyqa, opklib), Cint,
                (Cptrdiff_t, Cptrdiff_t, Ptr{Void}, Ptr{Any},
                 Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                 Cdouble, Cdouble, Cptrdiff_t, Cptrdiff_t,
                 Ptr{Cdouble}),
                   n, npt, bobyqa_objfun_c, pointer_from_objref(f),
                   x, xl, xu, rhobeg, rhoend, verbose, maxeval, w)
    if status != BOBYQA_SUCCESS
        error(bobyqa_reason(status))
    end
    return w[1]
end

function bobyqa_test()
    # The test function.
    function f(x::Vector{Cdouble})
        fx = 0.0
        n = length(x)
        for i in 4:2:n
            for j in 2:2:i-2
                tempa = x[i - 1] - x[j - 1]
                tempb = x[i] - x[j]
                temp = tempa*tempa + tempb*tempb
                temp = max(temp,1e-6)
                fx += 1.0/sqrt(temp)
            end
        end
        return fx
    end

    # Run the tests.
    bdl = -1.0
    bdu =  1.0
    rhobeg = 0.1
    rhoend = 1e-6
    for m in (5,10)
        q = 2.0*pi/m
        n = 2*m
        x = Array(Cdouble, n)
        xl = Array(Cdouble, n)
        xu = Array(Cdouble, n)
        for i in 1:n
            xl[i] = bdl
            xu[i] = bdu
        end
        for jcase in 1:2
            if jcase == 2
                npt = 2*n + 1
            else
                npt = n + 6
            end
            @printf("\n\n     2D output with M =%4ld,  N =%4ld  and  NPT =%4ld\n",
                    m, n, npt)
            for j in 1:m
                temp = q*j
                x[2*j - 1] = cos(temp)
                x[2*j]     = sin(temp)
            end
            fx = bobyqa!(f, x, xl, xu, rhobeg, rhoend, npt=npt, verbose=2, maxeval=500000)
            @printf("\n***** least function value: %.15e\n", fx)
        end
    end
end

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
