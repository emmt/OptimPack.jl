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
                 m::Integer, rhobeg::Real, rhoend::Real,
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
                   n, npt, cobyla_objfun_c, pointer_from_objref(f),
                   x, rhobeg, rhoend, verbose, maxfun, w, iact)
    if status != COBYLA_SUCCESS
        error(cobyla_reason(status))
    end
    #return w[1]
end

# Context for reverse communication variant of COBYLA.
immutable CobylaContext
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
# ctx = cobyla_create(n, m, rhobeg, rhoend, verbose=1, maxeval=500);
# status = cobyla_get_status(ctx);
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
        reason = (get_errno() == ENOMEM ? "insufficient memory"
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
                 npt::Integer, rhobeg::Real, rhoend::Real,
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
immutable NewuoaContext
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
                n, m, rhobeg, rhoend, verbose, maxeval)
    if ptr == C_NULL
        reason = (get_errno() == ENOMEM ? "insufficient memory"
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
                 npt::Union(Integer,Nothing)=nothing,
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

# -----------------------------------------------------------------------------
# ERRNO

get_errno() = cglobal((:errno,:libc), Cint)
const EPERM     = convert(Cint,  1) # Operation not permitted
const ENOENT    = convert(Cint,  2) # No such file or directory
const ESRCH     = convert(Cint,  3) # No such process
const EINTR     = convert(Cint,  4) # Interrupted system call
const EIO       = convert(Cint,  5) # I/O error
const ENXIO     = convert(Cint,  6) # No such device or address
const E2BIG     = convert(Cint,  7) # Argument list too long
const ENOEXEC   = convert(Cint,  8) # Exec format error
const EBADF     = convert(Cint,  9) # Bad file number
const ECHILD    = convert(Cint, 10) # No child processes
const EAGAIN    = convert(Cint, 11) # Try again
const ENOMEM    = convert(Cint, 12) # Out of memory
const EACCES    = convert(Cint, 13) # Permission denied
const EFAULT    = convert(Cint, 14) # Bad address
const ENOTBLK   = convert(Cint, 15) # Block device required
const EBUSY     = convert(Cint, 16) # Device or resource busy
const EEXIST    = convert(Cint, 17) # File exists
const EXDEV     = convert(Cint, 18) # Cross-device link
const ENODEV    = convert(Cint, 19) # No such device
const ENOTDIR   = convert(Cint, 20) # Not a directory
const EISDIR    = convert(Cint, 21) # Is a directory
const EINVAL    = convert(Cint, 22) # Invalid argument
const ENFILE    = convert(Cint, 23) # File table overflow
const EMFILE    = convert(Cint, 24) # Too many open files
const ENOTTY    = convert(Cint, 25) # Not a typewriter
const ETXTBSY   = convert(Cint, 26) # Text file busy
const EFBIG     = convert(Cint, 27) # File too large
const ENOSPC    = convert(Cint, 28) # No space left on device
const ESPIPE    = convert(Cint, 29) # Illegal seek
const EROFS     = convert(Cint, 30) # Read-only file system
const EMLINK    = convert(Cint, 31) # Too many links
const EPIPE     = convert(Cint, 32) # Broken pipe
const EDOM      = convert(Cint, 33) # Math argument out of domain of func
const ERANGE    = convert(Cint, 34) # Math result not representable

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
