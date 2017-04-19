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
# Copyright (C) 2015-2017, Éric Thiébaut.
#
# ----------------------------------------------------------------------------

module Powell

export iterate,
       restart,
       getstatus,
       getreason,
       getradius,
       getncalls,
       getlastf,
       cobyla_optimize!,
       cobyla_optimize,
       cobyla_minimize!,
       cobyla_minimize,
       cobyla_maximize!,
       cobyla_maximize,
       cobyla_create,
       COBYLA_INITIAL_ITERATE,
       COBYLA_ITERATE,
       COBYLA_SUCCESS,
       COBYLA_BAD_RHO_RANGE,
       COBYLA_BAD_SCALING,
       COBYLA_ROUNDING_ERRORS,
       COBYLA_TOO_MANY_EVALUATIONS,
       COBYLA_BAD_ADDRESS,
       COBYLA_CORRUPTED,
       newuoa_optimize!,
       newuoa_optimize,
       newuoa_minimize!,
       newuoa_minimize,
       newuoa_maximize!,
       newuoa_maximize,
       newuoa_create,
       NEWUOA_INITIAL_ITERATE,
       NEWUOA_ITERATE,
       NEWUOA_SUCCESS,
       NEWUOA_BAD_NPT,
       NEWUOA_BAD_RHO_RANGE,
       NEWUOA_BAD_SCALING,
       NEWUOA_ROUNDING_ERRORS,
       NEWUOA_TOO_MANY_EVALUATIONS,
       NEWUOA_STEP_FAILED,
       NEWUOA_BAD_ADDRESS,
       NEWUOA_CORRUPTED,
       bobyqa_optimize!,
       bobyqa_optimize,
       bobyqa_minimize!,
       bobyqa_minimize,
       bobyqa_maximize!,
       bobyqa_maximize,
       BOBYQA_SUCCESS,
       BOBYQA_BAD_NPT,
       BOBYQA_BAD_RHO_RANGE,
       BOBYQA_BAD_SCALING,
       BOBYQA_TOO_CLOSE,
       BOBYQA_ROUNDING_ERRORS,
       BOBYQA_TOO_MANY_EVALUATIONS,
       BOBYQA_STEP_FAILED

import Base: ==

import OptimPack: opklib

abstract Status

abstract Context

=={T<:Status}(a::T, b::T) = a._code == b._code
==(a::Status, b::Status) = false

doc"""
The `iterate(ctx, ...)` method performs the next iteration of the reverse
communication associated with the context `ctx`.  Other arguments depend on the
type of algorithm.

For **COBYLA** algorithm, the next iteration is performed by:

    iterate(ctx, f, x, c) -> status

or

    iterate(ctx, f, x) -> status

on entry, the workspace status must be `COBYLA_ITERATE`, `f` and `c` are the
function value and the constraints at `x`, the latter can be omitted if there
are no constraints.  On exit, the returned value (the new workspace status) is:
`COBYLA_ITERATE` if a new trial point has been stored in `x` and if user is
requested to compute the function value and the constraints on the new point;
`COBYLA_SUCCESS` if algorithm has converged and `x` has been set with the
variables at the solution (the corresponding function value can be retrieved
with `getlastf`); anything else indicates an error (see `getreason`
for an explanatory message).


For **NEWUOA** algorithm, the next iteration is performed by:

    iterate(ctx, f, x) -> status

on entry, the wokspace status must be `NEWUOA_ITERATE`, `f` is the function
value at `x`.  On exit, the returned value (the new wokspace status) is:
`NEWUOA_ITERATE` if a new trial point has been stored in `x` and if user is
requested to compute the function value for the new point; `NEWUOA_SUCCESS` if
algorithm has converged; anything else indicates an error (see `getreason` for
an explanatory message).

"""
function iterate end

doc"""

    restart(ctx) -> status

restarts the reverse communication algorithm associated with the context `ctx`
using the same parameters.  The return value is the new status of the
algorithm, see `getstatus` for details.

"""
function restart end

doc"""

    getstatus(ctx) -> status

get the current status of the reverse communication algorithm associated with
the context `ctx`.  Possible values are:

* for **COBYLA**: `COBYLA_ITERATE`, if user is requested to compute `f(x)` and
  `c(x)`; `COBYLA_SUCCESS`, if algorithm has converged;

* for **NEWUOA**: `NEWUOA_ITERATE`, if user is requested to compute `f(x)`;
  `NEWUOA_SUCCESS`, if algorithm has converged;

Anything else indicates an error (see `getreason` for an explanatory message).

"""
function getstatus end

doc"""

    getreason(ctx) -> msg

or

    getreason(status) -> msg

get an explanatory message about the current status of the reverse
communication algorithm associated with the context `ctx` or with the status
returned by an optimization method of by `getstatus(ctx)`.

"""
function getreason end

getreason(ctx::Context) = getreason(getstatus(ctx))

doc"""

    getlastf(ctx) -> fx

get the last function value in the reverse communication algorithm associated
with the context `ctx`.  Upon convergence of `iterate`, this value corresponds
to the function at the solution; otherwise, this value corresponds to the
previous set of variables.

"""
function getlastf end

doc"""

    getncalls(ctx) -> nevals

get the current number of function evaluations in the reverse communication
algorithm associated with the context `ctx`.  Result is -1 if something is
wrong, nonnegative otherwise.

"""
function getncalls end

doc"""

    getradius(ctx) -> rho

get the current size of the trust region of the reverse communication algorithm
associated with the context `ctx`.  Result is 0 if algorithm not yet started
(before first iteration), -1 if something is wrong, strictly positive
otherwise.

"""
function getradius end

# Wrapper for the objective function in NEWUOA or BOBYQA, the actual objective
# function is provided by the client data.
function _objfun(n::Cptrdiff_t, xptr::Ptr{Cdouble}, fptr::Ptr{Void})
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    convert(Cdouble, f(x))::Cdouble
end

const _objfun_c = cfunction(_objfun, Cdouble, (Cptrdiff_t, Ptr{Cdouble},
                                               Ptr{Void}))

# Wrapper for the objective function in COBYLA, the actual objective
# function is provided by the client data.
function _calcfc(n::Cptrdiff_t, m::Cptrdiff_t, xptr::Ptr{Cdouble},
                 _c::Ptr{Cdouble}, fptr::Ptr{Void})
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    convert(Cdouble, (m > 0 ? f(x, unsafe_wrap(Array, _c, m)) : f(x)))::Cdouble
end

const _calcfc_c = cfunction(_calcfc, Cdouble, (Cptrdiff_t, Cptrdiff_t,
                                               Ptr{Cdouble}, Ptr{Cdouble},
                                               Ptr{Void}))

#------------------------------------------------------------------------------
# COBYLA

immutable CobylaStatus <: Status
    _code::Cint
end

# Possible status values returned by COBYLA.
const COBYLA_INITIAL_ITERATE      = CobylaStatus( 2)
const COBYLA_ITERATE              = CobylaStatus( 1)
const COBYLA_SUCCESS              = CobylaStatus( 0)
const COBYLA_BAD_NVARS            = CobylaStatus(-1)
const COBYLA_BAD_NCONS            = CobylaStatus(-2)
const COBYLA_BAD_RHO_RANGE        = CobylaStatus(-3)
const COBYLA_BAD_SCALING          = CobylaStatus(-4)
const COBYLA_ROUNDING_ERRORS      = CobylaStatus(-5)
const COBYLA_TOO_MANY_EVALUATIONS = CobylaStatus(-6)
const COBYLA_BAD_ADDRESS          = CobylaStatus(-7)
const COBYLA_CORRUPTED            = CobylaStatus(-8)

# Get a textual explanation of the status returned by COBYLA.
function getreason(status::CobylaStatus)
    ptr = ccall((:cobyla_reason, opklib), Ptr{UInt8}, (Cint,), status._code)
    if ptr == C_NULL
        error("unknown COBYLA status: ", status._code)
    end
    bytestring(ptr)
end

_cobyla_wslen(n::Integer, m::Integer) = n*(3*n + 2*m + 11) + 4*m + 6

doc"""
The methods:

    cobyla_optimize!(fc, x, m, rhobeg, rhoend) -> (status, x, fx)
    cobyla_optimize(fc, x0, m, rhobeg, rhoend) -> (status, x, fx)

are identical to `cobyla_minimize!` and `cobyla_minimize` respectively but have
an additional `maximize` keyword which is `false` by default and which
specifies whether to maximize the objective function; otherwise, the method
attempts to minimize the objective function.

"""
function cobyla_optimize!(fc::Function, x::DenseVector{Cdouble},
                          m::Integer, rhobeg::Real, rhoend::Real;
                          scale::DenseVector{Cdouble} = Array(Cdouble, 0),
                          maximize::Bool = false,
                          check::Bool = false,
                          verbose::Integer = 0,
                          maxeval::Integer = 30*length(x))
    n = length(x)
    nscl = length(scale)
    if nscl == 0
        sclptr = convert(Ptr{Cdouble}, C_NULL)
    elseif nscl == n
        sclptr = pointer(scale)
    else
        error("bad number of scaling factors")
    end
    work = Array(Cdouble, _cobyla_wslen(n, m))
    iact = Array(Cptrdiff_t, m + 1)
    status = CobylaStatus(ccall((:cobyla_optimize, opklib), Cint,
                                (Cptrdiff_t, Cptrdiff_t, Cint, Ptr{Void},
                                 Ptr{Void}, Ptr{Cdouble}, Ptr{Cdouble},
                                 Cdouble, Cdouble, Cptrdiff_t, Cptrdiff_t,
                                 Ptr{Cdouble}, Ptr{Cptrdiff_t}), n, m,
                                maximize, _calcfc_c, pointer_from_objref(fc),
                                x, sclptr, rhobeg, rhoend, verbose, maxeval,
                                work, iact))
    if check && status != COBYLA_SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

cobyla_optimize(fc::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    cobyla_optimize!(fc, copy(x0), args...; kwds...)

doc"""
# Minimizing a function of many variables subject to inequality constraints

Mike Powell's **COBYLA** algorithm attempts to find the variables `x` which
solve the problem:

    min f(x)    s.t.   c(x) <= 0

where `x` is a vector of variables that has `n` components, `f(x)` is an
objective function and `c(x)` implement `m` inequality constraints.  The
algorithm employs linear approximations to the objective and constraint
functions, the approximations being formed by linear interpolation at `n+1`
points in the space of the variables.  We regard these interpolation points as
vertices of a simplex.  The parameter `rho` controls the size of the simplex
and it is reduced automatically from `rhobeg` to `rhoend`.  For each `rho`,
COBYLA tries to achieve a good vector of variables for the current size, and
then `rho` is reduced until the value `rhoend` is reached.  Therefore `rhobeg`
and `rhoend` should be set to reasonable initial changes to and the required
accuracy in the variables respectively, but this accuracy should be viewed as a
subject for experimentation because it is not guaranteed.  The subroutine has
an advantage over many of its competitors, however, which is that it treats
each constraint individually when calculating a change to the variables,
instead of lumping the constraints together into a single penalty function.
The name of the subroutine is derived from the phrase "Constrained Optimization
BY Linear Approximations".

The most simple version of the algorithm is called as:

    cobyla_minimize!(fc, x, m, rhobeg, rhoend) -> (status, x, fx)

where `x` is a Julia vector with the initial and final variables, `m`,
`rhobeg`, and `rhoend` have been defined already, while `fc` is a Julia
function which is called as:

    fc(x, cx) -> fx

to store in `cx` the values of the constraints at `x` and to return `fx` the
value of the objective function at `x`.  If there are no constraints
(i.e. `m=0`), then `fc` is called without the `cx` argument as:

    fc(x) -> fx

The method returns a tuple with `status` the termaintion condition (should be
`COBYLA_SUCCESS` unless keyword `check` is set `false`, see below), `x` the
solution found by the algorithm and `fx` the corresponding function value.

The method:

    cobyla_minimize(fc, x0, m, rhobeg, rhoend) -> (status, x, fx)

is identical but does not modify the vector of initial variables.


## Scaling of variables

The proper scaling of the variables is important for the success of the
algorithm and the optional `scale` keyword should be specified if the typical
precision is not the same for all variables.  If specified, `scale` is an array
of strictly nonnegative values and of same size is the variables `x`, such that
`scale[i]*rho` (with `rho` the trust region radius) is the size of the trust
region for the `i`-th variable.  If keyword `scale` is not specified, a unit
scaling for all the variables is assumed.


## Other keywords

The following keywords are available:

* `scale` specify the typical magnitudes of the variables.  If specified, it
  must have as many elements as `x`, all strictly positive.

* `check` (`true` by default) specifies whether to throw an exception if the
  algorithm is not fully successful.

* `verbose` (`0` by default) set the amount of printing.

* `maxeval` (`30*length(x)` by default) set maximum number of calls to the
  objective function.


## References

The algorithm is described in:

> M.J.D. Powell, "A direct search optimization method that models the objective
> and constraint functions by linear interpolation," in Advances in
> Optimization and Numerical Analysis Mathematics and Its Applications,
> vol. 275 (eds. Susana Gomez and Jean-Pierre Hennart), Kluwer Academic
> Publishers, pp. 51-67 (1994).

"""
cobyla_minimize!(args...; kwds...) =
    cobyla_optimize!(args...; maximize=false, kwds...)

cobyla_minimize(args...; kwds...) =
    cobyla_optimize(args...; maximize=false, kwds...)

doc"""
The methods:

    cobyla_maximize!(fc, x, m, rhobeg, rhoend) -> (status, x, fx)
    cobyla_maximize(fc, x0, m, rhobeg, rhoend) -> (status, x, fx)

are similar to `cobyla_minimize!` and `cobyla_minimize` respectively but
solve the contrained maximization problem:

    max f(x)    s.t.   c(x) <= 0

"""
cobyla_maximize!(args...; kwds...) =
    cobyla_optimize!(args...; maximize=true, kwds...)

cobyla_maximize(args...; kwds...) =
    cobyla_optimize(args...; maximize=true, kwds...)

@doc @doc(cobyla_optimize!) cobyla_optimize
@doc @doc(cobyla_minimize!) cobyla_minimize
@doc @doc(cobyla_maximize!) cobyla_maximize

# Simpler version, mostly for testing.

function cobyla!(f::Function, x::DenseVector{Cdouble},
                 m::Integer, rhobeg::Real, rhoend::Real;
                 check::Bool = true,
                 verbose::Integer = 0,
                 maxeval::Integer = 30*length(x))
    n = length(x)
    work = Array(Cdouble, _cobyla_wslen(n, m))
    iact = Array(Cptrdiff_t, m + 1)
    status = CobylaStatus(ccall((:cobyla, opklib), Cint,
                                (Cptrdiff_t, Cptrdiff_t, Ptr{Void}, Ptr{Void},
                                 Ptr{Cdouble}, Cdouble, Cdouble, Cptrdiff_t,
                                 Cptrdiff_t, Ptr{Cdouble}, Ptr{Cptrdiff_t}),
                                n, m, _calcfc_c, pointer_from_objref(f),
                                x, rhobeg, rhoend, verbose, maxeval, work, iact))
    if check && status != COBYLA_SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

cobyla(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    cobyla!(f, copy(x0), args...; kwds...)

# Context for reverse communication variant of COBYLA.
type CobylaContext <: Context
    ptr::Ptr{Void}
    n::Int
    m::Int
    rhobeg::Cdouble
    rhoend::Cdouble
    verbose::Int
    maxeval::Int
end

doc"""

    ctx = cobyla_create(n, m, rhobeg, rhoend; verbose=0, maxeval=500)

creates a new reverse communication workspace for COBYLA algorithm.  A typical
usage is:

    x = Array(Cdouble, n)
    c = Array(Cdouble, m)
    x[...] = ... # initial solution
    ctx = cobyla_create(n, m, rhobeg, rhoend, verbose=1, maxeval=500)
    status = getstatus(ctx)
    while status == COBYLA_ITERATE
        fx = ...       # compute function value at X
        c[...] = ...   # compute constraints at X
        status = iterate(ctx, fx, x, c)
    end
    if status != COBYLA_SUCCESS
        println("Something wrong occured in COBYLA: ", getreason(status))
    end


"""
function cobyla_create(n::Integer, m::Integer,
                       rhobeg::Real, rhoend::Real;
                       verbose::Integer=0, maxeval::Integer=500)
    if n < 2
        throw(ArgumentError("bad number of variables"))
    elseif m < 0
        throw(ArgumentError("bad number of constraints"))
    elseif rhoend < 0 || rhoend > rhobeg
        throw(ArgumentError("bad trust region radius parameters"))
    end
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

function iterate(ctx::CobylaContext, f::Real, x::DenseVector{Cdouble},
                        c::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    length(c) == ctx.m || error("bad number of constraints")
    CobylaStatus(ccall((:cobyla_iterate, opklib), Cint,
                       (Ptr{Void}, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}),
                       ctx.ptr, f, x, c))
end

function iterate(ctx::CobylaContext, f::Real, x::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    ctx.m == 0 || error("bad number of constraints")
    CobylaStatus(ccall((:cobyla_iterate, opklib), Cint,
                       (Ptr{Void}, Cdouble, Ptr{Cdouble}, Ptr{Void}),
                       ctx.ptr, f, x, C_NULL))
end

restart(ctx::CobylaContext) =
    CobylaStatus(ccall((:cobyla_restart, opklib), Cint, (Ptr{Void},), ctx.ptr))

getstatus(ctx::CobylaContext) =
    CobylaStatus(ccall((:cobyla_get_status, opklib), Cint, (Ptr{Void},),
                       ctx.ptr))

# Get the current number of function evaluations.  Result is -1 if
# something is wrong (e.g. CTX is NULL), nonnegative otherwise.
getncalls(ctx::CobylaContext) =
    Int(ccall((:cobyla_get_nevals, opklib), Cptrdiff_t, (Ptr{Void},), ctx.ptr))

getradius(ctx::CobylaContext) =
    ccall((:cobyla_get_rho, opklib), Cdouble, (Ptr{Void},), ctx.ptr)

getlastf(ctx::CobylaContext) =
    ccall((:cobyla_get_last_f, opklib), Cdouble, (Ptr{Void},), ctx.ptr)

function cobyla_test(;revcom::Bool = false, scale::Real = 1.0)
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
            ftest = (x::DenseVector{Cdouble}) -> begin
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
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
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
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
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
            ftest = (x::DenseVector{Cdouble}) -> begin
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
            ftest = (x::DenseVector{Cdouble}) -> begin
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
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
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
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
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
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
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
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
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
            xopt = fill!(Array(Cdouble, n), 0.0)
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
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
            fill!(x, 1.0)
            rhobeg = 0.5
            rhoend = (icase == 2 ? 1e-4 : 0.001)
            if revcom
                # Test the reverse communication variant.
                c = Array(Cdouble, max(m, 0))
                ctx = cobyla_create(n, m, rhobeg, rhoend;
                                    verbose = 1, maxeval = 2000)
                status = getstatus(ctx)
                while status == COBYLA_ITERATE
                    if m > 0
                        # Some constraints.
                        fx = ftest(x, c)
                        status = iterate(ctx, fx, x, c)
                    else
                        # No constraints.
                        fx = ftest(x)
                        status = iterate(ctx, fx, x)
                    end
                end
                if status != COBYLA_SUCCESS
                    println("Something wrong occured in COBYLA: ",
                            getreason(status))
                end
            elseif scale == 1
                cobyla!(ftest, x, m, rhobeg, rhoend;
                        verbose = 1, maxeval = 2000)
            else
                cobyla_minimize!(ftest, x, m, rhobeg/scale, rhoend/scale;
                                 scale = fill!(Array(Cdouble, n), scale),
                                 verbose = 1, maxeval = 2000)
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

#------------------------------------------------------------------------------
# NEWUOA

immutable NewuoaStatus <: Status
    _code::Cint
end

# Possible status values returned by NEWUOA.
const NEWUOA_INITIAL_ITERATE      = NewuoaStatus( 2)
const NEWUOA_ITERATE              = NewuoaStatus( 1)
const NEWUOA_SUCCESS              = NewuoaStatus( 0)
const NEWUOA_BAD_NVARS            = NewuoaStatus(-1)
const NEWUOA_BAD_NPT              = NewuoaStatus(-2)
const NEWUOA_BAD_RHO_RANGE        = NewuoaStatus(-3)
const NEWUOA_BAD_SCALING          = NewuoaStatus(-4)
const NEWUOA_ROUNDING_ERRORS      = NewuoaStatus(-5)
const NEWUOA_TOO_MANY_EVALUATIONS = NewuoaStatus(-6)
const NEWUOA_STEP_FAILED          = NewuoaStatus(-7)
const NEWUOA_BAD_ADDRESS          = NewuoaStatus(-8)
const NEWUOA_CORRUPTED            = NewuoaStatus(-9)

# Get a textual explanation of the status returned by NEWUOA.
function getreason(status::NewuoaStatus)
    ptr = ccall((:newuoa_reason, opklib), Ptr{UInt8}, (Cint,), status.status)
    if ptr == C_NULL
        error("unknown NEWUOA status: ", status._code)
    end
    bytestring(ptr)
end

_newuoa_wslen(n::Integer, npt::Integer) =
    (npt + 13)*(npt + n) + div(3*n*(n + 3),2)

function newuoa_optimize!(f::Function, x::DenseVector{Cdouble},
                          rhobeg::Real, rhoend::Real;
                          scale::DenseVector{Cdouble} = Array(Cdouble, 0),
                          maximize::Bool = false,
                          npt::Integer = 2*length(x) + 1,
                          check::Bool = true,
                          verbose::Integer = 0,
                          maxeval::Integer = 30*length(x))
    n = length(x)
    nw = _newuoa_wslen(n, npt)
    nscl = length(scale)
    if nscl == 0
        sclptr = convert(Ptr{Cdouble}, C_NULL)
    elseif nscl == n
        sclptr = pointer(scale)
        nw += n
    else
        error("bad number of scaling factors")
    end
    work = Array(Cdouble, nw)
    status = NewuoaStatus(ccall((:newuoa_optimize, opklib), Cint,
                                (Cptrdiff_t, Cptrdiff_t, Cint, Ptr{Void},
                                 Ptr{Void}, Ptr{Cdouble}, Ptr{Cdouble},
                                 Cdouble, Cdouble, Cptrdiff_t, Cptrdiff_t,
                                 Ptr{Cdouble}), n, npt, maximize, _objfun_c,
                                pointer_from_objref(f), x, sclptr, rhobeg,
                                rhoend, verbose, maxeval, work))
    if check && status != NEWUOA_SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

newuoa_optimize(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    newuoa_optimize(f, copy(x0), args...; kwds...)

newuoa_minimize!(args...; kwds...) =
    newuoa_optimize!(args...; maximize=false, kwds...)

newuoa_minimize(args...; kwds...) =
    newuoa_optimize(args...; maximize=false, kwds...)

newuoa_maximize!(args...; kwds...) =
    newuoa_optimize!(args...; maximize=true, kwds...)

newuoa_maximize(args...; kwds...) =
    newuoa_optimize(args...; maximize=true, kwds...)

function newuoa!(f::Function, x::DenseVector{Cdouble},
                 rhobeg::Real, rhoend::Real;
                 npt::Integer = 2*length(x) + 1,
                 verbose::Integer = 0,
                 maxeval::Integer = 30*length(x),
                 check::Bool = true)
    n = length(x)
    work = Array(Cdouble, _newuoa_wslen(n, npt))
    status = NewuoaStatus(ccall((:newuoa, opklib), Cint,
                                (Cptrdiff_t, Cptrdiff_t, Ptr{Void}, Ptr{Void},
                                 Ptr{Cdouble}, Cdouble, Cdouble, Cptrdiff_t,
                                 Cptrdiff_t, Ptr{Cdouble}), n, npt, _objfun_c,
                                pointer_from_objref(f), x, rhobeg, rhoend,
                                verbose, maxeval, work))
    if check && status != NEWUOA_SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

newuoa(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    newuoa!(f, copy(x0), args...; kwds...)

# Context for reverse communication variant of NEWUOA.
type NewuoaContext <: Context
    ptr::Ptr{Void}
    n::Int
    npt::Int
    rhobeg::Cdouble
    rhoend::Cdouble
    verbose::Int
    maxeval::Int
end

doc"""

    ctx = newuoa_create(n, rhobeg, rhoend; npt=..., verbose=..., maxeval=...)

creates a new reverse communication workspace for NEWUOA algorithm.  A typical
usage is:

    x = Array(Cdouble, n)
    x[...] = ... # initial solution
    ctx = newuoa_create(n, rhobeg, rhoend; verbose=1, maxeval=500)
    status = getstatus(ctx)
    while status == NEWUOA_ITERATE
        fx = ...       # compute function value at X
        status = iterate(ctx, fx, x)
    end
    if status != NEWUOA_SUCCESS
        println("Something wrong occured in NEWUOA: ", getreason(status))
    end

"""
function newuoa_create(n::Integer, rhobeg::Real, rhoend::Real;
                       npt::Integer = 2*length(x) + 1,
                       verbose::Integer = 0,
                       maxeval::Integer = 30*length(x))
    ptr = ccall((:newuoa_create, opklib), Ptr{Void},
                (Cptrdiff_t, Cptrdiff_t, Cdouble, Cdouble,
                 Cptrdiff_t, Cptrdiff_t),
                n, npt, rhobeg, rhoend, verbose, maxeval)
    if ptr == C_NULL
        reason = (errno() == Base.Errno.ENOMEM
                  ? "insufficient memory"
                  : "invalid argument")
        error(reason)
    end
    ctx = NewuoaContext(ptr, n, npt, rhobeg, rhoend, verbose, maxeval)
    finalizer(ctx, ctx -> ccall((:newuoa_delete, opklib), Void,
                                (Ptr{Void},), ctx.ptr))
    return ctx
end

function iterate(ctx::NewuoaContext, f::Real, x::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    NewuoaStatus(ccall((:newuoa_iterate, opklib), Cint,
                       (Ptr{Void}, Cdouble, Ptr{Cdouble}),
                       ctx.ptr, f, x))
end

restart(ctx::NewuoaContext) =
    NewuoaStatus(ccall((:newuoa_restart, opklib), Cint, (Ptr{Void},), ctx.ptr))

getstatus(ctx::NewuoaContext) =
    NewuoaStatus(ccall((:newuoa_get_status, opklib), Cint, (Ptr{Void},),
                       ctx.ptr))

getncalls(ctx::NewuoaContext) =
    Int(ccall((:newuoa_get_nevals, opklib), Cptrdiff_t, (Ptr{Void},), ctx.ptr))

getradius(ctx::NewuoaContext) =
    ccall((:newuoa_get_rho, opklib), Cdouble, (Ptr{Void},), ctx.ptr)

function newuoa_test(;revcom::Bool=false, scale::Real=1)
    # The Chebyquad test problem (Fletcher, 1965) for N = 2,4,6 and 8, with
    # NPT = 2N+1.
    function ftest(x::DenseVector{Cdouble})
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
            ctx = newuoa_create(n, rhobeg, rhoend;
                                npt = npt, verbose = 2, maxeval = 5000)
            status = getstatus(ctx)
            while status == NEWUOA_ITERATE
                fx = ftest(x)
                status = iterate(ctx, fx, x)
            end
            if status != NEWUOA_SUCCESS
                println("Something wrong occured in NEWUOA: ",
                        getreason(status))
            end
        elseif scale != 1
            newuoa_minimize!(ftest, x, rhobeg/scale, rhoend/scale;
                             scale = fill!(similar(x), scale),
                             npt = npt, verbose = 2, maxeval = 5000)
        else
            newuoa!(ftest, x, rhobeg, rhoend;
                    npt = npt, verbose = 2, maxeval = 5000)
        end
    end
end

#------------------------------------------------------------------------------
# BOBYQA

immutable BobyqaStatus <: Status
    _code::Cint
end

const BOBYQA_SUCCESS              = BobyqaStatus( 0)
const BOBYQA_BAD_NVARS            = BobyqaStatus(-1)
const BOBYQA_BAD_NPT              = BobyqaStatus(-2)
const BOBYQA_BAD_RHO_RANGE        = BobyqaStatus(-3)
const BOBYQA_BAD_SCALING          = BobyqaStatus(-4)
const BOBYQA_TOO_CLOSE            = BobyqaStatus(-5)
const BOBYQA_ROUNDING_ERRORS      = BobyqaStatus(-6)
const BOBYQA_TOO_MANY_EVALUATIONS = BobyqaStatus(-7)
const BOBYQA_STEP_FAILED          = BobyqaStatus(-8)

# Get a textual explanation of the status returned by BOBYQA.
function getreason(status::BobyqaStatus)
    ptr = ccall((:bobyqa_reason, opklib), Ptr{UInt8}, (Cint,), status._code)
    if ptr == C_NULL
        error("unknown BOBYQA status: ", status._code)
    end
    bytestring(ptr)
end

_bobyqa_wslen(n::Integer, npt::Integer) =
    (npt + 5)*(npt + n) + div(3*n*(n + 5),2)

function bobyqa_optimize!(f::Function, x::DenseVector{Cdouble},
                          xl::DenseVector{Cdouble}, xu::DenseVector{Cdouble},
                          rhobeg::Real, rhoend::Real;
                          scale::DenseVector{Cdouble}=Array(Cdouble, 0),
                          maximize::Bool=false,
                          npt::Integer=2*length(x) + 1,
                          check::Bool=false,
                          verbose::Integer=0,
                          maxeval::Integer=30*length(x))
    n = length(x)
    length(xl) == n || error("bad length for inferior bound")
    length(xu) == n || error("bad length for superior bound")
    nw = _bobyqa_wslen(n, npt)
    nscl = length(scale)
    if nscl == 0
        sclptr = convert(Ptr{Cdouble}, C_NULL)
    elseif nscl == n
        sclptr = pointer(scale)
        nw += 3*n
    else
        error("bad number of scaling factors")
    end
    work = Array(Cdouble, nw)
    status = BobyqaStatus(ccall((:bobyqa_optimize, opklib), Cint,
                                (Cptrdiff_t, Cptrdiff_t, Cint, Ptr{Void},
                                 Ptr{Void}, Ptr{Cdouble}, Ptr{Cdouble},
                                 Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble,
                                 Cptrdiff_t, Cptrdiff_t, Ptr{Cdouble}),
                                n, npt, (maximize ? Cint(1) : Cint(0)),
                                _objfun_c, pointer_from_objref(f),
                                x, xl, xu, sclptr, rhobeg, rhoend,
                                verbose, maxeval, work))
    if check && status != BOBYQA_SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

bobyqa_optimize(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    bobyqa_optimize(f, copy(x0), args...; kwds...)

bobyqa_minimize!(args...; kwds...) =
    bobyqa_optimize!(args...; maximize=false, kwds...)

bobyqa_minimize(args...; kwds...) =
    bobyqa_optimize(args...; maximize=false, kwds...)

bobyqa_maximize!(args...; kwds...) =
    bobyqa_optimize!(args...; maximize=true, kwds...)

bobyqa_maximize(args...; kwds...) =
    bobyqa_optimize(args...; maximize=true, kwds...)

function bobyqa!(f::Function, x::DenseVector{Cdouble},
                 xl::DenseVector{Cdouble}, xu::DenseVector{Cdouble},
                 rhobeg::Real, rhoend::Real;
                 npt::Integer = 2*length(x) + 1,
                 verbose::Integer = 0,
                 maxeval::Integer = 30*length(x),
                 check::Bool = true)
    n = length(x)
    length(xl) == n || error("bad length for inferior bound")
    length(xu) == n || error("bad length for superior bound")
    work = Array(Cdouble, _bobyqa_wslen(n, npt))
    status = BobyqaStatus(ccall((:bobyqa, opklib), Cint,
                                (Cptrdiff_t, Cptrdiff_t, Ptr{Void}, Ptr{Void},
                                 Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                                 Cdouble, Cdouble, Cptrdiff_t, Cptrdiff_t,
                                 Ptr{Cdouble}),
                                n, npt, _objfun_c,
                                pointer_from_objref(f), x, xl, xu,
                                rhobeg, rhoend, verbose, maxeval, work))
    if check && status != BOBYQA_SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

bobyqa(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    bobyqa!(f, copy(x0), args...; kwds...)

function bobyqa_test()
    # The test function.
    function f(x::DenseVector{Cdouble})
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
            fx = bobyqa!(f, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=2, maxeval=500000)[3]
            @printf("\n***** least function value: %.15e\n", fx)
        end
    end
end

#------------------------------------------------------------------------------

end # module Powell
