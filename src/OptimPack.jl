#
# OptimPack.jl --
#
# Julia wrapper for OptimPack.
#
# ----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT "Expat"
# License:
#
# Copyright (C) 2014-2019, Éric Thiébaut.
#
# ----------------------------------------------------------------------------

VERSION < v"0.7.0-beta2.199" && __precompile__(true)

module OptimPack

using Compat
using Compat: String, @debug, @error, @info, @warn
import Compat: LinearAlgebra
using Compat.Printf

export nlcg, vmlmb, spg2

export fzero, fmin, fmin_global

# Functions must be imported to be extended with new methods.
import Base: ENV, size, length, eltype, ndims, copy
import .LinearAlgebra: dot

isfile(joinpath(@__DIR__,"..","deps","deps.jl")) ||
    error("OptimPack not properly installed.  Please run Pkg.build(\"OptimPack\")")
include("../deps/deps.jl")

"""
`Floats` is any floating point type supported by the library.
"""
const Floats = Union{Cfloat,Cdouble}

#------------------------------------------------------------------------------
# CONSTANTS

"""
OptimPack Constants
===================
`get_constant(name)` yields the value of an OptimPack constant `name`.

"""
function get_constant(name::AbstractString)
    value = Ref{Clong}(0)
    status = ccall((:opk_get_integer_constant, opklib), Cint,
                   (Cstring, Ref{Clong}), name, value)
    status == 0 || throw(ArgumentError("unknown OptimPack constant \"$name\""))
    convert(Cint, value[])
end

for sym in (# status
            :SUCCESS, :INVALID_ARGUMENT, :INSUFFICIENT_MEMORY,
            :ILLEGAL_ADDRESS, :NOT_IMPLEMENTED, :CORRUPTED_WORKSPACE,
            :BAD_SPACE, :OUT_OF_BOUNDS_INDEX, :NOT_STARTED, :NOT_A_DESCENT,
            :STEP_CHANGED, :STEP_OUTSIDE_BRACKET, :STPMIN_GT_STPMAX,
            :STPMIN_LT_ZERO, :STEP_LT_STPMIN, :STEP_GT_STPMAX,
            :FTOL_TEST_SATISFIED, :GTOL_TEST_SATISFIED, :XTOL_TEST_SATISFIED,
            :STEP_EQ_STPMAX, :STEP_EQ_STPMIN,
            :ROUNDING_ERRORS_PREVENT_PROGRESS, :NOT_POSITIVE_DEFINITE,
            :BAD_PRECONDITIONER, :INFEASIBLE_BOUNDS, :WOULD_BLOCK,
            :UNDEFINED_VALUE, :TOO_MANY_EVALUATIONS, :TOO_MANY_ITERATIONS,
            # boolean
            :TRUE, :FALSE,
            # data type
            :FLOAT, :DOUBLE,
            # reverse communication task
            :TASK_ERROR, :TASK_START, :TASK_COMPUTE_FG, :TASK_NEW_X,
            :TASK_FINAL_X, :TASK_WARNING,
            # line search
            :LNSRCH_ERROR, :LNSRCH_SEARCH, :LNSRCH_CONVERGENCE,
            :LNSRCH_WARNING,
            # nonlinear conjugate gradient
            :NLCG_FLETCHER_REEVES, :NLCG_HESTENES_STIEFEL,
            :NLCG_POLAK_RIBIERE_POLYAK, :NLCG_FLETCHER, :NLCG_LIU_STOREY,
            :NLCG_DAI_YUAN, :NLCG_PERRY_SHANNO, :NLCG_HAGER_ZHANG,
            :NLCG_POWELL, :NLCG_SHANNO_PHUA,
            # bounds
            :BOUND_NONE, :BOUND_SCALAR_FLOAT, :BOUND_SCALAR_DOUBLE,
            :BOUND_STATIC_FLOAT, :BOUND_STATIC_DOUBLE, :BOUND_VOLATILE_FLOAT,
            :BOUND_VOLATILE_DOUBLE, :BOUND_VECTOR,
            # search direction
            :ASCENT, :DESCENT,
            # initial inverse Hessian approximation
            :SCALING_NONE, :SCALING_OREN_SPEDICATO, :SCALING_BARZILAI_BORWEIN,
            # algorithm
            :ALGORITHM_NLCG, :ALGORITHM_VMLMB)
    let name = "OPK_"*string(sym), value = get_constant(name)
        @eval begin
            const $sym = $value
        end
    end
end

#------------------------------------------------------------------------------
# ERROR MANAGEMENT
#
# We must provide an appropriate error handler to OptimPack in order to throw
# an error exception and avoid aborting the program in case of misuse of the
# library.

__error__(ptr::Ptr{UInt8}) = (ErrorException(unsafe_string(ptr)); nothing)
const __cerror__ = Ref{Ptr{Cvoid}}(0)

# With precompilation, `__init__()` carries on initializations that must occur
# at runtime like C-callable function pointers.
function __init__()
    __cerror__[] = @cfunction(__error__, Nothing, (Ptr{UInt8},))
    ccall((:opk_set_error_handler, opklib), Ptr{Cvoid}, (Ptr{Cvoid},),
          __cerror__[])
    nothing
end

"""
`get_reason(s)` yields the textual reason for status `s`.
"""
function get_reason(s::Integer)
    val = ccall((:opk_get_reason, opklib), Ptr{UInt8}, (Cint,), s)
    val == C_NULL ? "" : unsafe_string(val)
end

guess_status() = ccall((:opk_guess_status, opklib), Cint, ())

#------------------------------------------------------------------------------
# OBJECT MANAGEMENT

"""
OptimPack Objects
=================

All concrete types derived from the abstract `Object` type have a `handle`
member which stores the address of the OptimPack object.  To avoid conflicts
with Julia `Vector` type, an OptimPack vector (*i.e.* `opk_vector_t`)
corresponds to the type `Variable` in Julia.
"""
@compat abstract type Object end

"""
Reference Counting
==================

OptimPack use reference counting for managing the memory.  The number of
references of an object is given by `references(obj)`.

Two *private* functions are used for managing the reference count:
`__hold_object__(ptr)` set a reference on an OptimPack object while
`__drop_object__(ptr)` discards a reference on an OptimPack object.  The
argument `ptr` is the address of the object.  This functions shall not be
directly called by a user of the code.
"""
function references(obj::Object)
    ccall((:opk_get_object_references, opklib), Cptrdiff_t, (Ptr{Cvoid},),
          obj.handle)
end

function __hold_object__(ptr::Ptr{Cvoid})
    ccall((:opk_hold_object, opklib), Ptr{Cvoid}, (Ptr{Cvoid},), ptr)
end

function __drop_object__(ptr::Ptr{Cvoid})
    ccall((:opk_drop_object, opklib), Nothing, (Ptr{Cvoid},), ptr)
end

@doc (@doc references) __hold_object__
@doc (@doc references) __drop_object__

#------------------------------------------------------------------------------
# VARIABLE SPACE

@compat abstract type VariableSpace <: Object end
"""

Variable Space
==============

Abstract type `VariableSpace` corresponds to a *vector space* (type
`opk_vspace_t`) in OptimPack.

"""
mutable struct DenseVariableSpace{T,N} <: VariableSpace
    handle::Ptr{Cvoid}
    eltype::Type{T}
    size::NTuple{N,Int}
    length::Int
end

# Extend basic functions for arrays.
length(vsp::DenseVariableSpace) = vsp.length
eltype(vsp::DenseVariableSpace) = vsp.eltype
size(vsp::DenseVariableSpace) = vsp.size
size(vsp::DenseVariableSpace, n::Integer) = vsp.size[n]
ndims(vsp::DenseVariableSpace) = length(vsp.size)

DenseVariableSpace(T::Union{Type{Cfloat},Type{Cdouble}},
                   dims::Int...) = DenseVariableSpace(T, dims)

function checkdims(dims::NTuple{N,Int}) where {N}
    number::Int = 1
    for dim in dims
        if dim < 1
            error("invalid dimension")
        end
        number *= dim
    end
    return number
end

for (T, f) in ((Cfloat, :opk_new_simple_float_vector_space),
               (Cdouble, :opk_new_simple_double_vector_space))
    @eval begin
        function DenseVariableSpace(::Type{$T}, dims::NTuple{N,Int}) where {N}
            length::Int = checkdims(dims)
            ptr = ccall(($(string(f)), opklib), Ptr{Cvoid}, (Cptrdiff_t,), length)
            systemerror("failed to create vector space", ptr == C_NULL)
            return finalizer(obj -> __drop_object__(obj.handle),
                             DenseVariableSpace{$T,N}(ptr, $T, dims, length))
        end
    end
end

#------------------------------------------------------------------------------
# VARIABLES

@compat abstract type Variable <: Object end

"""

Variables
=========

Abstract type `Variable` correspond to *vectors* (type `opk_vector_t`) in
OptimPack.

"""
mutable struct DenseVariable{T<:Floats,N} <: Variable
    # Note: There are no needs to register a reference for the owner of a
    # vector (it already owns one internally).
    handle::Ptr{Cvoid}
    owner::DenseVariableSpace{T,N}
    array::Union{Array,Nothing}
end

length(v::DenseVariable) = length(v.owner)
eltype(v::DenseVariable) = eltype(v.owner)
size(v::DenseVariable) = size(v.owner)
size(v::DenseVariable, n::Integer) = size(v.owner, n)
ndims(v::DenseVariable) = ndims(v.owner)
owner(v::DenseVector) = v.owner
__handle__(v::DenseVector) = v.handle

"""
`v = create(s)` creates a new variable of the variable space `s`.
"""
function create(space::DenseVariableSpace{T,N}) where {T<:Floats,N<:Integer}
    ptr = ccall((:opk_vcreate, opklib), Ptr{Cvoid}, (Ptr{Cvoid},), space.handle)
    systemerror("failed to create vector", ptr == C_NULL)
    return finalizer(obj -> __drop_object__(obj.handle),
                     DenseVariable{T,N}(ptr, space, nothing))
end

for (T, ctype) in ((Cfloat, "float"),
                   (Cdouble, "double"))
    @eval begin
        function wrap(s::DenseVariableSpace{$T,N}, a::DenseArray{$T,N}) where {N}
            @assert size(a) == size(s)
            ptr = ccall(($("opk_wrap_simple_"*ctype*"_vector"), opklib),
                        Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{$T}, Ptr{Cvoid}, Ptr{Cvoid}),
                        s.handle, a, C_NULL, C_NULL)
            systemerror("failed to wrap vector", ptr == C_NULL)
            return finalizer(obj -> __drop_object__(obj.handle),
                             DenseVariable{$T,N}(ptr, s, a))
        end

        function wrap!(v::DenseVariable{$T,N}, a::DenseArray{$T,N}) where {N}
            @assert size(a) == size(v)
            @assert v.array != nothing
            status = ccall(($("opk_rewrap_simple_"*ctype*"_vector"), opklib),
                           Cint, (Ptr{Cvoid}, Ptr{$T}, Ptr{Cvoid}, Ptr{Cvoid}),
                           v.handle, a, C_NULL, C_NULL)
            systemerror("failed to re-wrap vector", status != SUCCESS)
            v.array = a
            return v
        end
    end
end

"""
`v = wrap(s, a)` wraps the Julia array `a` into a variable of the space `s` and
returns the resulting variable `v`.  Array `a` must have the correct dimensions
and element type.
""" wrap

"""
`wrap!(v, a)` rewraps the Julia array `a` into the variable `v` and returns
`v`.  Array `a` must have the correct dimensions and element type.
""" wrap!

#------------------------------------------------------------------------------
# OPERATIONS ON VARIABLES (AS VECTORS)

"""
`norm1(v)` returns the L1 norm (sum of absolute values) ov *variables* `v`.
"""
function norm1(v::Variable)
    ccall((:opk_vnorm1, opklib), Cdouble, (Ptr{Cvoid},), v.handle)
end

"""
`norm2(v)` returns the Euclidean (L2) norm (square root of the sum of squared
values) of *variables* `v`.
"""
function norm2(v::Variable)
    ccall((:opk_vnorm2, opklib), Cdouble, (Ptr{Cvoid},), v.handle)
end

"""
`norminf(v)` returns the infinite norm (maximum absolute value) of *variables*
`v`.
"""
function norminf(v::Variable)
    ccall((:opk_vnorminf, opklib), Cdouble, (Ptr{Cvoid},), v.handle)
end

"""
`zero!(v)` fills *variables* `v` with zeros.
"""
function zero!(v::Variable)
    ccall((:opk_vzero, opklib), Nothing, (Ptr{Cvoid},), v.handle)
end

"""
`fill!(v, alpha)` fills *variables* `v` with value `alpha`.
"""
function fill!(v::Variable, alpha::Real)
    ccall((:opk_vfill, opklib), Nothing, (Ptr{Cvoid},Cdouble),
          v.handle, alpha)
end

"""
`copyto!(dst, src)` copies source *variables* `src` into the destination
*variables* `dst`.
"""
function copyto!(dst::Variable, src::Variable)
    ccall((:opk_vcopy, opklib), Nothing, (Ptr{Cvoid},Ptr{Cvoid}),
          dst.handle, src.handle)
end

"""
`scale!(dst, alpha, src)` stores `alpha` times the source *variables* `src`
into the destination *variables* `dst`.
"""
function scale!(dst::Variable, alpha::Real, src::Variable)
    ccall((:opk_vscale, opklib), Nothing, (Ptr{Cvoid},Cdouble,Ptr{Cvoid}),
          dst.handle, alpha, src.handle)
end

"""
`swap!(x, y)` exchanges the contents of *variables* `x` and `y`.
"""
function swap!(x::Variable, y::Variable)
    ccall((:opk_vswap, opklib), Nothing, (Ptr{Cvoid},Ptr{Cvoid}),
          x.handle, y.handle)
end

"""
`dot(x, y)` returns the inner product of *variables* `x` and `y`.
"""
function dot(x::Variable, y::Variable)
    ccall((:opk_vdot, opklib), Cdouble, (Ptr{Cvoid},Ptr{Cvoid}),
          x.handle, y.handle)
end

"""
`combine!(dst, alpha, x, beta, y)` stores into the destination `dst` the linear
combination `alpha*x + beta*y`.

`combine!(dst, alpha, x, beta, y, gamma, z)` stores into the destination `dst`
the linear combination `alpha*x + beta*y + gamma*z`.
"""
function combine!(dst::Variable,
                  alpha::Real, x::Variable,
                  beta::Real,  y::Variable)
    ccall((:opk_vaxpby, opklib), Nothing,
          (Ptr{Cvoid},Cdouble,Ptr{Cvoid},Cdouble,Ptr{Cvoid}),
          dst.handle, alpha, x.handle, beta, y.handle)
end

function combine!(dst::Variable,
                  alpha::Real, x::Variable,
                  beta::Real,  y::Variable,
                  gamma::Real, z::Variable)
    ccall((:opk_vaxpbypcz, opklib), Nothing,
          (Ptr{Cvoid},Cdouble,Ptr{Cvoid},Cdouble,Ptr{Cvoid},Cdouble,Ptr{Cvoid}),
          dst.handle, alpha, x.handle, beta, y.handle, gamma, y.handle)
end

#------------------------------------------------------------------------------
# OPERATORS

@compat abstract type Operator <: Object end

for (jf, cf) in ((:apply_direct!, :opk_apply_direct),
                 (:apply_adoint!, :opk_apply_adjoint),
                 (:apply_inverse!, :opk_apply_inverse))
    @eval begin
        function $jf(op::Operator,
                     dst::Variable,
                     src::Variable)
            status = ccall(($(string(cf)), opklib), Cint,
                           (Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),
                           op.handle, dst.handle, src.handle)
            if status != SUCCESS
                error("something wrong happens")
            end
            nothing
        end
    end
end

#------------------------------------------------------------------------------
# CONVEX SETS

@compat abstract type ConvexSet <: Object end

function checkbound(name::AbstractString, b::Variable, space::VariableSpace)
    if owner(b) != space
        throw(ArgumentError(name *
                            " must belong to the same space as the variables"))
    end
    return (b, BOUND_VECTOR, __handle__(b))
end

function checkbound(name::AbstractString, b::Real, space::VariableSpace)
    return checkbound(name, convert(Cdouble, b), space)
end

function checkbound(name::AbstractString, b::Cfloat, space::VariableSpace)
    return (b, BOUND_SCALAR_FLOAT, [b])
end

function checkbound(name::AbstractString, b::Cdouble, space::VariableSpace)
    return (b, BOUND_SCALAR_DOUBLE, [b])
end

function checkbound(::AbstractString, ::Nothing, ::VariableSpace)
    return (nothing, BOUND_NONE, C_NULL)
end

for (T, boundtype) in ((Cfloat, BOUND_STATIC_FLOAT),
                       (Cdouble, BOUND_STATIC_DOUBLE))
    @eval begin
        function checkbound(name::AbstractString, b::Array{$T,N},
                            space::DenseVariableSpace{$T,N}) where {N}
            if size(b) != size(space)
                throw(ArgumentError(name *
                                    " must have the same size as the variables"))
            end
            return (b, $boundtype, b)
        end
    end
end

mutable struct BoxedSet <: ConvexSet
    handle::Ptr{Cvoid}
    space::VariableSpace
    lower::Any
    upper::Any
    function BoxedSet(space::VariableSpace,
                      lower, lower_type::Cint, lower_addr,
                      upper, upper_type::Cint, upper_addr)
        ptr = ccall((:opk_new_boxset, opklib), Ptr{Cvoid},
                    (Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}),
                    space.handle,
                    lower_type, lower_addr,
                    upper_type, upper_addr)
        systemerror("failed to create linesearch", ptr == C_NULL)
        return finalizer(obj -> __drop_object__(obj.handle),
                         new(ptr, space, lower, upper))
    end
end

function BoxedSet(space::VariableSpace, lower, upper)
    (lo_value, lo_type, lo_addr) = checkbound("lower bound", lower, space)
    (up_value, up_type, up_addr) = checkbound("upper bound", upper, space)
    return BoxedSet(space,
                    lo_value, lo_type, lo_addr,
                    up_value, up_type, up_addr);
end


#for f in (:project_direction, :get_free_variables)
#    let name = string(f), f! = Symbol(name*"!"), cf = Symbol("opk_"*name)
#        @eval begin
#            function $f(x::Variable, set::ConvexSet,
#                        d::Variable, orient::integer)
#                dst = create(owner(x))
#                $f!(dst, x, set, d, orient)
#                return dst
#            end
#            function $f!(dst::Variable, x::Variable, set::ConvexSet,
#                         d::Variable, orient::integer)
#                status = ccall(($(string(cf)), opklib), Cint,
#                               (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
#                                Ptr{Cvoid}, Cint),
#                               dst.handle, x.handle, set.handle,
#                               d.handle, orient)
#                status == SUCCESS || error(get_reason(status))
#            end
#        end
#    end
#end
#
#function get_step_limits(x::Variable, set::ConvexSet, d::Variable,
#                         orient::integer)
#    smin1 = Ref{Cdouble}(0)
#    smin2 = Ref{Cdouble}(0)
#    smax  = Ref{Cdouble}(0)
#    status = ccall((:opk_get_step_limits, opklib), Cint,
#                   (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
#                    Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cint),
#                   smin1, smin2, smax, x.handle, set.handle, d.handle, orient)
#    status == SUCCESS || error(get_reason(status))
#    return (smin1[], smin2[], smax[])
#end

for f in (:can_project_direction,
          :can_get_free_variables,
          :can_get_step_limits)
    @eval begin
        function $f(set::ConvexSet)
            ccall(($("opk_"*string(f)), opklib), Cint, (Ptr{Cvoid},),
                  set.handle) != 0
        end
    end
end

#------------------------------------------------------------------------------
# LINE SEARCH METHODS

@compat abstract type LineSearch <: Object end

mutable struct ArmijoLineSearch <: LineSearch
    handle::Ptr{Cvoid}
    ftol::Cdouble
    function ArmijoLineSearch(;ftol::Real=1e-4)
        @assert 0.0 <= ftol < 1.0
        ptr = ccall((:opk_lnsrch_new_backtrack, opklib), Ptr{Cvoid},
                    (Cdouble,), ftol)
        systemerror("failed to create linesearch", ptr == C_NULL)
        return finalizer(obj -> __drop_object__(obj.handle),
                         new(ptr, ftol))
    end
end

mutable struct MoreThuenteLineSearch <: LineSearch
    handle::Ptr{Cvoid}
    ftol::Cdouble
    gtol::Cdouble
    xtol::Cdouble
    function MoreThuenteLineSearch(;ftol::Real=1e-4, gtol::Real=0.9,
                                   xtol::Real=eps(Cdouble))
        @assert 0.0 <= ftol < gtol < 1.0
        @assert 0.0 <= xtol < 1.0
        ptr = ccall((:opk_lnsrch_new_csrch, opklib), Ptr{Cvoid},
                (Cdouble, Cdouble, Cdouble), ftol, gtol, xtol)
        systemerror("failed to create linesearch", ptr == C_NULL)
        return finalizer(obj -> __drop_object__(obj.handle),
                         new(ptr, ftol, gtol, xtol))
    end
end

mutable struct NonmonotoneLineSearch <: LineSearch
    handle::Ptr{Cvoid}
    mem::Int
    ftol::Cdouble
    amin::Cdouble
    amax::Cdouble
    function NonmonotoneLineSearch(;mem::Integer=10, ftol::Real=1e-4,
                                   amin::Real=0.1, amax::Real=0.9)
        @assert mem >= 1
        @assert 0.0 <= ftol < 1.0
        @assert 0.0 < amin < amax < 1.0
        ptr = ccall((:opk_lnsrch_new_nonmonotone, opklib), Ptr{Cvoid},
                (Cptrdiff_t, Cdouble, Cdouble, Cdouble), mem, ftol, amin, amax)
        systemerror("failed to create nonmonotone linesearch", ptr == C_NULL)
        return finalizer(obj -> __drop_object__(obj.handle),
                         new(ptr, mem, ftol, amin, amax))
    end
end

function start!(ls::LineSearch, f::Real, df::Real,
                stp::Real, stpmin::Real, stpmax::Real)
    ccall((:opk_lnsrch_start, opklib), Cint,
          (Ptr{Cvoid}, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble),
          ls, f, df, stp, stpmin, stpmax)
end

function iterate!(ls::LineSearch, stp::Real, f::Real, df::Real)
    _stp = Cdouble[stp]
    task = ccall((:opk_lnsrch_iterate, opklib), Cint,
                 (Ptr{Cvoid}, Ptr{Cdouble}, Cdouble, Cdouble),
                 ls, _stp, f, df)
    return (task, _stp[1])
end

for (jf, ct, cf) in ((:get_step,   Cdouble, :opk_lnsrch_get_step),
                     (:get_task,   Cint,    :opk_lnsrch_get_task),
                     (:get_status, Cint,    :opk_lnsrch_get_status))

    @eval begin
        $jf(ls::LineSearch) = ccall(($(string(cf)), opklib), $ct, (Ptr{Cvoid}, ), ls)
    end
end

for (jf, cf) in ((:has_errors,   :opk_lnsrch_has_errors),
                 (:has_warnings, :opk_lnsrch_has_warnings),
                 (:converged,    :opk_lnsrch_converged),
                 (:finished,     :opk_lnsrch_finished),
                 (:use_deriv,    :opk_lnsrch_use_deriv))
    @eval begin
        $jf(ls::LineSearch) = (ccall(($(string(cf)), opklib), Cint,
                                     (Ptr{Cvoid},), ls) != 0)
    end
end

get_ftol(ls::LineSearch) = ls.ftol
get_gtol(ls::MoreThuenteLineSearch) = ls.gtol
get_xtol(ls::MoreThuenteLineSearch) = ls.xtol


#------------------------------------------------------------------------------
# NON LINEAR LIMITED-MEMORY OPTIMIZERS

@compat abstract type LimitedMemoryOptimizer <: Object end

@compat abstract type LimitedMemoryOptimizerOptions end

@compat abstract type LimitedMemoryOptimizerDriver <: LimitedMemoryOptimizer end

mutable struct VMLMBoptions <: LimitedMemoryOptimizerOptions
    # Relative size for a small step.
    delta::Cdouble

    # Threshold to accept descent direction.
    epsilon::Cdouble

    # Relative and absolute thresholds for the norm or the gradient for
    # convergence.
    grtol::Cdouble
    gatol::Cdouble

    # Relative minimum and maximum step length.
    stpmin::Cdouble
    stpmax::Cdouble

    # Maximum number of memorized steps.
    mem::Cptrdiff_t

    # Emulate Benson & Moré BLMVM method?
    blmvm::Cint

    # Save some memory?
    savemem::Cint

    function VMLMBoptions(;
                          delta::Union{Real,Nothing}=nothing,
                          epsilon::Union{Real,Nothing}=nothing,
                          gatol::Union{Real,Nothing}=nothing,
                          grtol::Union{Real,Nothing}=nothing,
                          stpmin::Union{Real,Nothing}=nothing,
                          stpmax::Union{Real,Nothing}=nothing,
                          mem::Union{Integer,Nothing}=nothing,
                          blmvm::Union{Bool,Nothing}=nothing,
                          savemem::Union{Bool,Nothing}=nothing)
        opts = new()
        initialize!(opts)
        if delta   != nothing; opts.delta   = delta;   end
        if epsilon != nothing; opts.epsilon = epsilon; end
        if gatol   != nothing; opts.gatol   = gatol;   end
        if grtol   != nothing; opts.grtol   = grtol;   end
        if stpmin  != nothing; opts.stpmin  = stpmin;  end
        if stpmax  != nothing; opts.stpmax  = stpmax;  end
        if mem     != nothing; opts.mem     = mem;     end
        if blmvm   != nothing; opts.blmvm   = (blmvm   ? 1 : 0); end
        if savemem != nothing; opts.savemem = (savemem ? 1 : 0); end
        check(opts)
        return opts
    end

end

mutable struct NLCGoptions <: LimitedMemoryOptimizerOptions
    # Relative size for a small step.
    delta::Cdouble

    # Threshold to accept descent direction.
    epsilon::Cdouble

    # Relative and absolute thresholds for the norm or the gradient for
    # convergence.
    grtol::Cdouble
    gatol::Cdouble

    # Relative minimum and maximum step length.
    stpmin::Cdouble
    stpmax::Cdouble

    # Minimal function value if provided.
    fmin::Cdouble

    # A bitwise combination of the non-linear conjugate gradient update method
    # and options.
    flags::Cuint

     # Minimal function value is provided?
    fmin_given::Cint

    function NLCGoptions(;
                         delta::Union{Real,Nothing}=nothing,
                         epsilon::Union{Real,Nothing}=nothing,
                         gatol::Union{Real,Nothing}=nothing,
                         grtol::Union{Real,Nothing}=nothing,
                         stpmin::Union{Real,Nothing}=nothing,
                         stpmax::Union{Real,Nothing}=nothing,
                         fmin::Union{Real,Nothing}=nothing,
                         flags::Union{Integer,Nothing}=nothing)
        opts = new()
        initialize!(opts)
        if delta   != nothing; opts.delta   = delta;   end
        if epsilon != nothing; opts.epsilon = epsilon; end
        if gatol   != nothing; opts.gatol   = gatol;   end
        if grtol   != nothing; opts.grtol   = grtol;   end
        if stpmin  != nothing; opts.stpmin  = stpmin;  end
        if stpmax  != nothing; opts.stpmax  = stpmax;  end
        if flags   != nothing; opts.flags   = flags;   end
        if fmin != nothing
            opts.fmin = fmin
            opts.fmin_given = fmin_given
        end
        check(opts)
        return opts
    end
end

for (T, f1, f2) in ((VMLMBoptions,
                     :opk_get_vmlmb_default_options,
                     :opk_check_vmlmb_options),
                    (NLCGoptions,
                     :opk_get_nlcg_default_options,
                     :opk_check_nlcg_options))
    @eval begin
        function initialize!(opts::$T)
            ccall(($(string(f1)), opklib), Nothing, (Ptr{$T},), Ref(opts))
        end
        function check(opts::$T)
            status = ccall(($(string(f2)), opklib), Cint, (Ptr{$T},), Ref(opts))
            status == SUCCESS || throw(ArgumentError("bad option(s)"))
        end
    end
end

@doc "Set default parameters." initialize!
@doc "Check options." check

"""
Default settings for non linear conjugate gradient (should correspond to the
method which is, in general, the most successful).
"""
const VMLMB_DEFAULT = VMLMBoptions()

mutable struct VMLMB <: LimitedMemoryOptimizer
    handle::Ptr{Cvoid}
    options::VMLMBoptions
    space::VariableSpace
    lnsrch::LineSearch
    box::Union{ConvexSet,Nothing}
    function VMLMB(options::VMLMBoptions,
                   space::VariableSpace,
                   lnsrch::LineSearch,
                   box::Union{ConvexSet,Nothing})
        mem = options.mem
        mem ≥ 1 || error("illegal number of memorized steps")
        mem = min(mem, length(space))
        box_handle =
        ptr = ccall((:opk_new_vmlmb_optimizer, opklib), Ptr{Cvoid},
                    (Ptr{VMLMBoptions}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                    Ref(options), space.handle, lnsrch.handle,
                    (box == nothing ? C_NULL : box.handle))
        systemerror("failed to create optimizer", ptr == C_NULL)
        return finalizer(obj -> __drop_object__(obj.handle),
                         new(ptr, options, space, lnsrch, box))
    end
end

"""
Default settings for non linear conjugate gradient (should correspond to the
method which is, in general, the most successful).
"""
const NLCG_DEFAULT = NLCGoptions()

mutable struct NLCG <: LimitedMemoryOptimizer
    handle::Ptr{Cvoid}
    options::NLCGoptions
    space::VariableSpace
    lnsrch::LineSearch
    function NLCG(options::NLCGoptions,
                  space::VariableSpace,
                  lnsrch::LineSearch)
        ptr = ccall((:opk_new_nlcg_optimizer, opklib), Ptr{Cvoid},
                    (Ptr{NLCGoptions}, Ptr{Cvoid}, Ptr{Cvoid}),
                    Ref(options), space.handle, lnsrch.handle)
        systemerror("failed to create optimizer", ptr == C_NULL)
        return finalizer(obj -> __drop_object__(obj.handle),
                         new(ptr, options, space, lnsrch))
    end
end

for (T, sfx) in ((LimitedMemoryOptimizerDriver, ""),
                 (NLCG, "_nlcg"), (VMLMB, "_vmlmb"))
    @eval begin

        start!(opt::$T, x::Variable) = ccall(($("opk_start"*sfx), opklib),
                                             Cint, (Ptr{Cvoid}, Ptr{Cvoid}),
                                             opt.handle, x.handle)

        function iterate!(opt::$T, x::Variable, f::Real, g::Variable)
            ccall(($("opk_iterate"*sfx), opklib), Cint,
                  (Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Ptr{Cvoid}),
                  opt.handle, x.handle, f, g.handle)
        end

        get_task(opt::$T) = ccall(($(string("opk_get"*sfx*"_task")), opklib),
                                    Cint, (Ptr{Cvoid},), opt.handle)

        get_status(opt::$T) = ccall(($(string("opk_get"*sfx*"_status")),
                                     opklib), Cint, (Ptr{Cvoid},), opt.handle)

        evaluations(opt::$T) = ccall(($(string("opk_get"*sfx*"_evaluations")),
                                      opklib), Cptrdiff_t, (Ptr{Cvoid},),
                                     opt.handle)

        iterations(opt::$T) = ccall(($(string("opk_get"*sfx*"_iterations")),
                                     opklib), Cptrdiff_t, (Ptr{Cvoid},),
                                    opt.handle)

        restarts(opt::$T) = ccall(($(string("opk_get"*sfx*"_restarts")),
                                   opklib), Cptrdiff_t, (Ptr{Cvoid},),
                                  opt.handle)

        get_step(opt::$T) = ccall(($(string("opk_get"*sfx*"_step")), opklib),
                                   Cdouble, (Ptr{Cvoid},), opt.handle)

        get_gnorm(opt::$T) = ccall(($(string("opk_get"*sfx*"_gnorm")), opklib),
                                   Cdouble, (Ptr{Cvoid},), opt.handle)

        function get_name(opt::$T)
            nbytes = ccall(($(string("opk_get"*sfx*"_name")), opklib), Csize_t,
                           (Ptr{UInt8}, Csize_t, Ptr{Cvoid}),
                           C_NULL, 0, opt.handle)
            buf = Array{UInt8}(undef, nbytes)
            ccall(($(string("opk_get"*sfx*"_name")), opklib), Csize_t,
                  (Ptr{UInt8}, Csize_t, Ptr{Cvoid}),
                  buf, nbytes, opt.handle)
            unsafe_string(buf)
        end

        function get_description(opt::$T)
            nbytes = ccall(($(string("opk_get"*sfx*"_description")), opklib),
                           Csize_t, (Ptr{UInt8}, Csize_t, Ptr{Cvoid}),
                           C_NULL, 0, opt.handle)
            buf = Array{UInt8}(undef, nbytes)
            ccall(($(string("opk_get"*sfx*"_description")), opklib), Csize_t,
                  (Ptr{UInt8}, Csize_t, Ptr{Cvoid}), buf, nbytes, opt.handle)
            unsafe_string(buf)
        end
    end
end

"""
`task = start!(opt)` starts optimization with the nonlinear optimizer
`opt` and returns the next pending task.
""" start!

"""
`task = iterate!(opt, x, f, g)` performs one optimization step with
the nonlinear optimizer `opt` for variables `x`, function value `f`
and gradient `g`.  The method returns the next pending task.
""" iterate!

"""
`get_task(opt)` returns the current pending task for the nonlinear
optimizer `opt`.
""" get_task

"""
`get_status(opt)` returns the current status of the nonlinear optimizer `opt`.
""" get_status

"""
`evaluations(opt)` returns the number of function (and gradient)
evaluations requested by the nonlinear optimizer `opt`.
""" evaluations

"""
`iterations(opt)` returns the number of iterations performed by the
nonlinear optimizer `opt`.
""" iterations

"""
`restarts(opt)` returns the number of restarts performed by the nonlinear
optimizer `opt`.
""" restarts

"""
`get_step(opt)` returns the current step length along the search direction.
""" get_step

"""
`get_gnorm(opt)` returns the norm of the (projected) gradient of the last
iterate accept by the nonlinear optimizer `opt`.
""" get_gnorm


#------------------------------------------------------------------------------
# DRIVERS FOR NON-LINEAR OPTIMIZATION

default_nlcg_line_search() = MoreThuenteLineSearch(ftol=1E-4, gtol=0.1)
default_vmlmb_line_search() = MoreThuenteLineSearch(ftol=1E-4, gtol=0.9)


"""
Nonlinear Conjugate Gradient
============================

Minimizing the smooth mulivariate function `f(x)` by a nonlinear conjugate
gradient methods is done by:
```
x = nlcg(fg!, x0, method)
```
where `fg!` implements the objective function (and its gradient), `x0` gives
the initial value of the variables (as well as the data type and dimensions of
the solution) and optional argument `method` may be used to choose a specific
conjugate gradient method.

See `vmlmb` for more details.
"""
function nlcg(fg!::Function, x0::DenseArray{T,N},
              flags::Integer=NLCG_DEFAULT.flags;
              lnsrch::LineSearch=default_nlcg_line_search(),
              delta::Real=NLCG_DEFAULT.delta,
              epsilon::Real=NLCG_DEFAULT.epsilon,
              fmin::Union{Real,Nothing}=nothing,
              gatol::Real=NLCG_DEFAULT.gatol,
              grtol::Real=NLCG_DEFAULT.grtol,
              stpmin::Real=NLCG_DEFAULT.stpmin,
              stpmax::Real=NLCG_DEFAULT.stpmax,
              maxeval::Integer=-1, maxiter::Integer=-1,
              verb::Bool=false, debug::Bool=false) where {T<:Floats,N}
    # Create an optimizer and solve the problem.
    dims = size(x0)
    space = DenseVariableSpace(T, dims)
    options = NLCGoptions(delta=delta, epsilon=epsilon,
                          gatol=gatol, grtol=grtol,
                          stpmin=stpmin, stpmax=stpmax,
                          flags=flags, fmin=fmin)
    opt = NLCG(options, space, lnsrch)
    solve(opt, fg!, x0, maxeval=maxeval, maxiter=maxiter,
          verb=verb, debug=debug)
end

"""
Limited Memory Variable Metric
==============================

Minimizing the smooth mulivariate function `f(x)` by a limited-memory version
of the LBFGS variable metric method is done by:
```
x = vmlmb(fg!, x0, mem)
```
where `fg!` implements the objective function (see below), `x0` gives
the initial value of the variables (as well as the data type and dimensions of
the solution) and optional argument `mem` is the number of previous steps to
memorize (by default `mem = 3`).

The objective function is implemented by `fg!` which is called as:
```
f = fg!(x, g)
```
with `x` the current variables and `g` a Julia array (of same type and
simensions as `x`) to store the gradient of the function.  The value returned
by `fg!` is `f(x)`.
"""
function vmlmb(fg!::Function, x0::DenseArray{T,N};
               lower=nothing, upper=nothing,
               lnsrch::LineSearch=default_vmlmb_line_search(),
               mem::Integer=VMLMB_DEFAULT.mem,
               delta::Real=VMLMB_DEFAULT.delta,
               epsilon::Real=VMLMB_DEFAULT.epsilon,
               gatol::Real=VMLMB_DEFAULT.gatol,
               grtol::Real=VMLMB_DEFAULT.grtol,
               stpmin::Real=VMLMB_DEFAULT.stpmin,
               stpmax::Real=VMLMB_DEFAULT.stpmax,
               blmvm::Bool=(VMLMB_DEFAULT.blmvm != 0),
               savemem::Bool=(VMLMB_DEFAULT.savemem != 0),
               maxeval::Integer=-1,
               maxiter::Integer=-1,
               verb::Bool=false,
               debug::Bool=false) where {T<:Floats,N}
    # Create an optimizer and solve the problem.
    #options = VMLMBoptions(mem=mem)
    options = VMLMBoptions(delta=delta, epsilon=epsilon,
                           gatol=gatol, grtol=grtol,
                           stpmin=stpmin, stpmax=stpmax,
                           mem=mem, blmvm=blmvm, savemem=savemem)
    space = DenseVariableSpace(T, size(x0))
    if lower == nothing && upper == nothing
        box = nothing
    else
        box = BoxedSet(space, lower, upper)
    end
    opt = VMLMB(options, space, lnsrch, box)
    solve(opt, fg!, x0, maxeval=maxeval, maxiter=maxiter,
          verb=verb, debug=debug)
end

function solve(opt::LimitedMemoryOptimizer, fg!::Function, x0::DenseArray;
               maxeval::Integer=-1, maxiter::Integer=-1,
               verb::Bool=false, debug::Bool=false)
    if debug
        @printf("gatol=%E; grtol=%E; stpmin=%E; stpmax=%E\n",
                get_gatol(opt), get_grtol(opt),
                get_stpmin(opt), get_stpmax(opt))
    end
    dims = size(x0)
    space = opt.space
    x = copy(x0)
    g = similar(x)
    wx = wrap(space, x)
    wg = wrap(space, g)
    task = start!(opt, wx)
    local f
    while true
        if task == TASK_COMPUTE_FG
            f = fg!(x, g)
        elseif task == TASK_NEW_X || task == TASK_FINAL_X
            iter = iterations(opt)
            eval = evaluations(opt)
            if verb
                if iter == 0
                    @printf("%s\n%s\n",
                            " ITER   EVAL  RESTARTS          F(X)           ||G(X)||    STEP",
                            "-----------------------------------------------------------------")
                end
                @printf("%5d  %5d  %5d  %24.16E %9.2E %9.2E\n",
                        iter, eval, restarts(opt), f, get_gnorm(opt),
                        get_step(opt))
            end
            if task == TASK_FINAL_X
                return x
            end
            if maxiter >= 0 && iter >= maxiter
                @warn "exceeding maximum number of iterations ($maxiter)"
                return x
            end
            if maxeval >= 0 && eval >= maxeval
                @warn "exceeding maximum number of evaluations ($eval >= $maxeval)"
                return x
            end
        elseif task == TASK_WARNING
            @printf("some warnings...\n")
            return x
        elseif task == TASK_ERROR
            @printf("some errors...\n")
            return nothing
        else
            @printf("unexpected task...\n")
            return nothing
        end
        task = iterate!(opt, wx, f, wg)
    end
end

# Load other components.
include("Brent.jl")
include("powell.jl")
include("spg2.jl")

end # module
