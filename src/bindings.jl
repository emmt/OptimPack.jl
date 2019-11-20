#
# bindings.jl --
#
# Julia wrapper for the C library of OptimPack.
#
# ----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT "Expat"
# License:
#
# Copyright (C) 2014-2019, Éric Thiébaut.
#
# ----------------------------------------------------------------------------

# Functions must be imported to be extended with new methods.
import Base: ENV, size, length, eltype, ndims, copy, copyto!, fill!
import LinearAlgebra: dot

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
    status = ccall((:opk_get_integer_constant, libopk), Cint,
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
    let name = string("OPK_",sym), value = get_constant(name)
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

# FIXME: throw exception?
__error__(ptr::Ptr{UInt8}) = (ErrorException(unsafe_string(ptr)); nothing)
const __cerror__ = Ref{Ptr{Cvoid}}(0)

# With precompilation, `__init__()` carries on initializations that must occur
# at runtime like C-callable function pointers.
function __init__()
    __cerror__[] = @cfunction(__error__, Cvoid, (Ptr{UInt8},))
    ccall((:opk_set_error_handler, libopk), Ptr{Cvoid}, (Ptr{Cvoid},),
          __cerror__[])
    nothing
end

"""

`get_reason(s)` yields the textual reason for status `s`.

"""
function get_reason(s::Integer)
    val = ccall((:opk_get_reason, libopk), Ptr{UInt8}, (Cint,), s)
    val == C_NULL ? "" : unsafe_string(val)
end

"""

`guess_status()` yields an error code determined from the current value of
`Libc.errno()`.

"""
guess_status() = ccall((:opk_guess_status, libopk), Cint, ())

#------------------------------------------------------------------------------
# OBJECT MANAGEMENT

"""

OptimPack Objects
=================

All concrete types derived from the abstract `Object` type have a `handle`
member which stores the address of the OptimPack object (of C type
`opk_object_t`).  To avoid conflicts with Julia `Vector` type, an OptimPack
vector (*i.e.* `opk_vector_t`) corresponds to the type `Variable` in Julia.

"""
abstract type Object end

"""

Reference Counting
==================

OptimPack use reference counting for managing the memory.  The number of
references of an object is given by `references(obj)`.

Two *private* functions are used for managing the reference count: `_hold(ptr)`
set a reference on an OptimPack object while `_drop(ptr)` discards a reference
on an OptimPack object.  The argument `ptr` is the address of the object.  This
functions shall not be directly called by a user of the code.

"""
references(obj) =
    ccall((:opk_get_object_references, libopk), Cptrdiff_t,
          (Ptr{Object},), obj)

_hold(obj) = ccall((:opk_hold_object, libopk), Ptr{Object}, (Ptr{Object},), obj)

_drop(obj) = ccall((:opk_drop_object, libopk), Cvoid, (Ptr{Object},), obj)

@doc (@doc references) _hold
@doc (@doc references) _drop

#------------------------------------------------------------------------------
# VARIABLE SPACE

"""

Variable Space
==============

Abstract type `VariableSpace` corresponds to a *vector space* (type
`opk_vspace_t`) in OptimPack.

"""
abstract type VariableSpace <: Object end

mutable struct DenseVariableSpace{T<:Floats,N} <: VariableSpace
    handle::Ptr{DenseVariableSpace}
    size::NTuple{N,Int}

    DenseVariableSpace{T,N}(dims::NTuple{N,Int}) where {T<:Floats,N} =
        finalizer(_drop, new{T,N}(_new_simple_vector_space(T, dims), dims))
end

# Extend basic methods for variable spaces.
eltype(vsp::DenseVariableSpace{T}) where {T} = T
ndims(vsp::DenseVariableSpace{T,N}) where {T,N} = N
length(vsp::DenseVariableSpace) = prod(size(vsp))
size(vsp::DenseVariableSpace) = vsp.size
size(vsp::DenseVariableSpace{T,N}, d::Integer) where {T,N} =
    (d < 1 ? error("out of bounds dimension") :
     d ≤ N ? (@inbounds vsp.size[d]) : 1)

DenseVariableSpace(::Type{T}, dims::Integer...) where {T} =
    DenseVariableSpace(T, dims)

DenseVariableSpace(::Type{T}, dims::Tuple{Vararg{Integer}}) where {T} =
    DenseVariableSpace(T, map(Int, dims))

DenseVariableSpace(::Type{T}, dims::NTuple{N,Int}) where {T,N} =
    DenseVariableSpace{T,N}(dims)

for (T, func) in ((Cfloat, "opk_new_simple_float_vector_space"),
                  (Cdouble, "opk_new_simple_double_vector_space"))
    @eval function _new_simple_vector_space(::Type{$T},
                                            dims::NTuple{N,Int}) where {N}

        len = 1
        for dim in dims
            dim ≥ 1 || error("invalid dimension")
            len *= dim
        end
        ptr = ccall(($func, libopk), Ptr{DenseVariableSpace},
                    (Cptrdiff_t,), len)
        systemerror("failed to create vector space", ptr == C_NULL)
        return ptr
    end
end


#------------------------------------------------------------------------------
# VARIABLES

"""

Variables
=========

Abstract type `Variable` correspond to *vectors* (type `opk_vector_t`) in
OptimPack.

"""
abstract type Variable <: Object end

mutable struct DenseVariable{T<:Floats,N,A<:Union{Array{T},Nothing}} <: Variable
    # Note: There are no needs to register a reference for the owner of a
    # vector (it already owns one internally).
    handle::Ptr{DenseVariable}
    owner::DenseVariableSpace{T,N}
    values::A
end

eltype(v::DenseVariable{T,N}) where {T,N} = T
ndims(v::DenseVariable{T,N}) where {T,N} = N
length(v::DenseVariable) = length(v.owner)
size(v::DenseVariable) = size(v.owner)
size(v::DenseVariable, d::Integer) = size(v.owner, d)
owner(v::DenseVector) = v.owner

"""

```julia
create(vsp, monolithic=Val(false)) -> var
```

creates a new variable of the variable space `vsp`.  If `monolithic` is
`Val(true)`, the variable is allocated in one piece of memory by OptimPack C
library.  Otherwise, the variable is wrapped over an embedded Julia array.

"""
create(vsp::DenseVariableSpace) = create(vsp, Val(false))

create(vsp::DenseVariableSpace{T,N}, ::Val{false}) where {T<:Floats,N} =
    wrap(vsp, Array{T,N}(undef, size(vspc)))

function create(vsp::DenseVariableSpace{T,N},
                ::Val{true}) where {T<:Floats,N}
    ptr = ccall((:opk_vcreate, libopk), Ptr{Variable},
                (Ptr{VariableSpace},), vsp)
    systemerror("failed to create vector", ptr == C_NULL)
    return finalizer(_drop, DenseVariable{T,N,Nothing}(ptr, vsp, nothing))
end

"""

`var = wrap(vsp, arr)` wraps the Julia array `arr` into a variable of the space
`vsp` and returns the resulting variable `var`.  Array `arr` must have the
correct dimensions and element type.

""" wrap

"""

`wrap!(var, arr)` rewraps the Julia array `arr` into the variable `var` and
returns `var`.  Array `arr` must have the correct dimensions and element type.

""" wrap!

for (T, ctype) in ((Cfloat, "float"),
                   (Cdouble, "double"))
    @eval begin
        function wrap(vsp::DenseVariableSpace{$T,N},
                      arr::A) where {N,A<:DenseArray{$T,N}}
            assertflatarray(arr)
            size(arr) == size(vsp) || error("incompatible array dimensions")
            ptr = ccall(($("opk_wrap_simple_"*ctype*"_vector"), libopk),
                        Ptr{Variable},
                        (Ptr{VariableSpace}, Ptr{$T}, Ptr{Cvoid}, Ptr{Cvoid}),
                        vsp, arr, C_NULL, C_NULL)
            systemerror("failed to wrap vector", ptr == C_NULL)
            return finalizer(_drop, DenseVariable{$T,N,A}(ptr, vsp, arr))
        end

        function wrap!(var::DenseVariable{$T,N,A}, arr::A) where {N,A}
            assertflatarray(arr)
            size(arr) == size(vsp) || error("incompatible array dimensions")
            status = ccall(($("opk_rewrap_simple_"*ctype*"_vector"), libopk),
                           Cint,
                           (Ptr{Variable}, Ptr{$T}, Ptr{Cvoid}, Ptr{Cvoid}),
                           var, arr, C_NULL, C_NULL)
            systemerror("failed to re-wrap vector", status != SUCCESS)
            var.values = arr
            return var
        end
    end
end

"""

```julia
assertflatarray(A)
```

ensures that `A` is a *flat array* that is an array whose first element is
at index 1 and elements are contiguous and stored in comum-major order.

This is to make sure that the array can be accessed like a simple vector.

"""

function assertflatarray(A::DenseArray{T,N}) where {T,N}
    inds = axes(A)
    stds = strides(A)
    stride = 1
    @inbounds for d in 1:N
        stds[d] == stride || error("unsupported element ordering")
        first(inds[d]) == 1 || error("unsupported indexing")
        dim = last(inds[d])
        dim > 0 || error("invalid dimenson(s)")
        stride *= dim
    end
end

#------------------------------------------------------------------------------
# OPERATIONS ON VARIABLES (AS VECTORS)

"""

`norm1(var)` returns the L1 norm (sum of absolute values) of *variables* `var`.

"""
norm1(var::Variable) =
    ccall((:opk_vnorm1, libopk), Cdouble, (Ptr{Variable},), var)

"""

`norm2(var)` returns the Euclidean (L2) norm (square root of the sum of squared
values) of *variables* `var`.

"""
norm2(var::Variable) =
    ccall((:opk_vnorm2, libopk), Cdouble, (Ptr{Variable},), var)

"""

`norminf(var)` returns the infinite norm (maximum absolute value) of
*variables* `var`.

"""
norminf(var::Variable) =
    ccall((:opk_vnorminf, libopk), Cdouble, (Ptr{Variable},), var)

"""

`zero!(var)` fills *variables* `var` with zeros and returns `var`.

"""
zero!(var::Variable) = begin
    ccall((:opk_vzero, libopk), Cvoid, (Ptr{Variable},), var)
    return var
end

"""

`fill!(var, alpha)` fills *variables* `var` with value `alpha` and returns
`var`.

"""
fill!(var::Variable, alpha::Real) = begin
    ccall((:opk_vfill, libopk), Cvoid, (Ptr{Variable}, Cdouble), var, alpha)
    return var
end

"""

`copyto!(dst, src)` copies source *variables* `src` into the destination
*variables* `dst` and returns `dst`.

"""
copyto!(dst::Variable, src::Variable) = begin
    ccall((:opk_vcopy, libopk), Cvoid, (Ptr{Variable}, Ptr{Variable}),
          dst, src)
    return dst
end

"""

`scale!(dst, alpha, src)` stores `alpha` times the source *variables* `src`
into the destination *variables* `dst` and returns `dst`.  Operation can be
done in-place for variables `var` by calling `scale!(alpha, var)` or
`scale!(var, alpha)`.

"""
scale!(dst::Variable, alpha::Real, src::Variable) = begin
    ccall((:opk_vscale, libopk), Cvoid,
          (Ptr{Variable}, Cdouble, Ptr{Variable}), dst, alpha, src)
    return dst
end
scale!(alpha::Real, var::Variable) = scale!(var, alpha, var)
scale!(var::Variable, alpha::Real) = scale!(var, alpha, var)

"""

`swap!(x, y)` exchanges the contents of *variables* `x` and `y`.

"""
swap!(x::Variable, y::Variable) =
    ccall((:opk_vswap, libopk), Cvoid, (Ptr{Variable}, Ptr{Variable}), x, y)

"""

`dot(x, y)` returns the inner product of *variables* `x` and `y`.

"""
dot(x::Variable, y::Variable) =
    ccall((:opk_vdot, libopk), Cdouble, (Ptr{Variable}, Ptr{Variable}), x, y)

"""

`combine!(dst, alpha, x, beta, y)` stores into the destination `dst` the linear
combination `alpha*x + beta*y` and returns `dst`.

`combine!(dst, alpha, x, beta, y, gamma, z)` stores into the destination `dst`
the linear combination `alpha*x + beta*y + gamma*z` and returns `dst`.

"""
function combine!(dst::Variable,
                  alpha::Real, x::Variable,
                  beta::Real,  y::Variable)
    ccall((:opk_vaxpby, libopk), Cvoid,
          (Ptr{Variable}, Cdouble, Ptr{Variable}, Cdouble, Ptr{Variable}),
          dst, alpha, x, beta, y)
    return dst
end

function combine!(dst::Variable,
                  alpha::Real, x::Variable,
                  beta::Real,  y::Variable,
                  gamma::Real, z::Variable)
    ccall((:opk_vaxpbypcz, libopk), Cvoid,
          (Ptr{Variable}, Cdouble, Ptr{Variable}, Cdouble, Ptr{Variable},
           Cdouble, Ptr{Variable}),
          dst, alpha, x, beta, y, gamma, y)
    return dst
end

#------------------------------------------------------------------------------
# OPERATORS

"""

Abstract type `Operator` represents an OptimPack `opk_operator_t` opaque
structure.

"""
abstract type Operator <: Object end

for (jf, cf) in ((:apply_direct!, :opk_apply_direct),
                 (:apply_adoint!, :opk_apply_adjoint),
                 (:apply_inverse!, :opk_apply_inverse))
    @eval function $jf(op::Operator, dst::Variable, src::Variable)
        status = ccall(($(string(cf)), libopk), Cint,
                       (Ptr{Operator}, Ptr{Variable}, Ptr{Variable}),
                       op, dst, src)
        if status != SUCCESS
            error("something wrong happens")
        end
        return dst
    end
end

#------------------------------------------------------------------------------
# CONVEX SETS

"""

Abstract type `ConvexSet` represents an OptimPack `opk_convexset_t` opaque
structure.

"""
abstract type ConvexSet <: Object end

checkbound(name::AbstractString, b::Variable, space::VariableSpace) = begin
    owner(b) == space || throw_not_same_space(name) # FIXME: efficiency of the comparison
    return (b, BOUND_VECTOR, _handle(b))
end

checkbound(name::AbstractString, b::Real, space::VariableSpace) =
    checkbound(name, Cdouble(b), space)

checkbound(name::AbstractString, b::Cfloat, space::VariableSpace) =
    (b, BOUND_SCALAR_FLOAT, [b])

checkbound(name::AbstractString, b::Cdouble, space::VariableSpace) =
    (b, BOUND_SCALAR_DOUBLE, [b])

checkbound(::AbstractString, ::Nothing, ::VariableSpace) =
    (nothing, BOUND_NONE, C_NULL)

for (T, boundtype) in ((Cfloat, BOUND_STATIC_FLOAT),
                       (Cdouble, BOUND_STATIC_DOUBLE))
    @eval function checkbound(name::AbstractString, b::Array{$T,N},
                              space::DenseVariableSpace{$T,N}) where {N}
        size(b) == size(space) || throw_not_same_size(name)
        return (b, $boundtype, b)
    end
end

@noinline throw_not_same_space(name::AbstractString) =
    throw(ArgumentError(string(name,
                               " must belong to the same space as the variables")))

@noinline throw_not_same_size(name::AbstractString) =
    throw(ArgumentError(string(name,
                               " must belong to the same space as the variables")))

mutable struct BoxedSet <: ConvexSet
    handle::Ptr{BoxedSet}
    space::VariableSpace
    lower::Any
    upper::Any
    function BoxedSet(space::VariableSpace,
                      lower, lower_type::Cint, lower_addr,
                      upper, upper_type::Cint, upper_addr)
        ptr = ccall((:opk_new_boxset, libopk), Ptr{BoxedSet},
                    (Ptr{VariableSpace}, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}),
                    space,
                    lower_type, lower_addr,
                    upper_type, upper_addr)
        systemerror("failed to create boxed set", ptr == C_NULL)
        return finalizer(_drop, new(ptr, space, lower, upper))
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
#                status = ccall(($(string(cf)), libopk), Cint,
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
#    status = ccall((:opk_get_step_limits, libopk), Cint,
#                   (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
#                    Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cint),
#                   smin1, smin2, smax, x.handle, set.handle, d.handle, orient)
#    status == SUCCESS || error(get_reason(status))
#    return (smin1[], smin2[], smax[])
#end

for f in (:can_project_direction,
          :can_get_free_variables,
          :can_get_step_limits)
    @eval $f(obj::ConvexSet) = (ccall(($(string("opk_",f)), libopk),
                                      Cint, (Ptr{ConvexSet},), obj) != 0)
end

#------------------------------------------------------------------------------
# LINE SEARCH METHODS

"""

Abstract type `LineSearch` represents an OptimPack `opk_lnsrch_t` opaque
structure.

"""
abstract type LineSearch <: Object end

mutable struct ArmijoLineSearch <: LineSearch
    handle::Ptr{ArmijoLineSearch}
    ftol::Cdouble
    function ArmijoLineSearch(; ftol::Real=1e-4)
        @assert 0.0 <= ftol < 1.0
        ptr = ccall((:opk_lnsrch_new_backtrack, libopk),
                    Ptr{ArmijoLineSearch},
                    (Cdouble,), ftol)
        systemerror("failed to create linesearch", ptr == C_NULL)
        return finalizer(_drop, new(ptr, ftol))
    end
end

mutable struct MoreThuenteLineSearch <: LineSearch
    handle::Ptr{MoreThuenteLineSearch}
    ftol::Cdouble
    gtol::Cdouble
    xtol::Cdouble
    function MoreThuenteLineSearch(; ftol::Real=1e-4, gtol::Real=0.9,
                                   xtol::Real=eps(Cdouble))
        @assert 0.0 <= ftol < gtol < 1.0
        @assert 0.0 <= xtol < 1.0
        ptr = ccall((:opk_lnsrch_new_csrch, libopk),
                    Ptr{MoreThuenteLineSearch},
                    (Cdouble, Cdouble, Cdouble),
                    ftol, gtol, xtol)
        systemerror("failed to create linesearch", ptr == C_NULL)
        return finalizer(_drop, new(ptr, ftol, gtol, xtol))
    end
end

mutable struct NonmonotoneLineSearch <: LineSearch
    handle::Ptr{NonmonotoneLineSearch}
    mem::Int
    ftol::Cdouble
    amin::Cdouble
    amax::Cdouble
    function NonmonotoneLineSearch(; mem::Integer=10, ftol::Real=1e-4,
                                   amin::Real=0.1, amax::Real=0.9)
        @assert mem >= 1
        @assert 0.0 <= ftol < 1.0
        @assert 0.0 < amin < amax < 1.0
        ptr = ccall((:opk_lnsrch_new_nonmonotone, libopk),
                    Ptr{NonmonotoneLineSearch},
                    (Cptrdiff_t, Cdouble, Cdouble, Cdouble),
                    mem, ftol, amin, amax)
        systemerror("failed to create nonmonotone linesearch", ptr == C_NULL)
        return finalizer(_drop, new(ptr, mem, ftol, amin, amax))
    end
end

function start!(ls::LineSearch, f::Real, df::Real,
                stp::Real, stpmin::Real, stpmax::Real)
    ccall((:opk_lnsrch_start, libopk), Cint,
          (Ptr{LineSearch}, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble),
          ls, f, df, stp, stpmin, stpmax)
end

function iterate!(ls::LineSearch, stp::Real, f::Real, df::Real)
    refstp = Ref{Cdouble}(stp)
    task = ccall((:opk_lnsrch_iterate, libopk), Cint,
                 (Ptr{LineSearch}, Ptr{Cdouble}, Cdouble, Cdouble),
                 ls, refstp, f, df)
    return (task, refstp[])
end

for (jf, ct, cf) in ((:get_step,   Cdouble, "opk_lnsrch_get_step"),
                     (:get_task,   Cint,    "opk_lnsrch_get_task"),
                     (:get_status, Cint,    "opk_lnsrch_get_status"))

    @eval $jf(ls::LineSearch) = ccall(($cf, libopk), $ct,
                                      (Ptr{LineSearch}, ), ls)
end

for (jf, cf) in ((:has_errors,      "opk_lnsrch_has_errors"),
                 (:has_warnings,    "opk_lnsrch_has_warnings"),
                 (:converged,       "opk_lnsrch_converged"),
                 (:finished,        "opk_lnsrch_finished"),
                 (:use_derivatives, "opk_lnsrch_use_deriv"))
    @eval $jf(ls::LineSearch) = (ccall(($cf, libopk), Cint,
                                       (Ptr{LineSearch},), ls) != 0)
end

get_ftol(ls::LineSearch) = ls.ftol
get_gtol(ls::MoreThuenteLineSearch) = ls.gtol
get_xtol(ls::MoreThuenteLineSearch) = ls.xtol
get_mem(ls::NonmonotoneLineSearch) = ls.mem
get_amin(ls::NonmonotoneLineSearch) = ls.amin
get_amax(ls::NonmonotoneLineSearch) = ls.amax

#------------------------------------------------------------------------------
# NON LINEAR LIMITED-MEMORY OPTIMIZERS

abstract type LimitedMemoryOptimizer <: Object end

abstract type LimitedMemoryOptimizerOptions end

abstract type LimitedMemoryOptimizerDriver <: LimitedMemoryOptimizer end

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
        if delta   !== nothing; opts.delta   = delta;             end
        if epsilon !== nothing; opts.epsilon = epsilon;           end
        if gatol   !== nothing; opts.gatol   = gatol;             end
        if grtol   !== nothing; opts.grtol   = grtol;             end
        if stpmin  !== nothing; opts.stpmin  = stpmin;            end
        if stpmax  !== nothing; opts.stpmax  = stpmax;            end
        if mem     !== nothing; opts.mem     = mem;               end
        if blmvm   !== nothing; opts.blmvm   = (blmvm   ? 1 : 0); end
        if savemem !== nothing; opts.savemem = (savemem ? 1 : 0); end
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
        if delta   !== nothing; opts.delta   = delta;   end
        if epsilon !== nothing; opts.epsilon = epsilon; end
        if gatol   !== nothing; opts.gatol   = gatol;   end
        if grtol   !== nothing; opts.grtol   = grtol;   end
        if stpmin  !== nothing; opts.stpmin  = stpmin;  end
        if stpmax  !== nothing; opts.stpmax  = stpmax;  end
        if flags   !== nothing; opts.flags   = flags;   end
        if fmin    !== nothing
            opts.fmin = fmin
            opts.fmin_given = fmin_given
        end
        check(opts)
        return opts
    end
end

for (T, f1, f2) in ((VMLMBoptions,
                     "opk_get_vmlmb_default_options",
                     "opk_check_vmlmb_options"),
                    (NLCGoptions,
                     "opk_get_nlcg_default_options",
                     "opk_check_nlcg_options"))
    @eval begin
        initialize!(opts::$T) =
            ccall(($f1, libopk), Cvoid, (Ref{$T},), opts)
        check(opts::$T) = begin
            status = ccall(($f2, libopk), Cint, (Ref{$T},), opts)
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
    handle::Ptr{VMLMB}
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
        ptr = ccall((:opk_new_vmlmb_optimizer, libopk), Ptr{VMLMB},
                    (Ref{VMLMBoptions}, Ptr{VariableSpace}, Ptr{LineSearch},
                     Ptr{ConvexSet}), options, space, lnsrch,
                    (box === nothing ?
                     Ptr{ConvexSet}(0) :
                     Ptr{ConvexSet}(_handle(box))))
        systemerror("failed to create optimizer", ptr == C_NULL)
        return finalizer(_drop, new(ptr, options, space, lnsrch, box))
    end
end


"""

Default settings for non linear conjugate gradient (should correspond to the
method which is, in general, the most successful).

"""
const NLCG_DEFAULT = NLCGoptions()

mutable struct NLCG <: LimitedMemoryOptimizer
    handle::Ptr{NLCG}
    options::NLCGoptions
    space::VariableSpace
    lnsrch::LineSearch
    function NLCG(options::NLCGoptions,
                  space::VariableSpace,
                  lnsrch::LineSearch)
        ptr = ccall((:opk_new_nlcg_optimizer, libopk), Ptr{NLCG},
                    (Ref{NLCGoptions}, Ptr{VariableSpace}, Ptr{LineSearch}),
                    options, space, lnsrch)
        systemerror("failed to create optimizer", ptr == C_NULL)
        return finalizer(_drop, new(ptr, options, space, lnsrch))
    end
end

for (T, sfx) in ((LimitedMemoryOptimizerDriver, ""),
                 (NLCG, "_nlcg"), (VMLMB, "_vmlmb"))
    @eval begin

        start!(opt::$T, x::Variable) = ccall(($("opk_start"*sfx), libopk),
                                             Cint, (Ptr{$T}, Ptr{Variable}),
                                             opt, x)

        function iterate!(opt::$T, x::Variable, f::Real, g::Variable)
            ccall(($("opk_iterate"*sfx), libopk), Cint,
                  (Ptr{$T}, Ptr{Variable}, Cdouble, Ptr{Variable}),
                  opt, x, f, g)
        end

        get_task(opt::$T) = ccall(($("opk_get"*sfx*"_task"), libopk),
                                    Cint, (Ptr{$T},), opt)

        get_status(opt::$T) = ccall(($("opk_get"*sfx*"_status"), libopk),
                                    Cint, (Ptr{$T},), opt)

        evaluations(opt::$T) = ccall(($("opk_get"*sfx*"_evaluations"), libopk),
                                     Cptrdiff_t, (Ptr{$T},), opt)

        iterations(opt::$T) = ccall(($("opk_get"*sfx*"_iterations"), libopk),
                                    Cptrdiff_t, (Ptr{$T},), opt)

        restarts(opt::$T) = ccall(($("opk_get"*sfx*"_restarts"), libopk),
                                  Cptrdiff_t, (Ptr{$T},), opt)

        get_step(opt::$T) = ccall(($("opk_get"*sfx*"_step"), libopk),
                                   Cdouble, (Ptr{$T},), opt)

        get_gnorm(opt::$T) = ccall(($("opk_get"*sfx*"_gnorm"), libopk),
                                   Cdouble, (Ptr{$T},), opt)

    end

    for field in ("name", "description")
        jfunc = Symbol("get_"*field)
        cfunc = "opk_get"*sfx*"_"*field
        @eval function $jfunc(opt::$T)
            nbytes = ccall(($cfunc, libopk), Csize_t,
                           (Ptr{UInt8}, Csize_t, Ptr{$T}), C_NULL, 0, opt)
            buf = Array{UInt8}(undef, nbytes)
            ccall(($cfunc, libopk), Csize_t,
                  (Ptr{UInt8}, Csize_t, Ptr{$T}), buf, nbytes, opt)
            return unsafe_string(pointer(buf))
        end
    end

end

"""

`task = start!(opt)` starts optimization with the nonlinear optimizer `opt` and
returns the next pending task.

""" start!

"""

`task = iterate!(opt, x, f, g)` performs one optimization step with the
nonlinear optimizer `opt` for variables `x`, function value `f` and gradient
`g`.  The method returns the next pending task.

""" iterate!

"""

`get_task(opt)` returns the current pending task for the nonlinear optimizer
`opt`.

""" get_task

"""

`get_status(opt)` returns the current status of the nonlinear optimizer `opt`.

""" get_status

"""

`evaluations(opt)` returns the number of function (and gradient) evaluations
requested by the nonlinear optimizer `opt`.

""" evaluations

"""

`iterations(opt)` returns the number of iterations performed by the nonlinear
optimizer `opt`.

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

"""

`get_name(opt)` yields the name of the algorithm used by the nonlinear
optimizer `opt`.

""" get_name

"""

`get_description(opt)` yields a brief description of the algorithm used by the
nonlinear optimizer `opt`.

""" get_description


#------------------------------------------------------------------------------
# DRIVERS FOR NON-LINEAR OPTIMIZATION

# Methods to yield the default line search algorithm.
default_nlcg_line_search() = MoreThuenteLineSearch(ftol=1E-4, gtol=0.1)
default_vmlmb_line_search() = MoreThuenteLineSearch(ftol=1E-4, gtol=0.9)


"""

Nonlinear Conjugate Gradient
============================

Minimizing the smooth mulivariate function `f(x)` by a nonlinear conjugate
gradient methods is done by:

```julia
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

where `fg!` implements the objective function (see below), `x0` gives the
initial value of the variables (as well as the data type and dimensions of the
solution) and optional argument `mem` is the number of previous steps to
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

# Extend `unsafe_convert` for abstract types derived from `Object` to work with
# `ccall`.  The `_handle` method may have to be extended if its default
# definition does not match object implementation.
for T in (Object, VariableSpace, Variable, Operator, ConvexSet, LineSearch,
          LimitedMemoryOptimizer, LimitedMemoryOptimizerDriver)
    @eval Base.unsafe_convert(::Type{Ptr{$T}}, obj::$T) =
        Ptr{$T}(_handle(obj))
end

# Extend `unsafe_convert` for concrete types.
for T in (DenseVariableSpace, DenseVariable, BoxedSet,
          ArmijoLineSearch, NonmonotoneLineSearch, MoreThuenteLineSearch,
          VMLMB, NLCG)
    @eval Base.unsafe_convert(::Type{Ptr{$T}}, obj::$T) = _handle(obj)
end

# General method to retrieve the pointer to and OptimPack object.
_handle(obj::Object) = Base.getfield(obj, :handle)
