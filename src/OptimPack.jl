#
# OptimPack.jl --
#
# Optimization for Julia.
#
# ----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 2014-2016, Éric Thiébaut.
#
# ----------------------------------------------------------------------------

module OptimPack

export nlcg, vmlm, spg2

export fzero, fmin, fmin_global

# Functions must be imported to be extended with new methods.
import Base.size
import Base.length
import Base.eltype
import Base.ndims
import Base.copy
import Base.dot

if isfile(joinpath(dirname(@__FILE__),"..","deps","deps.jl"))
    include("../deps/deps.jl")
else
    error("OptimPack not properly installed. Please run Pkg.build(\"OptimPack\")")
end

"""
`Float` is any floating point type supported by the library.
"""
typealias Float Union{Cfloat,Cdouble}

cint(i::Integer) = convert(Cint, i)
cuint(i::Integer) = convert(Cuint, i)

const SUCCESS = cint( 0)
const FAILURE = cint(-1)

#------------------------------------------------------------------------------
# ERROR MANAGEMENT
#
# We must provide an appropriate error handler to OptimPack in order to throw
# an error exception and avoid aborting the program in case of misuse of the
# library.

__error__(ptr::Ptr{UInt8}) = (ErrorException(bytestring(ptr)); nothing)

const __cerror__ = cfunction(__error__, Void, (Ptr{UInt8},))

function __init__()
    ccall((:opk_set_error_handler, opklib), Ptr{Void}, (Ptr{Void},),
          __cerror__)
    nothing
end

__init__()

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
abstract Object

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
    ccall((:opk_get_object_references, opklib), Cptrdiff_t, (Ptr{Void},),
          obj.handle)
end

function __hold_object__(ptr::Ptr{Void})
    ccall((:opk_hold_object, opklib), Ptr{Void}, (Ptr{Void},), ptr)
end

function __drop_object__(ptr::Ptr{Void})
    ccall((:opk_drop_object, opklib), Void, (Ptr{Void},), ptr)
end

@doc (@doc references) __hold_object__
@doc (@doc references) __drop_object__

#------------------------------------------------------------------------------
# VARIABLE SPACE

abstract VariableSpace <: Object
"""
Variable Space
==============
Abstract type `VariableSpace` corresponds to a *vector space* (type
`opk_vspace_t`) in OptimPack.
"""

type DenseVariableSpace{T,N} <: VariableSpace
    handle::Ptr{Void}
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

function checkdims{N}(dims::NTuple{N,Int})
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
        function DenseVariableSpace{N}(::Type{$T}, dims::NTuple{N,Int})
            length::Int = checkdims(dims)
            ptr = ccall(($(string(f)), opklib), Ptr{Void}, (Cptrdiff_t,), length)
            systemerror("failed to create vector space", ptr == C_NULL)
            obj = DenseVariableSpace{$T,N}(ptr, $T, dims, length)
            finalizer(obj, obj -> __drop_object__(obj.handle))
            return obj
        end
    end
end


#------------------------------------------------------------------------------
# VARIABLES

abstract Variable <: Object
"""
Variables
=========
Abstract type `Variable` correspond to *vectors* (type `opk_vector_t`) in OptimPack.
"""

# Note: There are no needs to register a reference for the owner of a
# vector (it already owns one internally).
type DenseVariable{T<:Float,N} <: Variable
    handle::Ptr{Void}
    owner::DenseVariableSpace{T,N}
    array::Union{Array,Void}
end

length(v::DenseVariable) = length(v.owner)
eltype(v::DenseVariable) = eltype(v.owner)
size(v::DenseVariable) = size(v.owner)
size(v::DenseVariable, n::Integer) = size(v.owner, n)
ndims(v::DenseVariable) = ndims(v.owner)

# FIXME: add means to wrap a Julia array around this or (better?, simpler?)
#        just use allocate a Julia array and wrap a vector around it?
"""
`v = create(s)` creates a new variable of the variable space `s`.
"""
function create{T<:Float,N<:Integer}(space::DenseVariableSpace{T,N})
    ptr = ccall((:opk_vcreate, opklib), Ptr{Void}, (Ptr{Void},), space.handle)
    systemerror("failed to create vector", ptr == C_NULL)
    obj = DenseVariable{T,N}(ptr, space, nothing)
    finalizer(obj, obj -> __drop_object__(obj.handle))
    return obj
end

for (T, wrap, rewrap) in ((Cfloat, :opk_wrap_simple_float_vector,
                           :opk_rewrap_simple_float_vector),
                          (Cdouble, :opk_wrap_simple_double_vector,
                           :opk_rewrap_simple_double_vector))
    @eval begin
        function wrap{N}(s::DenseVariableSpace{$T,N}, a::DenseArray{$T,N})
            assert(size(a) == size(s))
            ptr = ccall(($(string(wrap)), opklib), Ptr{Void},
                        (Ptr{Void}, Ptr{$T}, Ptr{Void}, Ptr{Void}),
                        s.handle, a, C_NULL, C_NULL)
            systemerror("failed to wrap vector", ptr == C_NULL)
            obj = DenseVariable{$T,N}(ptr, s, a)
            finalizer(obj, obj -> __drop_object__(obj.handle))
            return obj
        end

        function wrap!{N}(v::DenseVariable{$T,N}, a::DenseArray{$T,N})
            assert(size(a) == size(v))
            assert(v.array != nothing)
            status = ccall(($(string(rewrap)), opklib), Cint,
                           (Ptr{Void}, Ptr{$T}, Ptr{Void}, Ptr{Void}),
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
""" wrap

#------------------------------------------------------------------------------
# OPERATIONS ON VARIABLES

"""
`norm1(v)` returns the L1 norm (sum of absolute values) ov *variables* `v`.
"""
function norm1(v::Variable)
    ccall((:opk_vnorm1, opklib), Cdouble, (Ptr{Void},), v.handle)
end

"""
`norm2(v)` returns the Euclidean (L2) norm (square root of the sum of squared
values) of *variables* `v`.
"""
function norm2(v::Variable)
    ccall((:opk_vnorm2, opklib), Cdouble, (Ptr{Void},), v.handle)
end

"""
`norminf(v)` returns the infinite norm (maximum absolute value) of *variables*
`v`.
"""
function norminf(v::Variable)
    ccall((:opk_vnorminf, opklib), Cdouble, (Ptr{Void},), v.handle)
end

"""
`zero!(v)` fills *variables* `v` with zeros.
"""
function zero!(v::Variable)
    ccall((:opk_vzero, opklib), Void, (Ptr{Void},), v.handle)
end

"""
`fill!(v, alpha)` fills *variables* `v` with value `alpha`.
"""
function fill!(v::Variable, alpha::Real)
    ccall((:opk_vfill, opklib), Void, (Ptr{Void},Cdouble),
          v.handle, alpha)
end

"""
`copy!(dst, src)` copies source *variables* `src` into the destination
*variables* `dst`.
"""
function copy!(dst::Variable, src::Variable)
    ccall((:opk_vcopy, opklib), Void, (Ptr{Void},Ptr{Void}),
          dst.handle, src.handle)
end

"""
`scale!(dst, alpha, src)` stores `alpha` times the source *variables* `src`
into the destination *variables* `dst`.
"""
function scale!(dst::Variable, alpha::Real, src::Variable)
    ccall((:opk_vscale, opklib), Void, (Ptr{Void},Cdouble,Ptr{Void}),
          dst.handle, alpha, src.handle)
end

"""
`swap!(x, y)` exchanges the contents of *variables* `x` and `y`.
"""
function swap!(x::Variable, y::Variable)
    ccall((:opk_vswap, opklib), Void, (Ptr{Void},Ptr{Void}),
          x.handle, y.handle)
end

"""
`dot(x, y)` returns the inner product of *variables* `x` and `y`.
"""
function dot(x::Variable, y::Variable)
    ccall((:opk_vdot, opklib), Cdouble, (Ptr{Void},Ptr{Void}),
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
    ccall((:opk_vaxpby, opklib), Void,
          (Ptr{Void},Cdouble,Ptr{Void},Cdouble,Ptr{Void}),
          dst.handle, alpha, x.handle, beta, y.handle)
end

function combine!(dst::Variable,
                  alpha::Real, x::Variable,
                  beta::Real,  y::Variable,
                  gamma::Real, z::Variable)
    ccall((:opk_vaxpbypcz, opklib), Void,
          (Ptr{Void},Cdouble,Ptr{Void},Cdouble,Ptr{Void},Cdouble,Ptr{Void}),
          dst.handle, alpha, x.handle, beta, y.handle, gamma, y.handle)
end

#------------------------------------------------------------------------------
# OPERATORS

abstract Operator <: Object

for s in (:apply_direct, :apply_adoint, :apply_inverse)
    @eval begin
        function $(Symbol(string(s)*"!"))(op::Operator,
                                          dst::Variable,
                                          src::Variable)
            status = ccall(($("opk_"*string(s)), opklib), Cint,
                           (Ptr{Void},Ptr{Void},Ptr{Void}),
                           op.handle, dst.handle, src.handle)
            if status != SUCCESS
                error("something wrong happens")
            end
            nothing
        end
    end
end

#------------------------------------------------------------------------------
# LINE SEARCH METHODS

abstract LineSearch <: Object

type ArmijoLineSearch <: LineSearch
    handle::Ptr{Void}
    ftol::Cdouble
    function ArmijoLineSearch(;ftol::Real=1e-4)
        assert(0.0 <= ftol < 1.0)
        ptr = ccall((:opk_lnsrch_new_backtrack, opklib), Ptr{Void},
                    (Cdouble,), ftol)
        systemerror("failed to create linesearch", ptr == C_NULL)
        obj = new(ptr, ftol)
        finalizer(obj, obj -> __drop_object__(obj.handle))
        return obj
    end
end

type MoreThuenteLineSearch <: LineSearch
    handle::Ptr{Void}
    ftol::Cdouble
    gtol::Cdouble
    xtol::Cdouble
    function MoreThuenteLineSearch(;ftol::Real=1e-4, gtol::Real=0.9,
                                   xtol::Real=eps(Cdouble))
        assert(0.0 <= ftol < gtol < 1.0)
        assert(0.0 <= xtol < 1.0)
        ptr = ccall((:opk_lnsrch_new_csrch, opklib), Ptr{Void},
                (Cdouble, Cdouble, Cdouble), ftol, gtol, xtol)
        systemerror("failed to create linesearch", ptr == C_NULL)
        obj = new(ptr, ftol, gtol, xtol)
        finalizer(obj, obj -> __drop_object__(obj.handle))
        return obj
    end
end

type NonmonotoneLineSearch <: LineSearch
    handle::Ptr{Void}
    mem::Int
    ftol::Cdouble
    amin::Cdouble
    amax::Cdouble
    function NonmonotoneLineSearch(;mem::Integer=10, ftol::Real=1e-4,
                                   amin::Real=0.1, amax::Real=0.9)
        assert(mem >= 1)
        assert(0.0 <= ftol < 1.0)
        assert(0.0 < amin < amax < 1.0)
        ptr = ccall((:opk_lnsrch_new_nonmonotone, opklib), Ptr{Void},
                (Cptrdiff_t, Cdouble, Cdouble, Cdouble), mem, ftol, amin, amax)
        systemerror("failed to create nonmonotone linesearch", ptr == C_NULL)
        obj = new(ptr, mem, ftol, amin, amax)
        finalizer(obj, obj -> __drop_object__(obj.handle))
        return obj
    end
end

const LNSRCH_ERROR_ILLEGAL_ADDRESS                    = cint(-12)
const LNSRCH_ERROR_CORRUPTED_WORKSPACE                = cint(-11)
const LNSRCH_ERROR_BAD_WORKSPACE                      = cint(-10)
const LNSRCH_ERROR_STP_CHANGED                        = cint( -9)
const LNSRCH_ERROR_STP_OUTSIDE_BRACKET                = cint( -8)
const LNSRCH_ERROR_NOT_A_DESCENT                      = cint( -7)
const LNSRCH_ERROR_STPMIN_GT_STPMAX                   = cint( -6)
const LNSRCH_ERROR_STPMIN_LT_ZERO                     = cint( -5)
const LNSRCH_ERROR_STP_LT_STPMIN                      = cint( -4)
const LNSRCH_ERROR_STP_GT_STPMAX                      = cint( -3)
const LNSRCH_ERROR_INITIAL_DERIVATIVE_GE_ZERO         = cint( -2)
const LNSRCH_ERROR_NOT_STARTED                        = cint( -1)
const LNSRCH_SEARCH                                   = cint(  0)
const LNSRCH_CONVERGENCE                              = cint(  1)
const LNSRCH_WARNING_ROUNDING_ERRORS_PREVENT_PROGRESS = cint(  2)
const LNSRCH_WARNING_XTOL_TEST_SATISFIED              = cint(  3)
const LNSRCH_WARNING_STP_EQ_STPMAX                    = cint(  4)
const LNSRCH_WARNING_STP_EQ_STPMIN                    = cint(  5)

function start!(ls::LineSearch, f0::Real, df0::Real,
                stp1::Real, stpmin::Real, stpmax::Real)
    ccall((:opk_lnsrch_start, opklib), Cint,
          (Ptr{Void}, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble),
          ls, f0, df0, stp1, stpmin, stpmax)
end

function iterate!(ls::LineSearch, stp::Real, f::Real, df::Real)
    _stp = Cdouble[stp]
    task = ccall((:opk_lnsrch_iterate, opklib), Cint,
                 (Ptr{Void}, Ptr{Cdouble}, Cdouble, Cdouble),
                 ls, _stp, f, df)
    return (task, _stp[1])
end

get_step(ls::LineSearch) = ccall((:opk_lnsrch_get_step, opklib),
                                 Cdouble, (Ptr{Void}, ), ls)
get_status(ls::LineSearch) = ccall((:opk_lnsrch_get_status, opklib),
                                   Cint, (Ptr{Void}, ), ls)
has_errors(ls::LineSearch) = (ccall((:opk_lnsrch_has_errors, opklib),
                                    Cint, (Ptr{Void}, ), ls) != 0)
has_warnings(ls::LineSearch) = (ccall((:opk_lnsrch_has_warnings, opklib),
                                      Cint, (Ptr{Void}, ), ls) != 0)
converged(ls::LineSearch) = (ccall((:opk_lnsrch_converged, opklib),
                                   Cint, (Ptr{Void}, ), ls) != 0)
finished(ls::LineSearch) = (ccall((:opk_lnsrch_finished, opklib),
                                  Cint, (Ptr{Void}, ), ls) != 0)
get_ftol(ls::LineSearch) = ls.ftol
get_gtol(ls::MoreThuenteLineSearch) = ls.gtol
get_xtol(ls::MoreThuenteLineSearch) = ls.xtol


#------------------------------------------------------------------------------
# NON LINEAR OPTIMIZERS

# Codes returned by the reverse communication version of optimzation
# algorithms.

const TASK_ERROR       = cint(-1) # An error has ocurred.
const TASK_PROJECT_X   = cint( 0) # Caller must project variables x.
const TASK_COMPUTE_FG  = cint( 1) # Caller must compute f(x) and g(x).
const TASK_PROJECT_D   = cint( 2) # Caller must project the direction d.
const TASK_FREE_VARS   = cint( 3) # Caller must update the subspace of free variables.
const TASK_NEW_X       = cint( 4) # A new iterate is available.
const TASK_FINAL_X     = cint( 5) # Algorithm has converged, solution is available.
const TASK_WARNING     = cint( 6) # Algorithm terminated with a warning.

abstract Optimizer <: Object
abstract Options


const NLCG_FLETCHER_REEVES        = cuint(1)
const NLCG_HESTENES_STIEFEL       = cuint(2)
const NLCG_POLAK_RIBIERE_POLYAK   = cuint(3)
const NLCG_FLETCHER               = cuint(4)
const NLCG_LIU_STOREY             = cuint(5)
const NLCG_DAI_YUAN               = cuint(6)
const NLCG_PERRY_SHANNO           = cuint(7)
const NLCG_HAGER_ZHANG            = cuint(8)
const NLCG_POWELL                 = cuint(1<<8) # force beta >= 0
const NLCG_SHANNO_PHUA            = cuint(1<<9) # compute scale from previous iterations

# For instance: (NLCG_POLAK_RIBIERE_POLYAK | NLCG_POWELL) merely
# corresponds to PRP+ (Polak, Ribiere, Polyak) while (NLCG_PERRY_SHANNO |
# NLCG_SHANNO_PHUA) merely corresponds to the conjugate gradient method
# implemented in CONMIN.

type NLCGoptions <: Options
    gatol::Cdouble
    grtol::Cdouble
    stpmin::Cdouble
    stpmax::Cdouble
    method::UInt
    function NLCGoptions(;gatol::Real=0.0, grtol::Real=1E-6,
                         stpmin::Real=1E-20, stpmax::Real=1E+20,
                         method::Integer=(NLCG_HAGER_ZHANG|NLCG_SHANNO_PHUA))
        new(convert(Cdouble, gatol),
            convert(Cdouble, grtol),
            convert(Cdouble, stpmin),
            convert(Cdouble, stpmax),
            convert(Cptrdiff_t, method))
    end
    NLCGoptions(method::Integer) = new(convert(UInt, method))
end

"""
Default settings for non linear conjugate gradient (should correspond to the
method which is, in general, the most successful).
"""
const NLCG_DEFAULT = NLCGoptions()

type NLCG <: Optimizer
    handle::Ptr{Void}
    space::VariableSpace
    method::Cuint
    lnsrch::LineSearch
    function NLCG(options::NLCGoptions,
                  space::VariableSpace,
                  lnsrch::LineSearch)
        ptr = ccall((:opk_new_nlcg_optimizer_with_line_search, opklib),
                    Ptr{Void}, (Ptr{Void}, Cuint, Ptr{Void}),
                    space.handle, options.method, lnsrch.handle)
        systemerror("failed to create optimizer", ptr == C_NULL)
        obj = new(ptr, space, options.method, lnsrch)
        finalizer(obj, obj -> __drop_object__(obj.handle))
        set_gatol!(obj, options.gatol)
        set_grtol!(obj, options.grtol)
        set_stpmin_and_stpmax!(obj, options.stpmin, options.stpmax)
        return obj
    end
end

const SCALING_NONE             = cint(0)
const SCALING_OREN_SPEDICATO   = cint(1) # gamma = <s,y>/<y,y>
const SCALING_BARZILAI_BORWEIN = cint(2) # gamma = <s,s>/<s,y>

type VMLMoptions <: Options
    gatol::Cdouble
    grtol::Cdouble
    stpmin::Cdouble
    stpmax::Cdouble
    mem::Cptrdiff_t
    scaling::Cint
    function VMLMoptions(;gatol::Real=0.0, grtol::Real=1E-6,
                         stpmin::Real=1E-20, stpmax::Real=1E+20,
                         mem::Integer=3,
                         scaling::Integer=SCALING_OREN_SPEDICATO)
        new(convert(Cdouble, gatol),
            convert(Cdouble, grtol),
            convert(Cdouble, stpmin),
            convert(Cdouble, stpmax),
            convert(Cptrdiff_t, mem),
            convert(Cint, scaling))
    end
end

"""
Default settings for non linear conjugate gradient (should correspond to the
method which is, in general, the most successful).
"""
const VMLM_DEFAULT = VMLMoptions()

type VMLM <: Optimizer
    handle::Ptr{Void}
    space::VariableSpace
    mem::Cptrdiff_t
    lnsrch::LineSearch
    function VMLM(options::VMLMoptions,
                  space::VariableSpace,
                  lnsrch::LineSearch)
        mem = options.mem
        mem ≥ 1 || error("illegal number of memorized steps")
        mem = min(mem, length(space))
        ptr = ccall((:opk_new_vmlm_optimizer_with_line_search, opklib),
                    Ptr{Void}, (Ptr{Void}, Cptrdiff_t, Ptr{Void},
                                Cdouble, Cdouble, Cdouble),
                    space.handle, mem, lnsrch.handle, 0.0, 0.0, 0.0)
        systemerror("failed to create optimizer", ptr == C_NULL)
        obj = new(ptr, space, mem, lnsrch)
        finalizer(obj, obj -> __drop_object__(obj.handle))
        set_gatol!(obj, options.gatol)
        set_grtol!(obj, options.grtol)
        set_stpmin_and_stpmax!(obj, options.stpmin, options.stpmax)
        set_scaling!(obj, options.scaling)
        return obj
    end
end

for (T, start, iterate,
     get_task,
     get_iterations,
     get_evaluations,
     get_restarts,
     get_gatol, get_grtol,
     set_gatol, set_grtol,
     get_stpmin, get_stpmax,
     set_stpmin_and_stpmax) in ((NLCG, :opk_start_nlcg, :opk_iterate_nlcg,
                                 :opk_get_nlcg_task, :opk_get_nlcg_iterations,
                                 :opk_get_nlcg_evaluations,
                                 :opk_get_nlcg_restarts,
                                 :opk_get_nlcg_gatol, :opk_get_nlcg_grtol,
                                 :opk_set_nlcg_gatol, :opk_set_nlcg_grtol,
                                 :opk_get_nlcg_stpmin, :opk_get_nlcg_stpmax,
                                 :opk_set_nlcg_stpmin_and_stpmax),
                                (VMLM, :opk_start_vmlm, :opk_iterate_vmlm,
                                 :opk_get_vmlm_task, :opk_get_vmlm_iterations,
                                 :opk_get_vmlm_evaluations,
                                 :opk_get_vmlm_restarts,
                                 :opk_get_vmlm_gatol, :opk_get_vmlm_grtol,
                                 :opk_set_vmlm_gatol, :opk_set_vmlm_grtol,
                                 :opk_get_vmlm_stpmin, :opk_get_vmlm_stpmax,
                                 :opk_set_vmlm_stpmin_and_stpmax))
    @eval begin

        start!(opt::$T) = ccall(($(string(start)), opklib), Cint,
                                  (Ptr{Void},), opt.handle)

        function iterate!(opt::$T, x::Variable, f::Real, g::Variable)
            ccall(($(string(iterate)), opklib), Cint,
                  (Ptr{Void}, Ptr{Void}, Cdouble, Ptr{Void}),
                  opt.handle, x.handle, f, g.handle)
        end

        get_task(opt::$T) = ccall(($(string(get_task)), opklib),
                                    Cint, (Ptr{Void},), opt.handle)

        iterations(opt::$T) = ccall(($(string(get_iterations)), opklib),
                                    Cptrdiff_t, (Ptr{Void},), opt.handle)

        evaluations(opt::$T) = ccall(($(string(get_evaluations)), opklib),
                                     Cptrdiff_t, (Ptr{Void},), opt.handle)

        restarts(opt::$T) = ccall(($(string(get_restarts)), opklib),
                                  Cptrdiff_t, (Ptr{Void},), opt.handle)

        get_gatol(opt::$T) = ccall(($(string(get_gatol)), opklib),
                                   Cdouble, (Ptr{Void},), opt.handle)

        get_grtol(opt::$T) = ccall(($(string(get_grtol)), opklib),
                                   Cdouble, (Ptr{Void},), opt.handle)

        function set_gatol!(opt::$T, gatol::Real)
            if ccall(($(string(set_gatol)), opklib),
                     Cint, (Ptr{Void},Cdouble), opt.handle, gatol) != SUCCESS
                e = errno()
                if e == Base.EINVAL
                    error("invalid value for parameter gatol")
                else
                    error("unexpected error while setting parameter gatol")
                end
            end
        end

        function set_grtol!(opt::$T, grtol::Real)
            if ccall(($(string(set_grtol)), opklib),
                     Cint, (Ptr{Void},Cdouble), opt.handle, grtol) != SUCCESS
                e = errno()
                if e == Base.EINVAL
                    error("invalid value for parameter grtol")
                else
                    error("unexpected error while setting parameter grtol")
                end
            end
        end

        get_stpmin(opt::$T) = ccall(($(string(get_stpmin)), opklib),
                                    Cdouble, (Ptr{Void},), opt.handle)

        get_stpmax(opt::$T) = ccall(($(string(get_stpmax)), opklib),
                                    Cdouble, (Ptr{Void},), opt.handle)

        function set_stpmin_and_stpmax!(opt::$T, stpmin::Real, stpmax::Real)
            if ccall(($(string(set_stpmin_and_stpmax)), opklib),
                     Cint, (Ptr{Void},Cdouble,Cdouble),
                     opt.handle, stpmin, stpmax) != SUCCESS
                e = errno()
                if e == Base.EINVAL
                    error("invalid values for parameters stpmin and stpmax")
                else
                    error("unexpected error while setting parameters stpmin and stpmax")
                end
            end
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
`iterations(opt)` returns the number of iterations performed by the
nonlinear optimizer `opt`.
""" iterations

"""
`evaluations(opt)` returns the number of function (and gradient)
evaluations requested by the nonlinear optimizer `opt`.
""" evaluations

"""
`restarts(opt)` returns the number of restarts performed by the
nonlinear optimizer `opt`.
""" restarts

"""
`get_gatol(opt)` returns the absolute gradient threshold used by the
nonlinear optimizer `opt` to check for the convergence.
""" get_gatol

"""
`get_grtol(opt)` returns the relative gradient threshold used by the
nonlinear optimizer `opt` to check for the convergence.
""" get_grtol

"""
`set_gatol!(opt, gatol)` set the absolute gradient threshold used by the
nonlinear optimizer `opt` to check for the convergence.
""" set_gatol!

"""
`set_grtol!(opt, grtol)` set the relative gradient threshold used by the
nonlinear optimizer `opt` to check for the convergence.
""" set_grtol!

"""
`get_stpmin(opt)` returns the minimum relative step length used by the
nonlinear optimizer `opt` during line searches.
""" get_stpmin

"""
`get_stpmax(opt)` returns the maximum relative step length used by the
nonlinear optimizer `opt` during line searches.
""" get_stpmax

"""
`set_stpmin_and_stpmax!(opt, stpmin, stpmax)` set the minimum and the
maximum relative step length used by the nonlinear optimizer `opt` during
line searches.
""" set_stpmin_and_stpmax!

get_scaling(opt::VMLM) = ccall((:opk_get_vmlm_scaling, opklib),
                                      Cint, (Ptr{Void},), opt.handle)

function set_scaling!(opt::VMLM, scaling::Integer)
    if ccall((:opk_set_vmlm_scaling, opklib),
             Cint, (Ptr{Void},Cint), opt.handle, scaling) != SUCCESS
        error("unexpected error while setting scaling scaling")
    end
end

#------------------------------------------------------------------------------
# DRIVERS FOR NON-LINEAR OPTIMIZATION

default_nlcg_line_search() = MoreThuenteLineSearch(ftol=1E-4, gtol=0.1)
default_vmlm_line_search() = MoreThuenteLineSearch(ftol=1E-4, gtol=0.9)


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

See `vmlm` for more details.
"""
function nlcg{T<:Float,N}(fg!::Function, x0::DenseArray{T,N},
                          method::Integer=NLCG_DEFAULT.method;
                          lnsrch::LineSearch=default_nlcg_line_search(),
                          gatol::Real=NLCG_DEFAULT.gatol,
                          grtol::Real=NLCG_DEFAULT.grtol,
                          stpmin::Real=NLCG_DEFAULT.stpmin,
                          stpmax::Real=NLCG_DEFAULT.stpmax,
                          maxeval::Integer=-1, maxiter::Integer=-1,
                          verb::Bool=false, debug::Bool=false)
    # Create an optimizer and solve the problem.
    space = DenseVariableSpace(T, size(x0))
    options = NLCGoptions(gatol=gatol, grtol=grtol, stpmin=stpmin,
                          stpmax=stpmax, method=method)
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
x = vmlm(fg!, x0, mem)
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
function vmlm{T<:Float,N}(fg!::Function, x0::DenseArray{T,N},
                          mem::Integer=VMLM_DEFAULT.mem;
                          lnsrch::LineSearch=default_vmlm_line_search(),
                          scaling::Integer=VMLM_DEFAULT.scaling,
                          gatol::Real=VMLM_DEFAULT.gatol,
                          grtol::Real=VMLM_DEFAULT.grtol,
                          stpmin::Real=VMLM_DEFAULT.stpmin,
                          stpmax::Real=VMLM_DEFAULT.stpmax,
                          maxeval::Integer=-1, maxiter::Integer=-1,
                          verb::Bool=false, debug::Bool=false)
    # Create an optimizer and solve the problem.
    options = VMLMoptions(gatol=gatol, grtol=grtol, stpmin=stpmin,
                          stpmax=stpmax, mem=mem, scaling=scaling)
    space = DenseVariableSpace(T, size(x0))
    opt = VMLM(options, space, lnsrch)
    solve(opt, fg!, x0, maxeval=maxeval, maxiter=maxiter,
          verb=verb, debug=debug)
end

function solve(opt::Optimizer, fg!::Function, x0::DenseArray;
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
    task = start!(opt)
    while true
        if task == TASK_COMPUTE_FG
            f = fg!(x, g)
        elseif task == TASK_NEW_X || task == TASK_FINAL_X
            iter = iterations(opt)
            eval = evaluations(opt)
            if verb
                if iter == 0
                    @printf("%s\n%s\n",
                            " ITER   EVAL  RESTARTS         F(X)             ||G(X)||",
                            "--------------------------------------------------------")
                end
                @printf("%5d  %5d  %5d  %24.16E %10.3E\n",
                        iter, eval, restarts(opt), f, norm2(wg))
            end
            if task == TASK_FINAL_X
                return x
            end
            if maxiter >= 0 && iter >= maxiter
                warn("exceeding maximum number of iterations ($maxiter)")
                return x
            end
            if maxeval >= 0 && eval >= maxeval
                warn("exceeding maximum number of evaluations ($eval >= $maxeval)")
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
include("Powell.jl")
include("spg2.jl")

end # module
