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
# Copyright (C) 2014, Éric Thiébaut.
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

__error__(ptr::Ptr{Uint8}) = (ErrorException(bytestring(ptr)); nothing)

#function __error__(str::String) = ErrorException(str)

const __cerror__ = cfunction(__error__, Void, (Ptr{Uint8},))

function __init__()
    ccall((:opk_set_error_handler,opklib),Ptr{Void},(Ptr{Void},),__cerror__)
    nothing
end

__init__()

#------------------------------------------------------------------------------
# OBJECT MANAGEMENT
#
# All concrete types derived from the abstract Object type have a `handle`
# member which stores the address of the OptimPack object.  To avoid conflicts
# with Julia `Vector` type, an OptimPack vector corresponds to the type
# `Variable`.
abstract Object
abstract VariableSpace <: Object
abstract Variable      <: Object
abstract LineSearch    <: Object

function references(obj::Object)
    ccall((:opk_get_object_references, opklib), Cptrdiff_t, (Ptr{Void},), obj.handle)
end

# __hold_object__() set a reference on an OptimPack object while
# __drop_object__() discards a reference on an OptimPack object.  The argument
# is the address of the object and these functions are low-level private
# functions.

function __hold_object__(ptr::Ptr{Void})
    ccall((:opk_hold_object, opklib), Ptr{Void}, (Ptr{Void},), ptr)
end

function __drop_object__(ptr::Ptr{Void})
    ccall((:opk_drop_object, opklib), Void, (Ptr{Void},), ptr)
end

#------------------------------------------------------------------------------
# VECTOR SPACES
#

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

DenseVariableSpace(T::Union(Type{Cfloat},Type{Cdouble}), dims::Int...) = DenseVariableSpace(T, dims)

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

function DenseVariableSpace{N}(::Type{Cfloat}, dims::NTuple{N,Int})
    length::Int = checkdims(dims)
    ptr = ccall((:opk_new_simple_float_vector_space, opklib),
                Ptr{Void}, (Cptrdiff_t,), length)
    systemerror("failed to create vector space", ptr == C_NULL)
    obj = DenseVariableSpace{Cfloat,N}(ptr, Cfloat, dims, length)
    finalizer(obj, obj -> __drop_object__(obj.handle))
    return obj
end

function DenseVariableSpace{N}(::Type{Cdouble}, dims::NTuple{N,Int})
    length::Int = checkdims(dims)
    ptr = ccall((:opk_new_simple_double_vector_space, opklib),
                Ptr{Void}, (Cptrdiff_t,), length)
    systemerror("failed to create vector space", ptr == C_NULL)
    obj = DenseVariableSpace{Cdouble,N}(ptr, Cdouble, dims, length)
    finalizer(obj, obj -> __drop_object__(obj.handle))
    return obj
end

# Note: There are no needs to register a reference for the owner of a
# vector (it already owns one internally).
type DenseVector{T<:Union(Cfloat,Cdouble),N} <: Variable
    handle::Ptr{Void}
    owner::DenseVariableSpace{T,N}
    array::Union(Array,Nothing)
end

length(v::DenseVector) = length(v.owner)
eltype(v::DenseVector) = eltype(v.owner)
size(v::DenseVector) = size(v.owner)
size(v::DenseVector, n::Integer) = size(v.owner, n)
ndims(v::DenseVector) = ndims(v.owner)

# FIXME: add means to wrap a Julia array around this or (better?, simpler?)
#        just use allocate a Julia array and wrap a vector around it?
function create{T<:Union(Cfloat,Cdouble),N<:Integer}(vspace::DenseVariableSpace{T,N})
    ptr = ccall((:opk_vcreate, opklib), Ptr{Void}, (Ptr{Void},), vspace.handle)
    systemerror("failed to create vector", ptr == C_NULL)
    obj = DenseVector{T,N}(ptr, vspace, nothing)
    finalizer(obj, obj -> __drop_object__(obj.handle))
    return obj
end

function wrap{T<:Cfloat,N}(s::DenseVariableSpace{T,N}, a::DenseArray{T,N})
    assert(size(a) == size(s))
    ptr = ccall((:opk_wrap_simple_float_vector, opklib), Ptr{Void},
                (Ptr{Void}, Ptr{Cfloat}, Ptr{Void}, Ptr{Void}),
                s.handle, a, C_NULL, C_NULL)
    systemerror("failed to wrap vector", ptr == C_NULL)
    obj = DenseVector{T,N}(ptr, s, a)
    finalizer(obj, obj -> __drop_object__(obj.handle))
    return obj
end

function wrap!{T<:Cfloat,N}(v::DenseVector{T,N}, a::DenseArray{T,N})
    assert(size(a) == size(v))
    assert(v.array != nothing)
    status = ccall((:opk_rewrap_simple_float_vector, opklib), Cint,
                   (Ptr{Void}, Ptr{Cfloat}, Ptr{Void}, Ptr{Void}),
                   v.handle, a, C_NULL, C_NULL)
    systemerror("failed to re-wrap vector", status != SUCCESS)
    v.array = a
    return v
end

function wrap{T<:Cdouble,N}(s::DenseVariableSpace{T,N}, a::DenseArray{T,N})
    assert(size(a) == size(s))
    ptr = ccall((:opk_wrap_simple_double_vector, opklib), Ptr{Void},
                (Ptr{Void}, Ptr{Cdouble}, Ptr{Void}, Ptr{Void}),
                s.handle, a, C_NULL, C_NULL)
    systemerror("failed to wrap vector", ptr == C_NULL)
    obj = DenseVector{T,N}(ptr, s, a)
    finalizer(obj, obj -> __drop_object__(obj.handle))
    return obj
end

function wrap!{T<:Cdouble,N}(v::DenseVector{T,N}, a::DenseArray{T,N})
    assert(size(a) == size(v))
    assert(v.array != nothing)
    status = ccall((:opk_rewrap_simple_double_vector, opklib), Cint,
                   (Ptr{Void}, Ptr{Cdouble}, Ptr{Void}, Ptr{Void}),
                   v.handle, a, C_NULL, C_NULL)
    systemerror("failed to re-wrap vector", status != SUCCESS)
    v.array = a
    return v
end

#------------------------------------------------------------------------------
# OPERATIONS ON VECTORS

function norm1(vec::Variable)
    ccall((:opk_vnorm1,opklib), Cdouble, (Ptr{Void},), vec.handle)
end

function norm2(vec::Variable)
    ccall((:opk_vnorm2,opklib), Cdouble, (Ptr{Void},), vec.handle)
end

function norminf(vec::Variable)
    ccall((:opk_vnorminf,opklib), Cdouble, (Ptr{Void},), vec.handle)
end

function zero!(vec::Variable)
    ccall((:opk_vzero,opklib), Void, (Ptr{Void},), vec.handle)
end

function fill!(vec::Variable, val::Real)
    ccall((:opk_vfill, opklib), Void, (Ptr{Void},Cdouble), vec.handle, val)
end

function copy!(dst::Variable, src::Variable)
    ccall((:opk_vcopy,opklib), Void, (Ptr{Void},Ptr{Void}), dst.handle, src.handle)
end

function scale!(dst::Variable, alpha::Real, src::Variable)
    ccall((:opk_vscale,opklib), Void, (Ptr{Void},Cdouble,Ptr{Void}),
          dst.handle, alpha, src.handle)
end

function swap!(x::Variable, y::Variable)
    ccall((:opk_vswap,opklib), Void, (Ptr{Void},Ptr{Void}), x.handle, y.handle)
end

function dot(x::Variable, y::Variable)
    ccall((:opk_vdot,opklib), Cdouble, (Ptr{Void},Ptr{Void}), x.handle, y.handle)
end

function axpby!(dst::Variable,
                alpha::Real, x::Variable,
                beta::Real,  y::Variable)
    ccall((:opk_vaxpby,opklib), Void,
          (Ptr{Void},Cdouble,Ptr{Void},Cdouble,Ptr{Void}),
          dst.handle, alpha, x.handle, beta, y.handle)
end

function axpbypcz!(dst::Variable,
                   alpha::Real, x::Variable,
                   beta::Real,  y::Variable,
                  gamma::Real, z::Variable)
    ccall((:opk_vaxpbypcz,opklib), Void,
          (Ptr{Void},Cdouble,Ptr{Void},Cdouble,Ptr{Void},Cdouble,Ptr{Void}),
          dst.handle, alpha, x.handle, beta, y.handle, gamma, y.handle)
end

#------------------------------------------------------------------------------
# OPERATORS

if false
    function apply_direct(op::Operator,
                          dst::Variable,
                          src::Variable)
        status = ccall((:opk_apply_direct,opklib), Cint,
                       (Ptr{Void},Ptr{Void},Ptr{Void}),
                       op.handle, dst.handle, src.handle)
        if status != SUCCESS
            error("something wrong happens")
        end
        nothing
    end

    function apply_adoint(op::Operator,
                          dst::Variable,
                          src::Variable)
        status = ccall((:opk_apply_adjoint,opklib), Cint,
                       (Ptr{Void},Ptr{Void},Ptr{Void}),
                       op.handle, dst.handle, src.handle)
        if status != SUCCESS
            error("something wrong happens")
        end
        nothing
    end
    function apply_inverse(op::Operator,
                           dst::Variable,
                           src::Variable)
        status = ccall((:opk_apply_inverse,opklib), Cint,
                       (Ptr{Void},Ptr{Void},Ptr{Void}),
                       op.handle, dst.handle, src.handle)
        if status != SUCCESS
            error("something wrong happens")
        end
        nothing
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

#------------------------------------------------------------------------------
# NON LINEAR CONJUGATE GRADIENTS

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
#
# Default settings for non linear conjugate gradient (should correspond to
# the method which is, in general, the most successful). */
const NLCG_DEFAULT  = (NLCG_HAGER_ZHANG | NLCG_SHANNO_PHUA)

type NLCG <: Optimizer
    handle::Ptr{Void}
    vspace::VariableSpace
    method::Cuint
    lnsrch::LineSearch
    function NLCG(space::VariableSpace,
                  method::Integer,
                  lnsrch::LineSearch)
        ptr = ccall((:opk_new_nlcg_optimizer_with_line_search, opklib), Ptr{Void},
                (Ptr{Void}, Cuint, Ptr{Void}), space.handle, method, lnsrch.handle)
        systemerror("failed to create optimizer", ptr == C_NULL)
        obj = new(ptr, space, method, lnsrch)
        finalizer(obj, obj -> __drop_object__(obj.handle))
        return obj
    end
end

start!(opt::NLCG) = ccall((:opk_start_nlcg, opklib), Cint,
                                   (Ptr{Void},), opt.handle)

function iterate!(opt::NLCG, x::Variable, f::Real, g::Variable)
    ccall((:opk_iterate_nlcg, opklib), Cint,
          (Ptr{Void}, Ptr{Void}, Cdouble, Ptr{Void}),
          opt.handle, x.handle, f, g.handle)
end

get_task(opt::NLCG) = ccall((:opk_get_nlcg_task, opklib),
                                     Cint, (Ptr{Void},), opt.handle)
iterations(opt::NLCG) = ccall((:opk_get_nlcg_iterations, opklib),
                                       Cptrdiff_t, (Ptr{Void},), opt.handle)
evaluations(opt::NLCG) = ccall((:opk_get_nlcg_evaluations, opklib),
                                        Cptrdiff_t, (Ptr{Void},), opt.handle)
restarts(opt::NLCG) = ccall((:opk_get_nlcg_restarts, opklib),
                                     Cptrdiff_t, (Ptr{Void},), opt.handle)
get_gatol(opt::NLCG) = ccall((:opk_get_nlcg_gatol, opklib),
                                      Cdouble, (Ptr{Void},), opt.handle)
get_grtol(opt::NLCG) = ccall((:opk_get_nlcg_grtol, opklib),
                                      Cdouble, (Ptr{Void},), opt.handle)
get_stpmin(opt::NLCG) = ccall((:opk_get_nlcg_stpmin, opklib),
                                       Cdouble, (Ptr{Void},), opt.handle)
get_stpmax(opt::NLCG) = ccall((:opk_get_nlcg_stpmax, opklib),
                                       Cdouble, (Ptr{Void},), opt.handle)
function set_gatol!(opt::NLCG, gatol::Real)
    if ccall((:opk_set_nlcg_gatol, opklib),
             Cint, (Ptr{Void},Cdouble), opt.handle, gatol) != SUCCESS
        e = errno()
        if e == Base.EINVAL
            error("invalid value for parameter gatol")
        else
            error("unexpected error while setting parameter gatol")
        end
    end
end
function set_grtol!(opt::NLCG, grtol::Real)
    if ccall((:opk_set_nlcg_grtol, opklib),
             Cint, (Ptr{Void},Cdouble), opt.handle, grtol) != SUCCESS
        e = errno()
        if e == Base.EINVAL
            error("invalid value for parameter grtol")
        else
            error("unexpected error while setting parameter grtol")
        end
    end
end
function set_stpmin_and_stpmax!(opt::NLCG, stpmin::Real, stpmax::Real)
    if ccall((:opk_set_nlcg_stpmin_and_stpmax, opklib),
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

#------------------------------------------------------------------------------
# VARIABLE METRIC OPTIMIZATION METHOD

type VMLM <: Optimizer
    handle::Ptr{Void}
    vspace::VariableSpace
    m::Cptrdiff_t
    lnsrch::LineSearch
    function VMLM(space::VariableSpace,
                  m::Integer,
                  lnsrch::LineSearch)
        if m < 1
            error("illegal number of memorized steps")
        end
        if m > length(space)
            m = length(space)
        end
        ptr = ccall((:opk_new_vmlm_optimizer_with_line_search, opklib),
                    Ptr{Void}, (Ptr{Void}, Cptrdiff_t, Ptr{Void},
                                Cdouble, Cdouble, Cdouble),
                    space.handle, m, lnsrch.handle, 0.0, 0.0, 0.0)
        systemerror("failed to create optimizer", ptr == C_NULL)
        obj = new(ptr, space, m, lnsrch)
        finalizer(obj, obj -> __drop_object__(obj.handle))
        return obj
    end
end

start!(opt::VMLM) = ccall((:opk_start_vmlm, opklib), Cint,
                          (Ptr{Void},), opt.handle)

function iterate!(opt::VMLM, x::Variable, f::Real, g::Variable)
    ccall((:opk_iterate_vmlm, opklib), Cint,
          (Ptr{Void}, Ptr{Void}, Cdouble, Ptr{Void}),
          opt.handle, x.handle, f, g.handle)
end

get_task(opt::VMLM) = ccall((:opk_get_vmlm_task, opklib),
                            Cint, (Ptr{Void},), opt.handle)
iterations(opt::VMLM) = ccall((:opk_get_vmlm_iterations, opklib),
                              Cptrdiff_t, (Ptr{Void},), opt.handle)
evaluations(opt::VMLM) = ccall((:opk_get_vmlm_evaluations, opklib),
                               Cptrdiff_t, (Ptr{Void},), opt.handle)
restarts(opt::VMLM) = ccall((:opk_get_vmlm_restarts, opklib),
                            Cptrdiff_t, (Ptr{Void},), opt.handle)
get_gatol(opt::VMLM) = ccall((:opk_get_vmlm_gatol, opklib),
                             Cdouble, (Ptr{Void},), opt.handle)
get_grtol(opt::VMLM) = ccall((:opk_get_vmlm_grtol, opklib),
                             Cdouble, (Ptr{Void},), opt.handle)
get_stpmin(opt::VMLM) = ccall((:opk_get_vmlm_stpmin, opklib),
                              Cdouble, (Ptr{Void},), opt.handle)
get_stpmax(opt::VMLM) = ccall((:opk_get_vmlm_stpmax, opklib),
                              Cdouble, (Ptr{Void},), opt.handle)
function set_gatol!(opt::VMLM, gatol::Real)
    if ccall((:opk_set_vmlm_gatol, opklib),
             Cint, (Ptr{Void},Cdouble), opt.handle, gatol) != SUCCESS
        e = errno()
        if e == Base.EINVAL
            error("invalid value for parameter gatol")
        else
            error("unexpected error while setting parameter gatol")
        end
    end
end
function set_grtol!(opt::VMLM, grtol::Real)
    if ccall((:opk_set_vmlm_grtol, opklib),
             Cint, (Ptr{Void},Cdouble), opt.handle, grtol) != SUCCESS
        e = errno()
        if e == Base.EINVAL
            error("invalid value for parameter grtol")
        else
            error("unexpected error while setting parameter grtol")
        end
    end
end
function set_stpmin_and_stpmax!(opt::VMLM, stpmin::Real, stpmax::Real)
    if ccall((:opk_set_vmlm_stpmin_and_stpmax, opklib),
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

const SCALING_NONE             = cint(0)
const SCALING_OREN_SPEDICATO   = cint(1) # gamma = <s,y>/<y,y>
const SCALING_BARZILAI_BORWEIN = cint(2) # gamma = <s,s>/<s,y>

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

# x = minimize(fg!, x0)
#
#   This driver minimizes a smooth multi-variate function.  `fg!` is a function
#   which takes two arguments, `x` and `g` and which, for the given variables x,
#   stores the gradient of the function in `g` and returns the value of the
#   function:
#
#      f = fg!(x, g)
#
#   `x0` are the initial variables, they are left unchanged, they must be a
#   single or double precision floating point array.
#
function nlcg{T,N}(fg!::Function, x0::DenseArray{T,N},
                   method::Integer=NLCG_DEFAULT;
                   lnsrch::LineSearch=MoreThuenteLineSearch(ftol=1E-4, gtol=0.1),
                   gatol::Real=0.0, grtol::Real=1E-6,
                   stpmin::Real=1E-20, stpmax::Real=1E+20,
                   maxeval::Integer=-1, maxiter::Integer=-1,
                   verb::Bool=false, debug::Bool=false)
    #assert(T == Type{Cdouble} || T == Type{Cfloat})

    # Allocate workspaces
    dims = size(x0)
    space = DenseVariableSpace(T, dims)
    x = copy(x0)
    g = Array(T, dims)
    wx = wrap(space, x)
    wg = wrap(space, g)
    opt = NLCG(space, method, lnsrch)
    set_gatol!(opt, gatol)
    set_grtol!(opt, grtol)
    set_stpmin_and_stpmax!(opt, stpmin, stpmax)
    if debug
        @printf("gatol=%E; grtol=%E; stpmin=%E; stpmax=%E\n",
                get_gatol(opt), get_grtol(opt),
                get_stpmin(opt), get_stpmax(opt))
    end
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

function vmlm{T,N}(fg!::Function, x0::DenseArray{T,N}, m::Integer=3;
                   scaling::Integer=SCALING_OREN_SPEDICATO,
                   lnsrch::LineSearch=MoreThuenteLineSearch(ftol=1E-4, gtol=0.9),
                   gatol::Real=0.0, grtol::Real=1E-6,
                   stpmin::Real=1E-20, stpmax::Real=1E+20,
                   maxeval::Integer=-1, maxiter::Integer=-1,
                   verb::Bool=false, debug::Bool=false)
    #assert(T == Type{Cdouble} || T == Type{Cfloat})

    # Allocate workspaces
    dims = size(x0)
    space = DenseVariableSpace(T, dims)
    x = copy(x0)
    g = Array(T, dims)
    wx = wrap(space, x)
    wg = wrap(space, g)
    opt = VMLM(space, m, lnsrch)
    set_scaling!(opt, scaling)
    set_gatol!(opt, gatol)
    set_grtol!(opt, grtol)
    set_stpmin_and_stpmax!(opt, stpmin, stpmax)
    if debug
        @printf("gatol=%E; grtol=%E; stpmin=%E; stpmax=%E\n",
                get_gatol(opt), get_grtol(opt),
                get_stpmin(opt), get_stpmax(opt))
    end
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

include("Brent.jl")
include("Powell.jl")
include("spg2.jl")

end # module

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
