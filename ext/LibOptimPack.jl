module LibOptimPack

using CEnum: CEnum, @cenum

using OptimPack_jll: libbobyqa, libcobyla, libnewuoa

using ..Bobyqa, ..Cobyla, ..Newuoa

# typedef double cobyla_calcfc ( ptrdiff_t n , ptrdiff_t m , const double x [ ] , double con [ ] , void * data )
const cobyla_calcfc = Cvoid


function cobyla(n, m, fc, data, x, rhobeg, rhoend, iprint, maxfun, work, iact)
    @ccall libcobyla.cobyla(n::Cptrdiff_t, m::Cptrdiff_t, fc::Ptr{cobyla_calcfc}, data::Ptr{Cvoid}, x::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::Cptrdiff_t, maxfun::Cptrdiff_t, work::Ptr{Cdouble}, iact::Ptr{Cptrdiff_t})::Cobyla.Status
end

function cobyla_optimize(n, m, maximize, fc, data, x, scl, rhobeg, rhoend, iprint, maxfun, work, iact)
    @ccall libcobyla.cobyla_optimize(n::Cptrdiff_t, m::Cptrdiff_t, maximize::Bool, fc::Ptr{cobyla_calcfc}, data::Ptr{Cvoid}, x::Ptr{Cdouble}, scl::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::Cptrdiff_t, maxfun::Cptrdiff_t, work::Ptr{Cdouble}, iact::Ptr{Cptrdiff_t})::Cobyla.Status
end

mutable struct cobyla_context_ end

const cobyla_context = cobyla_context_

function cobyla_create(n, m, rhobeg, rhoend, iprint, maxfun)
    @ccall libcobyla.cobyla_create(n::Cptrdiff_t, m::Cptrdiff_t, rhobeg::Cdouble, rhoend::Cdouble, iprint::Cptrdiff_t, maxfun::Cptrdiff_t)::Ptr{cobyla_context}
end

function cobyla_delete(ctx)
    @ccall libcobyla.cobyla_delete(ctx::Ptr{cobyla_context})::Cvoid
end

function cobyla_iterate(ctx, f, x, c)
    @ccall libcobyla.cobyla_iterate(ctx::Ptr{cobyla_context}, f::Cdouble, x::Ptr{Cdouble}, c::Ptr{Cdouble})::Cobyla.Status
end

function cobyla_restart(ctx)
    @ccall libcobyla.cobyla_restart(ctx::Ptr{cobyla_context})::Cobyla.Status
end

function cobyla_get_status(ctx)
    @ccall libcobyla.cobyla_get_status(ctx::Ptr{cobyla_context})::Cobyla.Status
end

function cobyla_get_nevals(ctx)
    @ccall libcobyla.cobyla_get_nevals(ctx::Ptr{cobyla_context})::Cptrdiff_t
end

function cobyla_get_rho(ctx)
    @ccall libcobyla.cobyla_get_rho(ctx::Ptr{cobyla_context})::Cdouble
end

function cobyla_get_last_f(ctx)
    @ccall libcobyla.cobyla_get_last_f(ctx::Ptr{cobyla_context})::Cdouble
end

function cobyla_reason(status)
    @ccall libcobyla.cobyla_reason(status::Cobyla.Status)::Cstring
end

# typedef double bobyqa_objfun ( const ptrdiff_t n , const double * x , void * data )
const bobyqa_objfun = Cvoid

function bobyqa(n, npt, objfun, data, x, xl, xu, rhobeg, rhoend, iprint, maxfun, w)
    @ccall libbobyqa.bobyqa(n::Cptrdiff_t, npt::Cptrdiff_t, objfun::Ptr{bobyqa_objfun}, data::Ptr{Cvoid}, x::Ptr{Cdouble}, xl::Ptr{Cdouble}, xu::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::Cptrdiff_t, maxfun::Cptrdiff_t, w::Ptr{Cdouble})::Bobyqa.Status
end

function bobyqa_optimize(n, npt, maximize, objfun, data, x, xl, xu, scl, rhobeg, rhoend, iprint, maxfun, w)
    @ccall libbobyqa.bobyqa_optimize(n::Cptrdiff_t, npt::Cptrdiff_t, maximize::Bool, objfun::Ptr{bobyqa_objfun}, data::Ptr{Cvoid}, x::Ptr{Cdouble}, xl::Ptr{Cdouble}, xu::Ptr{Cdouble}, scl::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::Cptrdiff_t, maxfun::Cptrdiff_t, w::Ptr{Cdouble})::Bobyqa.Status
end

function bobyqa_reason(status)
    @ccall libbobyqa.bobyqa_reason(status::Bobyqa.Status)::Cstring
end

function bobyqa_test()
    @ccall libbobyqa.bobyqa_test()::Cvoid
end

# typedef double newuoa_objfun ( const ptrdiff_t n , const double * x , void * data )
const newuoa_objfun = Cvoid

function newuoa(n, npt, objfun, data, x, rhobeg, rhoend, iprint, maxfun, work)
    @ccall libnewuoa.newuoa(n::Cptrdiff_t, npt::Cptrdiff_t, objfun::Ptr{newuoa_objfun}, data::Ptr{Cvoid}, x::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::Cptrdiff_t, maxfun::Cptrdiff_t, work::Ptr{Cdouble})::Newuoa.Status
end

function newuoa_optimize(n, npt, maximize, objfun, data, x, scl, rhobeg, rhoend, iprint, maxfun, work)
    @ccall libnewuoa.newuoa_optimize(n::Cptrdiff_t, npt::Cptrdiff_t, maximize::Bool, objfun::Ptr{newuoa_objfun}, data::Ptr{Cvoid}, x::Ptr{Cdouble}, scl::Ptr{Cdouble}, rhobeg::Cdouble, rhoend::Cdouble, iprint::Cptrdiff_t, maxfun::Cptrdiff_t, work::Ptr{Cdouble})::Newuoa.Status
end

function newuoa_reason(status)
    @ccall libnewuoa.newuoa_reason(status::Newuoa.Status)::Cstring
end

mutable struct newuoa_context_ end

const newuoa_context = newuoa_context_

function newuoa_create(n, npt, rhobeg, rhoend, iprint, maxfun)
    @ccall libnewuoa.newuoa_create(n::Cptrdiff_t, npt::Cptrdiff_t, rhobeg::Cdouble, rhoend::Cdouble, iprint::Cptrdiff_t, maxfun::Cptrdiff_t)::Ptr{newuoa_context}
end

function newuoa_delete(ctx)
    @ccall libnewuoa.newuoa_delete(ctx::Ptr{newuoa_context})::Cvoid
end

function newuoa_iterate(ctx, f, x)
    @ccall libnewuoa.newuoa_iterate(ctx::Ptr{newuoa_context}, f::Cdouble, x::Ptr{Cdouble})::Newuoa.Status
end

function newuoa_restart(ctx)
    @ccall libnewuoa.newuoa_restart(ctx::Ptr{newuoa_context})::Newuoa.Status
end

function newuoa_get_status(ctx)
    @ccall libnewuoa.newuoa_get_status(ctx::Ptr{newuoa_context})::Newuoa.Status
end

function newuoa_get_nevals(ctx)
    @ccall libnewuoa.newuoa_get_nevals(ctx::Ptr{newuoa_context})::Cptrdiff_t
end

function newuoa_get_rho(ctx)
    @ccall libnewuoa.newuoa_get_rho(ctx::Ptr{newuoa_context})::Cdouble
end

function newuoa_test()
    @ccall libnewuoa.newuoa_test()::Cvoid
end

# exports
const PREFIXES = ["BOBYQA_", "bobyqa_", "COBYLA_", "cobyla_", "NEWUOA_", "newuoa_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
