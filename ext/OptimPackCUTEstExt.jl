module OptimPackCUTEstExt

if isdefined(Base, :get_extension)
    using OptimPack, CUTEst
else
    using ..OptimPack, ..CUTEst
end

using NLPModels

function OptimPack.simplex(nlp::CUTEstModel, args...; kwds...)
    return simplex(nlp, nlp.meta.x0, args...; kwds...)
end

function OptimPack.simplex(nlp::CUTEstModel, x0::AbstractArray, args...; kwds...)
    return simplex(x -> obj(nlp, x), x0, args...; kwds...)
end

function OptimPack.newuoa(nlp::CUTEstModel, args...; kwds...)
    return newuoa(nlp, nlp.meta.x0, args...; kwds...)
end

function OptimPack.newuoa(nlp::CUTEstModel, x0::AbstractArray, args...; kwds...)
    return newuoa(x -> obj(nlp, x), x0, args...; kwds...)
end

function OptimPack.bobyqa(nlp::CUTEstModel, args...; kwds...)
    return bobyqa(nlp, nlp.meta.x0, args...; kwds...)
end

function OptimPack.bobyqa(nlp::CUTEstModel, x0::AbstractArray, args...; kwds...)
    return bobyqa(x -> obj(nlp, x), x0, args...;
                  lower = nlp.meta.lvar, upper = nlp.meta.uvar, kwds...)
end

function OptimPack.spg(nlp::CUTEstModel; kwds...)
    return spg(x -> obj(nlp, x), (g, x) -> copy!(g, grad(nlp, x)),
               OptimPack.Bounds(nlp.meta.lvar, nlp.meta.uvar), nlp.meta.x0; kwds...)
end

for alg in (:bobyqa, :cobyla, :newuoa, :simplex, :spg)
    # Strings after the name of the problem are considered as parameters of the problem.
    @eval function OptimPack.$alg(::Type{model}, name::AbstractString,
                                  args...; kwds...) where {model<:CUTEstModel}
        j = 0
        while j < length(args) && args[j+1] isa AbstractString
            j += 1
        end
        nlp = model(name, args[1:j]...)
        try
            return $alg(nlp, args[j+1:end]...; kwds...)
        finally
            finalize(nlp)
        end
    end
end

end # module
