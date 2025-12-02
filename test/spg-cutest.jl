module SPGCUTEst

using CUTEst
using NLPModels
using OptimPack
using OptimPack: BoundedSet
using Printf

default_problems = sort(select_sif_problems(; min_var=2,
                                            max_con=0, only_bnd_var=true, contype=:bounds))

function runtests(problems=default_problems; kwds...)
    println("Problem          n    iter    fcnt    gcnt    pcnt            f(x)         ‖gp(x)‖₀₀ status")
    println("---------- ------- ------- ------- ------- ------- ----------------------- --------- --------------------")
    for name in problems
        nlp = CUTEstModel{Cdouble}(name)
        try
            ctx = spg(x -> obj(nlp, x), (g, x) -> copy!(g, grad(nlp, x)),
                      BoundedSet(nlp.meta.lvar, nlp.meta.uvar), nlp.meta.x0;
                      kwds...)
            @printf("%-10s %7d %7d %7d %7d %7d %23.15e%10.2e ", name, length(ctx.x_best), ctx.iterations,
                    ctx.evaluations, ctx.gradients, ctx.projections, ctx.f_best, ctx.gpsupn)
            printstyled(ctx.status; color=(issuccess(ctx) ? :green : :red))
            println()
        finally
            finalize(nlp)
        end
    end
end

end # module
