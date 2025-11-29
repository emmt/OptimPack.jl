module BobyqaCUTEst

using Printf
using OptimPack, OptimPack_jll
using CUTEst, NLPModels

default_problems = sort(select_sif_problems(; min_var=2, max_var=100,
                                            max_con=0, only_bnd_var=true, contype=:bounds))

function runtests(problems=default_problems; rhobeg::Real=0.5, rhoend=1e-5*rhobeg, kwds...)
    println("Problem          n    fcnt            f(x)         status")
    println("---------- ------- ------- ----------------------- --------------------")
    for name in problems
        nlp = CUTEstModel{Cdouble}(name)
        try
            status, x, fx, nf = bobyqa(x -> obj(nlp, x), nlp.meta.x0;
                                       rhobeg=rhobeg, rhoend=rhoend,
                                       lower = nlp.meta.lvar, upper = nlp.meta.uvar,
                                       kwds...)
            @printf("%-10s %7d %7d %23.15e ", name, length(x), nf, fx)
            printstyled(status; color=(status == Bobyqa.SUCCESS ? :green : :red))
            println()
        finally
            finalize(nlp)
        end
    end
end

end # module
