module SimplexCUTEst

using Printf
using OptimPack, OptimPack_jll
using CUTEst, NLPModels

default_problems = sort(select_sif_problems(; min_var=2, max_var=40,
                                            max_con=0, only_free_var=true, contype="unc"))

function runtests(problems=default_problems; xsiz::Real=0.5, kwds...)
    println("Problem          n    fcnt            f(x)         status")
    println("---------- ------- ------- ----------------------- --------------------")
    for name in problems
        nlp = CUTEstModel{Cdouble}(name)
        try
            c = simplex(x -> obj(nlp, x), nlp.meta.x0, xsiz; kwds...)
            @printf("%-10s %7d %7d %23.15e ", name, c.n, c.evaluations, c.f_best)
            printstyled(c.status; color=(issuccess(c) ? :green : :red))
            println()
        finally
            finalize(nlp)
        end
    end
end

end # module
