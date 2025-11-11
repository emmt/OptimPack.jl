"""

Tests for BOBYQA. Usage:

    BobyqaTests.runtests(; scale=𝟙, verbose=1, maxevals::Integer=500000,
                           inplace::Bool=false)

"""
module BobyqaTests

using Printf, Test, Neutrals
using OptimPack.Bobyqa

const evals = Ref{Int}()

# The test function.
function ftest_1(x::AbstractArray{<:AbstractFloat})
    fx = 0.0
    n = length(x)
    for i in 4:2:n
        for j in 2:2:i-2
            tempa = x[i - 1] - x[j - 1]
            tempb = x[i] - x[j]
            temp = max(tempa*tempa + tempb*tempb, 1e-6)
            fx += oftype(fx, inv(sqrt(temp)))
        end
    end
    evals[] += 1
    return fx
end

# NOTE Default settings are to reproduce the output of the original software.
function runtests(; scale::Real=𝟙, verbose::Integer=2, maxevals::Integer=500000,
                  inplace::Bool=false)

    # Run the tests.
    ftest = ftest_1
    bdl = -1.0
    bdu =  1.0
    rhobeg = 0.1
    rhoend = 1e-6
    for m in (5,10)
        q = 2.0*pi/m
        n = 2*m
        x = Array{Cdouble}(undef, n)
        x0 = similar(x)
        xl = similar(x)
        xu = similar(x)
        for i in 1:n
            xl[i] = bdl
            xu[i] = bdu
        end
        for j in 1:m
            temp = q*j
            x0[2j - 1] = cos(temp)
            x0[2j]     = sin(temp)
        end
        for jcase in 1:2
            npt = jcase == 2 ? 2n + 1 : n + 6
            verbose > 0 && @printf("\n\n     2D output with M =%4ld,  N =%4ld  and  NPT =%4ld\n", m, n, npt)
            kwds = (; lower=xl, upper=xu, rhobeg=rhobeg, rhoend=rhoend,
                    npt=npt, verbose=verbose, maxevals=maxevals)
            x0sav = copy(x0) # to check that x0 is left unchanged
            evals[] = 0
            status, xm, fx, nf = if inplace
                copyto!(x, x0)
                bobyqa!(ftest, x; kwds...)
            else
                bobyqa(ftest, x0; kwds...)
            end
            verbose > 0 && @printf("\n***** least function value: %.15e\n", fx)
            if verbose > 0 && status != Bobyqa.SUCCESS
                printstyled("Something wrong occurred in BOBYQA: ", summary(status),
                            "\n"; color=:red)
            end
            @test_broken nf == evals[]
            @test issuccess(status) == (status == Bobyqa.SUCCESS)
            @test issuccess(status)
            @test summary(status) isa String
            @test (xm === x) == inplace
            @test x0 == x0sav
            @test ftest(xm) == fx
        end
    end
end

end # module BobyqaTests

nothing
