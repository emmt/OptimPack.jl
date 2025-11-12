"""

Tests for NEWUOA. Usage:

    NewuoaTests.runtests(; scale=𝟙, verbose=1, maxevals::Integer=500000,
                           inplace::Bool=false, recom=false)

"""
module NewuoaTests

using Printf, Test, Neutrals
using OptimPack.Newuoa
using OptimPack: configure!, restart!, iterate!
using OptimPack_jll

const evals = Ref{Int}()

# The Chebyquad test problem (Fletcher, 1965) for N = 2,4,6 and 8, with NPT = 2N+1.
function ftest(x::AbstractArray{T}) where {T<:AbstractFloat}
    n = length(x)
    np = n + 1
    y = Array{T}(undef, np, n)
    for j in 1:n
        y[1,j] = 1.0
        y[2,j] = x[j]*2.0 - 1.0
    end
    for i in 2:n
        for j in 1:n
            y[i+1,j] = y[2,j]*2.0*y[i,j] - y[i-1,j]
        end
    end
    f = 0.0
    iw = 1
    for i in 1:np
        sum = 0.0
        for j in 1:n
            sum += y[i,j]
        end
        sum /= n
        if iw > 0
            sum += 1.0/(i*i - 2*i)
        end
        iw = -iw
        f += sum*sum
    end
    evals[] += 1
    return f
end

# NOTE Default settings are to reproduce the output of the original software.
function runtests(; scale::Real=𝟙, verbose::Integer=2, maxevals::Integer=5000,
                  inplace::Bool=false, revcom::Bool=false)
    for n = 2:2:8
        npt = 2*n + 1
        x0 = Array{Cdouble}(undef, n)
        for i in 1:n
            x0[i] = i/(n + 1)
        end
        x0sav = copy(x0) # to check that x0 is left untouched
        x = similar(x0)
        rhobeg = x0[1]*0.2
        if scale != 1
            rhobeg /= scale
        end
        rhoend = 1e-6*rhobeg
        kwds = (; rhobeg=rhobeg, rhoend=rhoend, scale=scale, npt=npt,
                verbose=verbose, maxevals=maxevals)
        evals[] = 0
        verbose > 0 && @printf("\n\n    Results with N =%2d and NPT =%3d\n", n, npt)
        if revcom
            # Test the reverse communication variant.
            ctx = Newuoa.Context(copyto!(x, x0); kwds...)
            status = restart!(ctx)
            while status == Newuoa.ITERATE
                fx = ftest(x)
                status = iterate!(ctx, fx, x)
            end
            @test ctx.evals == evals[]
            @test ctx.fbest == ftest(x)
        else
            status, xm, fx, nf = if inplace
                newuoa!(ftest, copyto!(x, x0); kwds...)
            else
                newuoa(ftest, x0; kwds...)
            end
            @test nf == evals[]
            @test fx == ftest(xm)
            if inplace
                @test xm === x
                @test x0 == x0sav
            end
        end
        if verbose > 0 && status != Newuoa.SUCCESS
            printstyled("Something wrong occurred in NEWUOA: ", summary(status),
                        "\n"; color=:red)
        end
        @test issuccess(status) === (status === Newuoa.SUCCESS)
        @test summary(status) isa String
    end
end

end # module
