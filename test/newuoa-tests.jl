module NewuoaTests

using Printf, Test
using OptimPack.Newuoa

function runtests(;revcom::Bool=false, scale::Real=1)
    # The Chebyquad test problem (Fletcher, 1965) for N = 2,4,6 and 8, with
    # NPT = 2N+1.
    function ftest(x::DenseVector{Cdouble})
        n = length(x)
        np = n + 1
        y = Array{Cdouble}(undef, np, n)
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
        return f
    end

    # Run the tests.
    rhoend = 1e-6
    for n = 2:2:8
        npt = 2*n + 1
        x = Array{Cdouble}(undef, n)
        for i in 1:n
            x[i] = i/(n + 1)
        end
        rhobeg = x[1]*0.2
        @printf("\n\n    Results with N =%2d and NPT =%3d\n", n, npt)
        if revcom
            # Test the reverse communication variant.
            ctx = Newuoa.create(n, rhobeg, rhoend;
                                npt = npt, verbose = 2, maxeval = 5000)
            status = getstatus(ctx)
            while status == Newuoa.ITERATE
                fx = ftest(x)
                status = iterate(ctx, fx, x)
            end
            if status != Newuoa.SUCCESS
                println("Something wrong occured in NEWUOA: ",
                        getreason(status))
            end
        else
            status, _, fx = if scale != 1
                newuoa!(ftest, x; rhobeg=rhobeg/scale, rhoend=rhoend/scale,
                        scale = fill!(similar(x), scale),
                        npt, verbose = 2, maxeval = 5000)
            else
                newuoa!(ftest, x; rhobeg, rhoend, npt, verbose = 2, maxeval = 5000)
            end
            @test issuccess(status)
            @test status.code isa Integer
            @test status.reason isa String
        end
    end
end

isinteractive() && runtests()

end # module
