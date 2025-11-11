"""

Tests for COBYLA. Usage:

    CobylaTests.runtests(; scale=𝟙, verbose=1, maxevals::Integer=500000,
                           inplace::Bool=false, recom=false)

"""
module CobylaTests

using Printf, Test, Neutrals
using OptimPack.Cobyla
using OptimPack: configure!, restart!, iterate!

const evals = Ref{Int}()

# NOTE Default settings are to reproduce the output of the original software.
function runtests(; scale::Real=𝟙, verbose::Integer=1, maxevals::Integer=2000,
                  inplace::Bool=false, revcom::Bool=false)
    # Beware that order of operations may affect the result (within rounding errors). I have
    # tried to keep the same ordering as F2C which takes care of that, in particular when
    # converting expressions involving powers.
    prt(s) = println("\n       "*s)
    for nprob in 1:10
        if nprob == 1
            # Minimization of a simple quadratic function of two variables.
            verbose > 0 && prt("Output from test problem 1 (Simple quadratic)")
            n = 2
            m = 0
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = -1.0
            xopt[2] = 0.0
            ftest = (x::DenseVector{Cdouble}) -> begin
                r1 = x[1] + 1.0
                r2 = x[2]
                fc = 10.0*(r1*r1) + (r2*r2)
                evals[] += 1
                return fc
            end
        elseif nprob == 2
            # Easy two dimensional minimization in unit circle.
            verbose > 0 && prt("Output from test problem 2 (2D unit circle calculation)")
            n = 2
            m = 1
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = sqrt(0.5)
            xopt[2] = -xopt[1]
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = x[1]*x[2]
                con[1] = 1.0 - x[1]*x[1] - x[2]*x[2]
                evals[] += 1
                return fc
            end
        elseif nprob == 3
            # Easy three dimensional minimization in ellipsoid.
            verbose > 0 && prt("Output from test problem 3 (3D ellipsoid calculation)")
            n = 3
            m = 1
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = 1.0/sqrt(3.0)
            xopt[2] = 1.0/sqrt(6.0)
            xopt[3] = -0.33333333333333331
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = x[1]*x[2]*x[3]
                con[1] = 1.0 - (x[1]*x[1]) - 2.0*(x[2]*x[2]) - 3.0*(x[3]*x[3])
                evals[] += 1
                return fc
            end
        elseif nprob == 4
            # Weak version of Rosenbrock's problem.
            verbose > 0 && prt("Output from test problem 4 (Weak Rosenbrock)")
            n = 2
            m = 0
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = -1.0
            xopt[2] = 1.0
            ftest = (x::DenseVector{Cdouble}) -> begin
                r2 = x[1]
                r1 = r2*r2 - x[2]
                r3 = x[1] + 1.0
                fc = r1*r1 + r3*r3
                evals[] += 1
                return fc
            end
        elseif nprob == 5
            # Intermediate version of Rosenbrock's problem.
            verbose > 0 && prt("Output from test problem 5 (Intermediate Rosenbrock)")
            n = 2
            m = 0
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = -1.0
            xopt[2] = 1.0
            ftest = (x::DenseVector{Cdouble}) -> begin
                r2 = x[1]
                r1 = r2*r2 - x[2]
                r3 = x[1] + 1.0
                fc = r1*r1*10.0 + r3*r3
                evals[] += 1
                return fc
            end
        elseif nprob == 6
            # This problem is taken from Fletcher's book Practical Methods of Optimization
            # and has the equation number (9.1.15).
            verbose > 0 && prt("Output from test problem 6 (Equation (9.1.15) in Fletcher)")
            n = 2
            m = 2
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = sqrt(0.5)
            xopt[2] = xopt[1]
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = -x[1] - x[2]
                r1 = x[1]
                con[1] = x[2] - r1*r1
                r1 = x[1]
                r2 = x[2]
                con[2] = 1.0 - r1*r1 - r2*r2
                evals[] += 1
                return fc
            end
        elseif nprob == 7
            # This problem is taken from Fletcher's book Practical Methods of Optimization
            # and has the equation number (14.4.2).
            verbose > 0 && prt("Output from test problem 7 (Equation (14.4.2) in Fletcher)")
            n = 3
            m = 3
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = 0.0
            xopt[2] = -3.0
            xopt[3] = -3.0
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = x[3]
                con[1] = x[1]*5.0 - x[2] + x[3]
                r1 = x[1]
                r2 = x[2]
                con[2] = x[3] - r1*r1 - r2*r2 - x[2]*4.0
                con[3] = x[3] - x[1]*5.0 - x[2]
                evals[] += 1
                return fc
            end
        elseif nprob == 8
            # This problem is taken from page 66 of Hock and Schittkowski's book Test
            # Examples for Nonlinear Programming Codes. It is their test problem Number 43,
            # and has the name Rosen-Suzuki.
            verbose > 0 && prt("Output from test problem 8 (Rosen-Suzuki)")
            n = 4
            m = 3
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = 0.0
            xopt[2] = 1.0
            xopt[3] = 2.0
            xopt[4] = -1.0
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                fc = (r1*r1 + r2*r2 + r3*r3*2.0 + r4*r4 - x[1]*5.0
                      - x[2]*5.0 - x[3]*21.0 + x[4]*7.0)
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                con[1] = (8.0 - r1*r1 - r2*r2 - r3*r3 - r4*r4 - x[1]
                          + x[2] - x[3] + x[4])
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                con[2] = (10.0 - r1*r1 - r2*r2*2.0 - r3*r3 - r4*r4*2.0
                          + x[1] + x[4])
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                con[3] = (5.0 - r1*r1*2.0 - r2*r2 - r3*r3 - x[1]*2.0
                          + x[2] + x[4])
                evals[] += 1
                return fc
            end
        elseif nprob == 9
            # This problem is taken from page 111 of Hock and Schittkowski's book Test
            # Examples for Nonlinear Programming Codes. It is their test problem Number 100.
            verbose > 0 && prt("Output from test problem 9 (Hock and Schittkowski 100)")
            n = 7
            m = 4
            xopt = Array{Cdouble}(undef, n)
            xopt[1] =  2.330499
            xopt[2] =  1.951372
            xopt[3] = -0.4775414
            xopt[4] =  4.365726
            xopt[5] = -0.624487
            xopt[6] =  1.038131
            xopt[7] =  1.594227
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                r1 = x[1] - 10.0
                r2 = x[2] - 12.0
                r3 = x[3]
                r3 *= r3
                r4 = x[4] - 11.0
                r5 = x[5]
                r5 *= r5
                r6 = x[6]
                r7 = x[7]
                r7 *= r7
                fc = (r1*r1 + r2*r2*5.0 + r3*r3 + r4*r4*3.0
                      + r5*(r5*r5)*10.0 + r6*r6*7.0 + r7*r7
                      - x[6]*4.0*x[7] - x[6]*10.0 - x[7]*8.0)
                r1 = x[1]
                r2 = x[2]
                r2 *= r2
                r3 = x[4]
                con[1] = (127.0 - r1*r1*2.0 - r2*r2*3.0 - x[3]
                          - r3*r3*4.0 - x[5]*5.0)
                r1 = x[3]
                con[2] = (282.0 - x[1]*7.0 - x[2]*3.0 - r1*r1*10.0
                          - x[4] + x[5])
                r1 = x[2]
                r2 = x[6]
                con[3] = (196.0 - x[1]*23.0 - r1*r1 - r2*r2*6.0
                          + x[7]*8.0)
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                con[4] = (r1*r1*-4.0 - r2*r2 + x[1]*3.0*x[2]
                          - r3*r3*2.0 - x[6]*5.0 + x[7]*11.0)
                evals[] += 1
                return fc
            end
        elseif nprob == 10
            # This problem is taken from page 415 of Luenberger's book Applied Nonlinear
            # Programming. It is to maximize the area of a hexagon of unit diameter.
            verbose > 0 && prt("Output from test problem 10 (Hexagon area)")
            n = 9
            m = 14
            xopt = fill!(Array{Cdouble}(undef, n), 0.0)
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = -0.5*(x[1]*x[4] - x[2]*x[3] + x[3]*x[9] - x[5]*x[9]
                           + x[5]*x[8] - x[6]*x[7])
                r1 = x[3]
                r2 = x[4]
                con[1] = 1.0 - r1*r1 - r2*r2
                r1 = x[9]
                con[2] = 1.0 - r1*r1
                r1 = x[5]
                r2 = x[6]
                con[3] = 1.0 - r1*r1 - r2*r2
                r1 = x[1]
                r2 = x[2] - x[9]
                con[4] = 1.0 - r1*r1 - r2*r2
                r1 = x[1] - x[5]
                r2 = x[2] - x[6]
                con[5] = 1.0 - r1*r1 - r2*r2
                r1 = x[1] - x[7]
                r2 = x[2] - x[8]
                con[6] = 1.0 - r1*r1 - r2*r2
                r1 = x[3] - x[5]
                r2 = x[4] - x[6]
                con[7] = 1.0 - r1*r1 - r2*r2
                r1 = x[3] - x[7]
                r2 = x[4] - x[8]
                con[8] = 1.0 - r1*r1 - r2*r2
                r1 = x[7]
                r2 = x[8] - x[9]
                con[9] = 1.0 - r1*r1 - r2*r2
                con[10] = x[1]*x[4] - x[2]*x[3]
                con[11] = x[3]*x[9]
                con[12] = -x[5]*x[9]
                con[13] = x[5]*x[8] - x[6]*x[7]
                con[14] = x[9]
                evals[] += 1
                return fc
            end
        else
            error("bad problem number ($nprob)")
        end

        x0 = Array{Cdouble}(undef, n)
        x = similar(x0)
        c = Array{Cdouble}(undef, max(m, 0))
        for icase in 1:2
            # Initial solution and parameters.
            fill!(x0, 1.0)
            kwds = (rhobeg = 0.5/scale,
                    rhoend = (icase == 2 ? 1e-4 : 0.001)/scale,
                    verbose = verbose,
                    maxevals = maxevals,
                    scale = scale === 𝟙 ? scale : fill!(similar(x, Cdouble), scale))
            evals[] = 0
            if revcom
                # Test the reverse communication variant.
                ctx = Cobyla.Context(copyto!(x, x0), c; kwds...)
                status = restart!(ctx)
                while status == Cobyla.ITERATE
                    if m > 0
                        # Some constraints.
                        fx = ftest(x, c)
                        status = iterate!(ctx, fx, x, c)
                    else
                        # No constraints.
                        fx = ftest(x)
                        status = iterate!(ctx, fx, x)
                    end
                end
                @test ctx.evals == evals[]
            else
                status, xm, cm, fx, nf = if inplace
                    cobyla!(ftest, copyto!(x, x0), c; kwds...)
                else
                    cobyla(ftest, x0, size(c)...; kwds...)
                end
                @test nf == evals[]
                @test (xm === x) == inplace
                @test (cm === c) == inplace
                xm === x || copyto!(x, xm)
                cm === c || copyto!(c, cm)
            end
            if nprob == 10
                tempa = x[1] + x[3] + x[5] + x[7]
                tempb = x[2] + x[4] + x[6] + x[8]
                tempc = 0.5/sqrt(tempa*tempa + tempb*tempb)
                tempd = tempc*sqrt(3.0)
                xopt[1] = tempd*tempa + tempc*tempb
                xopt[2] = tempd*tempb - tempc*tempa
                xopt[3] = tempd*tempa - tempc*tempb
                xopt[4] = tempd*tempb + tempc*tempa
                for i in 1:4
                    xopt[i + 4] = xopt[i]
                end
            end
            if verbose > 0 && status != Cobyla.SUCCESS
                printstyled("Something wrong occurred in COBYLA: ", summary(status),
                            "\n"; color=:red)
            end
            @test issuccess(status) == (status == Cobyla.SUCCESS)
            @test issuccess(status)
            @test summary(status) isa String
            # Compare the solution to the known optimum.
            rtol = 0.13 # <- due to the choice of rhoend, the accuracy is rather poor...
            atol = 1e-6
            @test x ≈ xopt atol=atol rtol=rtol
            temp = 0.0
            for i in 1:n
                r1 = x[i] - xopt[i]
                temp += r1*r1
            end
            verbose > 0 && @printf("\n     Least squares error in variables =%16.6E\n",
                                   sqrt(temp))
        end
        verbose > 0 && @printf("  ------------------------------------------------------------------\n")
    end
end

end # module CobylaTests

nothing
