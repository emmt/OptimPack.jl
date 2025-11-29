module SimplexTests

using Test, Printf, Unitful, TypeUtils
using LinearAlgebra
using OptimPack
using OptimPack: Problems

@testset "Nelder-Mead Simplex method" begin
    @testset "Rosenbrock function" begin
        f = Problems.Rosenbrock(n=2)
        x0 = Problems.x_init(f)
        # Solve the problem with modest precision.
        c = @inferred simplex(f, x0, 0.5; xtol=1e-5, maxevals=1000)
        @test issuccess(c)
        @test c.x_best ≈ Problems.x_best(f) rtol=1e-5
        @test c.f_best ≈ Problems.f_best(f) atol=1e-10
        x1, f1, lvr1 = copy(c.x_best), c.f_best, c.LVR
        # Solve the problem with very high precision and a smaller initial simplex.
        @test Simplex.solve!(c, f, x0, [0.1, -0.1]; xtol=1e-10, ftol=0, maxevals=1000) === c
        @test issuccess(c)
        @test c.x_best ≈ Problems.x_best(f) rtol=1e-10
        @test c.f_best ≈ Problems.f_best(f) atol=1e-20
        @test c.LVR < lvr1
        @test norm(c.x_best - Problems.x_best(f)) < norm(x1 - Problems.x_best(f))
        @test abs(c.f_best - Problems.f_best(f)) < abs(f1 - Problems.f_best(f))
        # Test observer.
        Simplex.solve!(c, f, x0, 0.5;
                       observer = (c,f) -> c.iterations < 13 ? c.status : :stop13)
        @test c.status == :stop13
        @test c.iterations == 13
        # Test show.
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), c)
        str = String(take!(buf))
        @test startswith(str, "• Algorithm: Nelder-Mead Simplex method")
    end
end

end # module
