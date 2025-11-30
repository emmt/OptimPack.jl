"""
    CudaTests

Tests for basic operations on *"CUDA vectors"* remembering that any array is considered as a
vector of reals in these operations and complexes as pairs of reals. Typical usage:

    CudaTests.runtests(; kwds...)

"""
module CudaTests

using CUDA, LinearAlgebra, Neutrals, OptimPack, Printf, Test, TypeUtils, Unitful
using OptimPack: LoopStyles, one_norm, two_norm, sup_norm, inner, scale!, xpby!, axpby!

function runtests(; T::Type=Float32,
                  dims::Union{Integer,Tuple{Vararg{Integer}}}=10_000,
                  alphas=(-1, 0, 1, -𝟙, 𝟘, 𝟙, 2, -pi),
                  betas=(-1, 0, 1, -𝟙, 𝟘, 𝟙, -2, pi))
    x_cpu = rand(T, dims)
    y_cpu = rand(T, dims)
    z_cpu = similar(x_cpu)
    x_gpu = CuArray(x_cpu)
    y_gpu = CuArray(y_cpu)
    z_gpu = similar(x_gpu)
    x_vec = x_cpu[:]
    y_vec = y_cpu[:]
    n = length(x_cpu)
    x_cpu_cpy = copy(x_cpu) # to check that x is left untouched
    y_cpu_cpy = copy(y_cpu) # to check that y is left untouched
    x_gpu_cpy = copy(x_gpu) # to check that x is left untouched
    y_gpu_cpy = copy(y_gpu) # to check that y is left untouched
    @testset "Operations on vectors" begin
        @testset "1-norm" begin
            s = norm(x_vec, 1)
            @test @inferred(one_norm(x_cpu)) ≈ s
            @test @inferred(one_norm(x_gpu)) ≈ s
        end
        @testset "2-norm" begin
            s = norm(x_vec, 2)
            @test @inferred(two_norm(x_cpu)) ≈ s
            @test @inferred(two_norm(x_gpu)) ≈ s
        end
        @testset "sup-norm" begin
            s = norm(x_vec, Inf)
            @test @inferred(sup_norm(x_cpu)) ≈ s
            @test @inferred(sup_norm(x_gpu)) ≈ s
        end
        @testset "inner product" begin
            s = dot(x_vec, y_vec)
            @test @inferred(inner(x_cpu, y_cpu)) ≈ s
            @test @inferred(inner(x_gpu, y_gpu)) ≈ s
        end
        @testset "scale!(dst, $α, x)" for α in alphas
            @test @inferred(scale!(z_cpu, α, x_cpu)) === z_cpu
            @test x_cpu == x_cpu_cpy
            @test z_cpu ≈ α*x_cpu
            @test @inferred(scale!(z_gpu, α, x_gpu)) === z_gpu
            @test x_gpu == x_gpu_cpy
            @test z_gpu ≈ α*x_gpu
        end
        @testset "xpby!(dst, x, $β, y)" for β in betas
            @test @inferred(xpby!(z_cpu, x_cpu, β, y_cpu)) === z_cpu
            @test x_cpu == x_cpu_cpy
            @test y_cpu == y_cpu_cpy
            @test z_cpu ≈ x_cpu + β*y_cpu
            @test @inferred(xpby!(z_gpu, x_gpu, β, y_gpu)) === z_gpu
            @test x_gpu == x_gpu_cpy
            @test y_gpu == y_gpu_cpy
            @test z_gpu ≈ x_gpu + β*y_gpu
        end
        @testset "axpby!(dst, $α, x, $β, y)" for α in alphas, β in betas
            @test @inferred(axpby!(z_cpu, α, x_cpu, β, y_cpu)) === z_cpu
            @test x_cpu == x_cpu_cpy
            @test y_cpu == y_cpu_cpy
            @test z_cpu ≈ α*x_cpu + β*y_cpu
            @test @inferred(axpby!(z_gpu, α, x_gpu, β, y_gpu)) === z_gpu
            @test x_gpu == x_gpu_cpy
            @test y_gpu == y_gpu_cpy
            @test z_gpu ≈ α*x_gpu + β*y_gpu
        end
    end
end

end # module
