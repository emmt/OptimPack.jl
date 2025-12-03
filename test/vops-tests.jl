"""
    VectOpsTests

Tests for basic operations on *"vectors"* remembering that any array is considered as a
vector of reals in these operations and complexes as pairs of reals. Typical usage:

    VectOpsTests.runtests(; kwds...)

"""
module VectOpsTests

using LinearAlgebra, Neutrals, OptimPack, Printf, Test, TypeUtils, Unitful
using OptimPack: LoopStyles, one_norm, two_norm, sup_norm, inner, scale!, mult!, xpby!, axpby!

function runtests(; T::Type=Float32,
                  dims::Union{Integer,Tuple{Vararg{Integer}}}=10_000,
                  alphas=(-1, 0, 1, -𝟙, 𝟘, 𝟙, 2, -pi),
                  betas=(-1, 0, 1, -𝟙, 𝟘, 𝟙, -2, pi))
    w = rand(T, dims)
    x = rand(T, dims)
    y = rand(T, dims)
    z = similar(x)
    w_cpy = copy(w) # to check that w is left untouched
    x_cpy = copy(x) # to check that x is left untouched
    y_cpy = copy(y) # to check that y is left untouched
    @testset "Operations on vectors" begin
        @testset "1-norm" begin
            s = norm(view(x, :), 1)
            @test @inferred(one_norm(                       x)) ≈ s
            @test @inferred(one_norm(LoopStyles.Map(),      x)) ≈ s
            @test @inferred(one_norm(LoopStyles.Dot(),      x)) ≈ s
            @test @inferred(one_norm(LoopStyles.For(),      x)) ≈ s
            @test @inferred(one_norm(LoopStyles.InBounds(), x)) ≈ s
            @test @inferred(one_norm(LoopStyles.SIMD(),     x)) ≈ s
            @test @inferred(one_norm(LoopStyles.Turbo(),    x)) ≈ s
            @test isequal(@inferred(one_norm([1.0, NaN])), NaN)
            @test isequal(@inferred(one_norm([NaN, 1.0])), NaN)
            @test x == x_cpy
        end
        @testset "2-norm" begin
            s = norm(view(x, :), 2)
            @test @inferred(two_norm(                       x)) ≈ s
            @test @inferred(two_norm(LoopStyles.Map(),      x)) ≈ s
            @test @inferred(two_norm(LoopStyles.Dot(),      x)) ≈ s
            @test @inferred(two_norm(LoopStyles.For(),      x)) ≈ s
            @test @inferred(two_norm(LoopStyles.InBounds(), x)) ≈ s
            @test @inferred(two_norm(LoopStyles.SIMD(),     x)) ≈ s
            @test @inferred(two_norm(LoopStyles.Turbo(),    x)) ≈ s
            @test isequal(@inferred(two_norm([1.0, NaN])), NaN)
            @test isequal(@inferred(two_norm([NaN, 1.0])), NaN)
            @test x == x_cpy
        end
        @testset "sup-norm" begin
            s = norm(view(x, :), Inf)
            @test @inferred(sup_norm(                       x)) ≈ s
            @test @inferred(sup_norm(LoopStyles.Map(),      x)) ≈ s
            @test @inferred(sup_norm(LoopStyles.Dot(),      x)) ≈ s
            @test @inferred(sup_norm(LoopStyles.For(),      x)) ≈ s
            @test @inferred(sup_norm(LoopStyles.InBounds(), x)) ≈ s
            @test @inferred(sup_norm(LoopStyles.SIMD(),     x)) ≈ s
            @test @inferred(sup_norm(LoopStyles.Turbo(),    x)) ≈ s
            @test isequal(@inferred(sup_norm([1.0, NaN])), NaN)
            @test isequal(@inferred(sup_norm([NaN, 1.0])), NaN)
            @test x == x_cpy
        end
        @testset "inner product" begin
            s = dot(view(x, :), view(y, :))
            @test @inferred(inner(                       x, y)) ≈ s
            @test @inferred(inner(LoopStyles.Map(),      x, y)) ≈ s
            @test @inferred(inner(LoopStyles.Dot(),      x, y)) ≈ s
            @test @inferred(inner(LoopStyles.For(),      x, y)) ≈ s
            @test @inferred(inner(LoopStyles.InBounds(), x, y)) ≈ s
            @test @inferred(inner(LoopStyles.SIMD(),     x, y)) ≈ s
            @test @inferred(inner(LoopStyles.Turbo(),    x, y)) ≈ s
            @test x == x_cpy
            @test y == y_cpy
        end
        @testset "triple inner product" begin
            s = sum(w .* x .* y)
            @test @inferred(inner(                       w, x, y)) ≈ s
            @test @inferred(inner(LoopStyles.Map(),      w, x, y)) ≈ s
            @test @inferred(inner(LoopStyles.Dot(),      w, x, y)) ≈ s
            @test @inferred(inner(LoopStyles.For(),      w, x, y)) ≈ s
            @test @inferred(inner(LoopStyles.InBounds(), w, x, y)) ≈ s
            @test @inferred(inner(LoopStyles.SIMD(),     w, x, y)) ≈ s
            @test @inferred(inner(LoopStyles.Turbo(),    w, x, y)) ≈ s
            @test w == w_cpy
            @test x == x_cpy
            @test y == y_cpy
        end
        @testset "scale!(dst, $α, x)" for α in alphas
            s = α*x
            @test @inferred(scale!(                       z, α, x)) === z
            @test x == x_cpy
            @test z ≈ s
            @test @inferred(scale!(LoopStyles.Map(),      z, α, x)) === z
            @test x == x_cpy
            @test z ≈ s
            @test @inferred(scale!(LoopStyles.Dot(),      z, α, x)) === z
            @test x == x_cpy
            @test z ≈ s
            @test @inferred(scale!(LoopStyles.For(),      z, α, x)) === z
            @test x == x_cpy
            @test z ≈ s
            @test @inferred(scale!(LoopStyles.InBounds(), z, α, x)) === z
            @test x == x_cpy
            @test z ≈ s
            @test @inferred(scale!(LoopStyles.SIMD(),     z, α, x)) === z
            @test x == x_cpy
            @test z ≈ s
            @test @inferred(scale!(LoopStyles.Turbo(),    z, α, x)) === z
            @test x == x_cpy
            @test z ≈ s
        end
        @testset "element-wise multiplication" begin
            s = x .* y
            @test @inferred(mult!(                       z, x, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(mult!(LoopStyles.Map(),      z, x, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(mult!(LoopStyles.Dot(),      z, x, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(mult!(LoopStyles.For(),      z, x, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(mult!(LoopStyles.InBounds(), z, x, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(mult!(LoopStyles.SIMD(),     z, x, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(mult!(LoopStyles.Turbo(),    z, x, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
        end
        @testset "xpby!(dst, x, $β, y)" for β in betas
            s = x + β*y
            @test @inferred(xpby!(                       z, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(xpby!(LoopStyles.Map(),      z, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(xpby!(LoopStyles.Dot(),      z, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(xpby!(LoopStyles.For(),      z, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(xpby!(LoopStyles.InBounds(), z, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(xpby!(LoopStyles.SIMD(),     z, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(xpby!(LoopStyles.Turbo(),    z, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
        end
        @testset "axpby!(dst, $α, x, $β, y)" for α in alphas, β in betas
            s = α*x + β*y
            @test @inferred(axpby!(                       z, α, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(axpby!(LoopStyles.Map(),      z, α, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(axpby!(LoopStyles.Dot(),      z, α, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(axpby!(LoopStyles.For(),      z, α, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(axpby!(LoopStyles.InBounds(), z, α, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(axpby!(LoopStyles.SIMD(),     z, α, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
            @test @inferred(axpby!(LoopStyles.Turbo(),    z, α, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z ≈ s
        end
    end
end

end # module
