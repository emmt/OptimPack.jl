"""
    VectOpsBenchmarks

Tests for basic operations of *"vectors"* remembering that any array is considered as a
vector of reals in these operations and complexes as pairs of reals. Typical usage:

    VectOpsBenchmarks.runtests(args...; kwds...)

"""
module VectOpsBenchmarks

using BenchmarkTools
using LinearAlgebra
using LoopVectorization
using Neutrals
using OptimPack
using OptimPack: LoopStyles, one_norm, two_norm, sup_norm, inner, scale!, xpby!, axpby!
using Test
using ThreadPinning
using TypeUtils
using Unitful
using Unitful: AbstractQuantity

function runtests(funcs::Symbol... = :all;
                  T::Type=Float32,
                  dims::Union{Integer,Tuple{Vararg{Integer}}}=10_000,
                  alphas=(2, -pi, 0, -1, 1),
                  betas=(-2, pi, 0, -1, 1))
    if funcs == (:all,)
        funcs = (:one_norm, :two_norm, :sup_norm, :inner, :scale, :xpby, :axpby)
    end
    x = rand(T, dims)
    y = rand(T, dims)
    z = similar(x)
    x_flat = x[:]
    y_flat = y[:]
    pinthreads(:cores)
    if :one_norm ∈ funcs
        println()
        println("1-norm (T=$T, n=$(length(x)))")
        print(" norm(x, 1)              "); @btime norm($x_flat, 1)
        print(" one_norm(            x) "); @btime one_norm(                          $x)
        print(" one_norm(Map(),      x) "); @btime one_norm($(LoopStyles.Map()),      $x)
        print(" one_norm(Dot(),      x) "); @btime one_norm($(LoopStyles.Dot()),      $x)
        print(" one_norm(For(),      x) "); @btime one_norm($(LoopStyles.For()),      $x)
        print(" one_norm(InBounds(), x) "); @btime one_norm($(LoopStyles.InBounds()), $x)
        print(" one_norm(SIMD(),     x) "); @btime one_norm($(LoopStyles.SIMD()),     $x)
        print(" one_norm(Turbo(),    x) "); @btime one_norm($(LoopStyles.Turbo()),    $x)
    end
    if :two_norm ∈ funcs
        println()
        println("2-norm (T=$T, n=$(length(x)))")
        print(" norm(x, 2)              "); @btime norm($x_flat, 2)
        print(" two_norm(            x) "); @btime two_norm(                          $x)
        print(" two_norm(Map(),      x) "); @btime two_norm($(LoopStyles.Map()),      $x)
        print(" two_norm(Dot(),      x) "); @btime two_norm($(LoopStyles.Dot()),      $x)
        print(" two_norm(For(),      x) "); @btime two_norm($(LoopStyles.For()),      $x)
        print(" two_norm(InBounds(), x) "); @btime two_norm($(LoopStyles.InBounds()), $x)
        print(" two_norm(SIMD(),     x) "); @btime two_norm($(LoopStyles.SIMD()),     $x)
        print(" two_norm(Turbo(),    x) "); @btime two_norm($(LoopStyles.Turbo()),    $x)
    end
    if :sup_norm ∈ funcs
        println()
        println("sup-norm (T=$T, n=$(length(x)))")
        print(" norm(x, Inf)            "); @btime norm($x_flat, Inf)
        print(" sup_norm(            x) "); @btime sup_norm(                          $x)
        print(" sup_norm(Map(),      x) "); @btime sup_norm($(LoopStyles.Map()),      $x)
        print(" sup_norm(Dot(),      x) "); @btime sup_norm($(LoopStyles.Dot()),      $x)
        print(" sup_norm(For(),      x) "); @btime sup_norm($(LoopStyles.For()),      $x)
        print(" sup_norm(InBounds(), x) "); @btime sup_norm($(LoopStyles.InBounds()), $x)
        print(" sup_norm(SIMD(),     x) "); @btime sup_norm($(LoopStyles.SIMD()),     $x)
        print(" sup_norm(Turbo(),    x) "); @btime sup_norm($(LoopStyles.Turbo()),    $x)
    end
    if :inner ∈ funcs
        println()
        println("inner (T=$T, n=$(length(x)))")
        print(" dot(              x, y) "); @btime dot($x_flat, $y_flat)
        print(" inner(            x, y) "); @btime inner(                          $x, $y)
        print(" inner(Map(),      x, y) "); @btime inner($(LoopStyles.Map()),      $x, $y)
        print(" inner(Dot(),      x, y) "); @btime inner($(LoopStyles.Dot()),      $x, $y)
        print(" inner(For(),      x, y) "); @btime inner($(LoopStyles.For()),      $x, $y)
        print(" inner(InBounds(), x, y) "); @btime inner($(LoopStyles.InBounds()), $x, $y)
        print(" inner(SIMD(),     x, y) "); @btime inner($(LoopStyles.SIMD()),     $x, $y)
        print(" inner(Turbo(),    x, y) "); @btime inner($(LoopStyles.Turbo()),    $x, $y)
    end
    if :scale ∈ funcs
        for α in alphas
            println()
            println("scale! (T=$T, n=$(length(x)), α=$α)")
            print(" scale!(            z, α, x) "); @btime scale!(                          $z, $α, $x)
            print(" scale!(Map(),      z, α, x) "); @btime scale!($(LoopStyles.Map()),      $z, $α, $x)
            print(" scale!(Dot(),      z, α, x) "); @btime scale!($(LoopStyles.Dot()),      $z, $α, $x)
            print(" scale!(For(),      z, α, x) "); @btime scale!($(LoopStyles.For()),      $z, $α, $x)
            print(" scale!(InBounds(), z, α, x) "); @btime scale!($(LoopStyles.InBounds()), $z, $α, $x)
            print(" scale!(SIMD(),     z, α, x) "); @btime scale!($(LoopStyles.SIMD()),     $z, $α, $x)
            print(" scale!(Turbo(),    z, α, x) "); @btime scale!($(LoopStyles.Turbo()),    $z, $α, $x)
        end
    end
    if :xpby ∈ funcs
        for β in betas
            println()
            println("xpby! (T=$T, n=$(length(x)), β=$β)")
            print(" xpby!(            z, x, β, y) "); @btime xpby!(                          $z, $x, $β, $y)
            print(" xpby!(Map(),      z, x, β, y) "); @btime xpby!($(LoopStyles.Map()),      $z, $x, $β, $y)
            print(" xpby!(Dot(),      z, x, β, y) "); @btime xpby!($(LoopStyles.Dot()),      $z, $x, $β, $y)
            print(" xpby!(For(),      z, x, β, y) "); @btime xpby!($(LoopStyles.For()),      $z, $x, $β, $y)
            print(" xpby!(InBounds(), z, x, β, y) "); @btime xpby!($(LoopStyles.InBounds()), $z, $x, $β, $y)
            print(" xpby!(SIMD(),     z, x, β, y) "); @btime xpby!($(LoopStyles.SIMD()),     $z, $x, $β, $y)
            print(" xpby!(Turbo(),    z, x, β, y) "); @btime xpby!($(LoopStyles.Turbo()),    $z, $x, $β, $y)
        end
    end
    if :axpby ∈ funcs
        for α in alphas, β in betas
            println()
            println("axpby! (T=$T, n=$(length(x)), α=$α, β=$β)")
            print(" axpby!(            z, α, x, β, y) "); @btime axpby!(                          $z, $α, $x, $β, $y)
            print(" axpby!(Map(),      z, α, x, β, y) "); @btime axpby!($(LoopStyles.Map()),      $z, $α, $x, $β, $y)
            print(" axpby!(Dot(),      z, α, x, β, y) "); @btime axpby!($(LoopStyles.Dot()),      $z, $α, $x, $β, $y)
            print(" axpby!(For(),      z, α, x, β, y) "); @btime axpby!($(LoopStyles.For()),      $z, $α, $x, $β, $y)
            print(" axpby!(InBounds(), z, α, x, β, y) "); @btime axpby!($(LoopStyles.InBounds()), $z, $α, $x, $β, $y)
            print(" axpby!(SIMD(),     z, α, x, β, y) "); @btime axpby!($(LoopStyles.SIMD()),     $z, $α, $x, $β, $y)
            print(" axpby!(Turbo(),    z, α, x, β, y) "); @btime axpby!($(LoopStyles.Turbo()),    $z, $α, $x, $β, $y)
        end
    end
end

end # module
