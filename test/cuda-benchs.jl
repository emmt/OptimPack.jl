"""
    CudaBenchmarks

Tests for basic operations on *"vectors"* remembering that any array is considered as a
vector of reals in these operations and complexes as pairs of reals. Typical usage:

    CudaBenchmarks.runtests(args...; kwds...)

"""
module CudaBenchmarks

using BenchmarkTools
using LinearAlgebra
using CUDA
using Neutrals
using OptimPack
using OptimPack: LoopStyles, one_norm, two_norm, sup_norm, inner, scale!, xpby!, axpby!
using Test
using ThreadPinning
using TypeUtils

function runtests(funcs::Symbol... = :all;
                  T::Type=Float32,
                  dims::Union{Integer,Tuple{Vararg{Integer}}}=1_000_000,
                  alphas=(2, -pi, 0, -1, 1),
                  betas=(-2, pi, 0, -1, 1))
    if funcs == (:all,)
        funcs = (:one_norm, :two_norm, :sup_norm, :inner, :scale, :xpby, :axpby)
    end
    x_cpu = rand(T, dims)
    y_cpu = rand(T, dims)
    z_cpu = similar(x_cpu)
    x_gpu = CuArray(x_cpu)
    y_gpu = CuArray(y_cpu)
    z_gpu = similar(x_gpu)
    x_vec = x_cpu[:]
    y_vec = y_cpu[:]
    n = length(x_cpu)
    pinthreads(:cores)
    ENV["OPENBLAS_NUM_THREADS"] = 1
    if :one_norm ∈ funcs
        println()
        println("1-norm (T=$T, n=$n)")
        print(" norm(x_vec, 1)  "); @btime norm($x_vec, 1)
        print(" one_norm(x_cpu) "); @btime one_norm($x_cpu)
        print(" one_norm(x_gpu) "); @btime one_norm($x_gpu)
    end
    if :two_norm ∈ funcs
        println()
        println("2-norm (T=$T, n=$n)")
        print(" norm(x_vec, 2)  "); @btime norm($x_vec, 2)
        print(" two_norm(x_cpu) "); @btime two_norm($x_cpu)
        print(" two_norm(x_gpu) "); @btime two_norm($x_gpu)
    end
    if :sup_norm ∈ funcs
        println()
        println("sup-norm (T=$T, n=$n)")
        print(" norm(x_vec, Inf) "); @btime norm($x_vec, Inf)
        print(" sup_norm(x_cpu)  "); @btime sup_norm($x_cpu)
        print(" sup_norm(x_gpu)  "); @btime sup_norm($x_gpu)
    end
    if :inner ∈ funcs
        println()
        println("inner (T=$T, n=$n)")
        print(" dot(x_vec, y_vec)   "); @btime dot($x_vec, $y_vec)
        print(" inner(x_cpu, y_cpu) "); @btime inner($x_cpu, $y_cpu)
        print(" inner(x_gpu, y_gpu) "); @btime inner($x_gpu, $y_gpu)
    end
    if :scale ∈ funcs
        for α in alphas
            println()
            println("scale! (T=$T, n=$n, α=$α)")
            print(" scale!(z_cpu, α, x_cpu) "); @btime scale!($z_cpu, $α, $x_cpu)
            print(" scale!(z_gpu, α, x_gpu) "); @btime scale!($z_gpu, $α, $x_gpu)
        end
    end
    if :xpby ∈ funcs
        for β in betas
            println()
            println("xpby! (T=$T, n=$n, β=$β)")
            print(" xpby!(z_cpu, x_cpu, β, y_cpu) "); @btime xpby!($z_cpu, $x_cpu, $β, $y_cpu)
            print(" xpby!(z_gpu, x_gpu, β, y_gpu) "); @btime xpby!($z_gpu, $x_gpu, $β, $y_gpu)
        end
    end
    if :axpby ∈ funcs
        for α in alphas, β in betas
            println()
            println("axpby! (T=$T, n=$n, α=$α, β=$β)")
            print(" axpby!(z_cpu, α, x_cpu, β, y_cpu) "); @btime axpby!($z_cpu, $α, $x_cpu, $β, $y_cpu)
            print(" axpby!(z_gpu, α, x_gpu, β, y_gpu) "); @btime axpby!($z_gpu, $α, $x_gpu, $β, $y_gpu)
        end
    end
end

end # module
