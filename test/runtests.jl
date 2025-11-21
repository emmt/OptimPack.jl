module OptimPackTests

using OptimPack
using Test, Printf, Neutrals

include("simplex-tests.jl")

if false
include("brent-tests.jl")
include("rosenbrock.jl")
include("spg2-tests.jl")
end

@testset "COBYLA" begin
    include("cobyla-tests.jl")
    @testset "scale=$scale, revcom=$revcom, inplace=$inplace" for (
        scale, inplace, revcom) in ((  𝟙, false, false),
                                    (0.5, false, false),
                                    (  𝟙, true,  false),
                                    (3.0, true,  false),
                                    (  𝟙, false, true),
                                    (0.1, false, true))
        CobylaTests.runtests(; verbose=0, scale=scale, inplace=inplace, revcom=revcom)
    end
end

@testset "NEWUOA" begin
    include("newuoa-tests.jl")
    @testset "scale=$scale, revcom=$revcom, inplace=$inplace" for (
        scale, inplace, revcom) in ((  𝟙, false, false),
                                    (0.5, false, false),
                                    (  𝟙, true,  false),
                                    (3.0, true,  false),
                                    (  𝟙, false, true),
                                    (0.1, false, true))
        NewuoaTests.runtests(; verbose=0, scale=scale, inplace=inplace, revcom=revcom)
    end
end

@testset "BOBYQA" begin
    include("bobyqa-tests.jl")
    @testset "scale=$scale, inplace=$inplace" for (
        scale, inplace) in ((  𝟙, false),
                            (0.5, false),
                            (  𝟙, true),
                            (3.0, true))
        BobyqaTests.runtests(; verbose=0, scale=scale, inplace=inplace)
    end
end

end # module OptimPackTests

nothing
