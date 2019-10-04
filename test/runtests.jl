module OptimPackTests
using OptimPack
using Compat.Test
using Compat.Printf

VERBOSE = true

include("rosenbrock.jl")
include("cobyla-tests.jl")

nothing
end
