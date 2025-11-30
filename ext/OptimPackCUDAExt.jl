module OptimPackCUDAExt

if isdefined(Base, :get_extension)
    using CUDA
    using OptimPack
else
    using ..CUDA
    using ..OptimPack
end

using OptimPack: one_norm, two_norm, sup_norm, inner, unsafe_inner
using OptimPack: unsafe_scale!, unsafe_axpby!, unsafe_xpby!
using OptimPack.LoopStyles: GPU, Dot, Map

OptimPack.LoopStyle(::Type{<:CuArray}) = GPU{:CUDA}()

OptimPack.one_norm(::GPU{:CUDA}, x::CuArray) = one_norm(Map(), x)

OptimPack.two_norm(::GPU{:CUDA}, x::CuArray) = two_norm(Map(), x)

OptimPack.sup_norm(::GPU{:CUDA}, x::CuArray) = sup_norm(Map(), x)

OptimPack.unsafe_inner(::GPU{:CUDA}, x::CuArray, y::CuArray) = unsafe_inner(Map(), x, y)

function OptimPack.unsafe_xpby!(::GPU{:CUDA}, dst::CuArray, x::CuArray,
                                β::Number, y::CuArray)
    unsafe_xpby!(Map(), dst, x, β, y)
end

function OptimPack.unsafe_axpby!(::GPU{:CUDA}, dst::CuArray,
                                 α::Number, x::CuArray,
                                 β::Number, y::CuArray)
    unsafe_axpby!(Map(), dst, α, x, β, y)
end

end # module
