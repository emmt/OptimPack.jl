module OptimPackLoopVectorizationExt

if isdefined(Base, :get_extension)
    using LoopVectorization
    using OptimPack
else
    using ..LoopVectorization
    using ..OptimPack
end

using .OptimPack: @pass, LoopStyleTurbo, recode!
import .OptimPack: one_norm, two_norm, sup_norm, inner, unsafe_inner
import .OptimPack: unsafe_scale!, unsafe_xpby!, unsafe_axpby!

@eval $(recode!(OptimPack.one_norm_simd(),      OptimPack.simd_to_turbo...))
@eval $(recode!(OptimPack.two_norm_simd(),      OptimPack.simd_to_turbo...))
@eval $(recode!(OptimPack.sup_norm_simd(),      OptimPack.simd_to_turbo...))
@eval $(recode!(OptimPack.unsafe_inner_simd(),  OptimPack.simd_to_turbo...))
@eval $(recode!(OptimPack.unsafe_scale!_simd(), OptimPack.simd_to_turbo...))
@eval $(recode!(OptimPack.unsafe_xpby!_simd(),  OptimPack.simd_to_turbo...))
@eval $(recode!(OptimPack.unsafe_axpby!_simd(), OptimPack.simd_to_turbo...))

end # module
