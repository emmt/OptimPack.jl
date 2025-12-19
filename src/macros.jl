"""
    OptimPack.@pass ex -> ex

Return expression `ex` unchanged.

"""
macro pass(ex)
    esc(ex)
end
