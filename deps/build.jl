case = ((haskey(ENV, "OPTIMPACK_OPK_LIB")    ? 1 : 0) |
        (haskey(ENV, "OPTIMPACK_COBYLA_LIB") ? 2 : 0) |
        (haskey(ENV, "OPTIMPACK_BOBYQA_LIB") ? 4 : 0) |
        (haskey(ENV, "OPTIMPACK_NEWUOA_LIB") ? 8 : 0))
if case == 15
    open(joinpath(@__DIR__, "deps.jl"), "w") do io
        println(io, """
# Deal with compatibility issues.
if VERSION >= v"0.7.0-DEV.3382"
    using Libdl
end

# Macro to load a library.
macro checked_lib(libname, path)
    if Libdl.dlopen_e(path) == C_NULL
        error("Unable to load \\n\\n\$libname (\$path)\\n\\nPlease ",
              "re-run Pkg.build(package), and restart Julia.")
    end
    quote
        const \$(esc(libname)) = \$path
    end
end

# Libraries.
@checked_lib libopk    "$(ENV["OPTIMPACK_OPK_LIB"])"
@checked_lib libcobyla "$(ENV["OPTIMPACK_COBYLA_LIB"])"
@checked_lib libbobyqa "$(ENV["OPTIMPACK_BOBYQA_LIB"])"
@checked_lib libnewuoa "$(ENV["OPTIMPACK_NEWUOA_LIB"])"
""")
    end
else
    if case != 0
        println("WARNING: some but not all environment variables for OptimPack libraries have been provided, installation will assume none of them")
    end
    include("install_tarballs.jl")
end
