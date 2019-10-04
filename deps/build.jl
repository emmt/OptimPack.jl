using Compat.Libdl
let olddir = pwd(), newdir = @__DIR__
    cd(newdir)
    try
        run(`make EXT=$(Libdl.dlext)`)
    finally
        cd(olddir)
    end
end
