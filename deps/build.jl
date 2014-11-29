using BinDeps

const urioptimpack = URI("https://github.com/emmt/OptimPack/archive/master.zip")

@BinDeps.setup

optimpack = library_dependency("libOptimPack2")
provides(Sources,urioptimpack,optimpack,unpacked_dir="OptimPack-master")

prefix = joinpath(BinDeps.depsdir(optimpack),"usr")
srcdir = joinpath(BinDeps.depsdir(optimpack),"src","OptimPack-master/src")
@unix_only libfilename = "libOptimPack2.so"
@osx_only libfilename = "libOptimPack2.dylib"
destlib = joinpath(prefix,"lib",libfilename)

provides(SimpleBuild,
         (@build_steps begin
            GetSources(optimpack)
            CreateDirectory(prefix)
            CreateDirectory(joinpath(prefix,"lib"))
            @build_steps begin
                ChangeDirectory(srcdir)
                FileRule(destlib,
                        @build_steps begin
                        `make`
                        `cp libOptimPack2.so.1.0.0 $destlib`
                        end)
            end
         end),
         optimpack)

@BinDeps.install [:libOptimPack2 => :opklib]