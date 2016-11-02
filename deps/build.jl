using BinDeps

const version = "2.0.2"
const unpacked_dir = "optimpack-$version"

@BinDeps.setup

optimpack = library_dependency("libopk")

provides(Sources,
         URI("https://github.com/emmt/OptimPack/releases/download/v$version/optimpack-$version.tar.gz"),
         optimpack,
         unpacked_dir=unpacked_dir)

prefix = joinpath(BinDeps.depsdir(optimpack), "usr")
srcdir = joinpath(BinDeps.depsdir(optimpack), "src", unpacked_dir)
@static if is_unix() libfilename = "libopk.so"
end
@static if is_apple() libfilename = "libopk.dylib"
end
destlib = joinpath(prefix, "lib", libfilename)

provides(SimpleBuild,
         (@build_steps begin
             GetSources(optimpack)
             CreateDirectory(prefix)
             CreateDirectory(joinpath(prefix,"lib"))
             @build_steps begin
                 ChangeDirectory(srcdir)
                 FileRule(destlib,
                          @build_steps begin
                              `./configure --enable-shared --disable-static --prefix="$prefix"`
                              `make`
                              `make install`
                          end)
             end
         end),
         optimpack)

@BinDeps.install Dict(:libopk => :opklib)
