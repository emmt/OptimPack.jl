using BinDeps

const version = "master"
const unpacked_dir = "OptimPack-$version"

@BinDeps.setup

optimpack = library_dependency("libopk")

provides(Sources,
         URI("https://github.com/emmt/OptimPack/archive/$version.zip"),
         optimpack,
         unpacked_dir=unpacked_dir)

prefix = joinpath(BinDeps.depsdir(optimpack), "usr")
srcdir = joinpath(BinDeps.depsdir(optimpack), "src", unpacked_dir)

provides(SimpleBuild,
         (@build_steps begin
             GetSources(optimpack)
             CreateDirectory(prefix)
             CreateDirectory(joinpath(prefix,"lib"))
             @build_steps begin
                 ChangeDirectory(srcdir)
                 FileRule(destlib,
                          @build_steps begin
                              `./autogen.sh`
                              `./configure --enable-shared --disable-static --prefix="$prefix"`
                              `make`
                              `make install`
                          end)
             end
         end),
         optimpack)

@BinDeps.install [:libopk => :opklib]
