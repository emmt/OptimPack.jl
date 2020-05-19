using BinaryProvider

# Parse some basic command-line arguments
const verbose = "--verbose" in ARGS
const prefix = Prefix(get([a for a in ARGS if a != "--verbose"], 1, joinpath(@__DIR__, "usr")))

# These are the several binary objects we care about
products = Product[
    LibraryProduct(prefix, "libopk",    :libopk),
    LibraryProduct(prefix, "libbobyqa", :libbobyqa),
    LibraryProduct(prefix, "libcobyla", :libcobyla),
    LibraryProduct(prefix, "libnewuoa", :libnewuoa),
]

# Download binaries from hosted location
bin_prefix = "https://github.com/emmt/OptimPackBuilder/releases/download/v3.1.0"
download_info = Dict(
    Linux(:aarch64; libc=:glibc) => ("$bin_prefix/OptimPack.v3.1.0.aarch64-linux-gnu.tar.gz", "71e6a481776e9b5f2fed5e7d0ab8915d6acb0f1ad44562ab43f9931ea295c2f6"),
    Linux(:aarch64; libc=:musl) => ("$bin_prefix/OptimPack.v3.1.0.aarch64-linux-musl.tar.gz", "1a61e86921c530252ca40960b4c2441f0b8511e1c7e9e3adbccbd5509e1944f1"),
    Linux(:armv7l; libc=:glibc) => ("$bin_prefix/OptimPack.v3.1.0.armv7l-linux-gnueabihf.tar.gz", "b1dae85bde6686db7dad23c1533df2e4e33c9f84cdaa54573bd2d78e96a2f18a"),
    Linux(:armv7l; libc=:musl) => ("$bin_prefix/OptimPack.v3.1.0.armv7l-linux-musleabihf.tar.gz", "7f964d05429fecd7c20713d3672a15d4019f1f390f94c65282c713d96a0f5bc2"),
    Linux(:i686; libc=:glibc) => ("$bin_prefix/OptimPack.v3.1.0.i686-linux-gnu.tar.gz", "b2a548e7bc570ae493ce2bc9213d083675aaca28a34c429f4e0c2bf060e474d0"),
    Linux(:i686; libc=:musl) => ("$bin_prefix/OptimPack.v3.1.0.i686-linux-musl.tar.gz", "a8e830e01b5772e778111c2151755498fead4bd1e8c51281bce47c245857658a"),
    Linux(:x86_64; libc=:glibc) => ("$bin_prefix/OptimPack.v3.1.0.x86_64-linux-gnu.tar.gz", "f18bb702463eddc1e9219f92d1c8e0db184e8987d664d16bd7e7bfd2dfca0ebd"),
    Linux(:x86_64; libc=:musl) => ("$bin_prefix/OptimPack.v3.1.0.x86_64-linux-musl.tar.gz", "c3c02dde5ef26216706997312d01f80452c9a29b73698347ed7582ee86eabb47"),
    Linux(:powerpc64le; libc=:glibc) => ("$bin_prefix/OptimPack.v3.1.0.powerpc64le-linux-gnu.tar.gz", "1f2f9ea391488ee3362fad35c9ff9d33336c6b3091bff4882ba5a8291603e5bc"),
    FreeBSD(:x86_64) => ("$bin_prefix/OptimPack.v3.1.0.x86_64-unknown-freebsd11.1.tar.gz", "9ddfd26f6295828097f7eeeb38add123f309715f7f59aeca4c8cb737812b4e99"),
    MacOS(:x86_64) => ("$bin_prefix/OptimPack.v3.1.0.x86_64-apple-darwin14.tar.gz", "99dfdf40ad334f430981af2aa5df44e8efd25a475ca2c4d97f38a5066238909c"),
    Windows(:i686) => ("$bin_prefix/OptimPack.v3.1.0.i686-w64-mingw32.tar.gz", "9c78796fe2b58693b610e440aa5390b5fce670dad4250ba4542d79e72636a86c"),
    Windows(:x86_64) => ("$bin_prefix/OptimPack.v3.1.0.x86_64-w64-mingw32.tar.gz", "f839ab4f8d42f398bf089917a4581e1a7e37be5519e5516aec8154da09f6643c"),
)
# First, check to see if we're all satisfied
if any(!satisfied(p; verbose=verbose) for p in products)
    try
        # Download and install binaries
        url, tarball_hash = choose_download(download_info)
        install(url, tarball_hash; prefix=prefix, force=true, verbose=true)
    catch e
        if typeof(e) <: ArgumentError
            error("Your platform $(Sys.MACHINE) is not supported by this package!")
        else
            rethrow(e)
        end
    end

    # Finally, write out a deps.jl file
    write_deps_file(joinpath(@__DIR__, "deps.jl"), products, verbose=true)
end
