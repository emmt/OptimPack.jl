#! /bin/bash
#
# This is a simple shell script to generate the "build.jl" or
# "install_tarballs.jl" scripts with all components and hash codes.
#

name=OptimPack
version="3.1.0"
url="https://github.com/emmt/OptimPackBuilder/releases/download/v${version}"
base="${name}.v${version}"
sfx="tar.gz"
types="aarch64-linux-gnu aarch64-linux-musl armv7l-linux-gnueabihf armv7l-linux-musleabihf i686-linux-gnu i686-linux-musl x86_64-linux-gnu x86_64-linux-musl powerpc64le-linux-gnu x86_64-unknown-freebsd11.1 x86_64-apple-darwin14 i686-w64-mingw32 x86_64-w64-mingw32"

# Directory to download precompiled libraries.
dir="precompiled"

# Name of generated script.
dest="install_tarballs.jl"
auto="${dest}.auto"

die() {
    echo >&2 "$*"
    exit 1
}

mkdir -p "$dir"
echo
echo "Download precompiled libraries."
for T in $types; do
    file="${base}.${T}.${sfx}"
    test -r "${dir}/${file}" && continue
    echo "  Downloading ${dir}/${file}..."
    wget -O "${dir}/${file}" -N "${url}/${file}"
done

cat >"$auto" <<EOF
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
bin_prefix = "$url"
download_info = Dict(
EOF
echo
echo "Generate checksums."
for T in $types; do
    file="${base}.${T}.${sfx}"
    test -r "${dir}/${file}" || continue
    echo >&2 "  Processing ${file}..."
    cpu="unknown"
    case "$T" in
        aarch64-*)
            cpu=aarch64
            ;;
        armv7l-*)
            cpu=armv7l
            ;;
        i686-*)
            cpu=i686
            ;;
        powerpc64le-*)
            cpu=powerpc64le
            ;;
        x86_64-*)
            cpu=x86_64
            ;;
        *)
            die "unknown CPU in $T"
    esac
    ident=""
    case "$T" in
        *-linux-gnu*)
            ident="Linux(:${cpu}; libc=:glibc)"
            ;;
        *-linux-musl*)
            ident="Linux(:${cpu}; libc=:musl)"
            ;;
        *-freebsd*)
            ident="FreeBSD(:${cpu})"
            ;;
        *-darwin*)
            ident="MacOS(:${cpu})"
            ;;
        *-w64-mingw32)
            ident="Windows(:${cpu})"
            ;;
        *)
            die "unknown OS/C-lib in $T"
    esac
    hash=$(sha256sum -b "${dir}/${file}" | sed 's/ .*$//')
    echo >>"$auto" "    ${ident} => (\"\$bin_prefix/$file\", \"$hash\"),"
done
cat >>"$auto" <<EOF
)
# First, check to see if we're all satisfied
if any(!satisfied(p; verbose=verbose) for p in products)
    try
        # Download and install binaries
        url, tarball_hash = choose_download(download_info)
        install(url, tarball_hash; prefix=prefix, force=true, verbose=true)
    catch e
        if typeof(e) <: ArgumentError
            error("Your platform \$(Sys.MACHINE) is not supported by this package!")
        else
            rethrow(e)
        end
    end

    # Finally, write out a deps.jl file
    write_deps_file(joinpath(@__DIR__, "deps.jl"), products, verbose=true)
end
EOF

echo
echo "File \"${auto}\" has been generated."
echo "Check it and copy it to replace \"${dest}\"."
echo
