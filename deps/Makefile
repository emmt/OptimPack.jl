#
# Makefile to install OptimPack for Julia.
#
# There are two possibilities:
#
# 1. You want to use a pre-installed dynamic library, then:
#
#        make OPTIMPACK_DLL=...
#
#    with ... the absolute path to the dynamic library.
#
# 2. You want to build and install the dynamic library, then:
#
#        make EXT=...
#
#    with ... the file extension, e.g. "so", of dynamic libraries on your
#    system.
#
# If OPTIMPACK_DLL and EXT are both defined, the former takes precedence.
#
# Target can be nothing (as in above examples) or "install" or "reinstall" to
# force rebuilding the file "deps.jl".
#

VERSION=3.0.1
ARCHIVE=optimpack-$(VERSION).tar.gz
URL="https://github.com/emmt/OptimPack/releases/download/v$(VERSION)/$(ARCHIVE)"

# If OPTIMPACK_DLL is defined, just build deps.jl file.  Otherwise, EXT must be defined and
# the library is downloaded, configured, built and installed.
default: install

install:
	if test "$(OPTIMPACK_DLL)" != ""; then \
	    make deps.jl OPTIMPACK_DLL="$(OPTIMPACK_DLL)"; \
	elif test "$(EXT)" != ""; then \
	    make compile-and-install; \
	    make deps.jl OPTIMPACK_DLL="$$PWD/usr/lib/libopk.$(EXT)"; \
	else \
	    echo >&2 "Error: neither EXT nor OPTIMPACK_DLL defined"; \
	    false; \
	fi

reinstall:
	rm -f deps.jl
	make EXT="$(EXT)" OPTIMPACK_DLL="$(OPTIMPACK_DLL)"

clean:
	rm -f *~

distclean: clean
	rm -f deps.jl

downloads/$(ARCHIVE):
	mkdir -p downloads; \
	if curl --version >/dev/null 2>&1; then \
	    curl -L -o "$@" "$(URL)"; \
	elif wget --version >/dev/null 2>&1; then \
	    wget -O "$@" "$(URL)"; \
	else \
	    echo >&2 "Error: neither curl nor wget found"; \
	    false; \
	fi

src/%: downloads/%.tgz
	mkdir -p src && tar zxvf $< -C src

src/%: downloads/%.tar.gz
	mkdir -p src && tar zxvf $< -C src

src/%: downloads/%.tar.bz2
	mkdir -p src && tar jxvf $< -C src

src/%: downloads/%.tar.xz
	mkdir -p src && tar Jxvf $< -C src

src/%: downloads/%.txz
	mkdir -p src && tar Jxvf $< -C src

configure: build/config.status

build/config.status: src/optimpack-$(VERSION)
	prefix="$$PWD/usr"; \
	mkdir -p build; \
	cd build; \
	../src/optimpack-$(VERSION)/configure --enable-shared --disable-static --prefix="$$prefix"

compile-and-install: configure
	cd build; \
	make install

deps.jl: deps.jl.in
	if test "$(OPTIMPACK_DLL)" != "" -a -r "$(OPTIMPACK_DLL)"; then \
	    sed <$^ >$@ -e "s|^ *@checked_lib  *opklib .*|@checked_lib opklib \"$(OPTIMPACK_DLL)\"|"; \
	else \
	    echo >&2 "Error: OPTIMPACK_DLL is not readable"; \
	    false; \
	fi

.PHONY: default install reinstall clean distclean configure compile-and-install
