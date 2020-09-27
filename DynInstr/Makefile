# Copyright 2015, 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#   http://www.apache.org/licenses/LICENSE-2.0
#

PREFIX      ?= /usr/local
HELPER_PATH  = $(PREFIX)/dyn_instr

VERSION     = $(shell grep '^\#define VERSION ' config.h | cut -d '"' -f2)

LLVM_CONFIG ?= llvm-config

CFLAGS      ?= -O3 -funroll-loops 
CFLAGS      += -Wall -D_FORTIFY_SOURCE=2 -g -Wno-pointer-sign \
               -DDYN_INSTR_PATH=\"$(HELPER_PATH)\"  \
               -DVERSION=\"$(VERSION)\"

CXXFLAGS    ?= -O3 -funroll-loops -std=c++11
CXXFLAGS    += -Wall -D_FORTIFY_SOURCE=2 -g -Wno-pointer-sign \
               -DVERSION=\"$(VERSION)\" -Wno-variadic-macros

CLANG_CFL    = `$(LLVM_CONFIG) --cxxflags` -fpic $(CXXFLAGS)
CLANG_LFL    = `$(LLVM_CONFIG) --ldflags` $(LDFLAGS)

# We were using llvm-config --bindir to get the location of clang, but
# this seems to be busted on some distros, so using the one in $PATH is
# probably better.

ifeq "$(origin CC)" "default"
  CC         = clang
  CXX        = clang++
endif

ifndef AFL_TRACE_PC
  PROGS      = dyn-instr-clang dyn-instr-pass.so dyn-instr-rt.o dyn-instr-rt-32.o dyn-instr-rt-64.o
else
  PROGS      = dyn-instr-clang dyn-instr-rt.o dyn-instr-rt-32.o dyn-instr-rt-64.o
endif

all: $(PROGS) all_done

dyn-instr-clang: dyn-instr.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
	ln -sf dyn-instr-clang dyn-instr-clang++

dyn-instr-pass.so: dyn-instr-pass.so.cc
	$(CXX) $(CLANG_CFL) -shared $< -o $@ $(CLANG_LFL)

dyn-instr-rt.o: dyn-instr-rt.o.c
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

dyn-instr-rt-32.o: dyn-instr-rt.o.c
	@printf "[*] Building 32-bit variant of the runtime (-m32)... "
	@$(CC) $(CFLAGS) -m32 -fPIC -c $< -o $@ 2>/dev/null; if [ "$$?" = "0" ]; then echo "success!"; else echo "failed (that's fine)"; fi

dyn-instr-rt-64.o: dyn-instr-rt.o.c
	@printf "[*] Building 64-bit variant of the runtime (-m64)... "
	@$(CC) $(CFLAGS) -m64 -fPIC -c $< -o $@ 2>/dev/null; if [ "$$?" = "0" ]; then echo "success!"; else echo "failed (that's fine)"; fi

all_done:
	@echo "[+] All done! You can now use '../dyn-instr-clang' to compile programs."

.NOTPARALLEL: clean

clean:
	rm -f *.o *.so *~ a.out core core.[1-9][0-9]*
	rm -f $(PROGS)
