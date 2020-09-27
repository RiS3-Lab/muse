/*
   american fuzzy lop - LLVM-mode wrapper for clang
   ------------------------------------------------

   Written by Laszlo Szekeres <lszekeres@google.com> and
              Michal Zalewski <lcamtuf@google.com>

   LLVM integration design comes from Laszlo Szekeres.

   Copyright 2015, 2016 Google Inc. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at:

     http://www.apache.org/licenses/LICENSE-2.0

   This program is a drop-in replacement for clang, similar in most respects
   to ../afl-gcc. It tries to figure out compilation mode, adds a bunch
   of flags, and then calls the real compiler.

 */

#define INSTR_MAIN

#include "config.h"
#include "types.h"
#include "debug.h"
#include "alloc-inl.h"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/time.h>

static u8*  obj_path;               /* Path to runtime libraries         */

/* Try to find the runtime libraries. If that fails, abort. */

static void find_obj(u8* argv0) {

  u8 *instr_path = getenv("DYN_INSTR_PATH");
  u8 *slash, *tmp;

  if (instr_path) {

    tmp = alloc_printf("%s/dyn-instr-rt.o", instr_path);

    if (!access(tmp, R_OK)) {
      obj_path = instr_path;
      ck_free(tmp);
      return;
    }

    ck_free(tmp);

  }

  slash = strrchr(argv0, '/');

  if (slash) {

    u8 *dir;

    *slash = 0;
    dir = ck_strdup(argv0);
    *slash = '/';

    tmp = alloc_printf("%s/dyn-instr-rt.o", dir);

    if (!access(tmp, R_OK)) {
      obj_path = dir;
      ck_free(tmp);
      return;
    }

    ck_free(tmp);
    ck_free(dir);

  }

  if (!access(DYN_INSTR_PATH "/dyn-instr-rt.o", R_OK)) {
    obj_path = instr_path;
    return;
  }

  FATAL("Unable to find 'dyn-instr-rt.o' or 'dyn-instr-pass.so'. Please set INSTR_PATH");
}

/* Copy argv to cc_params, making the necessary edits. */

typedef struct { u8** cc_params; u8 print; } params;

static params edit_params(u32 argc, char** argv, int make_bitcode) {
  u32  cc_par_cnt = 1;         /* Param count, including argv0      */

  u8 x_set = 0, maybe_linking = 1, bit_mode = 0, print = 0;
  u8 *name;

  u8** cc_params = ck_alloc((argc + 128) * sizeof(u8*));

  name = strrchr(argv[0], '/');
  if (!name) name = argv[0]; else name++;

  if (!strcmp(name, "dyn-instr-clang++")) {
    u8* alt_cxx = getenv("AFL_CXX");
    cc_params[0] = alt_cxx ? alt_cxx : (u8*)"clang++";
  } else {
    u8* alt_cc = getenv("AFL_CC");
    cc_params[0] = alt_cc ? alt_cc : (u8*)"clang";
  }


  cc_params[cc_par_cnt++] = "-Xclang";
  cc_params[cc_par_cnt++] = "-load";
  cc_params[cc_par_cnt++] = "-Xclang";
  cc_params[cc_par_cnt++] = alloc_printf("%s/dyn-instr-pass.so", obj_path);

  cc_params[cc_par_cnt++] = "-Qunused-arguments";

  /* Detect stray -v calls from ./configure scripts. */

  if (argc == 1 && !strcmp(argv[1], "-v")) maybe_linking = 0;

  while (--argc) {
    u8* cur = *(++argv);

    if (!strcmp(cur, "-o") && make_bitcode)
      argv[1] = alloc_printf("%s.bc", argv[1]);

    if (!strcmp(cur, "-m32")) bit_mode = 32;
    if (!strcmp(cur, "-m64")) bit_mode = 64;

    if (!strcmp(cur, "-x")) x_set = 1;

    if (!strcmp(cur, "-c") || !strcmp(cur, "-S") || !strcmp(cur, "-E"))
      maybe_linking = 0;

    if (!strcmp(cur, "-shared")) maybe_linking = 0;

    if (!strcmp(cur, "-print-prog-name=ld") ||
        !strcmp(cur, "-print-search-dirs")  ||
        !strcmp(cur, "-print-multi-os-directory") ||
        !strcmp(cur, "--version") ||
        !strcmp(cur, "-V")) print = 1;

    if (!strcmp(cur, "-Wl,-z,defs") ||
        !strcmp(cur, "-Wl,--no-undefined")) continue;

    if (!strcmp(cur, "-fsanitize=address"))
        SAYF("MUSE have built-in support for memory operation, consider droping ASAN flag\n");

    cc_params[cc_par_cnt++] = cur;

  }

  if (getenv("INSERT_LAVA_LABEL")) {
    cc_params[cc_par_cnt++] = "-g";
    cc_params[cc_par_cnt++] = "-O0";
    cc_params[cc_par_cnt++] = "-funroll-loops";
    cc_params[cc_par_cnt++] = "-fno-optimize-sibling-calls";
  }else
  if (!getenv("AFL_DONT_OPTIMIZE")) {
    cc_params[cc_par_cnt++] = "-g";
    cc_params[cc_par_cnt++] = "-O3";
    cc_params[cc_par_cnt++] = "-funroll-loops";
  }

  if(make_bitcode) {
    // TODO: what if -c is already specified?
    cc_params[cc_par_cnt++] = "-c";
    cc_params[cc_par_cnt++] = "-emit-llvm";
    cc_params[cc_par_cnt++] = "-fno-vectorize";
    cc_params[cc_par_cnt++] = "-fno-slp-vectorize";
    cc_params[cc_par_cnt++] = "-fno-slp-vectorize";
  }

  if (getenv("NO_BUILTIN")) {

    cc_params[cc_par_cnt++] = "-fno-builtin-strcmp";
    cc_params[cc_par_cnt++] = "-fno-builtin-strncmp";
    cc_params[cc_par_cnt++] = "-fno-builtin-strcasecmp";
    cc_params[cc_par_cnt++] = "-fno-builtin-strncasecmp";
    cc_params[cc_par_cnt++] = "-fno-builtin-memcmp";

  }

  cc_params[cc_par_cnt++] = "-D__AFL_HAVE_MANUAL_CONTROL=1";
  cc_params[cc_par_cnt++] = "-D__AFL_COMPILER=1";
  cc_params[cc_par_cnt++] = "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION=1";


  cc_params[cc_par_cnt++] = "-D__AFL_INIT()="
    "do { static volatile char *_A __attribute__((used)); "
    " _A = (char*)\"" DEFER_SIG "\"; "
#ifdef __APPLE__
    "__attribute__((visibility(\"default\"))) "
    "void _I(void) __asm__(\"__manual_init\"); "
#else
    "__attribute__((visibility(\"default\"))) "
    "void _I(void) __asm__(\"__manual_init\"); "
#endif /* ^__APPLE__ */
    "_I(); } while (0)";

  if (maybe_linking) {

    if (x_set) {
      cc_params[cc_par_cnt++] = "-x";
      cc_params[cc_par_cnt++] = "none";
    }

    switch (bit_mode) {

      case 0:
        cc_params[cc_par_cnt++] = alloc_printf("%s/dyn-instr-rt.o", obj_path);
        break;

      case 32:
        cc_params[cc_par_cnt++] = alloc_printf("%s/dyn-instr-rt-32.o", obj_path);

        if (access(cc_params[cc_par_cnt - 1], R_OK))
          FATAL("-m32 is not supported by your compiler");

        break;

      case 64:
        cc_params[cc_par_cnt++] = alloc_printf("%s/dyn-instr-rt-64.o", obj_path);

        if (access(cc_params[cc_par_cnt - 1], R_OK))
          FATAL("-m64 is not supported by your compiler");

        break;

    }

  }

  cc_params[cc_par_cnt] = NULL;

  params pp;
  pp.cc_params = cc_params;
  pp.print = print;
  return pp;

}

void dumpEditedParams(u8** params) {
  u8 **tmp = params;
  while ((*tmp) != NULL) {
    u8* cur = *(tmp++);
    printf("%s ", cur);
  }
  printf("\n");
}

/* Main entry point */

int main(int argc, char** argv) {


  if (argc < 2) {

    SAYF("invalid parameter, use dyn-instr as a drop-in replacement for clang\n");
    exit(1);

  }


  // initialize random seed similar to afl-as
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  const unsigned int rand_seed = tv.tv_sec ^ tv.tv_usec ^ getpid();
  char rand_seed_str[20];
  snprintf(rand_seed_str, 20, "%u", rand_seed);
  setenv("RANDOM_SEED", rand_seed_str, 1);

  find_obj(argv[0]);

  params pparams = edit_params(argc, argv, 0);
  u8** cc_params = pparams.cc_params;

  pid_t pid = fork();
  if (pid == -1) {
    FATAL("Oops, failed to fork.");
  } else if (pid == 0) {
    // compile to binary
    OKF("In binary generation process...");
    unsetenv("ANNOTATE_FOR_SE");
    //dumpEditedParams(cc_params);
    execvp(cc_params[0], (char**)cc_params);
    FATAL("Oops, failed to execute '%s' - check your PATH", cc_params[0]);
  } else {
    int status;
    waitpid(pid, &status, 0);
  }

  // compile to llvm ir
  params pp = edit_params(argc, argv, 1);

  if (!pp.print) {
    u8** bc_params = pp.cc_params;
    OKF("In SE IR bitcode generation process...");
    // ANNOTATE_FOR_SE = 1 means symbolic exection specific instrumentation.
    setenv("ANNOTATE_FOR_SE", "1", 1);
    //dumpEditedParams(bc_params);
    execvp(bc_params[0], (char**)bc_params);
    FATAL("Oops, failed to execute '%s' - check your PATH", bc_params[0]);
  }

  return 0;
}
