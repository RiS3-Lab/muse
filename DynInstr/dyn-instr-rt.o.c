/*
   LLVM instrumentation bootstrap
   ---------------------------------------------------

   Copyright 2015, 2016 Google Inc. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at:

     http://www.apache.org/licenses/LICENSE-2.0

*/

#include "config.h"
#include "types.h"

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

#include <sys/types.h>

#define CONST_PRIO 0

static FILE* loc_trace_file = NULL;
static FILE* label_cov_file = NULL;
static FILE* trigger_label_cov_file = NULL;
static FILE* cmp_log_file = NULL;
static FILE* indcall_log_file = NULL;
static FILE* extcall_log_file = NULL;
static FILE* memop_log_file = NULL;
__thread u32 __afl_prev_loc;

/* log file setup. */

static void __prep_log(void) {

  char* loc_trace_file_name = getenv("AFL_LOC_TRACE_FILE");
  if (loc_trace_file_name) {
    loc_trace_file = fopen(loc_trace_file_name, "w");
  }
  char* label_cov_file_name = getenv("SAVIOR_LABEL_LOG");
  if (label_cov_file_name) {
    label_cov_file = fopen(label_cov_file_name, "w");
  }
  char* trigger_label_cov_file_name = getenv("SAVIOR_TRIGGER_LABEL_LOG");
  if (trigger_label_cov_file_name) {
    trigger_label_cov_file = fopen(trigger_label_cov_file_name, "w");
  }
  char* cmp_log_file_name = getenv("CMP_LOG");
  if (cmp_log_file_name)  {
      cmp_log_file = fopen(cmp_log_file_name, "w");
  }
  char* indcall_log_file_name = getenv("INDIRECT_CALL_LOG");
  if (indcall_log_file_name)  {
      indcall_log_file = fopen(indcall_log_file_name, "w");
  }
  char* extcall_log_file_name = getenv("EXTERNAL_CALL_LOG");
  if (extcall_log_file_name)  {
      extcall_log_file = fopen(extcall_log_file_name, "w");
  }
  char* memop_log_file_name = getenv("MEM_OP_LOG");
  if (memop_log_file_name)  {
      memop_log_file = fopen(memop_log_file_name, "w");
  }
}

void __afl_log_loc(u32 cur_loc, u32 xor) {
	if (loc_trace_file) {
		fprintf(loc_trace_file, "%d,%d\n", cur_loc, xor);
	}
}

void __afl_log_label(u32 cur_loc) {
	if (label_cov_file) {
		fprintf(label_cov_file, "%d\n", cur_loc);
	}
}

void __afl_log_memop(u32 cur_loc) {
    if (memop_log_file) {
        fprintf(memop_log_file, "%d\n", cur_loc);
    }
}

void __afl_log_triggered(u32 cur_loc) {
	if (trigger_label_cov_file) {
		fprintf(trigger_label_cov_file, "%d\n", cur_loc);
	}
}

void __afl_log_cmp(int cmp_num) {
    if (cmp_log_file) {
        fprintf(cmp_log_file, "%d\n", cmp_num);
    }
}

void __afl_log_ext_call(int ec_num) {
    if (extcall_log_file) {
        fprintf(extcall_log_file, "%d\n", ec_num);
    }
}

void __afl_log_ind_call(int ic_num) {
    if (indcall_log_file) {
        fprintf(indcall_log_file, "%d\n", ic_num);
    }
}

/* Proper initialization routine. */

__attribute__((constructor(CONST_PRIO))) void __auto_init(void) {

    static u8 init_done;

    if (!init_done) {

        __prep_log();
        init_done = 1;

    }

}
