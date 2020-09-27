#include "stdio.h"
#include "string.h"
#include <stdlib.h>
#include <assert.h>

#define SIZE 10

char * buffer;

void fill_buffer(FILE *fp) {
  int ch;
  size_t len = 0;
  while(EOF != (ch = fgetc(fp))) {
    buffer[len] = ch;
    len++;
    if (len == SIZE) {
      return;
    }
  }
}

int change_cond(int x) {
  return x*100;
}

int fork_point(int b, char * src, int len) {
  if (b) {
    assert(0);
  } else {
    memcpy(buffer, src, len);
  }
  return 1;
}

int main(int argc, char ** argv) {
  buffer = realloc(NULL, SIZE);
  char* buffer2 = realloc(NULL, SIZE);
  if (!buffer) return -1;
  if (!buffer2) return-1;
  fill_buffer(stdin);


  int cond = change_cond(buffer[1]-'1');
  const char* bad = "12345678911";
  if (cond == 200) {
    //    memcpy(buffer, bad, 11);
    fork_point(buffer[3]-'4', bad, 11);
  } else {
    memcpy(buffer, bad, 11);
  }
  return 0;

}
