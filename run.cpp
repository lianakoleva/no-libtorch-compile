#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>

int main() {
  void *handle = dlopen("foo.so", RTLD_NOW);
  if (handle == NULL) {
    fprintf(stderr, "Failed to open shared library: %s\n", dlerror());
    exit(1);
  }
  printf("handle is %p", handle);
  return 0;
}
