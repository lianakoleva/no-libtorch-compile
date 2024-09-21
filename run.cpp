#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>

extern "C" {
  int32_t aoti_torch_device_type_cuda() {
    return 1;
  }
}

int main() {
  dlopen(NULL, RTLD_GLOBAL);
  void *handle = dlopen("foo.so", RTLD_NOW | RTLD_GLOBAL);
  if (handle == NULL) {
    fprintf(stderr, "Failed to open shared library: %s\n", dlerror());
    exit(1);
  }
  printf("handle is %p", handle);
  return 0;
}
