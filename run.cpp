#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string>

extern "C" {
  int32_t aoti_torch_device_type_cuda() {
    return 1;
  }
  int32_t aoti_torch_grad_mode_is_enabled() {
    return false;
  }
  void aoti_torch_grad_mode_set_enabled(bool enabled) {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  void aoti_torch_get_data_ptr() {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  void aoti_torch_get_storage_offset() {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  void aoti_torch_get_strides() {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  void aoti_torch_create_tensor_from_blob_v2() {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  void aoti_torch_delete_cuda_stream_guard() {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  int aoti_torch_device_type_cpu() {
    return 0;
  }
  void aoti_torch_delete_tensor_object() {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  void aoti_torch_get_storage_size() {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  void aoti_torch_create_cuda_stream_guard() {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  void aoti_torch_create_tensor_from_blob() {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  int32_t aoti_torch_dtype_float32() {
    return 6;
  }
  void aoti_torch_empty_strided() {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }

  using AOTIRuntimeError = int32_t;

  struct AOTInductorModelContainerOpaque;
  using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;

  extern AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir);
}

int main() {
  AOTInductorModelContainerHandle container_handle_ = nullptr;

  AOTIRuntimeError err;

  err = AOTInductorModelContainerCreateWithDevice(
    &container_handle_,
    1,
    "cuda",
    nullptr);
  printf("container_handle=%p\n", container_handle_);
  return 0;
}
