#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string>
#include <iostream>

extern "C" {
  struct AtenTensorHandle {
    void* data_ptr;
  };


  using AOTIRuntimeError = int32_t;
  using AOTITorchError = int32_t;

  struct AOTInductorModelContainerOpaque;
  using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;

  AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir);

  AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

  int32_t aoti_torch_device_type_cuda() {
    return 1;
  }
  int32_t aoti_torch_grad_mode_is_enabled() {
    return false;
  }
  void aoti_torch_grad_mode_set_enabled(bool enabled) {
     std::cout << __func__ << " vous est ici!" << std::endl;
  }
  AOTITorchError aoti_torch_get_data_ptr(
    AtenTensorHandle tensor,
    void** ret_data_ptr) {
    *ret_data_ptr = tensor.data_ptr;
  }
  AOTITorchError aoti_torch_get_storage_offset(
    AtenTensorHandle tensor,
    int64_t* ret_storage_offset) {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  AOTITorchError aoti_torch_get_strides(
    AtenTensorHandle tensor,
    int64_t** ret_strides) {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  int aoti_torch_device_type_cpu() {
    return 0;
  }
  AOTITorchError aoti_torch_delete_tensor_object(AtenTensorHandle tensor) {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  AOTITorchError aoti_torch_get_storage_size(
    AtenTensorHandle tensor,
    int64_t* ret_size) {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard) {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
  AOTITorchError aoti_torch_create_tensor_from_blob(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor)
  {
    ret_new_tensor = new AtenTensorHandle();
    ret_new_tensor->data_ptr = (void*)1234;
    return 0;
  }
  int32_t aoti_torch_dtype_float32() {
    return 6;
  }
  AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor) {
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
  }
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

  size_t num_inputs;
  AOTInductorModelContainerGetNumInputs(
    container_handle_,
    &num_inputs);
  std::cout<<num_inputs<<std::endl;


  AtenTensorHandle* inputs;
  AtenTensorHandle* outputs;

  


  return 0;
}
