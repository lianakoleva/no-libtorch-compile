#include <stdio.h>
#include "cuda_runtime.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string>
#include <iostream>

extern "C" {
  struct AtenTensor {
    void* data_ptr;
  };
  using AtenTensorHandle = AtenTensor*;

  struct CUDAStreamGuardOpaque;
  using CUDAStreamGuardHandle = CUDAStreamGuardOpaque*;

  using AOTIRuntimeError = int32_t;
  using AOTITorchError = int32_t;

  struct AOTInductorModelContainerOpaque;
  using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;
  using AOTInductorStreamHandle = void*;
  using AOTIProxyExecutorHandle = void*;

  AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir);

  AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

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
    *ret_data_ptr = tensor->data_ptr;
   return 0;
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
    return 0;
  }
  int aoti_torch_device_type_cpu() {
    return 0;
  }
  AOTITorchError aoti_torch_delete_tensor_object(AtenTensorHandle tensor) {
    std::cout<<"Deleting "<<tensor<<std::endl;
    delete tensor;
    return 0;
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
    return 0;
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
    std::string msg = std::string(__func__) + " DNE";
    perror(msg.c_str());
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
    *ret_new_tensor = new AtenTensor();
    std::cout<<"Created "<<ret_new_tensor<<std::endl;

    (*ret_new_tensor)->data_ptr = (void*)123;
    return 0;
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

  int64_t sizes[2] = {32, 32};
  int64_t strides[2] = {32, 1};
  AtenTensorHandle inputs[2];
  aoti_torch_empty_strided(2, sizes, strides, 5, 1, 0, inputs + 0);
  aoti_torch_empty_strided(2, sizes, strides, 5, 1, 0, inputs + 1);
  AtenTensorHandle outputs[1] = {nullptr};
  AOTInductorModelContainerRun(container_handle_, inputs, 2, outputs, 1, nullptr, nullptr);

  cudaError_t code = cudaDeviceSynchronize();
  if (code != cudaSuccess) { 
    std::cerr << cudaGetErrorString(code) << std::endl;
    return -1;
  }


  return 0;
}
