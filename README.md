CUDA MODE IRL Project -- WIP

**Background:**

- AOTInductor is a specialized version of TorchInductor , designed to process exported PyTorch models, optimize them, and produce shared libraries as well as other relevant artifacts. 

- These compiled artifacts are specifically crafted for deployment in non-Python environments, which are frequently employed for inference deployments on the server side.

- A somewhat little known feature is that torch.compile can convert a PyTorch program into a C++ .so binary file. 

- We can then load the shared library, enabling us to conduct model predictions directly within a C++ environment.

- Unfortunately, today, running it still requires a libtorch dependency (very large filesize especially with CUDA). 

- Upon inspection of the symbol table and other attributes of the binary , we see that there is a fairly limited amount of APIs that need to be shimmed.

**Goal: Compile a PyTorch program into a (dependency-free) binary through torch.compile()**

- Started with a simple transpose kernel, then tried to expand support to Bert Maher's llama2.so project
  
- AOTInductor internals, patching ELF files, dynamic linking, ABI-compatibility

- lots of simplifying assumptions (no support for dynamic shapes)



APIs to shim:
- [X] aoti_torch_create_cuda_stream_guard 
- [X] aoti_torch_create_tensor_from_blob
- [X] aoti_torch_create_tensor_from_blob_v2
- [X] aoti_torch_delete_cuda_stream_guard 
- [X] aoti_torch_delete_tensor_object
- [X] aoti_torch_device_type_cpu
- [x] aoti_torch_device_type_cuda
- [x] aoti_torch_dtype_float32
- [X] aoti_torch_empty_strided
- [X] aoti_torch_get_data_ptr 
- [X] aoti_torch_get_storage_offset
- [X] aoti_torch_get_storage_size
- [X] aoti_torch_get_strides 
- [x] aoti_torch_grad_mode_is_enabled 
- [x] aoti_torch_grad_mode_set_enabled
