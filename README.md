CUDA MODE IRL Project -- WIP

Goal: Compile a PyTorch program into a (dependency-free) binary through torch.compile

APIs to shim:
- [ ] aoti_torch_create_cuda_stream_guard 
- [ ] aoti_torch_create_tensor_from_blob
- [ ] aoti_torch_create_tensor_from_blob_v2
- [ ] aoti_torch_delete_cuda_stream_guard 
- [ ] aoti_torch_delete_tensor_object
- [ ] aoti_torch_device_type_cpu
- [ ] aoti_torch_device_type_cuda
- [ ] aoti_torch_dtype_float32
- [ ] aoti_torch_empty_strided
- [ ] aoti_torch_get_data_ptr 
- [ ] aoti_torch_get_storage_offset
- [ ] aoti_torch_get_storage_size
- [ ] aoti_torch_get_strides 
- [ ] aoti_torch_grad_mode_is_enabled 
- [ ] aoti_torch_grad_mode_set_enabled