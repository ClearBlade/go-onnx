// Copyright (c) ClearBlade Inc. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNX_H
#define ONNX_H

#include "onnxruntime_c_api.h"

// describes a single output tensor
typedef struct TensorInfo {
    ONNXTensorElementDataType tensor_type;
    void* data;
    int64_t* shape_ptr;
    size_t shape_len;
} TensorInfo;

typedef struct SequenceInfo {
    size_t value_count;
    enum ONNXType value_type;
} SequenceInfo;

typedef struct StringTensorInfo {
    void* data;
    size_t* offsets;
} StringTensorInfo;

typedef struct OnnxRuntime {
    OrtEnv* env;
    OrtSession* session;
	OrtMemoryInfo* memory_info;
    OrtAllocator* allocator;
} OnnxRuntime;

typedef struct OrtReturn {
    void* value;
    size_t count;
    OrtStatus* status;
} OrtReturn;

typedef struct IOInfo {
    char* name;
    enum ONNXType value_type;
    ONNXTensorElementDataType tensor_type;
    size_t dim_count;
    int64_t* shape;
} IOInfo;

void init_api();
const char* error_message(OrtStatus* ret);
void free_status(OrtStatus* status);
void load_model(void* data, size_t data_len, OrtReturn* ret);
void make_c_tensor(OnnxRuntime* runtime, void* input, size_t input_len, int64_t* input_shape, size_t input_shape_len, ONNXTensorElementDataType dtype, OrtReturn* ret);
void run(OnnxRuntime* runtime, OrtValue** input_tensors, size_t input_len, char** input_names, size_t input_names_len, char** output_names, size_t output_names_len, OrtReturn *ret);
void cleanup_runtime(OnnxRuntime* runtime);

enum ONNXType value_type(OrtValue* value);

void get_tensor_info(OrtValue* value, OrtReturn* ret);
void free_tensor_info(TensorInfo* info);

void get_sequence_info(OrtValue* value, OrtReturn* ret);
void free_sequence_info(SequenceInfo* info);
void get_sequence_element(OnnxRuntime* runtime, OrtValue* sequence, size_t index, OrtReturn* ret);

void get_map_keys(OnnxRuntime* runtime, OrtValue* map, OrtReturn* ret);

size_t get_string_tensor_data_len(OrtValue* string_tensor);
void get_string_tensor_data(OrtValue* string_tensor, size_t num_elements, size_t data_len, OrtReturn* ret);
void free_string_tensor_data(StringTensorInfo* info);

void get_child_value(OnnxRuntime* runtime, OrtValue* parent, size_t index, OrtReturn* ret);
void free_value(OrtValue* value);

void get_io_count(OnnxRuntime* runtime, size_t type, OrtReturn* ret);
void get_io_info(OnnxRuntime* runtime, size_t type, size_t index, OrtReturn* ret);
void free_io_info(OnnxRuntime* runtime, IOInfo* info);

#endif // ONNX_H
