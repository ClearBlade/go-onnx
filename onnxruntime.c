// Copyright (c) ClearBlade Inc. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#include "onnxruntime.h"
#include <stdio.h>

// C API docs: https://onnxruntime.ai/docs/api/c/struct_ort_api.html
const OrtApi* g_ort = NULL;

void init_api() {
	g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	setbuf(stdout, NULL);
}
//JOHN DEBUGGING
void CheckStatus(OrtStatus* status)
{
    if (status != NULL) {
      const char* msg = g_ort->GetErrorMessage(status);
      fprintf(stderr, "%s\n", msg);
      g_ort->ReleaseStatus(status);
      exit(1);
    }
}
//JOHN DEBUGGING
// returns an OnnxRuntime*
void load_model(void* data, size_t data_len, OrtReturn* ret) {
	OnnxRuntime* runtime = malloc(sizeof(OnnxRuntime));

  ret->status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &runtime->env);
	
  if (ret->status) {
    free(runtime);
    return;
  }

	OrtSessionOptions* session_options;
	ret->status = g_ort->CreateSessionOptions(&session_options);
	if (ret->status) {
    free(session_options);
    free(runtime);
    return;
  }

	ret->status = g_ort->CreateSessionFromArray(runtime->env, data, data_len, session_options, &runtime->session);
	if (ret->status) {
    free(session_options);
    free(runtime);
    return;
  }

  ret->status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &runtime->memory_info);
  if (ret->status) {
    free(session_options);
    free(runtime);
    return;
  }

  ret->status = g_ort->CreateAllocator(runtime->session, runtime->memory_info, &runtime->allocator);
  if (ret->status) {
    free(session_options);
    free(runtime);
    return;
  }

	ret->value = runtime;

	g_ort->ReleaseSessionOptions(session_options);
}

// returns an OrtValue*
void make_c_tensor(OnnxRuntime* runtime, void* input, size_t input_len, int64_t* input_shape, size_t input_shape_len, ONNXTensorElementDataType dtype, OrtReturn* ret) {
  char arr[1][3] = {"Six"};
  //printf("%s", new_chars[0]);
  void* p = &arr;
  input = &p;
  char **new_chars=(char**)(input);
  printf("%s\n", new_chars[0]);
	ret->status = g_ort->CreateTensorWithDataAsOrtValue(runtime->memory_info, input, input_len, input_shape, input_shape_len, dtype, (OrtValue**)&ret->value);
  char** floatarr;
  CheckStatus(g_ort->GetTensorMutableData(*(OrtValue**)&ret->value, (void**)&floatarr));
  printf("ORT Input Tensor: %s \n***END***\n", floatarr[0]);
	if (ret->status) return;
}

void get_io_count(OnnxRuntime* runtime, size_t type, OrtReturn* ret) {
    if (type) ret->status = g_ort->SessionGetInputCount(runtime->session, &ret->count);
    else ret->status = g_ort->SessionGetOutputCount(runtime->session, &ret->count);
}

// returns an IOInfo*
void get_io_info(OnnxRuntime* runtime, size_t type, size_t index, OrtReturn* ret) {
    OrtTypeInfo* type_info;
    if (type) ret->status = g_ort->SessionGetInputTypeInfo(runtime->session, index, &type_info);
    else ret->status = g_ort->SessionGetOutputTypeInfo(runtime->session, index, &type_info);
    if (ret->status) {
      free(type_info);
      return;
    }

    IOInfo* info = malloc(sizeof(IOInfo));
    ret->value = info;

    ret->status = g_ort->GetOnnxTypeFromTypeInfo(type_info, &info->value_type);
    if (ret->status) {
        free(info);
        free(type_info);
        return;
    }

    if (type) ret->status = g_ort->SessionGetInputName(runtime->session, index, runtime->allocator, &info->name);
    else ret->status = g_ort->SessionGetOutputName(runtime->session, index, runtime->allocator, &info->name);
    if (ret->status) {
        free(info);
        free(type_info);
        return;
    }

    if (info->value_type == ONNX_TYPE_TENSOR) {
        const OrtTensorTypeAndShapeInfo* shape_info;
        ret->status = g_ort->CastTypeInfoToTensorInfo(type_info, &shape_info);
        if (ret->status) {
            free(type_info);
            free(info);
            return;
        }

        ret->status = g_ort->GetTensorElementType(shape_info, &info->tensor_type);
        if (ret->status) {
            free(info);
            return;
        }

        ret->status = g_ort->GetDimensionsCount(shape_info, &info->dim_count);
        if (ret->status) {
            free(info);
            return;
        }

        info->shape = malloc(info->dim_count*sizeof(int64_t));
        ret->status = g_ort->GetDimensions(shape_info, info->shape, info->dim_count);
        if (ret->status) {
            free(info->shape);
            free(info);
            return;
        }
    }

    g_ort->ReleaseTypeInfo(type_info);
}

void free_io_info(OnnxRuntime* runtime, IOInfo* info) {
    if(!info) return;
    if(info->name) runtime->allocator->Free(runtime->allocator, info->name);
    free(info);
}

const char* error_message(OrtStatus* ret) {
    return g_ort->GetErrorMessage(ret);
}

void free_status(OrtStatus* status) {
    g_ort->ReleaseStatus(status);
}

// returns an OrtValue**
void run(OnnxRuntime* runtime, OrtValue** input_tensors, size_t input_len, char** input_names, size_t input_names_len, char** output_names, size_t output_names_len, OrtReturn *ret) {
  OrtValue** outputs = malloc(output_names_len*sizeof(OrtValue*));
	for (int i=0; i<output_names_len; i++) outputs[i] = NULL;

  ret->status = g_ort->Run(
      runtime->session,
      NULL,
      (const char* const*)input_names,
      (const OrtValue* const*)input_tensors,
      input_names_len,
      (const char* const*)output_names,
      output_names_len,
      outputs);

  if (ret->status) {
      free(outputs);
      return;
  }

  ret->value = outputs;
}

void cleanup_runtime(OnnxRuntime* runtime) {
	if (!runtime) return;
	if (runtime->env) g_ort->ReleaseEnv(runtime->env);
	if (runtime->session) g_ort->ReleaseSession(runtime->session);
	if (runtime->memory_info) g_ort->ReleaseMemoryInfo(runtime->memory_info);
    if (runtime->allocator) g_ort->ReleaseAllocator(runtime->allocator);
	free(runtime);
}

enum ONNXType value_type(OrtValue* value) {
    OrtStatus* status;
    OrtTypeInfo* type_info;
    status = g_ort->GetTypeInfo(value, &type_info);
    if (status) {
      free(status);
      free(type_info);
      return ONNX_TYPE_UNKNOWN;
    }

    enum ONNXType type;
    status = g_ort->GetOnnxTypeFromTypeInfo(type_info, &type);
    if (status) {
      free(status);
      free(type_info);
      return ONNX_TYPE_UNKNOWN;
    }

    g_ort->ReleaseTypeInfo(type_info);
    return type;
}

// returns a TensorInfo*
void get_tensor_info(OrtValue* value, OrtReturn* ret) {

    OrtTensorTypeAndShapeInfo* shape_info;
    ret->status = g_ort->GetTensorTypeAndShape(value, &shape_info);
    if (ret->status) {
      free(shape_info);
      return;
    }

    size_t dim_count;
    ret->status = g_ort->GetDimensionsCount(shape_info, &dim_count);
    if (ret->status) {
      free(shape_info);
      return;
    }
    
    TensorInfo* out_info = malloc(sizeof(TensorInfo));
    out_info->shape_len = dim_count;

    ONNXTensorElementDataType output_type;
    ret->status = g_ort->GetTensorElementType(shape_info, &output_type);
    if (ret->status) {
      free(shape_info);
      free(out_info);
      return;
    }
    out_info->tensor_type = output_type;

    int64_t* dims = malloc(dim_count*sizeof(int64_t));
    ret->status = g_ort->GetDimensions(shape_info, dims, dim_count);
    if (ret->status) {
      free(shape_info);
      free(out_info);
      free(dims);
      return;
    }
    out_info->shape_ptr = dims;

    void* data;
    ret->status = g_ort->GetTensorMutableData(value, &data);
    if (ret->status) return;
    out_info->data = data;

    g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    ret->value = out_info;
}

void free_tensor_info(TensorInfo* info){
    if (!info) return;
    if (info->shape_ptr) free(info->shape_ptr);
    free(info);
}

// returns a SequenceInfo*
void get_sequence_info(OrtValue* value, OrtReturn* ret) {
    size_t value_count;
    ret->status = g_ort->GetValueCount(value, &value_count);
    if (ret->status) return;

    OrtTypeInfo* type_info;
    ret->status = g_ort->GetTypeInfo(value, &type_info);
    if (ret->status) {
      free(type_info);
      return;
    }

    const OrtSequenceTypeInfo* sequence_type_info;
    ret->status = g_ort->CastTypeInfoToSequenceTypeInfo(type_info, &sequence_type_info);
    if (ret->status) {
      free(type_info);
      return;
    }

    OrtTypeInfo* element_type_info;
    ret->status = g_ort->GetSequenceElementType(sequence_type_info, &element_type_info);

    enum ONNXType type;
    ret->status = g_ort->GetOnnxTypeFromTypeInfo(element_type_info, &type);
    if (ret->status) {
      free(type_info);
      free(element_type_info);
      return;
    }

    g_ort->ReleaseTypeInfo(type_info);
    g_ort->ReleaseTypeInfo(element_type_info);

    SequenceInfo* info = malloc(sizeof(SequenceInfo));
    info->value_count = value_count;
    info->value_type = type;

    ret->value = info;
}

void free_sequence_info(SequenceInfo* info) {
    if (info) free(info);
}

void get_sequence_element(OnnxRuntime* runtime, OrtValue* sequence, size_t index, OrtReturn* ret) {
    ret->status = g_ort->GetValue(sequence, index, runtime->allocator, (OrtValue**)&ret->value);
}

void get_map_keys(OnnxRuntime* runtime, OrtValue* map, OrtReturn* ret) {
    ret->status = g_ort->GetValue(map, 0, runtime->allocator, (OrtValue**)&ret->value);
}

void get_child_value(OnnxRuntime* runtime, OrtValue* parent, size_t index, OrtReturn* ret) {
    ret->status = g_ort->GetValue(parent, index, runtime->allocator, (OrtValue**)&ret->value);
}

size_t get_string_tensor_data_len(OrtValue* string_tensor) {
    size_t len;
    OrtStatus* status = g_ort->GetStringTensorDataLength(string_tensor, &len);
    if (status) {
      free(status);
      return -1;
    }
    return len;
}

void get_string_tensor_data(OrtValue* string_tensor, size_t num_elements, size_t data_len, OrtReturn* ret) {
    size_t* offsets = malloc(num_elements*sizeof(size_t));

    void* data = malloc(data_len);
    ret->status=g_ort->GetStringTensorContent(string_tensor, data, data_len, offsets, num_elements);
    if (ret->status) {
      free(offsets);
      free(data);
      return;
    }

    StringTensorInfo* info = malloc(sizeof(StringTensorInfo));
    info->offsets = offsets;
    info->data = data;
    ret->value = info;
}

void free_string_tensor_data(StringTensorInfo* info) {
    if (!info) return;
    if (info->offsets) free(info->offsets);
    free(info);
}

void free_value(OrtValue* value) {
    g_ort->ReleaseValue(value);
}