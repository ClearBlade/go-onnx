package onnx

/*
#include "onnxruntime_c_api.h"
#include "onnx.h"
*/
import "C"
import (
	"fmt"
	"gorgonia.org/tensor"
	"reflect"
)

type ValueType int

const (
	ValueTypeUnknown ValueType = iota
	ValueTypeTensor
	ValueTypeSequence
	ValueTypeMap
	ValueTypeOpaque
	ValueTypeSparseTensor
	ValueTypeOptional
)

func (t ValueType) String() string {
	switch t {
	case ValueTypeUnknown:
		return "unknown"
	case ValueTypeTensor:
		return "tensor"
	case ValueTypeSequence:
		return "sequence"
	case ValueTypeMap:
		return "map"
	case ValueTypeOpaque:
		return "opaque"
	case ValueTypeSparseTensor:
		return "sparse tensor"
	case ValueTypeOptional:
		return "optional"
	default:
		return "invalid"
	}
}

func cToGoValueType(t C.enum_ONNXType) ValueType {
	switch t {
	case C.ONNX_TYPE_UNKNOWN:
		return ValueTypeUnknown
	case C.ONNX_TYPE_TENSOR:
		return ValueTypeTensor
	case C.ONNX_TYPE_SEQUENCE:
		return ValueTypeSequence
	case C.ONNX_TYPE_MAP:
		return ValueTypeMap
	case C.ONNX_TYPE_OPAQUE:
		return ValueTypeOpaque
	case C.ONNX_TYPE_SPARSETENSOR:
		return ValueTypeSparseTensor
	case C.ONNX_TYPE_OPTIONAL:
		return ValueTypeOptional
	default:
		return ValueTypeUnknown
	}
}

func valueType(val *C.OrtValue) ValueType {
	return cToGoValueType(C.value_type(val))
}

type TensorElementDataType int

const (
	TensorElementDataTypeUndefined TensorElementDataType = iota
	TensorElementDataTypeFloat
	TensorElementDataTypeDouble
	TensorElementDataTypeUint8
	TensorElementDataTypeUint16
	TensorElementDataTypeUint32
	TensorElementDataTypeUint64
	TensorElementDataTypeInt8
	TensorElementDataTypeInt16
	TensorElementDataTypeInt32
	TensorElementDataTypeInt64
	TensorElementDataTypeBool
	TensorElementDataTypeString
	TensorElementDataTypeUnsupported
	TensorElementDataTypeInvalid
)

func cToGoTensorElementDataType(t C.ONNXTensorElementDataType) TensorElementDataType {
	switch t {
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
		return TensorElementDataTypeUndefined
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		return TensorElementDataTypeFloat
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		return TensorElementDataTypeDouble
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		return TensorElementDataTypeUint8
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		return TensorElementDataTypeUint16
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
		return TensorElementDataTypeUint32
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
		return TensorElementDataTypeUint64
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		return TensorElementDataTypeInt8
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		return TensorElementDataTypeInt16
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		return TensorElementDataTypeInt32
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		return TensorElementDataTypeInt64
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
		return TensorElementDataTypeString
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
		return TensorElementDataTypeBool
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128, C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
		return TensorElementDataTypeUnsupported
	default:
		return TensorElementDataTypeInvalid
	}
}

func (t TensorElementDataType) String() string {
	switch t {
	case TensorElementDataTypeUndefined:
		return "undefined"
	case TensorElementDataTypeFloat:
		return "float32"
	case TensorElementDataTypeDouble:
		return "float64"
	case TensorElementDataTypeUint8:
		return "uint8"
	case TensorElementDataTypeUint16:
		return "uint16"
	case TensorElementDataTypeUint32:
		return "uint32"
	case TensorElementDataTypeUint64:
		return "uint64"
	case TensorElementDataTypeInt8:
		return "int8"
	case TensorElementDataTypeInt16:
		return "int16"
	case TensorElementDataTypeInt32:
		return "int32"
	case TensorElementDataTypeInt64:
		return "int64"
	case TensorElementDataTypeBool:
		return "bool"
	case TensorElementDataTypeString:
		return "string"
	case TensorElementDataTypeUnsupported:
		return "unsupported"
	default:
		return "invalid"
	}
}

func (t TensorElementDataType) Dtype() tensor.Dtype {
	switch t {
	case TensorElementDataTypeFloat:
		return tensor.Float32
	case TensorElementDataTypeDouble:
		return tensor.Float64
	case TensorElementDataTypeUint8:
		return tensor.Uint8
	case TensorElementDataTypeUint16:
		return tensor.Uint16
	case TensorElementDataTypeUint32:
		return tensor.Uint32
	case TensorElementDataTypeUint64:
		return tensor.Uint64
	case TensorElementDataTypeInt8:
		return tensor.Int8
	case TensorElementDataTypeInt16:
		return tensor.Int16
	case TensorElementDataTypeInt32:
		return tensor.Int32
	case TensorElementDataTypeInt64:
		return tensor.Int64
	case TensorElementDataTypeBool:
		return tensor.Bool
	case TensorElementDataTypeString:
		return tensor.String
	default:
		return tensor.Dtype{}
	}
}

func getTensorType(ten *tensor.Dense) (C.ONNXTensorElementDataType, error) {
	switch ten.Dtype().Type.Kind() {
	case reflect.Int8:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, nil
	case reflect.Int16:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, nil
	case reflect.Int32:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, nil
	case reflect.Int64:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, nil
	case reflect.Uint8:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, nil
	case reflect.Uint16:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, nil
	case reflect.Uint32:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, nil
	case reflect.Uint64:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, nil
	case reflect.Float32:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, nil
	case reflect.Float64:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, nil
	case reflect.String:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, nil
	case reflect.Bool:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, nil
	default:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, fmt.Errorf("invalid tensor type: %s", ten.Dtype().Type.Kind())
	}
}
