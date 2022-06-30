package onnx

/*
#include "onnxruntime_c_api.h"
#include "onnx.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"reflect"
	"unsafe"

	"gorgonia.org/tensor"
)

func init() {
	C.init_api()
}

type OnnxRuntime struct {
	runtime *C.OnnxRuntime
	Inputs  []IOInfo
	Outputs []IOInfo
}

type IOInfo struct {
	Name string
	Type ValueType

	// these fields are only valid if Type == ValueTypeTensor
	DataType TensorElementDataType
	Shape    []int
}

func NewOnnxRuntime(model []byte) (*OnnxRuntime, error) {
	o := &OnnxRuntime{}
	ret := C.OrtReturn{}
	C.load_model(unsafe.Pointer(&model[0]), C.size_t(len(model)), &ret)
	if ret.status != nil {
		return nil, getError(ret.status)
	}
	o.runtime = (*C.OnnxRuntime)(ret.value)

	numInputs, err := o.ioCount(ioTypeInput)
	if err != nil {
		return nil, err
	}
	numOutputs, err := o.ioCount(ioTypeOutput)
	if err != nil {
		return nil, err
	}

	for i := 0; i < numInputs; i++ {
		info, err := o.ioInfo(ioTypeInput, i)
		if err != nil {
			return nil, err
		}
		o.Inputs = append(o.Inputs, info)
	}
	for i := 0; i < numOutputs; i++ {
		info, err := o.ioInfo(ioTypeOutput, i)
		if err != nil {
			return nil, err
		}
		o.Outputs = append(o.Outputs, info)
	}

	return o, nil
}

type ioType int

const (
	ioTypeOutput ioType = iota
	ioTypeInput
)

func (o *OnnxRuntime) ioCount(t ioType) (int, error) {
	ret := C.OrtReturn{}
	C.get_io_count(o.runtime, C.size_t(t), &ret)
	if ret.status != nil {
		return -1, getError(ret.status)
	}
	return int(ret.count), nil
}

func (o *OnnxRuntime) ioInfo(t ioType, index int) (IOInfo, error) {
	ret := C.OrtReturn{}
	C.get_io_info(o.runtime, C.size_t(t), C.size_t(index), &ret)
	if ret.status != nil {
		return IOInfo{}, getError(ret.status)
	}
	ioInfo := (*C.IOInfo)(ret.value)
	defer C.free_io_info(o.runtime, ioInfo)

	info := IOInfo{
		Name:     C.GoString(ioInfo.name),
		Type:     cToGoValueType(ioInfo.value_type),
		DataType: cToGoTensorElementDataType(ioInfo.tensor_type),
	}

	dimCount := int(ioInfo.dim_count)
	if info.Type == ValueTypeTensor && dimCount > 0 {
		tmpDims := (*[1 << 30]int64)(unsafe.Pointer(ioInfo.shape))[:dimCount:dimCount]
		info.Shape = make([]int, dimCount)
		for i := 0; i < dimCount; i++ {
			info.Shape[i] = int(tmpDims[i])
		}
	}

	return info, nil
}

func (o *OnnxRuntime) RunSimple(inputs ...*tensor.Dense) (map[string]interface{}, error) {
	if len(inputs) != len(o.Inputs) {
		return nil, fmt.Errorf("mismatched input lengths: expected %d got %d", len(o.Inputs), len(inputs))
	}
	in := make(map[string]*tensor.Dense, len(o.Inputs))
	for i, input := range o.Inputs {
		in[input.Name] = inputs[i]
	}

	want := make([]string, len(o.Outputs))
	for i, output := range o.Outputs {
		want[i] = output.Name
	}

	return o.Run(want, in)
}

func (o *OnnxRuntime) Run(desiredOutputs []string, inputs map[string]*tensor.Dense) (out map[string]interface{}, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panicked during model run: %+v", r)
		}
	}()
	if o.runtime == nil {
		return nil, fmt.Errorf("onnx runtime not initialized")
	}
	if o.runtime.session == nil {
		return nil, fmt.Errorf("onnx model not loaded")
	}
	if len(desiredOutputs) == 0 {
		for _, output := range o.Outputs {
			desiredOutputs = append(desiredOutputs, output.Name)
		}
	}
	inputTensorNames := make([]string, len(inputs))
	inputTensors := make([]*C.OrtValue, len(inputs))
	for i, in := range o.Inputs {
		inputTensor, ok := inputs[in.Name]
		if !ok {
			continue
		}
		inputTensors[i], err = o.makeCTensor(inputTensor)
		if err != nil {
			return nil, fmt.Errorf("failed to make input tensor[%d]: %s", i, err)
		}
		inputTensorNames[i] = in.Name
	}
	defer freeValues(inputTensors)

	outputNames := cStringSlice(desiredOutputs)
	defer freeCStringSlice(outputNames, len(desiredOutputs))
	inputNames := cStringSlice(inputTensorNames)
	defer freeCStringSlice(inputNames, len(inputTensorNames))

	ret := C.OrtReturn{}
	C.run(o.runtime, &inputTensors[0], C.size_t(len(inputTensors)),
		inputNames, C.size_t(len(inputTensorNames)), outputNames, C.size_t(len(desiredOutputs)), &ret)
	if ret.status != nil {
		return nil, getError(ret.status)
	}
	numOutputs := len(desiredOutputs)
	outputs := (*[1 << 30]*C.OrtValue)(ret.value)[:numOutputs:numOutputs]
	defer freeValues(outputs)

	out = make(map[string]interface{}, numOutputs)
	for i := 0; i < numOutputs; i++ {
		val, err := o.getValue(outputs[i])
		if err != nil {
			return nil, err
		}
		out[o.Outputs[i].Name] = val
	}
	return out, nil
}

func cStringSlice(in []string) **C.char {
	cArray := C.malloc(C.size_t(len(in)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	tmpSlice := (*[1<<30 - 1]*C.char)(cArray)

	for i, s := range in {
		tmpSlice[i] = C.CString(s)
	}

	return (**C.char)(cArray)
}

func freeCStringSlice(cArray **C.char, len int) {
	tmpSlice := (*[1<<30 - 1]*C.char)(unsafe.Pointer(cArray))

	for i := 0; i < len; i++ {
		C.free(unsafe.Pointer(tmpSlice[i]))
	}
	C.free(unsafe.Pointer(cArray))
}

func (o *OnnxRuntime) Cleanup() {
	C.cleanup_runtime(o.runtime)
	o.runtime = nil
}

func (o *OnnxRuntime) makeCTensor(ten *tensor.Dense) (*C.OrtValue, error) {
	shape := make([]int64, len(ten.Shape()))
	for i, s := range ten.Shape() {
		shape[i] = int64(s)
	}
	typ, err := getTensorType(ten)
	if err != nil {
		return nil, err
	}
	ret := C.OrtReturn{}
	C.make_c_tensor(
		o.runtime,
		ten.Pointer(),
		C.size_t(ten.DataSize()*int(unsafe.Sizeof(ten.Dtype().Type))),
		(*C.int64_t)(&shape[0]),
		C.size_t(len(shape)),
		typ,
		&ret)
	if ret.status != nil {
		return nil, getError(ret.status)
	}
	return (*C.OrtValue)(ret.value), nil
}

func ortValueToTensor(info *C.TensorInfo, val *C.OrtValue) (interface{}, error) {
	shapeLength := int(info.shape_len)
	cShapeSlice := (*[1 << 30]int64)(unsafe.Pointer(info.shape_ptr))[:shapeLength:shapeLength]
	ptr := info.data
	ty := cToGoTensorElementDataType(info.tensor_type)
	shape := toIntSlice(cShapeSlice)
	flattenedLength := getFlattenedLength(cShapeSlice)
	switch ty {
	case TensorElementDataTypeFloat:
		cData := (*[1 << 30]float32)(ptr)[:flattenedLength:flattenedLength]
		data := make([]float32, flattenedLength)
		copy(data, cData)
		return unflattenTensor(data, shape), nil
	case TensorElementDataTypeUint8:
		cData := (*[1 << 30]uint8)(ptr)[:flattenedLength:flattenedLength]
		data := make([]uint8, flattenedLength)
		copy(data, cData)
		return unflattenTensor(data, shape), nil
	case TensorElementDataTypeInt8:
		cData := (*[1 << 30]int8)(ptr)[:flattenedLength:flattenedLength]
		data := make([]int8, flattenedLength)
		copy(data, cData)
		return unflattenTensor(data, shape), nil
	case TensorElementDataTypeUint16:
		cData := (*[1 << 30]uint16)(ptr)[:flattenedLength:flattenedLength]
		data := make([]uint16, flattenedLength)
		copy(data, cData)
		return unflattenTensor(data, shape), nil
	case TensorElementDataTypeInt16:
		cData := (*[1 << 30]int16)(ptr)[:flattenedLength:flattenedLength]
		data := make([]int16, flattenedLength)
		copy(data, cData)
		return unflattenTensor(data, shape), nil
	case TensorElementDataTypeUint32:
		cData := (*[1 << 30]uint32)(ptr)[:flattenedLength:flattenedLength]
		data := make([]uint32, flattenedLength)
		copy(data, cData)
		return unflattenTensor(data, shape), nil
	case TensorElementDataTypeInt32:
		cData := (*[1 << 30]int32)(ptr)[:flattenedLength:flattenedLength]
		data := make([]int32, flattenedLength)
		copy(data, cData)
		return unflattenTensor(data, shape), nil
	case TensorElementDataTypeUint64:
		cData := (*[1 << 30]uint64)(ptr)[:flattenedLength:flattenedLength]
		data := make([]uint64, flattenedLength)
		copy(data, cData)
		return unflattenTensor(data, shape), nil
	case TensorElementDataTypeInt64:
		cData := (*[1 << 30]int64)(ptr)[:flattenedLength:flattenedLength]
		data := make([]int64, flattenedLength)
		copy(data, cData)
		return unflattenTensor(data, shape), nil
	case TensorElementDataTypeBool:
		cData := (*[1 << 30]bool)(ptr)[:flattenedLength:flattenedLength]
		data := make([]bool, flattenedLength)
		copy(data, cData)
		return unflattenTensor(data, shape), nil
	case TensorElementDataTypeDouble:
		cData := (*[1 << 30]float64)(ptr)[:flattenedLength:flattenedLength]
		data := make([]float64, flattenedLength)
		copy(data, cData)
		return unflattenTensor(data, shape), nil
	case TensorElementDataTypeString:
		totalLength := int(C.get_string_tensor_data_len(val))
		ret := C.OrtReturn{}
		C.get_string_tensor_data(val, C.size_t(flattenedLength), C.size_t(totalLength), &ret)
		if ret.status != nil {
			return nil, getError(ret.status)
		}
		info := (*C.StringTensorInfo)(ret.value)
		defer C.free_string_tensor_data(info)
		offsets := (*[1 << 30]C.size_t)(unsafe.Pointer(info.offsets))[:flattenedLength:flattenedLength]
		cData := (*[1 << 30]byte)(info.data)[:totalLength:totalLength]
		data := make([]string, flattenedLength)
		for i := 0; i < flattenedLength; i++ {
			begin := int(offsets[i])
			end := totalLength
			if i < flattenedLength-1 {
				end = int(offsets[i+1])
			}
			s := make([]byte, end-begin)
			copy(s, cData[begin:end])
			data[i] = string(s)
		}
		return unflattenTensor(data, shape), nil
	default:
		return nil, fmt.Errorf("unsupported tensor type: %s", ty)
	}
}

func sum(in []int) int {
	x := 0
	for _, y := range in {
		x += y
	}
	return x
}

func unflattenTensor[T any](in []T, shape []int) any {
	if len(shape) == 1 {
		return in[:shape[0]]
	}
	x := shape[0]
	y := sum(shape[1:])
	ret := make([]any, x)
	for i := 0; i < x; i++ {
		ret[i] = unflattenTensor(in[i*y:], shape[1:])
	}
	return ret
}

func toIntSlice(data []int64) []int {
	res := make([]int, len(data))
	for i, d := range data {
		res[i] = int(d)
	}
	return res
}

func getFlattenedLength(data []int64) int {
	res := 1
	for _, d := range data {
		res *= int(d)
	}
	return res
}

func (o *OnnxRuntime) getValue(val *C.OrtValue) (interface{}, error) {
	t := valueType(val)
	switch t {
	case ValueTypeTensor:
		return o.getTensor(val)
	case ValueTypeSequence:
		return o.getSequence(val)
	case ValueTypeMap:
		return o.getMap(val)
	default:
		return nil, fmt.Errorf("getValue unsupported type %q", t)
	}
}

func (o *OnnxRuntime) getChildValue(parent *C.OrtValue, index int) (interface{}, error) {
	ret := C.OrtReturn{}
	C.get_child_value(o.runtime, parent, C.size_t(index), &ret)
	if ret.status != nil {
		return nil, getError(ret.status)
	}
	child := (*C.OrtValue)(ret.value)
	defer C.free_value(child)
	return o.getValue(child)
}

func (o *OnnxRuntime) getTensor(val *C.OrtValue) (interface{}, error) {
	ret := C.OrtReturn{}
	C.get_tensor_info(val, &ret)
	if ret.status != nil {
		return nil, getError(ret.status)
	}
	info := (*C.TensorInfo)(ret.value)
	defer C.free_tensor_info(info)
	return ortValueToTensor(info, val)
}

func (o *OnnxRuntime) getSequence(val *C.OrtValue) ([]interface{}, error) {
	ret := C.OrtReturn{}
	C.get_sequence_info(val, &ret)
	if ret.status != nil {
		return nil, getError(ret.status)
	}
	info := (*C.SequenceInfo)(ret.value)
	defer C.free_sequence_info(info)

	sequence := make([]interface{}, int(info.value_count))
	for i := 0; i < int(info.value_count); i++ {
		elem, err := o.getChildValue(val, i)
		if err != nil {
			return nil, err
		}
		sequence[i] = elem
	}

	return sequence, nil
}

func (o *OnnxRuntime) getMap(val *C.OrtValue) (map[interface{}]interface{}, error) {
	keys, err := o.getChildValue(val, 0)
	if err != nil {
		return nil, err
	}
	values, err := o.getChildValue(val, 1)
	if err != nil {
		return nil, err
	}

	keysValue := reflect.ValueOf(keys)
	valuesValue := reflect.ValueOf(values)

	if keysValue.Kind() != reflect.Slice && keysValue.Kind() != reflect.Array {
		return nil, fmt.Errorf("map keys not slice, got %s", keysValue.Kind())
	}
	if valuesValue.Kind() != reflect.Slice && valuesValue.Kind() != reflect.Array {
		return nil, fmt.Errorf("map values not slice, got %s", valuesValue.Kind())
	}
	if keysValue.Len() != valuesValue.Len() {
		return nil, fmt.Errorf("map keys and values have mismatched lengths: %d %d", keysValue.Len(), valuesValue.Len())
	}

	ret := make(map[interface{}]interface{}, keysValue.Len())
	for i := 0; i < keysValue.Len(); i++ {
		k := keysValue.Index(i)
		v := valuesValue.Index(i)
		ret[k.Interface()] = v.Interface()
	}

	return ret, nil
}

func getError(status *C.OrtStatus) error {
	if status == nil {
		return errors.New("getError called on nil status")
	}
	defer C.free_status(status)
	return errors.New(C.GoString(C.error_message(status)))
}

func freeValues(vals []*C.OrtValue) {
	for _, val := range vals {
		C.free_value(val)
	}
}
