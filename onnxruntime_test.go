package onnxruntime

import (
	"gorgonia.org/tensor"
	"io/ioutil"
	"testing"
)

func TestNewOnnxRuntime(t *testing.T) {
	model, err := ioutil.ReadFile("./samples/test-models/scaler.onnx")
	if err != nil {
		t.Fatal(err)
	}
	ort, err := NewOnnxRuntime(model)
	if err != nil {
		t.Fatal(err)
	}
	// Logging expected input shape and type
	t.Logf("Expected Input Type: %+v", ort.Inputs[0].DataType)
	t.Logf("Expected Input Shape: %+v", ort.Inputs[0].Shape)

	t.Logf("Output layer name: %+v", ort.Outputs[0].Name)

}

func TestRunSimple(t *testing.T) {
	model, err := ioutil.ReadFile("./samples/test-models/scaler.onnx")
	if err != nil {
		t.Fatal(err)
	}
	ort, err := NewOnnxRuntime(model)
	if err != nil {
		t.Fatal(err)
	}

	inputData := []float32{
		6, 148, 72, 35, 0, 33.6, 0.627, 50,
		1, 85, 66, 29, 0, 26.6, 0.351, 3,
		8, 183, 64, 0, 0, 23.3, 0.672, 3,
		1, 89, 66, 23, 94, 28.1, 0.167, 2,
		0, 137, 40, 35, 168, 43.1, 2.288, 3,
	}
	in := tensor.NewDense(tensor.Float32, []int{5, 8}, tensor.WithBacking(inputData))
	out, err := ort.RunSimple(in)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("OUTPUT: %+v\n", out)
}
