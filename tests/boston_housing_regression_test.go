package tests

import (
	"io/ioutil"
	"testing"

	onnxruntime "github.com/ClearBlade/go-onnx"
	"gorgonia.org/tensor"
)

func TestLinearRegression(t *testing.T) {

	// Loading an ONNX model
	model, err := ioutil.ReadFile("./models/boston_housing_regression.onnx")
	if err != nil {
		t.Fatal(err)
	}
	ort, err := onnxruntime.NewOnnxRuntime(model)
	if err != nil {
		t.Fatal(err)
	}

	// Actual input data
	inputData := []float64{
		0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98,
		0.02731, 0, 7.07, 0, 0.469, 6.421, 78.9, 4.9671, 2, 242, 17.8, 396.9, 9.14,
		0.02729, 0, 7.07, 0, 0.469, 7.185, 61.1, 4.9671, 2, 242, 17.8, 800.83, 4.03,
	}

	// Creating an input tensor
	in := tensor.NewDense(tensor.Float64, []int{3, 13}, tensor.WithBacking(inputData))

	// Run model
	out, err := ort.RunSimple(in)
	if err != nil {
		t.Fatal(err)
	}
	_ = out
}
