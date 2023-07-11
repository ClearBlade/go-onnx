package tests

import (
	"os"
	"testing"

	ort "github.com/ClearBlade/go-onnx"
	"gorgonia.org/tensor"
)

func TestString(t *testing.T) {
	model, err := os.ReadFile("./models/string-convert.onnx")
	if err != nil {
		t.Fatal("failed to read model file: ", err)
	}

	rt, err := ort.NewOnnxRuntime(model)
	if err != nil {
		t.Fatal("failed to create runtime: ", err)
	}
	t.Logf("Expected Input Type: %+v", rt.Inputs[0].DataType)
	t.Logf("Expected Input Shape: %+v", rt.Inputs[0].Shape)

	t.Logf("Output layer name: %+v", rt.Outputs[0].Name)
	t.Logf("Output layer Shape: %+v", rt.Outputs[0].Shape)
	backing := []string{"Seven"}
	ten := tensor.NewDense(tensor.String, []int{1}, tensor.WithBacking(backing))
	out, err := rt.RunSimple(ten)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("OUTPUT: %+v\n", out)

}
