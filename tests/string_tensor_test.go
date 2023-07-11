package tests

import (
	"os"
	"testing"

	ort "github.com/ClearBlade/go-onnx"
	"gorgonia.org/tensor"
)

// uses test struct from mnist_cnn_test
func generate_string_tests() []test {
	tests := []test{{"Zero", 0},
		{"One", 1},
		{"Two", 2},
		{"Three", 3},
		{"Four", 4},
		{"Five", 5},
		{"Six", 6},
		{"Seven", 7},
		{"Eight", 8},
		{"Nine", 9}}
	return tests
}

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
	t.Logf("Output layer Type: %+v", rt.Outputs[0].Type)

	tests := generate_string_tests()

	for i := 0; i < len(tests); i++ {

		backing := []string{tests[i].input}
		ten := tensor.NewDense(tensor.String, []int{1, len(backing)}, tensor.WithBacking(backing))
		out, err := rt.RunSimple(ten)
		if err != nil {
			t.Fatal(err)
		}
		predicted := int(out[rt.Outputs[0].Name].([]interface{})[0].([]int64)[0])
		t.Logf("Input: %+v -> Output: %+v", backing, out[rt.Outputs[0].Name])
		if predicted != tests[i].output {
			t.Fatalf("Incorrect Prediction: Recieved %+v but expected %+v", predicted, tests[i].output)
		}
	}
}
