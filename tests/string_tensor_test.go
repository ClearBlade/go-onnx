package tests

import (
	"os"
	"testing"

	ort "github.com/ClearBlade/go-onnx"
	"gorgonia.org/tensor"
)

type testString struct {
	input  []string
	output []int
}

func generate_string_tests() []testString {
	tests := []testString{{[]string{"Zero", "One", "Two"}, []int{0, 1, 2}},
		{[]string{"Three"}, []int{3}},
		{[]string{"Four"}, []int{4}},
		{[]string{"Five"}, []int{5}},
		{[]string{"Six"}, []int{6}},
		{[]string{"Seven"}, []int{7}},
		{[]string{"Eight"}, []int{8}},
		{[]string{"Nine"}, []int{9}}}
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

		backing := tests[i].input
		ten := tensor.NewDense(tensor.String, []int{1, len(backing)}, tensor.WithBacking(backing))
		out, err := rt.RunSimple(ten)
		if err != nil {
			t.Fatal(err)
		}
		for j := 0; j < len(tests[i].output); j++ {
			predicted := int(out[rt.Outputs[0].Name].([]interface{})[0].([]int64)[j])
			t.Logf("Input: %+v -> Output: %+v", backing[j], out[rt.Outputs[0].Name])
			if predicted != tests[i].output[j] {
				t.Fatalf("Incorrect Prediction: Recieved %+v but expected %+v", predicted, tests[i].output[j])
			}
		}
	}
}
