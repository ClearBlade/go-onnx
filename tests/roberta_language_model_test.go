package tests

import (
	"os"

	"testing"

	onnxruntime "github.com/ClearBlade/go-onnx"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	"gorgonia.org/tensor"
)

type testcase struct {
	input_string string
	expected     int
}

func generate_test_cases() []testcase {
	cases := []testcase{{"This is great", 1},
		{"This is awful", 0},
		{"I'm happy", 1},
		{"I'm sad", 0}}
	// 					1 = good, 0 = bad

	return cases
}
func TestLanguageModel(t *testing.T) {
	model, err := os.ReadFile("./models/roberta-sequence-classification-9.onnx")
	if err != nil {
		t.Fatal(err)
	}

	ort, err := onnxruntime.NewOnnxRuntime(model)
	defer ort.Cleanup()
	if err != nil {
		t.Fatal(err)
	}
	configFile, err := tokenizer.CachedPath("roberta-base", "tokenizer.json")
	if err != nil {
		panic(err)
	}

	tk, err := pretrained.FromFile(configFile)
	if err != nil {
		panic(err)
	}
	cases := generate_test_cases()
	for i := 0; i < len(cases); i++ {
		tokens, err := tk.EncodeSingle(cases[i].input_string)
		if err != nil {
			t.Fatal(err)
		}
		inputData := []int64{}
		for i := 0; i < tokens.Len(); i++ {
			inputData = append(inputData, int64(tokens.Ids[i]))
		}
		// Creating an input tensor
		in := tensor.NewDense(tensor.Int64, []int{1, tokens.Len()}, tensor.WithBacking(inputData))
		// Run Model
		out, err := ort.RunSimple(in)
		if err != nil {
			t.Fatal(err)
		}
		results := out[ort.Outputs[0].Name].([]interface{})
		var expected string
		if cases[i].expected == 1 {
			expected = "Positive"
		} else if cases[i].expected == 0 {
			expected = "Negative"
		}
		var predicted string
		if results[0].([]float32)[0] > results[0].([]float32)[1] {
			predicted = "Negative"
		} else {
			predicted = "Positive"
		}
		if predicted != expected {
			t.Fatal("Expected: ", expected, "but predicted: ", predicted)
		}
	}
}
