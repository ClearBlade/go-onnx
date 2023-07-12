package tests

import (
	"os"
	"strings"
	"testing"

	ort "github.com/ClearBlade/go-onnx"
	"gorgonia.org/tensor"
	"gorgonia.org/vecf32"
)

func TestMultithreading(t *testing.T) {

	tests := generate_mnist_tests()
	tests1 := []test{}
	tests2 := []test{}
	for i := 0; i < len(tests); i++ {
		if i%2 == 0 {
			tests1 = append(tests1, tests[i])
		} else {
			tests2 = append(tests2, tests[i])
		}
	}
	c1 := make(chan string)
	c2 := make(chan string)
	go InferenceMnist(tests1, c1)
	go InferenceMnist(tests2, c2)
	for i := 0; i < 2; i++ {
		select {
		case err1 := <-c1:
			if strings.Contains(err1, "FAIL") {
				t.Fatal("Failed in goroutine: " + err1)
			}
		case err2 := <-c2:
			if strings.Contains(err2, "FAIL") {
				t.Fatal("Failed in goroutine: " + err2)
			}
		}
	}
}
func InferenceMnist(tests []test, channel chan string) {
	model, err := os.ReadFile("./models/mnist-8.onnx")
	if err != nil {
		channel <- "FAIL: " + err.Error()
		return
	}

	rt, err := ort.NewOnnxRuntime(model)
	defer rt.Cleanup()
	if err != nil {
		channel <- "FAIL: " + err.Error()
		return
	}
	for i := 0; i < len(tests); i++ {
		processedImg := readImage(tests[i].input)
		if err := processedImg[0].Reshape([]int64{28 * 28}); err != nil {
			channel <- "FAIL: " + err.Error()
			return
		}
		input := tensor.NewDense(tensor.Float32, []int{1, 1, 28, 28}, tensor.WithBacking(processedImg[0].Value()))
		out, err := rt.RunSimple(input)
		if err != nil {
			channel <- "FAIL: " + err.Error()
			return
		}
		results := out[rt.Outputs[0].Name].([]interface{})
		max := vecf32.Argmax(results[0].([]float32))
		expected := tests[i].output
		if expected != max {
			channel <- "FAIL: " + err.Error()
			return
		}
	}
	channel <- "SUCCESS"
}
