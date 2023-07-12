package tests

import (
	"os"
	"sync"
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
	var wg sync.WaitGroup
	wg.Add(2)
	go InferenceMnist(tests1, t, &wg)
	go InferenceMnist(tests2, t, &wg)
	wg.Wait()
}
func InferenceMnist(tests []test, t *testing.T, wg *sync.WaitGroup) {
	defer wg.Done()
	model, err := os.ReadFile("./models/mnist-8.onnx")
	if err != nil {
		t.Fatal("failed to read model file: ", err)
	}

	rt, err := ort.NewOnnxRuntime(model)
	if err != nil {
		t.Fatal("failed to create runtime: ", err)
	}
	for i := 0; i < len(tests); i++ {
		processedImg := readImage(tests[i].input)
		if err := processedImg[0].Reshape([]int64{28 * 28}); err != nil {
			t.Fatal("Got an error while reshaping: ", err)
		}
		input := tensor.NewDense(tensor.Float32, []int{1, 1, 28, 28}, tensor.WithBacking(processedImg[0].Value()))
		out, err := rt.RunSimple(input)
		if err != nil {
			t.Fatal(err)
		}
		results := out[rt.Outputs[0].Name].([]interface{})
		max := vecf32.Argmax(results[0].([]float32))
		expected := tests[i].output
		if expected != max {
			t.Fatal("Prediction Failed, expected", expected, "but predicted ", max)
		}
	}
}
