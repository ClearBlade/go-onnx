package tests

import (
	"os"
	"testing"

	ort "github.com/ClearBlade/go-onnx"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image"
	"gorgonia.org/tensor"
	"gorgonia.org/vecf32"
)

type test struct {
	input  string
	output int
}

func generate_tests() []test {
	tests := []test{{"./test_data/img_1.jpg", 2},
		{"./test_data/img_2.jpg", 0},
		{"./test_data/img_3.jpg", 9},
		{"./test_data/img_4.jpg", 0},
		{"./test_data/img_5.jpg", 3},
		{"./test_data/img_6.jpg", 7},
		{"./test_data/img_7.jpg", 0},
		{"./test_data/img_8.jpg", 3},
		{"./test_data/img_9.jpg", 0},
		{"./test_data/img_10.jpg", 3}}
	return tests
}

func readImage(filename string) []*tf.Tensor {
	root := tg.NewRoot()
	img := image.Read(root, filename, 3)
	testImg := img.Clone().ResizeBicubic(image.Size{Height: 28, Width: 28}).RGBToGrayscale().Normalize().Value()
	results := tg.Exec(root, []tf.Output{testImg}, nil, &tf.SessionOptions{})
	return results
}

func TestImageProcessingModel(t *testing.T) {
	model, err := os.ReadFile("./models/mnist-8.onnx")
	if err != nil {
		t.Fatal("failed to read model file: ", err)
	}

	rt, err := ort.NewOnnxRuntime(model)
	if err != nil {
		t.Fatal("failed to create runtime: ", err)
	}

	tests := generate_tests()

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
