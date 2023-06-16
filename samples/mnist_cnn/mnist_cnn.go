package main

import (
	"fmt"
	ort "github.com/ClearBlade/go-onnx"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image"
	"gorgonia.org/tensor"
	"log"
	"os"
)

func readImage(filename string) []*tf.Tensor {
	root := tg.NewRoot()
	img := image.Read(root, filename, 3)
	testImg := img.Clone().ResizeBicubic(image.Size{Height: 28, Width: 28}).RGBToGrayscale().Normalize().Value()
	results := tg.Exec(root, []tf.Output{testImg}, nil, &tf.SessionOptions{})
	return results
}

func main() {
	model, err := os.ReadFile("./mnist-8.onnx")
	if err != nil {
		log.Fatalln("failed to read model file: ", err)
	}

	rt, err := ort.NewOnnxRuntime(model)
	if err != nil {
		log.Fatalln("failed to runtime: ", err)
	}

	log.Println("Expected Input Type: ", rt.Inputs[0].DataType)
	log.Println("Expected Input Shape: ", rt.Inputs[0].Shape)

	log.Println("Output layer name: ", rt.Outputs[0].Name)

	processedImg := readImage("./test_data_set_0/test-img-1.png")

	fmt.Println("Shape before: ", processedImg[0].Shape())

	if err := processedImg[0].Reshape([]int64{1, 28 * 28}); err != nil {
		log.Fatalln("Got an error while reshaping: ", err)
	}

	fmt.Println("Shape after: ", processedImg[0].Value())

	input := tensor.NewDense(tensor.Float32, []int{1, 1, 28, 28}, tensor.WithBacking(processedImg[0].Value()))

	out, err := rt.RunSimple(input)
	if err != nil {
		log.Fatal(err)
	}

	log.Println("OUTPUT: ", out[rt.Outputs[0].Name])

	// inputs := make([]*tensor.Dense, len(ort.Inputs))
	// for i, input := range ort.Inputs {
	// 	if input.Type != onnxruntime.ValueTypeTensor {
	// 		t.Fatalf("model has unsupported input type! %+v\n", input)
	// 	}
	// 	shape := make([]int, len(input.Shape))
	// 	for i, dim := range input.Shape {
	// 		if dim < 0 {
	// 			shape[i] = 1
	// 		} else {
	// 			shape[i] = dim
	// 		}
	// 	}
	// 	inputs[i] = tensor.NewDense(input.DataType.Dtype(), shape)
	// }

	// out, err := ort.RunSimple(inputs...)
	// if err != nil {
	// 	t.Errorf("%s failed to run: \n%+v\n", err, ort.Inputs)
	// }
	// t.Logf("OUTPUT: %+v\n", out)
}
