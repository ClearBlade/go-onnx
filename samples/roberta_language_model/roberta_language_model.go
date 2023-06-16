package main

import (
	onnxruntime "github.com/ClearBlade/go-onnx"
	"gorgonia.org/tensor"
	"log"
	"os"
)

func main() {

	// Loading an ONNX model
	model, err := os.ReadFile("./roberta-sequence-classification-9.onnx")
	if err != nil {
		log.Fatal(err)
	}
	ort, err := onnxruntime.NewOnnxRuntime(model)
	if err != nil {
		log.Fatal(err)
	}

	// Logging expected input shape and type
	log.Println("Expected Input Type: ", ort.Inputs[0].DataType)
	log.Println("Expected Input Shape: ", ort.Inputs[0].Shape)

	log.Println("Output layer name: ", ort.Outputs[0].Name)

	// Actual input data
	inputData := []int64{0, 713, 822, 16, 98, 205, 2}

	// Creating an input tensor
	in := tensor.NewDense(tensor.Int64, []int{1, 7}, tensor.WithBacking(inputData))

	// Run model
	out, err := ort.RunSimple(in)
	if err != nil {
		log.Fatal(err)
	}

	log.Fatalln(out[ort.Outputs[0].Name])

	// Log output
	log.Println("OUTPUT: ", out[ort.Outputs[0].Name])
}
