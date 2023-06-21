package main

import (
	"bytes"
	ioutil "io/ioutil"
	"log"
	"os"

	ort "github.com/ClearBlade/go-onnx"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image"
	"gorgonia.org/tensor"
	"gorgonia.org/vecf32"
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

	pb_byte, errr := ioutil.ReadFile("./test_data_set_0/input_0.pb")
	if errr != nil {
		log.Fatalln("Failed to read protobuf input")
	}
	//log.Println("PROTOBUF BYTE VALUES: ", pb_byte)
	reader := bytes.NewReader(pb_byte)
	tfTensor, err := tf.ReadTensor(tf.Float, []int64{1, 1, 28, 28}, reader)
	if err != nil {
		log.Fatal("READ ERROR: ", err)
	}
	err = tfTensor.Reshape([]int64{28 * 28})
	if err != nil {
		log.Fatal("RESHAPE ERROR: ", err)
	}
	//log.Println("TF TENSOR: ", tfTensor)
	log.Println("TF Tensor Values: ", tfTensor.Value())
	//log.Println("TF TENSOR DT: ", tfTensor.DataType())
	//t1 is the Gorgonia tensor
	t1 := tensor.NewDense(tensor.Float32, []int{1, 1, 28, 28}, tensor.WithBacking(tfTensor.Value()))
	//log.Println(t1)
	//log.Println(t1.Shape())
	//log.Println(t1.Data().([]float32))
	if err != nil {
		log.Fatal(err)
	}
	out, err := rt.RunSimple(t1)
	if err != nil {
		log.Fatal(err)
	}
	log.Println(out)
	results := out[rt.Outputs[0].Name].([]interface{})
	log.Println("RESULTS", results)
	//RESULTS currently prints many NaN's - presumably related to the incorrect tfTensor values
	//everything below this works
	tests := generate_tests()

	for i := 0; i < 10; i++ {
		processedImg := readImage(tests[i].input)
		//fmt.Println("Shape before: ", processedImg[0].Shape())
		log.Println("Data Type: ", processedImg[0].DataType())
		if err := processedImg[0].Reshape([]int64{28 * 28}); err != nil {
			log.Fatalln("Got an error while reshaping: ", err)
		}

		//fmt.Println("Shape after: ", processedImg[0].Shape())
		//fmt.Println("Value after: ", processedImg[0].Value())
		//input := tensor.NewDense(tensor.Float32, []int{1, 1, 28, 28}, tensor.WithBacking(processedImg[0].Value()))
		//log.Println(input)

		//log.Printf("%T", input.Data().([]float32))
		out, err := rt.RunSimple(t1)
		if err != nil {
			log.Fatal(err)
		}
		results := out[rt.Outputs[0].Name].([]interface{})
		//log.Println(results)
		max := vecf32.Argmax(results[0].([]float32))
		log.Println("============")
		log.Println("Test Image", i+1)
		log.Println("Predicted: ", max)
		log.Println("Expected: ", tests[i].output)
	}

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

type test struct {
	input  string
	output int
}

func generate_tests() []test {
	var tests []test
	tests = append(tests, test{"./test_data_set_0/img_1.jpg", 2})
	tests = append(tests, test{"./test_data_set_0/img_2.jpg", 0})
	tests = append(tests, test{"./test_data_set_0/img_3.jpg", 9})
	tests = append(tests, test{"./test_data_set_0/img_4.jpg", 0})
	tests = append(tests, test{"./test_data_set_0/img_5.jpg", 3})
	tests = append(tests, test{"./test_data_set_0/img_6.jpg", 7})
	tests = append(tests, test{"./test_data_set_0/img_7.jpg", 0})
	tests = append(tests, test{"./test_data_set_0/img_8.jpg", 3})
	tests = append(tests, test{"./test_data_set_0/img_9.jpg", 0})
	tests = append(tests, test{"./test_data_set_0/img_10.jpg", 3})
	return tests
}
