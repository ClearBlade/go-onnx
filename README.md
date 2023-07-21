# Go-Onnx

Go-Onnx is a Golang package with CGo bindings for ONNX Runtime.

- [Go-Onnx](#go-onnx)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Creating a Runtime](#creating-a-runtime)
    - [Inferencing](#inferencing)
      - [Run](#run)
      - [RunSimple](#runsimple)
    - [Cleanup](#cleanup)
  - [Testing](#testing)
    - [Models](#models)

## Installation

Import Go-Onnx and the Gorgonia tensor package.

```go
import (
	ort "github.com/ClearBlade/go-onnx"
	"gorgonia.org/tensor"
	"os"
	"fmt"
)
```

## Usage
This package allows a user to create Runtimes. Analogous to Inference Sessions in other ONNX Runtime formats, Runtimes use Gorgonia tensors for an ONNX model's input and output.

### Creating a Runtime
Creating a Runtime can be done with NewOnnxRuntime. This should be paired with a "defer rt.Cleanup()" to be memory safe.
```go
// Read the model as a []byte
model, err := os.ReadFile("./models/mnist-8.onnx")
if err != nil {
	fmt.Fatal("failed to read model file: ", err)
}
// Instantiating a new Runtime
rt, err := ort.NewOnnxRuntime(model)
defer rt.Cleanup()
if err != nil {
	fmt.Fatal("failed to create runtime: ", err)
}
```
To read more about memory safety, refer to **Cleanup** below.

### Inferencing
There are two different functions for inferencing on a tensor, Run() and RunSimple().

#### Run
Run lets the use pass in requested model output names as a []string and a map[string]*tensor.Dense for inputs.
```go
// Create an input Map and Output String slice
outputs := []string{outputName}
in := tensor.NewDense(tensor.Float64, []int{3, 13}, tensor.WithBacking(inputData))
tensorMap := make(map[string]*tensor.Dense, 1)
tensorMap[inputName] = in
// Run inferencing
out, err := rt.Run(outputs, tensorMap)
if err != nil {
	log.Fatal(err)
}
// Print results
log.Println("PREDICTED: ", out[outputName])
```

#### RunSimple
RunSimple lets users pass in tensors without creating the map or designating specific outputs. Instead, this function assumes that the tensors passed in are in the same order as the model's inputs. It will also return all the model's outputs.
```go
in := tensor.NewDense(tensor.Float64, []int{3, 13}, tensor.WithBacking(inputData))
out, err := rt.RunSimple(in)
if err != nil {
	t.Fatal(err)
}
log.Println("PREDICTION: ", out[rt.Outputs[0].Name])
```

### Cleanup
Runtimes are, by necessity, memory unsafe. The Cleanup function should be deferred to prevent memory leaks whenever creating a new Runtime. If this isn't called, there will be memory leaks from the CGo calls. This should typically be called with "defer," to avoid leaks if there is a panic or fatal error.

## Testing
This package uses the go testing package. The tests are stored in the /tests/ folder and can be called with "go test -run ^TestToRun ./relative/path/to/tests"

>Note: Tests utilize other dependencies, such as github.com/sugarme/tokenizer and github.com/galeone/tfgo.

### Models
Some models are not included in this repository. The [mnist-8](https://github.com/onnx/models/blob/69c5d3751dda5349fd3fc53f525395d180420c07/vision/classification/mnist/model/mnist-8.onnx) and [roberta-sequence-classification-9](https://github.com/onnx/models/blob/69c5d3751dda5349fd3fc53f525395d180420c07/text/machine_comprehension/roberta/model/roberta-sequence-classification-9.onnx) models are available at the [ONNX Model Zoo](https://github.com/onnx/models). The boston_housing_regression, scaler, and string-convert models are included.