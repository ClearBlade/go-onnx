# Memory Leak Testing
## Prelude
PProf doesn't support leak testing any CGo code, so Valgrind was used in it's place.

Valgrind doesn't fully support MacOS. [This version](https://github.com/LouisBrunner/valgrind-macos) was tried, but was throwing several segmentation faults when running CGo Executables.

CGo's Cross Compilation requirements also caused some issues when trying to debug the executable on Linux, so eventually the whole repo was cloned to a Linux machine and built and valgrind tested there.
## Steps For Testing:
1. A Main package/function was created that would inference on different kinds of models. This code was all derived from the test cases, and can easily be remade by replacing calls to the testing package with logs inside of the new Main.
2. Print statements using print(sizeof(x)) were made for each Malloc and Free in onnxruntime.C. These aren't neccesary given a working valgrind, but due to the issues with Mac, these were used to double check that all memory that was allocated was also freed.
3. Build the executable with ```go build```
4. Leak test the executable with this call to Valgrind: ```G_SLICE=always-malloc G_DEBUG=gc-friendly  valgrind -v --tool=memcheck --leak-check=full --num-callers=40 --log-file=valgrind.log ./go-onnx``` as described in [this post](https://stackoverflow.com/questions/71500390/can-valigrind-memcheck-be-used-with-cgo)
5. Look through the output valgrind.log and shell output for leaks. In this case, a leak was found when allocating info->shape on line (at the time) 162 of onnxruntime.c