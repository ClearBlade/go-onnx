package onnx

// #cgo darwin,amd64 LDFLAGS: -L./darwin_amd64 -Wl,-rpath,./lib -lonnxruntime
// #cgo darwin,arm64 LDFLAGS: -L./darwin_arm64 -Wl,-rpath,./lib -lonnxruntime
// #cgo linux,arm64 LDFLAGS: -L./linux_arm64 -Wl,-rpath,./lib -lonnxruntime
// #cgo linux,amd64 LDFLAGS: -L./linux_amd64 -Wl,-rpath,./lib -lonnxruntime
import "C"
