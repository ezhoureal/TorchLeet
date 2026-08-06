#include <torch/extension.h>
#include <cuda_runtime.h>

// out[0] = sum(x[i]) for i in [0, n)
// You may use more than one kernel if you want.
__global__ void reduce_sum_kernel(const float* x, float* out, int n) {
    // TODO: implement the kernel
}

torch::Tensor reduce_sum(torch::Tensor x) {
    // TODO: allocate the output tensor and launch the kernel(s)
    TORCH_CHECK(false, "reduce_sum: not implemented");
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_sum);
}
