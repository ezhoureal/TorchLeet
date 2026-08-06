#include <torch/extension.h>
#include <cuda_runtime.h>

// Row-wise softmax: out[r][c] = exp(x[r][c] - max_r) / sum_c exp(x[r][c] - max_r)
// x, out: (rows, cols), contiguous, fp32.
__global__ void softmax_kernel(const float* x, float* out, int rows, int cols) {
    // TODO: implement the kernel
}

torch::Tensor softmax(torch::Tensor x) {
    // TODO: allocate the output tensor and launch the kernel
    TORCH_CHECK(false, "softmax: not implemented");
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax);
}
