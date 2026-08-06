#include <torch/extension.h>
#include <cuda_runtime.h>

// LayerNorm over the last dimension:
//   out[r][c] = (x[r][c] - mean_r) / sqrt(var_r + eps) * gamma[c] + beta[c]
// x, out: (rows, cols) contiguous fp32; gamma, beta: (cols,) fp32.
// var_r is the biased variance (divide by cols).
__global__ void layernorm_kernel(
    const float* x, const float* gamma, const float* beta, float* out,
    int rows, int cols, float eps) {
    // TODO: implement the kernel
}

torch::Tensor layernorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    // TODO: allocate the output tensor and launch the kernel
    TORCH_CHECK(false, "layernorm: not implemented");
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm", &layernorm);
}
