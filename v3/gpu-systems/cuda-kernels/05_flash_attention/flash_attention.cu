#include <torch/extension.h>
#include <cuda_runtime.h>

// Single-query-group scaled dot-product attention, causal masking NOT applied:
//   S = Q K^T / sqrt(d),  P = softmax(S) row-wise,  O = P V
// q, k, v, out: (batch, heads, seq, dim) contiguous fp32.
//
// The reference PyTorch implementation materializes the full (seq, seq)
// score matrix. Your kernel should NOT: tile the computation in SRAM and
// use the online (running max / running sum) softmax so HBM traffic stays
// O(seq * dim) instead of O(seq^2).
__global__ void flash_attention_kernel(
    const float* q, const float* k, const float* v, float* out,
    int batch, int heads, int seq, int dim) {
    // TODO: implement the kernel
}

torch::Tensor flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    // TODO: allocate the output tensor and launch the kernel
    TORCH_CHECK(false, "flash_attention: not implemented");
    return q;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", &flash_attention);
}
