#include <torch/extension.h>
#include <cuda_runtime.h>

// c[i] = a[i] + b[i] for i in [0, n)
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // TODO: implement the kernel
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    // TODO: allocate the output tensor and launch the kernel
    TORCH_CHECK(false, "vector_add: not implemented");
    return a;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add);
}
