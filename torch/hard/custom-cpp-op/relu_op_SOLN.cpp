#include <torch/extension.h>

// ============================================================================
// Step 1: torch::autograd::Function subclass for ReLU
// ============================================================================
// Subclassing torch::autograd::Function gives you direct control over both
// forward() and backward(). The dispatcher calls ReLUFunction::apply(),
// which routes to forward() and — during backprop — to backward().
//
// dReLU/dx = 1 if x > 0, else 0.  So grad_input = grad_output * (x > 0).

class ReLUFunction : public torch::autograd::Function<ReLUFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input)
    {
        // Save the original input so backward has access to it
        ctx->save_for_backward({input});

        // Compute ReLU: max(0, x) — manual loop for full control
        auto output = torch::empty_like(input);
        auto in_ptr  = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        for (int64_t i = 0; i < input.numel(); ++i) {
            out_ptr[i] = in_ptr[i] > 0.0f ? in_ptr[i] : 0.0f;
        }
        return output;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        // Retrieve the tensors forward() saved
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];

        // The op has one output, so backward receives one grad_output
        auto grad_output = grad_outputs[0];

        // dReLU/dx = 1 if x > 0, else 0
        auto mask = (input > 0).to(grad_output.dtype());
        auto grad_input = grad_output * mask;

        // Return one gradient per forward input
        return {grad_input};
    }
};

// ============================================================================
// Step 2: Register the operator schema with the dispatcher
// ============================================================================

TORCH_LIBRARY(custom_relu, m) {
    m.def("relu(Tensor input) -> Tensor");
}

// ============================================================================
// Step 3: Bind the Function class to the AutogradCPU dispatch key
// ============================================================================
// When using torch::autograd::Function, register on AutogradCPU — not CPU.
// The Function class handles both forward AND backward; AutogradCPU is the
// dispatch key for the full autograd path.  No separate CPU registration
// is needed — AutogradCPU handles non-differentiable tensors too.

TORCH_LIBRARY_IMPL(custom_relu, AutogradCPU, m) {
    // &ReLUFunction::apply is ambiguous (inherits multiple overloads from
    // Function<>), so we wrap in a lambda that pins the Tensor -> Tensor
    // signature the dispatcher should call.
    m.impl("relu", [](const torch::Tensor& input) -> torch::Tensor {
        return ReLUFunction::apply(input);
    });
}

// ============================================================================
// Step 4: Provide a Python module entry point
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // TORCH_LIBRARY already registered the op — nothing else needed.
}
