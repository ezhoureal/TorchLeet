#include <torch/extension.h>

// ============================================================================
// Step 1: Write a torch::autograd::Function subclass for ReLU
// ============================================================================
//
// Subclassing torch::autograd::Function gives you direct control over both
// forward() and backward(). This is the "proper" way to write a custom op that
// needs its own gradient formula — no fallthrough hacks, no guesswork.
//
// The pattern:
//
//   class MyOp : public torch::autograd::Function<MyOp> {
//     static Tensor forward(AutogradContext* ctx, const Tensor& input) {
//       ctx->save_for_backward({input});   // stash what backward needs
//       return ...;                         // compute output
//     }
//     static tensor_list backward(AutogradContext* ctx, tensor_list grads) {
//       auto saved = ctx->get_saved_variables();  // retrieve stashed tensors
//       auto input = saved[0];
//       auto grad_output = grads[0];
//       return { ... };                    // compute grad_input
//     }
//   };
//
//   // Register the apply() static method (not your forward function):
//   TORCH_LIBRARY_IMPL(my_ns, AutogradCPU, m) {
//     m.impl("my_op", &MyOp::apply);
//   }
//
// Key API:
//   ctx->save_for_backward({t1, t2, ...})   — save tensors during forward
//   ctx->get_saved_variables()              — get them back during backward
//
//   Backward returns a tensor_list with one gradient per forward input.
//   For an op with signature "relu(Tensor) -> Tensor":
//     forward has 1 input  → backward returns {grad_input}
//   For "add(Tensor, Tensor) -> Tensor":
//     forward has 2 inputs → backward returns {grad_a, grad_b}
//
// dReLU/dx = 1 if x > 0, else 0.  So grad_input = grad_output * (x > 0).
// ============================================================================

// TODO: Define the class and implement both forward() and backward().
//       Use the pattern above. Your forward should do the same element-wise
//       max(0, x) logic you already have — just inside a Function subclass now.

class ReLUFunction : public torch::autograd::Function<ReLUFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input)
    {
        // TODO: Save the input for backward.
        ctx->save_for_backward({input});

        // TODO: Compute ReLU — same loop you already wrote.
        auto output = torch::empty_like(input);
        auto output_ptr = output.data_ptr<float>();
        auto input_ptr = input.data_ptr<float>();
        for (int64_t i = 0; i < output.numel(); ++i) {
            output_ptr[i] = input_ptr[i] > 0 ? input_ptr[i] : 0;
        }
        return output;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        // TODO: Retrieve the saved tensors.
        // Hint: auto saved = ctx->get_saved_variables();
        //       auto input = saved[0];
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];

        // TODO: Get the gradient from the output (there's only one).
        // Hint: auto grad_output = grad_outputs[0];
        auto grad_output = grad_outputs[0];

        // TODO: Compute grad_input = grad_output * (input > 0).
        // Hint: dReLU/dx = 1 where x>0, 0 elsewhere. Use element-wise
        //       multiplication: grad_input = grad_output * mask
        //       Cast the mask to the right dtype with .to(grad_output.dtype()).
        auto mask = (input > 0).to(grad_output.dtype());
        auto grad_input = grad_output.mul(mask);
        // TODO: Return as tensor_list with one element.
        // Hint: return {grad_input};
        return {grad_input};
    }
};

// ============================================================================
// Step 2: Register the operator schema with the dispatcher via TORCH_LIBRARY
// ============================================================================
// TORCH_LIBRARY(namespace, m) declares what ops exist and their signatures.

// TODO: Uncomment and fill in:
TORCH_LIBRARY(custom_relu, m) {
    m.def("relu(Tensor input) -> Tensor");
}

// ============================================================================
// Step 3: Bind the Function class to the AutogradCPU dispatch key
// ============================================================================
// When using torch::autograd::Function, you register on AutogradCPU — not on
// CPU. The Function class handles both forward AND backward in one place.
// The dispatcher routes all autograd-enabled calls through your apply() method.
//
  TORCH_LIBRARY_IMPL(custom_relu, AutogradCPU, m) {
      // &ReLUFunction::apply is ambiguous (inherits multiple overloads),
      // so wrap in a lambda that pins the Tensor -> Tensor signature.
      m.impl("relu", [](const torch::Tensor& input) -> torch::Tensor {
          return ReLUFunction::apply(input);
      });
  }
//
// Note: you do NOT need a separate CPU registration. AutogradCPU covers
// both differentiable and non-differentiable tensors.

// TODO: Uncomment and fill in:
// TORCH_LIBRARY_IMPL(_____, AutogradCPU, m) {
//     m.impl("_____", &_____);
// }

// ============================================================================
// Step 4: Provide a Python module entry point
// ============================================================================
// torch.utils.cpp_extension.load() needs a PyInit_<name> symbol to import
// the .so. TORCH_LIBRARY handles operator registration as a side effect of
// loading, so the module body stays empty.

// TODO: Uncomment this block:
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // TORCH_LIBRARY already registered the op — nothing else needed.
}
