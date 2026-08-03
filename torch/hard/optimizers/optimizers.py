"""
Exercise: Implement Adam, AdamW, and Muon Optimizers from Scratch

In this exercise, you'll implement three important optimizers:
1. **Adam**  — Adaptive Moment Estimation (Kingma & Ba, 2014)
2. **AdamW** — Adam with Decoupled Weight Decay (Loshchilov & Hutter, 2017)
3. **Muon**  — Momentum with Newton–Schulz Orthogonalization (Jordan et al., 2024)

Each optimizer follows PyTorch's `torch.optim.Optimizer` interface.
Fill in the sections marked with `# TODO`.

References:
- Adam paper:  https://arxiv.org/abs/1412.6980
- AdamW paper: https://arxiv.org/abs/1711.05101
- Muon blog:   https://kellerjordan.github.io/posts/muon/
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Newton–Schulz orthogonalization (needed for Muon)
# ═══════════════════════════════════════════════════════════════════════════════

def newton_schulz(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Approximate orthogonalization of a matrix via Newton–Schulz iteration.

    Given X (m × n), this returns an approximately orthogonal matrix O
    such that O @ O.T ≈ I (if m ≤ n) or O.T @ O ≈ I (if n ≤ m).

    Uses the quintic (5th-order) iteration:
        X_{k+1} = a·X_k + b·X_k @ X_k.T @ X_k + c·(X_k @ X_k.T)² @ X_k
    with coefficients:
        a = 15/8,  b = -5/4,  c = 3/8

    Args:
        X:     input matrix, shape (m, n)
        steps: number of Newton–Schulz iterations (default 5)

    Returns:
        Orthogonalized matrix of same shape as X
    """
    # TODO: Normalize X so that its spectral norm ≤ 1
    #   X = X / (Frobenius_norm(X) + 1e-10)
    #   (Frobenius norm is an upper bound on the spectral norm)
    X = X / (torch.linalg.matrix_norm(X) + 1e-10)

    # Coefficients for the quintic Newton–Schulz iteration
    a = 15.0 / 8.0
    b = -5.0 / 4.0
    c = 3.0 / 8.0

    for _ in range(steps):
        if X.shape[0] <= X.shape[1]:
            X_sq = X @ X.T
            X = a * X + b * X_sq @ X + c * (X_sq ** 2) @ X
        else:
            X_sq = X.T @ X
            X = a * X + b * X.T @ X_sq + c * X.T @ (X_sq ** 2)

    return X


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Adam Optimizer
# ═══════════════════════════════════════════════════════════════════════════════
#
# Adam combines two ideas:
#   1. Momentum (exponential moving average of past gradients)
#   2. RMSProp (adaptive per-parameter learning rates via second-moment estimate)
#
# Algorithm:
#   m_t = β₁·m_{t-1} + (1 − β₁)·g_t          ← biased 1st moment
#   v_t = β₂·v_{t-1} + (1 − β₂)·g_t²         ← biased 2nd moment
#   m̂_t = m_t / (1 − β₁ᵗ)                     ← bias correction
#   v̂_t = v_t / (1 − β₂ᵗ)                     ← bias correction
#   θ_t = θ_{t-1} − lr·m̂_t / (√v̂_t + ε)       ← parameter update
#
# Key detail: β₁ᵗ = β₁^step, not β₁^t.  The step counter starts at 1.

class MyAdam(Optimizer):
    """
    Adam optimizer.

    Args:
        params:  iterable of parameters to optimize
        lr:      learning rate (default 1e-3)
        betas:   coefficients for 1st- and 2nd-moment estimates (default (0.9, 0.999))
        eps:     term added to denominator for numerical stability (default 1e-8)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: optional callable that re-evaluates the model and returns loss
        Returns:
            loss from closure, if one was provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                assert isinstance(p, torch.Tensor)
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("MyAdam does not support sparse gradients")

                state = self.state[p]

                # --- State initialization on first step ---
                if len(state) == 0:
                    # TODO: initialize state dict
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # state['step'] = 0
                    # state['exp_avg']   = torch.zeros_like(p)   # m — 1st moment
                    # state['exp_avg_sq'] = torch.zeros_like(p)   # v — 2nd moment

                exp_avg: torch.Tensor = state['exp_avg']
                exp_avg_sq: torch.Tensor = state['exp_avg_sq']

                # TODO: increment step counter: state['step'] += 1
                state['step'] += 1

                # TODO: update biased first moment estimate:
                #   exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # TODO: update biased second moment estimate:
                #   exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # TODO: compute bias corrections
                #   bias_correction1 = 1 - beta1 ** state['step']
                #   bias_correction2 = 1 - beta2 ** state['step']
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # TODO: compute step size with bias correction
                #   denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                #   step_size = lr / bias_correction1
                nom = exp_avg / bias_correction1
                denom = (exp_avg_sq / bias_correction2).sqrt() + eps
                # TODO: update parameters
                p.sub_(nom / denom * lr)

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: AdamW Optimizer
# ═══════════════════════════════════════════════════════════════════════════════
#
# AdamW fixes a flaw in the original Adam: in Adam, L2 regularization
# (weight decay) is coupled with the adaptive learning rates, so weights
# with small gradient magnitudes get less regularization.  AdamW *decouples*
# weight decay by applying it directly to the parameters.
#
# The weight-decay update is:
#   θ ← θ − lr·w·θ     (decoupled from the adaptive step)
#
# Adam with L2 reg (wrong):  g ← g + w·θ    ← weight decay in gradient
# AdamW (correct):            θ ← θ − lr·w·θ ← direct parameter decay
#
# Algorithm:
#   m_t, v_t  ← same as Adam
#   m̂_t, v̂_t  ← same bias correction
#   θ ← θ − lr·w·θ − lr·m̂_t / (√v̂_t + ε)
#
# Observe that weight decay is multiplied by lr here — the effective
# decay rate is lr * weight_decay.

class MyAdamW(Optimizer):
    """
    AdamW optimizer — Adam with decoupled weight decay.

    Args:
        params:       iterable of parameters to optimize
        lr:           learning rate (default 1e-3)
        betas:        coefficients for moment estimates (default (0.9, 0.999))
        eps:          term for numerical stability (default 1e-8)
        weight_decay: weight decay coefficient (default 1e-2)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("MyAdamW does not support sparse gradients")

                state = self.state[p]

                # --- State initialization (same as Adam) ---
                if len(state) == 0:
                    # TODO: initialize state dict
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['step'] = 0

                m: torch.Tensor = state['m']
                v: torch.Tensor = state['v']

                # TODO: increment step counter
                state['step'] += 1
                # TODO: update biased moment estimates (same as Adam)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # TODO: compute bias corrections and denom (same as Adam)
                m_corrected = m / (1 - beta1 ** state['step'])
                v_corrected = v / (1 - beta2 ** state['step'])
                # TODO: apply decoupled weight decay to parameters:
                #   p.mul_(1 - lr * weight_decay)
                p.mul_(1 - lr * weight_decay)
                # TODO: apply the Adam update (same as Adam):
                #   p.addcdiv_(exp_avg, denom, value=-step_size)
                p.addcdiv_(m_corrected, v_corrected.sqrt() + eps, value=-lr)

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3: Muon Optimizer
# ═══════════════════════════════════════════════════════════════════════════════
#
# Muon ("Momentum Orthogonalized by Newton–Schulz") is a modern optimizer
# designed for training large transformers.  Key ideas:
#
# 1. For *matrix* parameters (ndim ≥ 2):
#      a. Momentum update:  M ← μ·M + g          (μ = momentum coefficient)
#      b. Orthogonalize M via Newton–Schulz → O
#      c. Scale the update:  O ← O · √max(m,n) · lr
#      d. Update:  θ ← θ − lr · O
#
# 2. For *1D* parameters (biases, LayerNorm weights, etc.):
#      Fall back to AdamW-style updates.
#
# The LR scaling: for an (m × n) weight matrix, scale by √(max(m,n)/min(m,n))
# to account for the fact that orthogonalization changes the effective step size.
#
# Typical hyperparameters for LLM training:
#   lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.01

class MyMuon(Optimizer):
    """
    Muon optimizer — Momentum with Newton–Schulz Orthogonalization.

    Args:
        params:           iterable of parameters to optimize
        lr:               learning rate (default 2e-2 — higher than Adam)
        momentum:         momentum coefficient μ (default 0.95)
        nesterov:         use Nesterov-style momentum (default True)
        ns_steps:         Newton–Schulz iteration steps (default 5)
        weight_decay:     weight decay for 1D params (default 1e-2)
        adamw_betas:      betas for AdamW fallback on 1D params (default (0.9, 0.95))
        adamw_eps:        epsilon for AdamW fallback (default 1e-8)
    """

    def __init__(self, params, lr=2e-2, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=1e-2,
                 adamw_betas=(0.9, 0.95), adamw_eps=1e-8):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
            adamw_betas=adamw_betas, adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                assert isinstance(p, torch.Tensor)
                grad = p.grad

                state = self.state[p]

                # --- State initialization ---
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    # AdamW state for 1D fallback
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                buf = state['momentum_buffer']
                exp_avg: torch.Tensor = state['exp_avg']
                exp_avg_sq: torch.Tensor = state['exp_avg_sq']

                if p.ndim >= 2:
                    # ============================================================
                    # Matrix parameter — use Muon update
                    # ============================================================

                    # TODO: Update momentum buffer:
                    #   buf.mul_(momentum).add_(grad)
                    #
                    # If nesterov=True (look-ahead), compute:
                    #   update = grad.add(buf, alpha=momentum)
                    # Otherwise:
                    #   update = buf
                    buf.mul_(momentum).add_(grad)
                    if nesterov:
                        update = grad.add(buf, alpha=momentum)
                    else:
                        update = buf

                    # TODO: Orthogonalize the update via Newton–Schulz
                    #   update = newton_schulz(update, steps=ns_steps)
                    update = newton_schulz(update, steps=ns_steps)
                    # TODO: Scale the update by √(max(m, n)) to account for the
                    # fact that orthogonalization shrinks the norm.
                    #   scale = math.sqrt(max(update.shape[0], update.shape[1]))
                    #   (The LR already provides the main step size.)
                    scale = math.sqrt(max(p.shape[0], p.shape[1]))
                    # TODO: Apply decoupled weight decay:
                    #   p.mul_(1 - lr * weight_decay)
                    p.mul(1 - lr * weight_decay)
                    # TODO: Apply the update:
                    #   p.add_(update, alpha=-lr * scale)
                    p.add_(update, alpha=-lr * scale)
                else:
                    # ============================================================
                    # 1D parameter (bias, norm weight) — AdamW fallback
                    # ============================================================

                    # TODO: Update biased moment estimates (same as Adam/AdamW)
                    #   exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    #   exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # TODO: Bias corrections
                    #   bias1 = 1 - beta1 ** state['step']
                    #   bias2 = 1 - beta2 ** state['step']
                    bias1 = 1 - beta1 ** state['step']
                    bias2 = 1 - beta2 ** state['step']

                    # TODO: Compute denom and step_size
                    denom = (exp_avg_sq / bias2).sqrt() + eps
                    step_size = lr / bias1
                    # TODO: Apply weight decay: p.mul_(1 - lr * weight_decay)
                    p.mul_(1 - lr * weight_decay)
                    # TODO: Apply the AdamW update:
                    #   p.addcdiv_(exp_avg, denom, value=-step_size)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — verify your implementations against PyTorch's built-in optimizers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_quadratic():
    """Create a simple quadratic objective: f(x) = (1/D)·||x - target||²."""
    D = 64
    target = torch.linspace(-1, 1, D)
    return target


def test_adam():
    """Compare MyAdam with torch.optim.Adam on a quadratic objective."""
    print("Testing Adam...", end=" ")

    torch.manual_seed(42)
    D = 64
    target = torch.linspace(-1, 1, D)

    # Reference: PyTorch Adam
    x_ref = torch.zeros(D, requires_grad=True)
    opt_ref = torch.optim.Adam([x_ref], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

    # Ours
    x_my = torch.zeros(D, requires_grad=True)
    opt_my = MyAdam([x_my], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(100):
        # PyTorch step
        loss_ref = ((x_ref - target) ** 2).mean()
        opt_ref.zero_grad()
        loss_ref.backward()
        opt_ref.step()

        # Our step
        loss_my = ((x_my - target) ** 2).mean()
        opt_my.zero_grad()
        loss_my.backward()
        opt_my.step()

    diff = (x_ref - x_my).abs().max().item()
    if diff < 1e-5:
        print(f"✓  PASS  (max parameter diff: {diff:.2e})")
    else:
        print(f"✗  FAIL  (max parameter diff: {diff:.2e})")


def test_adamw():
    """Compare MyAdamW with torch.optim.AdamW.  Key: AdamW ≠ Adam with wd."""
    print("Testing AdamW...", end=" ")

    torch.manual_seed(42)
    D = 64
    target = torch.linspace(-1, 1, D)

    # Reference: PyTorch AdamW
    x_ref = torch.zeros(D, requires_grad=True)
    opt_ref = torch.optim.AdamW([x_ref], lr=0.1, betas=(0.9, 0.999),
                                eps=1e-8, weight_decay=0.01)

    # Ours
    x_my = torch.zeros(D, requires_grad=True)
    opt_my = MyAdamW([x_my], lr=0.1, betas=(0.9, 0.999),
                     eps=1e-8, weight_decay=0.01)

    for _ in range(100):
        loss_ref = ((x_ref - target) ** 2).mean()
        opt_ref.zero_grad()
        loss_ref.backward()
        opt_ref.step()

        loss_my = ((x_my - target) ** 2).mean()
        opt_my.zero_grad()
        loss_my.backward()
        opt_my.step()

    diff = (x_ref - x_my).abs().max().item()
    if diff < 1e-5:
        print(f"✓  PASS  (max parameter diff: {diff:.2e})")
    else:
        print(f"✗  FAIL  (max parameter diff: {diff:.2e})")


def test_adamw_weight_decay():
    """Verify that AdamW weight decay differs from Adam + L2 regularization."""
    print("Testing AdamW decoupled weight decay...", end=" ")

    torch.manual_seed(42)
    D = 16
    target = torch.randn(D)
    wd = 0.1

    # AdamW
    x_adamw = torch.zeros(D, requires_grad=True)
    opt_adamw = torch.optim.AdamW([x_adamw], lr=0.1, weight_decay=wd)

    # Adam with L2 in grad (the "wrong" way)
    x_adam_l2 = torch.zeros(D, requires_grad=True)
    opt_adam_l2 = torch.optim.Adam([x_adam_l2], lr=0.1)

    for _ in range(50):
        # AdamW
        loss_adamw = ((x_adamw - target) ** 2).mean()
        opt_adamw.zero_grad()
        loss_adamw.backward()
        opt_adamw.step()

        # Adam + L2 (add wd to grad manually — this is NOT the same as AdamW)
        loss_l2 = ((x_adam_l2 - target) ** 2).mean()
        opt_adam_l2.zero_grad()
        loss_l2.backward()
        # Manually add weight decay to gradient (simulates L2 reg)
        with torch.no_grad():
            x_adam_l2.grad.add_(wd * x_adam_l2)
        opt_adam_l2.step()

    diff = (x_adamw - x_adam_l2).abs().max().item()
    if diff > 1e-6:
        print(f"✓  PASS  (AdamW ≠ Adam+L2, max diff: {diff:.4f})")
    else:
        print(f"✗  FAIL  (AdamW and Adam+L2 gave identical results — "
              f"weight decay is not properly decoupled)")


def test_muon():
    """Test Muon: verify it reduces the loss on a small linear model."""
    print("Testing Muon...", end=" ")

    torch.manual_seed(42)
    B, D_in, D_out = 32, 16, 8

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(D_in, D_out)
            self.bias = nn.Parameter(torch.zeros(D_out))

        def forward(self, x):
            return self.linear(x) + self.bias

    model = TinyModel()
    opt = MyMuon(model.parameters(), lr=0.02, momentum=0.95, weight_decay=1e-3)

    X = torch.randn(B, D_in)
    y = torch.randn(B, D_out)

    initial_loss = ((model(X) - y) ** 2).mean().item()

    for _ in range(200):
        pred = model(X)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_loss = ((model(X) - y) ** 2).mean().item()

    if final_loss < initial_loss * 0.5:
        print(f"✓  PASS  (loss: {initial_loss:.4f} → {final_loss:.4f})")
    else:
        print(f"✗  FAIL  (loss did not decrease enough: "
              f"{initial_loss:.4f} → {final_loss:.4f})")


def test_newton_schulz():
    """Verify Newton–Schulz orthogonalization produces an orthogonal matrix."""
    print("Testing newton_schulz...", end=" ")

    torch.manual_seed(42)
    m, n = 32, 64
    X = torch.randn(m, n)

    O = newton_schulz(X, steps=5)

    # For m ≤ n, O @ O.T should be close to I
    I_approx = O @ O.T
    diag_mean = I_approx.diag().mean().item()
    # Off-diagonal RMS
    mask = ~torch.eye(m, dtype=torch.bool)
    offdiag_rms = I_approx[mask].square().mean().sqrt().item()

    if abs(diag_mean - 1.0) < 0.1 and offdiag_rms < 0.1:
        print(f"✓  PASS  (diag mean: {diag_mean:.4f}, offdiag RMS: {offdiag_rms:.4f})")
    else:
        print(f"✗  FAIL  (diag mean: {diag_mean:.4f}, offdiag RMS: {offdiag_rms:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-grader: run all tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Optimizer Implementation Tests")
    print("=" * 60 + "\n")
    test_adam()
    test_adamw()
    test_adamw_weight_decay()
    test_newton_schulz()
    test_muon()
    print("\n" + "=" * 60)
