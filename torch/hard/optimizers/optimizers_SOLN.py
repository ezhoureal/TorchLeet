"""
Solution: Implement Adam, AdamW, and Muon Optimizers from Scratch

This file contains the completed implementations.  Compare with your own
after you've attempted the exercises in optimizers.py.
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
    """
    # Normalize so spectral norm ≤ 1 (Frobenius norm ≥ spectral norm)
    norm = X.norm() + 1e-10
    X = X / norm

    a, b, c = 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0

    for _ in range(steps):
        if X.shape[0] <= X.shape[1]:
            # m ≤ n: work with (m × m) Gram matrix
            A = X @ X.T       # (m × m)
            B = A @ A         # (m × m) = (X X^T)²
            X = a * X + b * (A @ X) + c * (B @ X)
        else:
            # m > n: work with (n × n) Gram matrix (smaller)
            A = X.T @ X       # (n × n)
            B = A @ A         # (n × n)
            X = a * X + b * (X @ A) + c * (X @ (A @ A))

    return X


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Adam
# ═══════════════════════════════════════════════════════════════════════════════

class MyAdam(Optimizer):
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
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("MyAdam does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias corrections
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute denominator with bias-corrected second moment
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: AdamW
# ═══════════════════════════════════════════════════════════════════════════════

class MyAdamW(Optimizer):
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

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                # Update biased moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias corrections
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                # Decoupled weight decay — applied directly to parameters
                # (not through the gradient)
                p.mul_(1 - lr * weight_decay)

                # Adam update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3: Muon
# ═══════════════════════════════════════════════════════════════════════════════

class MyMuon(Optimizer):
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
                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                buf = state['momentum_buffer']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                if p.ndim >= 2:
                    # ── Matrix parameter: Muon update ──

                    # Update momentum buffer
                    buf.mul_(momentum).add_(grad)

                    # Look-ahead (Nesterov) or plain momentum
                    if nesterov:
                        update = grad.add(buf, alpha=momentum)
                    else:
                        update = buf

                    # Orthogonalize the update
                    update = newton_schulz(update, steps=ns_steps)

                    # Scale to compensate for orthogonalization shrinkage.
                    # The orthogonal matrix has unit spectral norm, but the
                    # original gradient's magnitude scales with matrix size.
                    m, n = update.shape
                    scale = math.sqrt(max(m, n) / min(m, n))

                    # Weight decay (decoupled, same pattern as AdamW)
                    if weight_decay > 0:
                        p.mul_(1 - lr * weight_decay)

                    # Apply the orthogonalized update
                    p.add_(update, alpha=-lr * scale)

                else:
                    # ── 1D parameter: AdamW fallback ──

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias1 = 1 - beta1 ** state['step']
                    bias2 = 1 - beta2 ** state['step']

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)
                    step_size = lr / bias1

                    if weight_decay > 0:
                        p.mul_(1 - lr * weight_decay)

                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_adam():
    """Compare MyAdam with torch.optim.Adam on a quadratic objective."""
    print("Testing Adam...", end=" ")

    torch.manual_seed(42)
    D = 64
    target = torch.linspace(-1, 1, D)

    x_ref = torch.zeros(D, requires_grad=True)
    opt_ref = torch.optim.Adam([x_ref], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

    x_my = torch.zeros(D, requires_grad=True)
    opt_my = MyAdam([x_my], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

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


def test_adamw():
    """Compare MyAdamW with torch.optim.AdamW."""
    print("Testing AdamW...", end=" ")

    torch.manual_seed(42)
    D = 64
    target = torch.linspace(-1, 1, D)

    x_ref = torch.zeros(D, requires_grad=True)
    opt_ref = torch.optim.AdamW([x_ref], lr=0.1, betas=(0.9, 0.999),
                                eps=1e-8, weight_decay=0.01)

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

    x_adamw = torch.zeros(D, requires_grad=True)
    opt_adamw = torch.optim.AdamW([x_adamw], lr=0.1, weight_decay=wd)

    x_adam_l2 = torch.zeros(D, requires_grad=True)
    opt_adam_l2 = torch.optim.Adam([x_adam_l2], lr=0.1)

    for _ in range(50):
        loss_adamw = ((x_adamw - target) ** 2).mean()
        opt_adamw.zero_grad()
        loss_adamw.backward()
        opt_adamw.step()

        loss_l2 = ((x_adam_l2 - target) ** 2).mean()
        opt_adam_l2.zero_grad()
        loss_l2.backward()
        with torch.no_grad():
            x_adam_l2.grad.add_(wd * x_adam_l2)
        opt_adam_l2.step()

    diff = (x_adamw - x_adam_l2).abs().max().item()
    if diff > 1e-6:
        print(f"✓  PASS  (AdamW ≠ Adam+L2, max diff: {diff:.4f})")
    else:
        print(f"✗  FAIL  (AdamW and Adam+L2 gave identical results)")


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
        print(f"✗  FAIL  (loss: {initial_loss:.4f} → {final_loss:.4f})")


def test_newton_schulz():
    """Verify Newton–Schulz orthogonalization produces an orthogonal matrix."""
    print("Testing newton_schulz...", end=" ")

    torch.manual_seed(42)
    m, n = 32, 64
    X = torch.randn(m, n)

    O = newton_schulz(X, steps=5)

    I_approx = O @ O.T
    diag_mean = I_approx.diag().mean().item()
    mask = ~torch.eye(m, dtype=torch.bool)
    offdiag_rms = I_approx[mask].square().mean().sqrt().item()

    if abs(diag_mean - 1.0) < 0.1 and offdiag_rms < 0.1:
        print(f"✓  PASS  (diag mean: {diag_mean:.4f}, offdiag RMS: {offdiag_rms:.4f})")
    else:
        print(f"✗  FAIL  (diag mean: {diag_mean:.4f}, offdiag RMS: {offdiag_rms:.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("Optimizer Implementation — Solution Tests")
    print("=" * 60 + "\n")
    test_newton_schulz()
    test_adam()
    test_adamw()
    test_adamw_weight_decay()
    test_muon()
    print("\n" + "=" * 60)
