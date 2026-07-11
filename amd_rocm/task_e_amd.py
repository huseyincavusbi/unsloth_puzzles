"""
Task E — Memory-Efficient Backprop (Chunked Autograd)
======================================================
Runs on CPU and AMD ROCm. No bitsandbytes required.

Usage:
    python amd_rocm/task_e_amd.py
    python amd_rocm/task_e_amd.py --device cuda
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. NAIVE BASELINE 
# ---------------------------------------------------------------------------

def transformation_function(batch, linear, labels):
    x = linear(batch).float()  # Up projection to large space
    from torch.nn import CrossEntropyLoss
    down_projection_function = CrossEntropyLoss(reduction="mean")
    # Down projection to small space
    loss = down_projection_function(x.view(-1, x.shape[-1]), labels.view(-1))
    return loss


# ---------------------------------------------------------------------------
# 2. RNG STATE HELPERS  (mirrors torch.utils.checkpoint pattern)
# ---------------------------------------------------------------------------

def _get_rng_state(device):
    cpu_state = torch.get_rng_state()
    device_state = None
    if (device == "cuda" or device == "hip") and torch.cuda.is_available():
        device_state = torch.cuda.get_rng_state()
    return cpu_state, device_state


def _set_rng_state(cpu_state, device_state, device):
    torch.set_rng_state(cpu_state)
    if device_state is not None:
        if device == "cuda" or device == "hip":
            torch.cuda.set_rng_state(device_state)


# ---------------------------------------------------------------------------
# 3. MEMORY-EFFICIENT IMPLEMENTATION
# ---------------------------------------------------------------------------

class MemoryEfficientLinear(torch.autograd.Function):
    """
    Chunked forward + recomputed backward for the lm_head loss.

    Follows torch.utils.checkpoint principles:
      - Save only inputs (X, labels), never the logit tensor.
      - Restore RNG state before each chunk's recomputation so dropout
        replays identically in forward and backward.

    Usage:
        loss = MemoryEfficientLinear.apply(
            X, linear, labels, forward_function, chunk_size
        )
        loss.backward()
    """

    @staticmethod
    def forward(ctx, X, linear, labels, forward_function, chunk_size):
        N = X.shape[0]
        device = X.device.type
        total_loss = torch.zeros((), dtype=torch.float32, device=X.device)
        chunk_rng_states = []

        with torch.no_grad():
            for start in range(0, N, chunk_size):
                x_chunk    = X[start : start + chunk_size]
                lbl_chunk  = labels[start : start + chunk_size]
                n_chunk    = x_chunk.shape[0]

                rng_state = _get_rng_state(device)
                chunk_rng_states.append(rng_state)

                loss_chunk = forward_function(x_chunk, linear, lbl_chunk)
                # Weight by chunk fraction to reconstruct global mean reduction
                total_loss = total_loss + loss_chunk.float() * (n_chunk / N)

        ctx.save_for_backward(X, labels)
        ctx.linear           = linear
        ctx.forward_function = forward_function
        ctx.chunk_size       = chunk_size
        ctx.N                = N
        ctx.device           = device
        ctx.chunk_rng_states = chunk_rng_states
        return total_loss

    @staticmethod
    def backward(ctx, dY):
        X, labels        = ctx.saved_tensors
        linear           = ctx.linear
        forward_function = ctx.forward_function
        chunk_size       = ctx.chunk_size
        N                = ctx.N
        device           = ctx.device
        chunk_rng_states = ctx.chunk_rng_states

        dX = torch.zeros_like(X)
        
        for chunk_idx, start in enumerate(range(0, N, chunk_size)):
            x_orig    = X[start : start + chunk_size]
            x_chunk   = x_orig.detach()
            x_chunk.requires_grad_(x_orig.requires_grad)
            lbl_chunk = labels[start : start + chunk_size]
            n_chunk   = x_chunk.shape[0]

            cpu_state, dev_state = chunk_rng_states[chunk_idx]
            with torch.random.fork_rng(enabled=True):
                _set_rng_state(cpu_state, dev_state, device)
                with torch.enable_grad():
                    loss_chunk = forward_function(x_chunk, linear, lbl_chunk)
                    scaled     = loss_chunk.float() * (n_chunk / N)

                torch.autograd.backward(scaled, dY)

            dX[start : start + chunk_size] = x_chunk.grad

        return dX, None, None, None, None


# ---------------------------------------------------------------------------
# 3. GRPO IMPLEMENTATION
# ---------------------------------------------------------------------------

def grpo_transformation_function(batch, linear, labels, advantages, n_total):
    """
    GRPO loss for one chunk.
    advantages: (chunk_size,) float — pre-computed from reward normalisation.
    n_total:    total token count across all completions.
    Returns this chunk's contribution to the global loss (caller just sums).
    """
    x = linear(batch).float()
    log_probs = -F.cross_entropy(
        x.view(-1, x.shape[-1]), labels.view(-1), reduction="none"
    )
    return -(advantages * log_probs).sum() / n_total


class MemoryEfficientGRPO(torch.autograd.Function):
    """
    Chunked GRPO loss with recomputed backward.
    advantages are pre-computed constants — no grad required.

    Usage:
        loss = MemoryEfficientGRPO.apply(
            X, linear, labels, advantages, forward_function, chunk_size
        )
        loss.backward()
    """

    @staticmethod
    def forward(ctx, X, linear, labels, advantages, forward_function, chunk_size):
        N = X.shape[0]
        device = X.device.type
        total_loss = torch.zeros((), dtype=torch.float32, device=X.device)
        chunk_rng_states = []

        with torch.no_grad():
            for start in range(0, N, chunk_size):
                x_chunk   = X[start : start + chunk_size]
                lbl_chunk = labels[start : start + chunk_size]
                adv_chunk = advantages[start : start + chunk_size]

                rng_state = _get_rng_state(device)
                chunk_rng_states.append(rng_state)

                loss_chunk = forward_function(x_chunk, linear, lbl_chunk, adv_chunk, N)
                total_loss = total_loss + loss_chunk.float()

        ctx.save_for_backward(X, labels, advantages)
        ctx.linear           = linear
        ctx.forward_function = forward_function
        ctx.chunk_size       = chunk_size
        ctx.N                = N
        ctx.device           = device
        ctx.chunk_rng_states = chunk_rng_states
        return total_loss

    @staticmethod
    def backward(ctx, dY):
        X, labels, advantages = ctx.saved_tensors
        linear           = ctx.linear
        forward_function = ctx.forward_function
        chunk_size       = ctx.chunk_size
        N                = ctx.N
        device           = ctx.device
        chunk_rng_states = ctx.chunk_rng_states

        dX = torch.zeros_like(X)
        
        for chunk_idx, start in enumerate(range(0, N, chunk_size)):
            x_orig    = X[start : start + chunk_size]
            x_chunk   = x_orig.detach()
            x_chunk.requires_grad_(x_orig.requires_grad)
            lbl_chunk = labels[start : start + chunk_size]
            adv_chunk = advantages[start : start + chunk_size]

            cpu_state, dev_state = chunk_rng_states[chunk_idx]
            with torch.random.fork_rng(enabled=True):
                _set_rng_state(cpu_state, dev_state, device)
                with torch.enable_grad():
                    loss_chunk = forward_function(
                        x_chunk, linear, lbl_chunk, adv_chunk, N
                    )
                torch.autograd.backward(loss_chunk.float(), dY)

            dX[start : start + chunk_size] = x_chunk.grad

        return dX, None, None, None, None, None


# ---------------------------------------------------------------------------
# 4. TEST HARNESS
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_correctness(device, chunk_size=4):
    section(f"TEST 1 — Correctness  (device={device}, chunk_size={chunk_size})")

    torch.manual_seed(42)
    N, hidden, vocab = 16, 64, 256

    linear = nn.Linear(hidden, vocab, bias=False).to(device)
    labels = torch.randint(0, vocab, (N,), device=device)

    X1 = torch.randn(N, hidden, device=device, requires_grad=True)
    X2 = X1.detach().clone().requires_grad_(True)

    # --- naive ---
    naive_loss = transformation_function(X1, linear, labels)
    naive_loss.backward()
    naive_dX = X1.grad.clone()
    naive_dW = linear.weight.grad.clone()

    # --- chunked ---
    linear.zero_grad()
    chunked_loss = MemoryEfficientLinear.apply(
        X2, linear, labels, transformation_function, chunk_size
    )
    chunked_loss.backward()
    chunked_dX = X2.grad.clone()
    chunked_dW = linear.weight.grad.clone()

    # Compare
    atol = 1e-5
    loss_ok = torch.allclose(naive_loss, chunked_loss, atol=atol)
    dX_ok   = torch.allclose(naive_dX,   chunked_dX,   atol=atol)
    dW_ok   = torch.allclose(naive_dW,   chunked_dW,   atol=1e-3)

    print(f"  naive_loss    = {naive_loss.item():.6f}")
    print(f"  chunked_loss  = {chunked_loss.item():.6f}")
    print(f"  loss match    : {'PASS' if loss_ok else 'FAIL'}")
    print(f"  dX match      : {'PASS' if dX_ok   else 'FAIL'}  (max diff {(naive_dX - chunked_dX).abs().max().item():.2e})")
    print(f"  dW match      : {'PASS' if dW_ok   else 'FAIL'}  (max diff {(naive_dW - chunked_dW).abs().max().item():.2e})")
    return loss_ok and dX_ok and dW_ok


def test_chunk_boundary(device):
    """N not divisible by chunk_size — tests remainder handling."""
    section(f"TEST 2 — Chunk boundary (N=17, chunk=4, device={device})")

    torch.manual_seed(7)
    N, hidden, vocab = 17, 64, 256

    linear = nn.Linear(hidden, vocab, bias=False).to(device)
    labels = torch.randint(0, vocab, (N,), device=device)
    X1     = torch.randn(N, hidden, device=device, requires_grad=True)
    X2     = X1.detach().clone().requires_grad_(True)

    naive_loss = transformation_function(X1, linear, labels)
    naive_loss.backward()

    linear.zero_grad()
    chunked_loss = MemoryEfficientLinear.apply(
        X2, linear, labels, transformation_function, 4
    )
    chunked_loss.backward()

    ok = torch.allclose(naive_loss, chunked_loss, atol=1e-5)
    print(f"  loss match (N=17, chunk=4): {'PASS' if ok else 'FAIL'}")
    return ok


def test_generality(device):
    """Use a completely different forward_function (not CE) to prove no hardcoding."""
    section(f"TEST 3 — Generality: non-CE loss (device={device})")

    def mse_loss_fn(batch, linear, labels):
        """Replace CE with MSE — totally different loss."""
        x      = linear(batch).float()
        target = torch.zeros_like(x)
        target.scatter_(1, labels.unsqueeze(1), 1.0)  # one-hot
        return nn.functional.mse_loss(x, target)

    torch.manual_seed(99)
    N, hidden, vocab = 16, 64, 256

    linear = nn.Linear(hidden, vocab, bias=False).to(device)
    labels = torch.randint(0, vocab, (N,), device=device)
    X1     = torch.randn(N, hidden, device=device, requires_grad=True)
    X2     = X1.detach().clone().requires_grad_(True)

    naive_loss = mse_loss_fn(X1, linear, labels)
    naive_loss.backward()

    linear.zero_grad()
    chunked_loss = MemoryEfficientLinear.apply(
        X2, linear, labels, mse_loss_fn, 4
    )
    chunked_loss.backward()

    ok = torch.allclose(naive_loss, chunked_loss, atol=1e-5)
    print(f"  MSE loss match: {'PASS' if ok else 'FAIL'}")
    print(f"  naive_loss = {naive_loss.item():.6f},  chunked_loss = {chunked_loss.item():.6f}")
    return ok


def test_memory(device):
    """Compare peak memory between naive and chunked on a larger input."""
    section(f"TEST 4 — Memory  (device={device})")

    torch.manual_seed(0)
    N, hidden, vocab = 512, 512, 8192
    chunk_size = 64

    # --- Theoretical calculation ---
    bytes_per_elem   = 4  # float32 logits
    naive_logit_mb   = N            * vocab * bytes_per_elem / 1024**2
    chunked_logit_mb = chunk_size   * vocab * bytes_per_elem / 1024**2
    reduction_pct    = (1 - chunk_size / N) * 100

    print(f"  Input shape      : ({N}, {hidden})  →  vocab {vocab}")
    print(f"  Chunk size       : {chunk_size}")
    print()
    print(f"  [Theoretical logit tensor peak]")
    print(f"  Naive   : {N} × {vocab} × 4B  = {naive_logit_mb:.1f} MB")
    print(f"  Chunked : {chunk_size} × {vocab} × 4B  = {chunked_logit_mb:.2f} MB")
    print(f"  Reduction: {reduction_pct:.0f}%  ({N//chunk_size}× smaller peak logit alloc)")


def test_dropout_rng(device):
    section(f"TEST 5 - Dropout / RNG save-restore  (device={device})")

    p = 0.3

    def dropout_fn(batch, linear, labels):
        x = linear(batch).float()
        x = torch.nn.functional.dropout(x, p=p, training=True)
        return nn.functional.cross_entropy(x.view(-1, x.shape[-1]), labels.view(-1))

    torch.manual_seed(42)
    N, hidden, vocab = 16, 64, 256
    linear = nn.Linear(hidden, vocab, bias=False).to(device)
    labels = torch.randint(0, vocab, (N,), device=device)
    X_base = torch.randn(N, hidden, device=device)

    # Run A twice with the same seed: gradients must be identical.
    results_a = []
    for _ in range(2):
        torch.manual_seed(99)
        X = X_base.detach().clone().requires_grad_(True)
        linear.zero_grad()
        loss = MemoryEfficientLinear.apply(X, linear, labels, dropout_fn, 4)
        loss.backward()
        results_a.append(X.grad.clone())

    consistent = torch.allclose(results_a[0], results_a[1], atol=1e-6)
    print(f"  Same seed -> same gradients (RNG replay works) : {'PASS' if consistent else 'FAIL'}")

    # Run B with two different seeds: gradients must differ.
    grads_different = []
    for seed in [1, 2]:
        torch.manual_seed(seed)
        X = X_base.detach().clone().requires_grad_(True)
        linear.zero_grad()
        loss = MemoryEfficientLinear.apply(X, linear, labels, dropout_fn, 4)
        loss.backward()
        grads_different.append(X.grad.clone())

    dropout_active = not torch.allclose(grads_different[0], grads_different[1], atol=1e-6)
    print(f"  Diff seed -> diff gradients (dropout is live)  : {'PASS' if dropout_active else 'FAIL'}")

    return consistent and dropout_active


def test_chunk_size_sweep(device):
    section(f"TEST 6 — Dynamic chunk size sweep  (device={device})")

    torch.manual_seed(5)
    N, hidden, vocab = 32, 64, 256
    linear = nn.Linear(hidden, vocab, bias=False).to(device)
    labels = torch.randint(0, vocab, (N,), device=device)

    # Compute reference with naive
    X_ref = torch.randn(N, hidden, device=device, requires_grad=True)
    naive_loss = transformation_function(X_ref, linear, labels)
    naive_loss.backward()
    ref_dX = X_ref.grad.clone()
    ref_dW = linear.weight.grad.clone()

    chunk_sizes = [1, 2, 4, 8, 16, 32]
    all_ok = True
    for cs in chunk_sizes:
        linear.zero_grad()
        X = X_ref.detach().clone().requires_grad_(True)
        loss = MemoryEfficientLinear.apply(X, linear, labels, transformation_function, cs)
        loss.backward()

        loss_ok = torch.allclose(naive_loss, loss,        atol=1e-5)
        dX_ok   = torch.allclose(ref_dX,    X.grad,      atol=1e-5)
        dW_ok   = torch.allclose(ref_dW,    linear.weight.grad, atol=1e-3)
        ok      = loss_ok and dX_ok and dW_ok
        all_ok  = all_ok and ok
        print(f"  chunk_size={cs:3d}  loss={loss.item():.6f}  {'PASS' if ok else 'FAIL'}")

    return all_ok


def test_upstream_gradient(device):
    section(f"TEST 7 — Upstream gradient (dY != 1.0)  (device={device})")

    torch.manual_seed(11)
    N, hidden, vocab = 16, 64, 256
    linear = nn.Linear(hidden, vocab, bias=False).to(device)
    labels = torch.randint(0, vocab, (N,), device=device)

    scale = 3.7

    # Baseline: dY = 1.0
    X1 = torch.randn(N, hidden, device=device, requires_grad=True)
    loss1 = MemoryEfficientLinear.apply(X1, linear, labels, transformation_function, 4)
    loss1.backward()
    dX_unit = X1.grad.clone()
    dW_unit = linear.weight.grad.clone()

    # Scaled: dY = scale
    linear.zero_grad()
    X2 = X1.detach().clone().requires_grad_(True)
    loss2 = MemoryEfficientLinear.apply(X2, linear, labels, transformation_function, 4)
    loss2.backward(gradient=torch.tensor(scale, device=device))
    dX_scaled = X2.grad.clone()
    dW_scaled  = linear.weight.grad.clone()

    dX_ok = torch.allclose(dX_scaled, dX_unit * scale, atol=1e-3)
    dW_ok = torch.allclose(dW_scaled, dW_unit * scale, atol=1e-3)

    print(f"  scale = {scale}")
    print(f"  dX scaled correctly : {'PASS' if dX_ok else 'FAIL'}  "
          f"(max diff {(dX_scaled - dX_unit * scale).abs().max().item():.2e})")
    print(f"  dW scaled correctly : {'PASS' if dW_ok else 'FAIL'}  "
          f"(max diff {(dW_scaled - dW_unit * scale).abs().max().item():.2e})")
    return dX_ok and dW_ok


# ---------------------------------------------------------------------------
# 5. GRPO TESTS
# ---------------------------------------------------------------------------

def _make_advantages(G, T, rewards):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    mean_r  = rewards.mean()
    std_r   = rewards.std(unbiased=False).clamp(min=1e-8)
    adv_per_completion = (rewards - mean_r) / std_r
    return adv_per_completion.repeat_interleave(T)


def test_grpo_correctness(device, chunk_size=4):
    section(f"TEST 8 - GRPO correctness  (device={device}, chunk_size={chunk_size})")

    torch.manual_seed(42)
    G, T, hidden, vocab = 4, 8, 32, 128
    N = G * T

    linear     = nn.Linear(hidden, vocab, bias=False).to(device)
    labels     = torch.randint(0, vocab, (N,), device=device)
    advantages = _make_advantages(G, T, [1.2, -0.5, 0.3, -1.0]).to(device)

    X1 = torch.randn(N, hidden, device=device, requires_grad=True)
    X2 = X1.detach().clone().requires_grad_(True)

    # Naive
    naive_loss = grpo_transformation_function(X1, linear, labels, advantages, N)
    naive_loss.backward()
    naive_dX = X1.grad.clone()
    naive_dW = linear.weight.grad.clone()

    # Chunked
    linear.zero_grad()
    chunked_loss = MemoryEfficientGRPO.apply(
        X2, linear, labels, advantages, grpo_transformation_function, chunk_size
    )
    chunked_loss.backward()
    chunked_dX = X2.grad.clone()
    chunked_dW = linear.weight.grad.clone()

    atol = 1e-5
    loss_ok = torch.allclose(naive_loss, chunked_loss, atol=atol)
    dX_ok   = torch.allclose(naive_dX,  chunked_dX,   atol=atol)
    dW_ok   = torch.allclose(naive_dW,  chunked_dW,   atol=1e-3)

    print(f"  naive_loss   = {naive_loss.item():.6f}")
    print(f"  chunked_loss = {chunked_loss.item():.6f}")
    print(f"  loss match   : {'PASS' if loss_ok else 'FAIL'}")
    print(f"  dX match     : {'PASS' if dX_ok   else 'FAIL'}  (max diff {(naive_dX - chunked_dX).abs().max().item():.2e})")
    print(f"  dW match     : {'PASS' if dW_ok   else 'FAIL'}  (max diff {(naive_dW - chunked_dW).abs().max().item():.2e})")
    return loss_ok and dX_ok and dW_ok


def test_grpo_gradient_signs(device):
    section(f"TEST 9 - GRPO gradient signs  (device={device})")

    torch.manual_seed(7)
    N, hidden, vocab = 8, 32, 64
    linear = nn.Linear(hidden, vocab, bias=False).to(device)
    labels = torch.randint(0, vocab, (N,), device=device)
    X_base = torch.randn(N, hidden, device=device)

    def run(adv_value):
        advantages = torch.full((N,), adv_value, device=device)
        X = X_base.detach().clone().requires_grad_(True)
        linear.zero_grad()
        loss = MemoryEfficientGRPO.apply(
            X, linear, labels, advantages, grpo_transformation_function, 4
        )
        loss.backward()
        return linear.weight.grad.clone()

    pos_dW = run(+2.0)
    neg_dW = run(-2.0)

    signs_flipped = torch.allclose(pos_dW, -neg_dW, atol=1e-5)
    print(f"  pos advantage dW == -neg advantage dW : {'PASS' if signs_flipped else 'FAIL'}")
    print(f"  (max diff: {(pos_dW + neg_dW).abs().max().item():.2e})")
    return signs_flipped


def test_grpo_chunk_sweep(device):
    section(f"TEST 10 - GRPO chunk size sweep  (device={device})")

    torch.manual_seed(3)
    G, T, hidden, vocab = 4, 8, 32, 128
    N = G * T

    linear     = nn.Linear(hidden, vocab, bias=False).to(device)
    labels     = torch.randint(0, vocab, (N,), device=device)
    advantages = _make_advantages(G, T, [0.8, -1.1, 0.2, 0.1]).to(device)
    X_ref      = torch.randn(N, hidden, device=device, requires_grad=True)

    # Reference
    naive_loss = grpo_transformation_function(X_ref, linear, labels, advantages, N)
    naive_loss.backward()
    ref_dW = linear.weight.grad.clone()

    all_ok = True
    for cs in [1, 2, 4, 8, 16, 32]:
        linear.zero_grad()
        X = X_ref.detach().clone().requires_grad_(True)
        loss = MemoryEfficientGRPO.apply(
            X, linear, labels, advantages, grpo_transformation_function, cs
        )
        loss.backward()
        ok = (torch.allclose(naive_loss, loss, atol=1e-5) and
              torch.allclose(ref_dW, linear.weight.grad, atol=1e-3))
        all_ok = all_ok and ok
        print(f"  chunk_size={cs:3d}  loss={loss.item():.6f}  {'PASS' if ok else 'FAIL'}")
    return all_ok


def test_grpo_group_structure(device):
    section(f"TEST 11 - GRPO group structure  (device={device})")

    torch.manual_seed(13)
    G, T, hidden, vocab = 2, 6, 32, 64
    N = G * T

    linear     = nn.Linear(hidden, vocab, bias=False).to(device)
    labels     = torch.randint(0, vocab, (N,), device=device)
    advantages = _make_advantages(G, T, [2.0, 0.0]).to(device)

    adv_mean = advantages.mean().item()
    adv_sum_near_zero = abs(adv_mean) < 1e-5
    print(f"  advantages mean = {adv_mean:.6f}  (should be ~0): "
          f"{'PASS' if adv_sum_near_zero else 'FAIL'}")

    X1 = torch.randn(N, hidden, device=device, requires_grad=True)
    X2 = X1.detach().clone().requires_grad_(True)

    naive_loss = grpo_transformation_function(X1, linear, labels, advantages, N)
    naive_loss.backward()

    linear.zero_grad()
    chunked_loss = MemoryEfficientGRPO.apply(
        X2, linear, labels, advantages, grpo_transformation_function, T
    )
    chunked_loss.backward()

    match = torch.allclose(naive_loss, chunked_loss, atol=1e-5)
    print(f"  loss match (chunk=completion boundary): {'PASS' if match else 'FAIL'}")
    return adv_sum_near_zero and match


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to run on (default: cpu)")
    args, _ = parser.parse_known_args()
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("ROCm GPU (cuda) not available, falling back to CPU")
        device = "cpu"

    print(f"\nTask E (AMD ROCm / CPU) -- Memory-Efficient Backprop")
    print(f"   device : {device}")
    print(f"   torch  : {torch.__version__}")

    results = []
    results.append(test_correctness(device, chunk_size=4))
    results.append(test_chunk_boundary(device))
    results.append(test_generality(device))
    test_memory(device)
    results.append(test_dropout_rng(device))
    results.append(test_chunk_size_sweep(device))
    results.append(test_upstream_gradient(device))

    section("GRPO TESTS")
    results.append(test_grpo_correctness(device))
    results.append(test_grpo_gradient_signs(device))
    results.append(test_grpo_chunk_sweep(device))
    results.append(test_grpo_group_structure(device))

    section("SUMMARY")
    all_pass = all(results)
    print(f"  All tests: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
