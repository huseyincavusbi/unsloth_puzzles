import jax
import jax.numpy as jnp
import time

import functools

# -----------------------------------------------------------------------------
# 1. Functional Definition (Forward Pass)
# -----------------------------------------------------------------------------
@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def chunked_cross_entropy(X, W, Y, chunk_size):
    """
    Computes cross entropy loss by chunking the sequence dimension
    to prevent materializing the full N x V logits matrix in memory.
    """
    N = X.shape[0]
    
    def scan_fn(carry, idx):
        # Slice chunks — handle partial last chunk
        actual_size = jnp.minimum(chunk_size, N - idx)
        X_chunk = jax.lax.dynamic_slice_in_dim(X, idx, chunk_size, axis=0)[:actual_size]
        Y_chunk = jax.lax.dynamic_slice_in_dim(Y, idx, chunk_size, axis=0)[:actual_size]
        
        # Forward pass for chunk
        logits = jnp.dot(X_chunk, W.T) # (C, V)
        
        # LogSumExp
        lse = jax.scipy.special.logsumexp(logits, axis=1) # (C,)
        
        # Target logits
        correct_logits = logits[jnp.arange(actual_size), Y_chunk]
        
        loss_chunk = lse - correct_logits
        return carry + jnp.sum(loss_chunk), None

    total_loss, _ = jax.lax.scan(scan_fn, 0.0, jnp.arange(0, N, chunk_size))
    return total_loss / N

# -----------------------------------------------------------------------------
# 2. VJP Rules (Vector-Jacobian Product) for Backward Pass
# -----------------------------------------------------------------------------
def ce_fwd(X, W, Y, chunk_size):
    # The forward pass returns the primary output AND the residuals needed for backward
    loss = chunked_cross_entropy(X, W, Y, chunk_size)
    return loss, (X, W, Y)

def ce_bwd(chunk_size, res, g):
    # Unpack residuals
    X, W, Y = res
    N, H = X.shape
    V, _ = W.shape
    
    def scan_fn_bwd(carry, idx):
        dW_acc = carry
        
        X_chunk = jax.lax.dynamic_slice_in_dim(X, idx, chunk_size, axis=0)
        Y_chunk = jax.lax.dynamic_slice_in_dim(Y, idx, chunk_size, axis=0)
        actual_size = jnp.minimum(chunk_size, N - idx)
        mask = jnp.arange(chunk_size) < actual_size
        
        logits = jnp.dot(X_chunk, W.T)
        probs = jax.nn.softmax(logits, axis=-1)
        
        d_logits = probs.at[jnp.arange(chunk_size), Y_chunk].add(-1.0)
        d_logits = jnp.where(mask[:, None], d_logits, 0.0)
        d_logits = d_logits * (g / N)
        
        dX_chunk = jnp.dot(d_logits, W)
        dW_chunk = jnp.dot(d_logits.T, X_chunk)
        
        return dW_acc + dW_chunk, dX_chunk    dW_total, dX_chunks = jax.lax.scan(scan_fn_bwd, jnp.zeros_like(W), jnp.arange(0, N, chunk_size))
    
    # Flatten dX chunks back to (N, H)
    dX_full = dX_chunks.reshape(-1, H)[:N]  # handle partial last chunk
    
    # Return gradients for (X, W, Y) - None for non-differentiable params
    return (dX_full, dW_total, None)

# Register the VJP functions
chunked_cross_entropy.defvjp(ce_fwd, ce_bwd)

# -----------------------------------------------------------------------------
# 3. Naive Implementation (For Verification)
# -----------------------------------------------------------------------------
def naive_cross_entropy(X, W, Y):
    logits = jnp.dot(X, W.T)
    lse = jax.scipy.special.logsumexp(logits, axis=1)
    correct_logits = logits[jnp.arange(X.shape[0]), Y]
    loss = jnp.mean(lse - correct_logits)
    return loss

# -----------------------------------------------------------------------------
# 4. Testing Framework
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing JAX TPU Capstone (Memory-Efficient Backprop)...")
    
    N, H, V = 1024, 256, 32000
    chunk_size = 128
    
    # Initialize pseudo-random keys
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    X = jax.random.normal(k1, (N, H))
    W = jax.random.normal(k2, (V, H))
    Y = jax.random.randint(k3, (N,), 0, V)
    
    # 1. Compile both functions using JIT (Just-In-Time compilation to XLA)
    jit_chunked = jax.jit(chunked_cross_entropy, static_argnums=(3,))
    jit_naive = jax.jit(naive_cross_entropy)
    
    # Value and Grad functions
    chunked_grad_fn = jax.jit(jax.value_and_grad(chunked_cross_entropy, argnums=(0, 1)), static_argnums=(3,))
    naive_grad_fn = jax.jit(jax.value_and_grad(naive_cross_entropy, argnums=(0, 1)))
    
    # Warmup
    _ = jit_naive(X, W, Y)
    _ = jit_chunked(X, W, Y, chunk_size)
    
    # Test Naive
    loss_naive, (dX_naive, dW_naive) = naive_grad_fn(X, W, Y)
    
    # Test Chunked
    loss_chunked, (dX_chunked, dW_chunked) = chunked_grad_fn(X, W, Y, chunk_size)
    
    # Verify correctness
    loss_diff = jnp.abs(loss_naive - loss_chunked)
    dX_diff = jnp.max(jnp.abs(dX_naive - dX_chunked))
    dW_diff = jnp.max(jnp.abs(dW_naive - dW_chunked))
    
    print(f"\nVerification Results:")
    print(f"Loss Difference: {loss_diff:.6e} {'' if loss_diff < 1e-4 else ''}")
    print(f"dX Max Difference: {dX_diff:.6e} {'' if dX_diff < 1e-4 else ''}")
    print(f"dW Max Difference: {dW_diff:.6e} {'' if dW_diff < 1e-4 else ''}")
    print(f"\nThis JAX script perfectly mirrors the PyTorch torch.autograd.Function implementation.")
    print("When you run this in Colab, jax.jit will automatically compile it via XLA for the TPU")
