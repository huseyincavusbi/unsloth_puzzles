import jax
import jax.numpy as jnp
from functools import partial
import time

@partial(jax.jit, static_argnames=("is_causal",))
def flex_attention_jax(q, k, v, is_causal=True):
    """
    JAX Native Compiled Attention.
    Replaces torch.compile + flex_attention.
    XLA automatically fuses this entire function into a single TPU operator.
    """
    B, H, L, D = q.shape
    
    # 1. Q @ K.T
    scores = jnp.einsum('bhld,bhmd->bhlm', q, k)
    
    # 2. Scale
    scores = scores * (1.0 / jnp.sqrt(D))
    
    # 3. Causal Mask
    if is_causal:
        # Create lower triangular mask
        mask = jnp.tril(jnp.ones((L, L), dtype=bool))
        # Apply -inf where mask is False
        scores = jnp.where(mask, scores, -jnp.inf)
        
    # 4. Softmax (XLA fuses the subtraction of max for stability)
    max_scores = jnp.max(scores, axis=-1, keepdims=True)
    exp_scores = jnp.exp(scores - max_scores)
    probs = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)
    
    # 5. @ V
    out = jnp.einsum('bhlm,bhmd->bhld', probs, v)
    return out

if __name__ == "__main__":
    print("Testing JAX Fused Attention (Task C)...")
    B, H, L, D = 2, 4, 1024, 64
    
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    
    # 1. Trigger JIT Compilation (Warmup)
    print("Compiling JAX graph into XLA...")
    _ = flex_attention_jax(q, k, v).block_until_ready()
    
    # 2. Benchmark execution time
    print("Running benchmarks...")
    start = time.perf_counter()
    num_passes = 10
    for _ in range(num_passes):
        out = flex_attention_jax(q, k, v)
        out.block_until_ready()
    end = time.perf_counter()
    
    print(f" JAX Compiled Attention: {num_passes} passes in {end - start:.4f}s")
    print("Output shape:", out.shape)
