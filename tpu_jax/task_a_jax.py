import jax
import jax.numpy as jnp
from functools import partial

# Standard NF4 exact values
NF4_VALUES = jnp.array([
    -1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0
], dtype=jnp.bfloat16)

@partial(jax.jit, static_argnames=("block_size",))
def dequantize_nf4_jax(weight_uint8, absmax, block_size=64):
    """
    JAX / XLA native NF4 dequantization.
    This replaces the Triton kernel from the official challenge.
    """
    # 1. Unpack 8-bit into two 4-bit values (nibbles)
    # Bits 0-3: first value, Bits 4-7: second value
    low_nibble = jnp.bitwise_and(weight_uint8, jnp.uint8(0x0F))
    high_nibble = jnp.right_shift(weight_uint8, jnp.uint8(4))
    
    # 2. Interleave to form (N*2,) int32 array for lookup
    # e.g. [low0, high0, low1, high1, ...]
    unpacked = jnp.stack([low_nibble, high_nibble], axis=1).flatten()
    
    # 3. Table lookup against exact NF4 floats
    dequantized = jnp.take(NF4_VALUES, unpacked)
    
    # 4. Scale by block-wise absmax
    dequantized_blocks = dequantized.reshape(-1, block_size)
    scaled = dequantized_blocks * absmax[:, None]
    
    return scaled.flatten()

if __name__ == "__main__":
    print("Testing JAX NF4 Dequantization (Task A)...")
    N_bytes = 4096
    block_size = 64
    
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    
    # Mock data
    weight_uint8 = jax.random.randint(key1, (N_bytes,), 0, 256, dtype=jnp.uint8)
    absmax = jax.random.uniform(key2, (N_bytes * 2 // block_size,), dtype=jnp.bfloat16)
    
    # Compile and run
    out = dequantize_nf4_jax(weight_uint8, absmax)
    print("Dequantized shape:", out.shape)
    print("Dequantized preview:", out[:8])
    print(" JAX NF4 kernel compiled and executed successfully")
