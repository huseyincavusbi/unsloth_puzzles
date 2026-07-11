import os
# Force JAX to simulate 2 distinct hardware devices on a single CPU
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import time

def simulate_fsdp2_in_jax():
    print(" Initializing JAX Multi-Device Cluster...")
    devices = jax.devices()
    print(f"Detected Devices: {devices}")
    assert len(devices) >= 2, "Failed to simulate 2 devices"

    # 1. Create a Hardware Mesh (equivalent to torch.distributed.init_process_group)
    # We lay out our 2 virtual devices in a 1D grid named 'fsdp_axis'
    mesh = Mesh(mesh_utils.create_device_mesh((2,)), axis_names=('fsdp_axis',))

    # 2. Define our Sharding Strategy (equivalent to FullyShardedDataParallel)
    # We tell JAX to shard the first dimension of our weight matrix across the 'fsdp_axis'
    fsdp_sharding = NamedSharding(mesh, P('fsdp_axis', None))
    
    # Replicated strategy (for inputs/outputs that every device needs a full copy of)
    replicated = NamedSharding(mesh, P(None, None))

    print("\n Generating massive weight matrix (Simulating 8B parameters)...")
    # We generate a large matrix on the CPU...
    key = jax.random.PRNGKey(0)
    W_full = jax.random.normal(key, (8192, 8192))
    
    # 3. Apply FSDP Sharding
    # This physically chops the matrix in half and sends 4096 rows to Device 0 and 4096 to Device 1
    W_sharded = jax.device_put(W_full, fsdp_sharding)
    
    print("Weight Matrix Sharding Profile:")
    jax.debug.visualize_array_sharding(W_sharded)
    print(f"Is it sharded? {W_sharded.is_fully_replicated == False}")

    # 4. Define our Training Step
    # In JAX, we just write normal math. XLA automatically inserts the NCCL All-Gather and Reduce-Scatter networking
    @jax.jit
    def fsdp_forward_backward(W, X):
        # Forward Pass (XLA automatically inserts All-Gather if needed)
        logits = jnp.dot(X, W.T)
        loss = jnp.sum(logits ** 2)
        
        # Backward Pass (XLA automatically calculates gradients and Reduce-Scatters them back to the mesh)
        dW = jax.grad(lambda w: jnp.sum(jnp.dot(X, w.T) ** 2))(W)
        return loss, dW

    # Dummy Input (replicated across all devices)
    X = jax.device_put(jax.random.normal(key, (128, 8192)), replicated)

    print("\n Running Distributed FSDP Forward/Backward pass...")
    start = time.perf_counter()
    loss, dW = fsdp_forward_backward(W_sharded, X)
    
    # Block until both devices finish
    loss.block_until_ready()
    dW.block_until_ready()
    end = time.perf_counter()
    
    print(f"Loss: {loss}")
    print("Gradient Matrix Sharding Profile (Notice how gradients are automatically sharded to match the weights):")
    jax.debug.visualize_array_sharding(dW)
    print(f"Completed distributed pass in {end - start:.4f}s")
    print(" Simulated FSDP2 JAX training successfully on Apple Silicon")

if __name__ == "__main__":
    simulate_fsdp2_in_jax()
