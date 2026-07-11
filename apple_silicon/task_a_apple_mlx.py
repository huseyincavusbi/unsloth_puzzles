import torch
import mlx.core as mx
from torch.utils.dlpack import from_dlpack
from bitsandbytes.functional import quantize_4bit, dequantize_4bit

NF4_VALS = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534,  0.16093020141124725,  0.24611230194568634, 0.33791524171829224,
     0.44070982933044434,  0.5626170039176941,   0.7229568362236023,  1.0,
]
NF4_MX = mx.array(NF4_VALS, dtype=mx.float32)

def mlx_dequantize(weight_module):
    """
    Apple Silicon MLX implementation of bitsandbytes NF4 dequantization.
    Directly converts 4-bit weights into PyTorch bfloat16/float16 buffers 
    using a zero-copy dlpack bridge.
    """
    qs = weight_module.weight.quant_state
    W = mx.array(weight_module.weight.numpy().flatten())
    A = mx.array(qs.absmax.numpy().flatten())
    C = mx.array(qs.state2.code.numpy().flatten())
    S = mx.array(qs.state2.absmax.numpy().flatten())
    offset = qs.offset.item()
    
    # Extract nibbles
    hi = mx.bitwise_and(mx.right_shift(W, 4), 15)
    lo = mx.bitwise_and(W, 15)
    
    # Lookup NF4
    nf4_hi = NF4_MX[hi]
    nf4_lo = NF4_MX[lo]
    
    # Level-2 absmax lookup
    absmax_fp = C[A]
    S_expanded = mx.repeat(S, 256)
    real_absmax = absmax_fp * S_expanded + offset
    
    real_absmax_expanded = mx.repeat(real_absmax, 32)
    w_hi = nf4_hi * real_absmax_expanded
    w_lo = nf4_lo * real_absmax_expanded
    
    # Interleave hi and lo
    w_combined = mx.stack([w_hi, w_lo], axis=1)
    w_combined = mx.flatten(w_combined)
    w_combined = mx.reshape(w_combined, list(qs.shape))
    
    out_dtype_str = str(qs.dtype).split('.')[-1]
    if out_dtype_str == 'float16':
        w_combined = w_combined.astype(mx.float16)
    elif out_dtype_str == 'bfloat16':
        w_combined = w_combined.astype(mx.bfloat16)
    else:
        w_combined = w_combined.astype(mx.float32)
    
    mx.eval(w_combined)
    
    # GC fix: keep a python reference alive on the module 
    weight_module._mlx_w = w_combined
    return from_dlpack(w_combined)

if __name__ == "__main__":
    import time
    from bitsandbytes.nn import Linear4bit, Params4bit
    from peft.utils.integrations import dequantize_module_weight as peft_dequantize
    
    print("Testing MLX Native NF4 Dequantization...")
    layer = Linear4bit(2048, 8192, bias=None, compute_dtype=torch.float16, compress_statistics=True, quant_type='nf4')
    W = torch.randn(8192, 2048, dtype=torch.float32)
    qW, qs = quantize_4bit(W, quant_type='nf4', compress_statistics=True)
    layer.weight = Params4bit(qW, requires_grad=False, quant_state=qs)
    layer.weight.bnb_quantized = True
    
    # Run once to compile / warmup
    out_mlx = mlx_dequantize(layer)
    out_pt = peft_dequantize(layer).to('mps')
    
    # Check max diff
    max_diff = (out_pt - out_mlx).abs().max().item()
    print(f"Max difference between MLX and PyTorch bitsandbytes: {max_diff:.6f}")
    
    start = time.time()
    for _ in range(10):
        mlx_dequantize(layer)
    end = time.time()
    print(f" Executed 10 MLX dequantize passes in {end - start:.4f}s")
