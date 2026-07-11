import torch
import triton
import triton.language as tl
from triton import cdiv as triton_cdiv

NF4_VALS = [
    -1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0
]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _dequantize_nf4_kernel(
    weight_ptr,        # uint8  packed 4-bit weights
    absmax_ptr,        # uint8  level-1 absmax (indices into state2.code)
    state2_code_ptr,   # float32  level-2 codebook (256 entries)
    state2_absmax_ptr, # float32  level-2 absmax (1 per 256 level-1 blocks)
    out_ptr,           # output (fp16 or bf16)
    nf4_lut_ptr,       # bf16/fp16 NF4 codebook (16 entries)
    n_elements,        # total number of output weight elements
    offset,            # float32 global offset
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Unpack nibbles: 2 nibbles per byte
    byte_offsets = offsets // 2
    packed = tl.load(weight_ptr + byte_offsets, mask=mask)
    is_odd = offsets & 1
    nibble = tl.where(is_odd == 1, packed & 0x0F, (packed >> 4) & 0x0F)

    # NF4 codebook lookup
    nf4_val = tl.load(nf4_lut_ptr + nibble, mask=mask)

    # Double-dequant of absmax:
    #   level-1 absmax (uint8) → index into state2.code (256-entry dynamic map)
    #       code_val = state2.code[int(quant_state.absmax[block])]
    #   level-2 absmax (float32, 1 per 256 level-1 blocks) → scale factor
    #       real_absmax = code_val * state2.absmax[l1_block // 256] + offset
    l1_block = offsets // 64
    absmax_uint8 = tl.load(absmax_ptr + l1_block, mask=mask)
    code_val = tl.load(state2_code_ptr + absmax_uint8)
    l2_block = l1_block // 256
    s2_scale = tl.load(state2_absmax_ptr + l2_block, mask=mask)
    real_absmax = code_val * s2_scale + offset

    out = nf4_val * real_absmax
    tl.store(out_ptr + offsets, out, mask=mask)


def dequantize_nf4_cuda(weight_uint8, quant_state):
    """
    Single Triton kernel that fully dequantizes NF4 double-quantized weights.
    Takes (weight.data, quant_state) as passed by the notebook harness.
    """
    device = weight_uint8.device
    target_dtype = quant_state.dtype

    n_elements = weight_uint8.numel() * 2
    nf4_lut = torch.tensor(NF4_VALS, dtype=target_dtype, device=device)

    state2_code = quant_state.state2.code.to(device)
    state2_absmax = quant_state.state2.absmax.flatten().to(device)
    offset = float(quant_state.offset.item())
    absmax = quant_state.absmax.flatten().contiguous()

    out = torch.empty(n_elements, dtype=target_dtype, device=device)

    BLOCK_SIZE = 512
    grid = (triton_cdiv(n_elements, BLOCK_SIZE),)
    _dequantize_nf4_kernel[grid](
        weight_uint8, absmax, state2_code, state2_absmax,
        out, nf4_lut, n_elements, offset,
    )
    return out.reshape(quant_state.shape)


def your_dequantize_nf4(weight_module):
    """
    Public entry point matching the notebook harness:
        test_dequantize(your_dequantize_nf4)
    weight_module is a Linear4bit instance.
    """
    return dequantize_nf4_cuda(
        weight_module.weight.data,
        weight_module.weight.quant_state,
    )


if __name__ == "__main__":
    import time
    from bitsandbytes.nn import Linear4bit, Params4bit
    from bitsandbytes.functional import quantize_4bit
    from peft.utils.integrations import dequantize_module_weight as peft_dequantize
    from unsloth.kernels.utils import fast_dequantize

    print("Testing NF4 Double-Dequant Triton Kernel...")

    def unsloth_dequantize(weight):
        return fast_dequantize(weight.weight, weight.weight.quant_state)

    layer = Linear4bit(
        2048, 8192, bias=None,
        compute_dtype=torch.float16,
        compress_statistics=True,
        quant_type='nf4',
    )
    W = torch.randn(8192, 2048, dtype=torch.float32)
    qW, qs = quantize_4bit(W, quant_type='nf4', compress_statistics=True)
    layer.weight = Params4bit(qW, requires_grad=False, quant_state=qs)
    layer.weight.bnb_quantized = True

    n_iters = 100
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        out_triton = your_dequantize_nf4(layer)
        torch.cuda.synchronize()
    triton_time = time.time() - start

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        out_unsloth = unsloth_dequantize(layer)
        torch.cuda.synchronize()
    unsloth_time = time.time() - start

    out_peft = peft_dequantize(layer)

    max_diff_unsloth = (out_unsloth - out_triton).abs().max().item()
    max_diff_peft = (out_peft - out_triton).abs().max().item()
    print(f"  Max diff vs Unsloth : {max_diff_unsloth:.6e}")
    print(f"  Max diff vs PEFT    : {max_diff_peft:.6e}")
    print(f"  Triton  time ({n_iters} iters) : {triton_time:.4f}s")
    print(f"  Unsloth time ({n_iters} iters) : {unsloth_time:.4f}s")
    print(f"  Speedup vs Unsloth             : {unsloth_time / triton_time:.2f}x")
