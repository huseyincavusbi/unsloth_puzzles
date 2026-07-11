import torch
import triton
import triton.language as tl

NF4_VALS = [
    -1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0
]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def _dequantize_nf4_kernel(
    weight_ptr,
    absmax_ptr,
    state2_code_ptr,
    state2_absmax_ptr,
    out_ptr,
    nf4_lut_ptr,
    n_elements,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    byte_offsets = offsets // 2
    packed = tl.load(weight_ptr + byte_offsets, mask=mask)
    is_high = offsets % 2
    nibble = tl.where(is_high == 1, (packed >> 4) & 0x0F, packed & 0x0F)

    nf4_val = tl.load(nf4_lut_ptr + nibble, mask=mask)

    l1_block = offsets // 64
    absmax_uint8 = tl.load(absmax_ptr + l1_block, mask=mask)
    code_val = tl.load(state2_code_ptr + absmax_uint8)
    l2_block = l1_block // 256
    s2_scale = tl.load(state2_absmax_ptr + l2_block, mask=mask)
    real_absmax = code_val * s2_scale + offset

    out = nf4_val * real_absmax
    tl.store(out_ptr + offsets, out, mask=mask)


def dequantize_nf4_amd(weight_uint8, quant_state):
    device = weight_uint8.device
    target_dtype = quant_state.dtype

    n_elements = weight_uint8.numel() * 2
    nf4_lut = torch.tensor(NF4_VALS, dtype=target_dtype, device=device)

    state2_code = quant_state.state2.code.to(device)
    state2_absmax = quant_state.state2.absmax.flatten().to(device)
    offset = float(quant_state.offset.item())
    absmax = quant_state.absmax.flatten().to(torch.int32)

    out = torch.empty(n_elements, dtype=target_dtype, device=device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _dequantize_nf4_kernel[grid](
        weight_uint8, absmax, state2_code, state2_absmax,
        out, nf4_lut, n_elements, offset,
    )
    return out.reshape(quant_state.shape)


def your_dequantize_nf4(weight_module):
    return dequantize_nf4_amd(
        weight_module.weight.data,
        weight_module.weight.quant_state,
    )


if __name__ == "__main__":
    print("Testing AMD-Optimized Triton NF4 Kernel")
    print(" Ready to compile via HIP on AMD ROCm")
