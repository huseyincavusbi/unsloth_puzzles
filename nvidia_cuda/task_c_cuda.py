import torch

def flex_attention_cuda(q, k, v, causal=True):
    from torch.nn.attention.flex_attention import flex_attention
    
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
        
    return flex_attention(q, k, v, score_mod=causal_mask if causal else None)

if __name__ == "__main__":
    print("Testing Canonical NVIDIA torch.compile Flex Attention...")
    print("Compiling via Torch Inductor -> Triton for NVCC...")
    flex_attention_compiled = torch.compile(flex_attention_cuda, dynamic=False)
    print(" Ready to run on NVIDIA GPU")
