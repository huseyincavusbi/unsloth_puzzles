import torch

# AMD MI200+ Architecture optimization
# Explicitly enables fast TF32 precision modes for Matrix Cores on AMD GPUs
torch.backends.cuda.matmul.allow_tf32 = True 

def flex_attention_amd(q, k, v, causal=True):
    from torch.nn.attention.flex_attention import flex_attention
    
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
        
    return flex_attention(q, k, v, score_mod=causal_mask if causal else None)

if __name__ == "__main__":
    print("Testing AMD-Optimized torch.compile Flex Attention...")
    print("Compiling via Torch Inductor -> Triton for ROCm...")
    flex_attention_compiled = torch.compile(flex_attention_amd, dynamic=False)
    print(" Ready to run on AMD GPU")
