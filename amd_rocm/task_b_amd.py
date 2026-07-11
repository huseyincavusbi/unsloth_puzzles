import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# AMD ROCm Note:
# PyTorch on ROCm aliases torch.cuda to HIP automatically.
# RCCL (ROCm Communication Collectives Library) is API-compatible with NCCL,
# so we use backend="nccl" here and ROCm will transparently route it through RCCL.
# Launch with: torchrun --nproc_per_node=<num_gpus> amd_rocm/task_b_amd.py

def main():
    # 1. Initialize Distributed Environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"Initialized Rank {local_rank} / {dist.get_world_size()} on {device}")

    # 2. Memory Optimizations for AMD GPUs
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

    # 3. Load QLoRA 4-bit Model
    # Crucial: device_map cannot be "auto" with FSDP, bind to this rank's GPU
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # Store 4-bit weights as bfloat16 so FSDP can shard them without crashing
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        quantization_config=quant_config,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
    )

    # Freeze 4-bit weights and prepare for training so FSDP does not assign gradients to uint8 tensors
    model = prepare_model_for_kbit_training(model)

    # 4. Apply LoRA Adapters
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # 5. Load Dataset
    dataset = load_dataset("imdb", split="train[:100]")

    # 6. Train via SFTTrainer using Accelerate native FSDP config
    # AMD-specific notes:
    # - AMD wavefront size is 64 vs NVIDIA warp 32, but FSDP sharding is
    #   hardware-agnostic at the Python level so no changes needed here.
    # - Do NOT add "offload" to fsdp, CPU offload breaks RCCL (GPU-only backend)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            output_dir="./fsdp2_amd_outputs",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=10,
            logging_steps=1,
            save_strategy="no",  # Params4bit FSDP serialization is broken
            learning_rate=2e-4,
            optim="adamw_torch",  # adamw_8bit is incompatible with FSDP2 DTensors
            ddp_find_unused_parameters=False,
            fsdp=["full_shard", "auto_wrap"],
            fsdp_config={
                "backward_prefetch": "backward_pre",
                "forward_prefetch": "False",
                "use_orig_params": "False",
            }
        )
    )

    print(f"Starting FSDP QLoRA Training on Rank {local_rank}...")
    trainer.train()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
