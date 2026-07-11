import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

def main():
    # 1. Initialize Distributed Environment (NCCL for multi-GPU)
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f" Initialized Rank {local_rank} / {dist.get_world_size()} on {device}")

    # 2. Memory Optimizations for Kaggle (T4 16GB)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 3. Load QLoRA 4-bit Model
    # Crucial: device_map cannot be "auto" with FSDP, we must load to our specific rank's GPU
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
        device_map={"": local_rank}, # Bind entirely to this process's GPU
        torch_dtype=torch.bfloat16,
    )

    # Freeze 4-bit weights and prepare for training
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
    
    # 6. Train via SFTTrainer using Accelerate's native FSDP config
    # Accelerate (PR 3394) natively handles FSDP + QLoRA by intelligently
    # wrapping the layers and bypassing the uint8 parameters safely.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            output_dir="./fsdp2_outputs",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=10,
            logging_steps=1,
            save_strategy="no",  # Disable checkpointing - Params4bit FSDP serialization is broken
            learning_rate=2e-4,
            optim="adamw_torch",  # adamw_8bit is incompatible with FSDP2 DTensors
            ddp_find_unused_parameters=False,
            # Native HuggingFace FSDP parameters:
            # NOTE: Do NOT add "offload" here - offloading to CPU breaks nccl which is GPU-only
            fsdp=["full_shard", "auto_wrap"],
            fsdp_config={
                "backward_prefetch": "backward_pre",
                "forward_prefetch": "False",
                "use_orig_params": "False",
            }
        )
    )

    print(f" Starting FSDP QLoRA Training on Rank {local_rank}...")
    trainer.train()
    dist.destroy_process_group()

if __name__ == "__main__":
    # Launch instruction:
    # torchrun --nproc_per_node=2 nvidia_cuda/task_b_fsdp2.py
    main()
