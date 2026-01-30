import sys
import os
import mlx.core as mx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aether.model.configuration import AetherConfig
from aether.model.transformer import AetherForCausalLM
from aether.training.optimizer import SovereignAdamW
from aether.training.trainer import SovereignTrainer

def dry_run():
    print("🚀 Initiating Dry Run: The Sovereign Training Loop...")
    
    # 1. Config
    config = AetherConfig(
        vocab_size=1000,
        dim=128,
        n_layers=2, # Small for speed
        n_heads=2,
        hidden_dim=256
    )
    
    # 2. Model & Init
    print("   2. Building Model & Applying Sovereign Init...")
    model = AetherForCausalLM(config) 
    # Init was applied in __init__ of AetherForCausalLM
    
    # 3. Optimizer
    print("   3. Preparing Sovereign AdamW...")
    optimizer = SovereignAdamW(learning_rate=3e-4, weight_decay=1e-2)
    
    # 4. Trainer
    print("   4. Assembling Sovereign Trainer...")
    trainer = SovereignTrainer(model, optimizer, grad_accumulation_steps=2, max_grad_norm=1.0)
    
    # 5. Synthetic Data
    # 10 batches, batch size 4, seq len 16
    print("   5. Generating Synthetic Stream...")
    dataset = []
    for _ in range(10):
        # random tokens [0, 1000)
        batch_tokens = mx.random.randint(0, 1000, shape=(4, 16))
        dataset.append({'input_ids': batch_tokens})
        
    val_dataset = []
    for _ in range(2):
        batch_tokens = mx.random.randint(0, 1000, shape=(4, 16))
        val_dataset.append({'input_ids': batch_tokens})
        
    # 6. Run
    print("\n⚡ EXECUTION BEGINS...")
    trainer.train(dataset, eval_dataset=val_dataset, steps=5, report_interval=1, eval_interval=2)
    
    print("\n✅ Verification SUCCESS: The Perpetual Engine is Alive.")

if __name__ == "__main__":
    dry_run()
