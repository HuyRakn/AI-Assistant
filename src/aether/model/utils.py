import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Any
import os

def init_weights(model: nn.Module, std: float = 0.02):
    """
    Initialize weights Titan-style (Llama/GPT).
    """
    def _init_fn(m):
        if isinstance(m, nn.Linear):
            # Normal initialization for weights
            m.weight = mx.random.normal(m.weight.shape) * std
            if "bias" in m:
                 m.bias = mx.zeros(m.bias.shape)
        elif isinstance(m, nn.Embedding):
            m.weight = mx.random.normal(m.weight.shape) * std
        elif isinstance(m, nn.LayerNorm) or "RMSNorm" in str(type(m)):
            m.weight = mx.ones(m.weight.shape)
            if "bias" in m:
                m.bias = mx.zeros(m.bias.shape)

    # MLX models don't have .apply() like PyTorch yet easily?
    # We can iterate parameters easily.
    # Actually MLX nn.Module stores params in .parameters() dict. 
    # But to update them in place cleanly with structure logic:
    
    # Simple recursive walker or just update leafs? 
    # MLX nn.Module tree walker:
    
    # For now, let's just use a simple leaf iterator if possible or just assume default MLX init is 'good enough' for now 
    # but the user asked for implementations.
    # Let's iterate named modules.
    
    for name, m in model.named_modules():
        _init_fn(m)

class CheckpointManager:
    """
    Governance Module: Checkpoint Manager.
    Uses .safetensors for efficient, zero-copy loading.
    """
    def __init__(self, directory: str = "checkpoints/titan"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        
    def save(self, model: nn.Module, step: int):
        path = os.path.join(self.directory, f"step_{step}.safetensors")
        print(f"💾 Saving checkpoint: {path}")
        model.save_weights(path)
        
    def load(self, model: nn.Module, path: str):
        if not os.path.exists(path):
            print(f"⚠️ Checkpoint not found: {path}")
            return False
        
        print(f"♻️  Loading architecture state from {path}...")
        try:
            model.load_weights(path)
            return True
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return False
