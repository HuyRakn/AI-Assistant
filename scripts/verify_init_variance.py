import sys
import os
import math
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import mlx.core as mx
from aether.model.transformer import AetherForCausalLM
from aether.model.configuration import AetherConfig

def verify_variance():
    print("🧪 Verifying Sovereign Initialization...")
    
    # 1. Config (Standard Small LLaMA)
    # dim=128 (small for check), layers=12
    config = AetherConfig(
        vocab_size=1000,
        dim=256,
        n_layers=12,
        n_heads=4,
        hidden_dim=256*4
    )
    
    model = AetherForCausalLM(config)
    
    # Eval mode (though init is modifying state directly)
    mx.eval(model.parameters())
    
    print(f"Model initialized with L={config.n_layers}.")
    
    # 2. Check Residual Layers (o_proj, down_proj)
    # Target Scale: Xavier * (1 / sqrt(2*L))
    # Xavier Limit (Uniform): sqrt(6 / (fan_in + fan_out))
    # Xavier StdDev (Uniform): Limit / sqrt(3) = sqrt(2 / (fan_in + fan_out))
    
    # For o_proj: fan_in=256, fan_out=256
    fan_in = 256
    fan_out = 256
    xavier_std = math.sqrt(2 / (fan_in + fan_out)) # ~0.0625
    
    # GPT-2 Scale
    scale_factor = 1.0 / math.sqrt(2 * config.n_layers) # 1 / sqrt(24) = 1/4.899 ~ 0.204
    
    expected_std = xavier_std * scale_factor # ~0.0127
    
    # Get last layer o_proj
    # Params dict structure is nested.
    # But layer list is accessible via model.model.layers
    last_layer = model.model.layers[11]
    
    o_proj_w = last_layer.self_attn.o_proj.weight
    o_proj_std = np.std(np.array(o_proj_w))
    
    print(f"\n--- Residual Projection (o_proj) ---")
    print(f"Expected Xavier Std: {xavier_std:.6f}")
    print(f"Scaling Factor (1/sqrt(2L)): {scale_factor:.6f}")
    print(f"Expected Final Std: {expected_std:.6f}")
    print(f"Actual Std: {o_proj_std:.6f}")
    
    if abs(o_proj_std - expected_std) < 0.005:
        print("✅ Correctly Scaled.")
    else:
        print("❌ Scaling Mismatch!")

    # 3. Check SwiGLU Gate (gate_proj)
    # He Normal (Gain=sqrt(2))
    # Std = sqrt(2) / sqrt(fan_in) = sqrt(2/fan_in)
    gate_proj_w = last_layer.mlp.gate_proj.weight
    # fan_in is dim=256.
    he_std = math.sqrt(2 / 256) # sqrt(0.0078) ~ 0.088
    
    actual_gate_std = np.std(np.array(gate_proj_w))
    
    print(f"\n--- SwiGLU Gating (gate_proj) ---")
    print(f"Expected He Std (Gain=sqrt(2)): {he_std:.6f}")
    print(f"Actual Std: {actual_gate_std:.6f}")
    
    if abs(actual_gate_std - he_std) < 0.005:
        print("✅ Correctly Initialized (He Normal).")
    else:
        print("❌ Initialization Mismatch!")

if __name__ == "__main__":
    verify_variance()
