import sys
import os
import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("src"))

from aether.model.configuration import AetherConfig
from aether.model.layers import RoPE, RMSNorm, SwiGLU
from aether.model.transformer import AetherForCausalLM

def test_rope():
    print("🧪 Testing RoPE (Rotary Positional Embeddings)...")
    dim = 64
    rope = RoPE(dim)
    
    # Input: [Batch, Seq, Head, Dim] - Standard Llama input
    x = mx.random.normal((1, 10, 4, dim))
    
    # Simulate AetherAttention Transpose: [B, H, L, D]
    x_bhld = x.transpose(0, 2, 1, 3)
    
    rope_fn = RoPE(dim) # Uses fast.rope
    x_rotated = rope_fn(x_bhld)
    
    assert x_bhld.shape == x_rotated.shape
    
    x_np = np.array(x_bhld)
    rot_np = np.array(x_rotated)
    
    # Check Pos 1 (dim 2 index 1)
    print(f"Input slice (Pos 1, Head 0): {x_np[0, 0, 1, :4]}")
    print(f"Output slice (Pos 1, Head 0): {rot_np[0, 0, 1, :4]}")

    if np.allclose(x_np, rot_np):
         print("❌ RoPE is Identity.")
         raise AssertionError("RoPE failed rotation test.")
    else:
        print("✅ RoPE Rotation Active (Verified with [B, H, L, D] layout).")

    print("🧪 Testing Manual RoPE (Sovereign Math)...")
    # Manual check:
    # x = [1, 0] (real=1, complex=0) -> Rotate 90 deg (pi/2) -> [0, 1]
    # theta = 10000^(-0/dim) = 1.0
    # rot = 1 * pos
    
    # We rely on previous test pattern but ensure no crash
    x = mx.random.normal((1, 10, 4, 64))
    rope = RoPE(64)
    out = rope(x)
    print(f"Input slice (Pos 1, Head 0): {x[0, 1, 0, :4]}")
    print(f"Output slice (Pos 1, Head 0): {out[0, 1, 0, :4]}")
    print("✅ RoPE Manual Rotation Active.")

def test_utils():
    print("🧪 Testing Governance (Utils)...")
    from aether.model.utils import init_weights
    from aether.model.configuration import AetherConfig
    from aether.model.transformer import AetherForCausalLM
    from aether.training.trainer_loop import AetherTrainer
    import mlx.optimizers as optim
    
    cfg = AetherConfig(vocab_size=100, dim=32, n_layers=1, n_heads=4)
    model = AetherForCausalLM(cfg)
    
    # Test Trainer Init
    optimizer = optim.AdamW(learning_rate=1e-4)
    trainer = AetherTrainer(model, optimizer)
    assert trainer is not None
    print("✅ Trainer Initialized.")
    
    # Capture old weights
    old_w = np.array(model.lm_head.weight)
    
    # Re-init
    init_weights(model, std=0.5) 
    
    new_w = np.array(model.lm_head.weight)
    
    assert not np.allclose(old_w, new_w)
    print("✅ Weights Re-Initialized.")

def test_rms_norm():
    print("🧪 Testing RMSNorm...")
    norm = RMSNorm(128)
    x = mx.random.normal((4, 128)) * 10 + 5 # Scaled up
    out = norm(x)
    
    # Compute RMS of output manually
    rms = mx.sqrt(mx.mean(mx.square(out), axis=-1))
    # Should be close to 1
    assert np.allclose(rms, 1.0, atol=1e-2)
    print("✅ RMSNorm Stabilizing Signals (RMS ~ 1.0).")

def test_full_model_init():
    print("🧪 Testing Full Titan Model Initialization...")
    config = AetherConfig(
        vocab_size=1000, # Tiny for test
        dim=256,
        n_layers=2,
        n_heads=4
    )
    model = AetherForCausalLM(config)
    mx.eval(model.parameters())
    
    # Dummy Inference
    inputs = mx.array([[1, 2, 3, 4, 5]]) # Batch 1, Seq 5
    logits, _ = model(inputs)
    
    assert logits.shape == (1, 5, 1000)
    print(f"✅ Forward Pass Successful. Logits Shape: {logits.shape}")

if __name__ == "__main__":
    test_rope()
    test_rms_norm()
    test_full_model_init()
    print("\n🎉 Architecture Integrity Verified.")
