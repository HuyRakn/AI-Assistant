import mlx.core as mx
import mlx.nn as nn
import numpy as np

def debug_rope():
    print("🔍 DEBUG: RoPE Investigation")
    dim = 64
    # Shape: [Batch, Seq, Head, Dim]
    x = mx.random.normal((1, 10, 4, dim))
    
    print(f"Shape: {x.shape}")
    
    # 1. Test Offset=1 (Maybe offset=0 is identity?)
    print("\n--- Test 1: mx.fast.rope (offset=1) ---")
    try:
        y1 = mx.fast.rope(x, dim, traditional=False, base=10000.0, scale=1.0, offset=1)
        check_change(x, y1, "Offset=1")
    except Exception as e:
        print(f"❌ Default failed: {e}")

    # 4. Test Shape Transpose: [Batch, Head, Seq, Dim]
    print("\n--- Test 4: Shape [Batch, Head, Seq, Dim] ---")
    x_bhld = mx.transpose(x, (0, 2, 1, 3)) # (1, 4, 10, 64)
    print(f"Shape: {x_bhld.shape}")
    try:
        y4 = mx.fast.rope(x_bhld, dim, traditional=False, base=10000.0, scale=1.0, offset=0)
        
        x_np = np.array(x_bhld)
        y_np = np.array(y4)
        
        # Check Pos 1 (dim 2 index 1)
        diff = np.abs(x_np[0,0,1,:4] - y_np[0,0,1,:4]).max()
        print(f"[BHLD] Diff at Pos 1: {diff:.6f}")
        if diff > 1e-5:
            print("✅ [BHLD] Rotation confirmed.")
        else:
            print("⚠️  [BHLD] IDENTITY.")
            
    except Exception as e:
        print(f"❌ BHLD failed: {e}")
        
    # 3. Test nn.RoPE (Official Module)
    print("\n--- Test 3: nn.RoPE Module ---")
    rope_mod = nn.RoPE(dim)
    y3 = rope_mod(x)
    check_change(x, y3, "nn.RoPE")
    
    # 4. Debug Frequencies
    # Maybe dim is interpreted wrong?
    # mx.fast.rope(x, xw, ...) 
    # Document says: xw is "The size of the feature dimension to be rotated."
    # If dims matches the last dimension, it rotates.
    
def check_change(x, y, name):
    x_np = np.array(x)
    y_np = np.array(y)
    
    # Check Pos 0 (Should be identity)
    diff0 = np.abs(x_np[0,0,0,:4] - y_np[0,0,0,:4]).max()
    print(f"[{name}] Diff at Pos 0: {diff0:.6f}")
    
    # Check Pos 1 (Should rotate)
    diff1 = np.abs(x_np[0,1,0,:4] - y_np[0,1,0,:4]).max()
    print(f"[{name}] Diff at Pos 1: {diff1:.6f}")
    
    if diff1 < 1e-5:
        print(f"⚠️  [{name}] IDENTITY DETECTED at Pos 1!")
    else:
        print(f"✅ [{name}] Rotation confirmed.")

if __name__ == "__main__":
    debug_rope()
