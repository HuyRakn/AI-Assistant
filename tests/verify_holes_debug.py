import unittest
import sys
import os
sys.path.append(os.getcwd())

import mlx.core as mx
from src.aether.model.transformer import AetherModel, AetherConfig

class TestCriticalHolesDebug(unittest.TestCase):
    
    def test_initialization_debug(self):
        print("\n🧪 Testing Sovereign Initialization (DEBUG)...")
        config = AetherConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            vocab_size=100
        )
        model = AetherModel(config)
        
        # Pick all weights
        weights = {}
        for k, v in model.parameters().items():
             # recursively flatten for inspection logic? 
             pass
             
        # Just check one specific known path
        # In AetherModel, structure is likely: layers -> list
        # MLX flatten keys: "layers.0.self_attn.q_proj.weight"
        
        # Let's traverse to find a leaf
        # Assuming model.layers is a list
        target_layer = model.layers[0]
        target_linear = target_layer.self_attn.q_proj
        old_w = target_linear.weight
        old_val = old_w[0, 0].item()
        print(f"Old Value: {old_val}")
        
        # APPLY
        model.apply_initialization()
        
        # CHECK
        new_w = target_linear.weight
        new_val = new_w[0, 0].item()
        print(f"New Value: {new_val}")
        
        if old_val == new_val:
            print("❌ FAILURE: Value matched.")
            # Verify if param tree keys match what we expect
            # We can print keys encountered by init_fn if we had logging
        else:
            print("✅ SUCCESS: Value changed.")
            
        self.assertNotEqual(old_val, new_val)

if __name__ == "__main__":
    unittest.main()
