import unittest
import sys
import os
import shutil
import mlx.core as mx
import mlx.nn as nn
sys.path.append(os.getcwd())

from src.aether.model.transformer import AetherModel, AetherConfig
from src.aether.tokenization.gluer import CompoundGluer

class TestCriticalHoles(unittest.TestCase):
    
    def test_initialization(self):
        print("\n🧪 Testing Sovereign Initialization...")
        config = AetherConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            vocab_size=100
        )
        model = AetherModel(config)
        
        # Capture old weights (snapshot)
        # We pick one weight to check
        old_weight = model.layers[0].self_attn.q_proj.weight
        old_val = old_weight[0, 0].item()
        
        # Apply Init
        model.apply_initialization()
        
        # Capture new weights
        new_weight = model.layers[0].self_attn.q_proj.weight
        new_val = new_weight[0, 0].item()
        
        print(f"Weight Check: {old_val} -> {new_val}")
        self.assertNotEqual(old_val, new_val, "Weights did not change after initialization!")
        
        # Check Distribution (Roughly)
        # Xavier Uniform for Attn: limit = sqrt(6 / (64+64)) = sqrt(6/128) ~ sqrt(0.046) ~ 0.21
        # Values should be within [-0.22, 0.22]
        max_val = mx.max(mx.abs(new_weight)).item()
        print(f"Max Weight Value: {max_val}")
        self.assertTrue(max_val < 0.5, "Weights initialized too large for Xavier?")
        print("✅ Initialization Logic Active.")

    def test_gluer(self):
        print("\n🧪 Testing Compound Gluer...")
        # create temp file
        os.makedirs("tests/artifacts", exist_ok=True)
        tpath = "tests/artifacts/glue_test.txt"
        with open(tpath, "w") as f:
            # Repeat "Hà Nội" many times to boost PMI
            for _ in range(20):
                f.write("Hà Nội là thủ đô của Việt Nam\n")
            # Add noise
            f.write("Hà khác Nội khác\n")
            
        gluer = CompoundGluer(min_count=5, threshold=0.1)
        gluer.train(tpath)
        
        # Output
        opath = "tests/artifacts/glue_out.txt"
        gluer.glue(tpath, opath)
        
        with open(opath, "r") as f:
            content = f.read()
            
        print(f"Glued Sample: {content[:100]}...")
        self.assertIn("Hà_Nội", content, "Failed to glue 'Hà Nội'")
        self.assertIn("Việt_Nam", content, "Failed to glue 'Việt Nam'")
        print("✅ Compound Gluer Active.")

if __name__ == "__main__":
    unittest.main()
