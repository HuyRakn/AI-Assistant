import unittest
import sys
import os
sys.path.append(os.path.abspath("src"))

# We mock AetherBPE to verify it's utilized
from unittest.mock import MagicMock, patch

class TestFinalIntegration(unittest.TestCase):
    
    @patch('aether.tokenization.bpe.AetherBPE') 
    def test_factory_uses_sovereign_bpe(self, MockBPE):
        from aether.preprocessing.pipeline import AetherDataFactory
        
        # Setup Mock
        mock_instance = MockBPE.return_value
        mock_instance.encode.return_value = [1, 2, 3]
        
        # Init Factory
        factory = AetherDataFactory()
        
        # Check initialization
        print("🏭 Checking Data Factory Components...")
        self.assertTrue(hasattr(factory, 'tokenizer'), "Factory missing tokenizer")
        self.assertTrue(hasattr(factory, 'eng_norm'), "Factory missing EnglishNormalizer")
        
        # Test Tokenization Flow
        tokens = factory._tokenize("test input")
        
        # Verify call
        mock_instance.encode.assert_called_with("test input")
        print("✅ Factory delegated tokenization to Sovereign AetherBPE.")
        
        # Test Process Text Flow (Eng Norm)
        # We can't easily mock inner attributes unless we patch the class used in init
        # But we can check results if we rely on actual implementation for norm
        res = factory._process_text("Devices running.")
        # "Devices" -> "devic" (Porter) or "device" (Simple)
        # "running" -> "run"
        print(f"Processed: {res}")
        self.assertIn("run", res) 
        # "Devices" -> "device" (Simple)
        self.assertIn("device", res)
        print("✅ Factory applied English Normalization.")

if __name__ == "__main__":
    unittest.main()
