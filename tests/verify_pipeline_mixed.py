import unittest
import sys
import os
sys.path.append(os.path.abspath("src"))

from aether.preprocessing.english import EnglishNormalizer
from aether.data.normalization import ViToneNormalizer

class TestPipeline(unittest.TestCase):
    def test_mixed_normalization(self):
        print("🧪 Testing Mixed Language Normalization...")
        
        vi_norm = ViToneNormalizer()
        eng_norm = EnglishNormalizer()
        
        # Scenario: Vietnamese sentence with English technical terms
        input_text = "Hệ thống đang running rất smoothly trên các devices."
        # Expect: "running" -> "run", "smoothly" -> "smoothli", "devices" -> "devic" (Porter stem)
        # Vietnamese words should be preserved (assuming they don't look like english suffixes)
        
        # 1. VN Norm (Tone calc)
        # "Hệ" -> "Hệ", "thống" -> "thống" (already clean)
        step1 = vi_norm.normalize(input_text)
        
        # 2. EN Norm (Stemming)
        step2 = eng_norm.normalize(step1).lower()
        
        print(f"Input:    {input_text}")
        print(f"Pipeline: {step2}")
        
        self.assertIn("run", step2.split())
        self.assertIn("device", step2.split())
        
        # Ensure VN text integrity
        self.assertIn("hệ", step2)
        self.assertIn("thống", step2)
        
        print("✅ Pipeline Successfully processes Mixed Content.")

if __name__ == "__main__":
    unittest.main()
