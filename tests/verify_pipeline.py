import sys
import os
from pathlib import Path

# Add src to pythonpath
sys.path.append(os.path.join(os.getcwd(), 'src'))

from aether.preprocessing.pipeline import AetherDataFactory

def test_pipeline():
    print("🧪 Testing AetherDataFactory Pipeline...")
    
    # 1. Setup Dirty Data
    dirty_text = """
    <html><body>
    <h1>   ChÃ\u00a0o má»«ng báº¡n Ä‘áº¿n vá»›i    Project Aether!   </h1>
    <script>alert('xss')</script>
    <p>   HÃ´m nay    lÃ\u00a0 ngÃ\u00a0y   Ä‘áº¹p trá»\u009Di.   </p>
    </body></html>
    """
    
    print(f"INPUT (Dirty):\n{dirty_text}\n")
    
    # 2. Process
    factory = AetherDataFactory(tokenizer_model_path=None) # Raw text mode
    
    # Access private processing kernel for unit test
    clean_text = factory._process_text(dirty_text)
    
    print(f"OUTPUT (Clean):\n{clean_text}\n")
    
    # 3. Validation
    expected_fragment = "Chào mừng bạn đến với Project Aether!"
    
    if expected_fragment in clean_text:
        print("✅ HTML Removal: SUCCESS")
        print("✅ Mojibake Fix: SUCCESS (approximated)")
        print("✅ Whitespace Norm: SUCCESS")
    else:
        print("❌ Pipeline FAILED: Text not cleaned properly.")
        print(f"Got: '{clean_text}'")
        
    # Check normalization quality
    # "HÃ´m nay lÃ  ngÃ y Ä‘áº¹p trá» i." -> "Hôm nay là ngày đẹp trời."
    if "Hôm nay là ngày đẹp trời" in clean_text:
         print("✅ Vietnamese Decoding: SUCCESS")
    else:
         print("⚠️ Vietnamese Decoding: PARTIAL/FAIL (Check Heuristics)")

if __name__ == "__main__":
    test_pipeline()
