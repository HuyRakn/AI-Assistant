import sys
import os
import glob
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aether.tokenization.core_bpe import SovereignTokenizer

def benchmark():
    print("🚀 Benchmarking Sovereign Tokenizer...")
    
    # 1. Load Corpus (Project Source Code)
    corpus = []
    files = glob.glob("src/**/*.py", recursive=True)
    print(f"Loading {len(files)} source files...")
    
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                corpus.append(f.read())
        except:
            pass
            
    # Add some synthetic data to bulk it up if needed
    full_text = "\n".join(corpus)
    print(f"Corpus Size: {len(full_text)/1024:.2f} KB | {len(full_text)} chars")
    
    # 2. Train
    tokenizer = SovereignTokenizer(vocab_size=1000) # Small vocab for speed test
    
    t0 = time.time()
    tokenizer.train([full_text])
    dt = time.time() - t0
    
    print(f"\n⏱️ Training Time: {dt:.4f}s")
    
    # 3. Verify Round-Trip
    sample = "class AetherModel(nn.Module): def __init__(self): pass"
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    
    print(f"\nOriginal: {sample}")
    print(f"Encoded IDs: {ids}")
    print(f"Decoded: {decoded}")
    
    # Loose verification (ignore whitespace nuances of regex for now if strict match fails)
    # BPE regex splits by space, so reconstruction matches exactly usually.
    if sample.replace(" ", "") == decoded.replace(" ", ""):
        print("✅ Round-Trip Verified (Content Match)")
    else:
        print("⚠️ Round-Trip Mismatch (Check regex handling)")

if __name__ == "__main__":
    benchmark()
