import time
import numpy as np
import mlx.core as mx
import sys
import os
sys.path.append(os.path.abspath("src"))
from aether.dedup.minhash import MinHashLSH

def benchmark_minhash():
    print("⚔️  Titan Arena: MinHash Benchmark")
    
    # Setup
    minhash = MinHashLSH(num_perm=128)
    
    # Large document simulation (10,000 words ~ 100,000 characters)
    # This generates ~10,000 shingles
    doc_length = 10000
    text = " ".join([f"word{i}" for i in range(doc_length)])
    
    print(f"📄 Document Length: {len(text)} chars (~{doc_length} n-grams)")
    
    # Warn up
    _ = minhash.compute_signature(text[:100])
    
    # Run
    start_time = time.time()
    iterations = 100
    
    for _ in range(iterations):
        sig = minhash.compute_signature(text)
        
    duration = time.time() - start_time
    avg_time = duration / iterations
    
    print(f"🚀 Titan (MLX Vectorized): {avg_time*1000:.2f} ms per doc")
    print(f"   Throughput: {1/avg_time:.2f} docs/sec")
    
    # Note: We can't easily run the old loop code since I overwrote it, 
    # but typical python loop for 128 perms * 10000 shingles = 1.28M ops
    # Python does ~10-20M ops/sec simplistic -> ~100ms per doc.
    # MLX should be <5ms.

if __name__ == "__main__":
    benchmark_minhash()
